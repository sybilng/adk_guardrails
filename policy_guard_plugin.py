"""
ADK Policy Guard Plugin
=======================
A deterministic, rule-based runtime guardrail Plugin for Google ADK.
Registered ONCE on the Runner — applies globally to every agent, tool,
and LLM call in the workflow (including A2A sub-agents).

Coverage:
  1. before_model_callback  → blocks secrets/PII leaking INTO the LLM
  2. after_model_callback   → blocks secrets leaking OUT of the LLM response
  3. before_tool_callback   → blocks dangerous tool calls (DROP TABLE, etc.)
  4. after_tool_callback    → blocks secrets leaking OUT of tool results
  5. before_agent_callback  → blocks secrets in A2A messages to sub-agents

Usage:
    from policy_guard_plugin import PolicyGuardPlugin, PolicyConfig

    guard = PolicyGuardPlugin(config=PolicyConfig(
        secret_values=["mypassword", os.environ["DB_PASS"]],
        secret_env_vars=["DB_PASSWORD", "API_KEY", "AWS_SECRET_ACCESS_KEY"],
    ))

    runner = Runner(
        agent=root_agent,
        app_name="my_app",
        session_service=session_service,
        plugins=[guard],          # <-- registered here, fires everywhere
    )
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

logger = logging.getLogger("policy_guard")


# ---------------------------------------------------------------------------
# Violation record
# ---------------------------------------------------------------------------

@dataclass
class Violation:
    check: str          # which check fired
    location: str       # where (prompt / tool_args / tool_result / a2a / llm_response)
    detail: str         # human-readable detail
    blocked: bool       # True = execution was halted


# ---------------------------------------------------------------------------
# Policy configuration
# ---------------------------------------------------------------------------

@dataclass
class PolicyConfig:
    # ── Secret detection ────────────────────────────────────────────────────
    # Literal secret values that must NEVER appear in any payload.
    # Add passwords, tokens, connection strings, etc. at runtime.
    secret_values: list[str] = field(default_factory=list)

    # Environment variable names whose values should be treated as secrets.
    secret_env_vars: list[str] = field(default_factory=list)

    # Additional regex patterns to flag as secrets (compiled at init).
    # Defaults cover common credential shapes.
    secret_patterns: list[str] = field(default_factory=lambda: [
        r"(?i)password\s*[:=]\s*\S+",          # password=xxx
        r"(?i)passwd\s*[:=]\s*\S+",
        r"(?i)secret\s*[:=]\s*\S+",
        r"(?i)api[_-]?key\s*[:=]\s*\S+",
        r"(?i)access[_-]?token\s*[:=]\s*\S+",
        r"(?i)bearer\s+[A-Za-z0-9\-._~+/]+=*", # Bearer tokens
        r"[A-Z0-9]{20,40}",                     # AWS-style keys
        r"sk-[A-Za-z0-9]{32,}",                 # OpenAI-style keys
        r"ghp_[A-Za-z0-9]{36}",                 # GitHub PAT
        r"-----BEGIN [A-Z ]+-----",             # PEM private keys
    ])

    # ── Tool call policy ────────────────────────────────────────────────────
    # SQL / shell keywords that must never appear in any tool argument.
    forbidden_keywords: list[str] = field(default_factory=lambda: [
        "DROP TABLE", "DROP DATABASE", "TRUNCATE", "DELETE FROM",
        "ALTER TABLE", "GRANT ALL", "REVOKE",
        "rm -rf", "sudo", "shutdown", "mkfs",
        "; DROP", "'; DROP", "\"; DROP",          # SQL-injection shapes
    ])

    # If set, ONLY these tool names are allowed. Everything else is blocked.
    # Leave empty to allow all tools (use forbidden_tools instead).
    allowed_tools: set[str] = field(default_factory=set)

    # Tool names that are explicitly blocked regardless of allowed_tools.
    forbidden_tools: set[str] = field(default_factory=set)

    # ── Behaviour flags ─────────────────────────────────────────────────────
    # When True a violation halts execution; when False it only logs.
    block_on_violation: bool = True

    # Log full payloads for debugging (never enable in production).
    debug_log_payloads: bool = False


# ---------------------------------------------------------------------------
# The Plugin
# ---------------------------------------------------------------------------

class PolicyGuardPlugin(BasePlugin):
    """
    Deterministic, rule-based runtime guardrail for Google ADK.

    Registered on the Runner so it fires for EVERY agent, sub-agent,
    tool call, and LLM interaction within that runner's scope.
    """

    name = "policy_guard"

    def __init__(self, config: Optional[PolicyConfig] = None) -> None:
        self.config = config or PolicyConfig()
        self.violations: list[Violation] = []

        # Build compiled secret patterns
        self._compiled_patterns: list[re.Pattern] = [
            re.compile(p) for p in self.config.secret_patterns
        ]

        # Collect literal secrets from env vars at startup
        for var in self.config.secret_env_vars:
            val = os.environ.get(var, "")
            if val:
                self.config.secret_values.append(val)

        # Pre-compile forbidden keyword patterns (case-insensitive)
        self._forbidden_kw_patterns: list[re.Pattern] = [
            re.compile(re.escape(kw), re.IGNORECASE)
            for kw in self.config.forbidden_keywords
        ]

        logger.info(
            "[PolicyGuard] Initialized | block=%s | secret_values=%d "
            "| secret_patterns=%d | forbidden_kw=%d | allowed_tools=%s",
            self.config.block_on_violation,
            len(self.config.secret_values),
            len(self._compiled_patterns),
            len(self._forbidden_kw_patterns),
            self.config.allowed_tools or "ALL",
        )

    # -----------------------------------------------------------------------
    # Public: inspect recorded violations
    # -----------------------------------------------------------------------

    def get_violations(self) -> list[Violation]:
        return list(self.violations)

    def clear_violations(self) -> None:
        self.violations.clear()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _to_text(self, obj: Any) -> str:
        """Flatten any object to a string for pattern matching."""
        if isinstance(obj, str):
            return obj
        try:
            return json.dumps(obj, default=str)
        except Exception:
            return str(obj)

    def _contains_secret(self, text: str) -> Optional[str]:
        """
        Return a description of the first secret found, or None.
        Checks literal values first (fast), then regex patterns.
        """
        # Literal secrets — exact substring search
        for secret in self.config.secret_values:
            if secret and secret in text:
                masked = secret[:2] + "***" + secret[-2:] if len(secret) > 4 else "***"
                return f"literal secret detected (masked: {masked})"

        # Regex patterns
        for pattern in self._compiled_patterns:
            m = pattern.search(text)
            if m:
                snippet = m.group(0)[:30]
                return f"pattern '{pattern.pattern[:40]}' matched '{snippet}…'"

        return None

    def _contains_forbidden_keyword(self, text: str) -> Optional[str]:
        """Return the first forbidden keyword found, or None."""
        for pattern in self._forbidden_kw_patterns:
            if pattern.search(text):
                return pattern.pattern  # original keyword string
        return None

    def _block_response(self, reason: str) -> LlmResponse:
        """Return a canned LlmResponse that short-circuits the pipeline."""
        msg = f"[PolicyGuard] Request blocked: {reason}"
        logger.warning(msg)
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=msg)],
            )
        )

    def _block_tool_response(self, reason: str) -> dict:
        """Return a canned tool result dict that short-circuits tool execution."""
        msg = f"[PolicyGuard] Tool call blocked: {reason}"
        logger.warning(msg)
        return {"error": msg, "blocked": True}

    def _record(self, check: str, location: str, detail: str, blocked: bool) -> None:
        v = Violation(check=check, location=location, detail=detail, blocked=blocked)
        self.violations.append(v)
        level = logging.ERROR if blocked else logging.WARNING
        logger.log(level, "[PolicyGuard][%s] %s @ %s | blocked=%s", check, detail, location, blocked)

    # -----------------------------------------------------------------------
    # Hook 1: before_model_callback
    # Fires before EVERY LLM call (including sub-agents in A2A flows).
    # Inspects the full prompt that is about to be sent to the LLM.
    # -----------------------------------------------------------------------

    def before_model_callback(
        self,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> Optional[LlmResponse]:
        """Block secrets leaking INTO the LLM prompt."""

        # Serialise the entire request to a single string for scanning
        payload_parts: list[str] = []
        if llm_request.contents:
            for content in llm_request.contents:
                for part in (content.parts or []):
                    if hasattr(part, "text") and part.text:
                        payload_parts.append(part.text)
        if llm_request.system_instruction:
            payload_parts.append(self._to_text(llm_request.system_instruction))

        full_text = "\n".join(payload_parts)

        if self.config.debug_log_payloads:
            logger.debug("[PolicyGuard] LLM prompt:\n%s", full_text[:500])

        hit = self._contains_secret(full_text)
        if hit:
            detail = f"Secret in LLM prompt: {hit}"
            self._record("SECRET_IN_PROMPT", "llm_request", detail, self.config.block_on_violation)
            if self.config.block_on_violation:
                return self._block_response(detail)

        return None  # allow

    # -----------------------------------------------------------------------
    # Hook 2: after_model_callback
    # Fires after the LLM responds — catches secrets the model echoes back.
    # -----------------------------------------------------------------------

    def after_model_callback(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> Optional[LlmResponse]:
        """Block secrets leaking OUT of LLM responses."""

        text = ""
        if llm_response.content and llm_response.content.parts:
            text = "\n".join(
                p.text for p in llm_response.content.parts if hasattr(p, "text") and p.text
            )

        hit = self._contains_secret(text)
        if hit:
            detail = f"Secret in LLM response: {hit}"
            self._record("SECRET_IN_LLM_RESPONSE", "llm_response", detail, self.config.block_on_violation)
            if self.config.block_on_violation:
                return self._block_response(detail)

        return None  # pass through

    # -----------------------------------------------------------------------
    # Hook 3: before_tool_callback
    # Fires before ANY tool is executed.
    # -----------------------------------------------------------------------

    def before_tool_callback(
        self,
        tool: BaseTool,
        args: dict[str, Any],
        tool_context: ToolContext,
    ) -> Optional[dict]:
        """
        Enforce tool allowlist/blocklist and scan args for:
          - Secrets
          - Dangerous SQL / shell keywords
        """
        tool_name = tool.name

        # ── Allowlist check ──────────────────────────────────────────────
        if self.config.allowed_tools and tool_name not in self.config.allowed_tools:
            detail = f"Tool '{tool_name}' not in allowed_tools allowlist"
            self._record("TOOL_NOT_ALLOWED", f"tool:{tool_name}", detail, True)
            return self._block_tool_response(detail)

        # ── Blocklist check ──────────────────────────────────────────────
        if tool_name in self.config.forbidden_tools:
            detail = f"Tool '{tool_name}' is explicitly forbidden"
            self._record("TOOL_FORBIDDEN", f"tool:{tool_name}", detail, True)
            return self._block_tool_response(detail)

        # ── Scan arguments ───────────────────────────────────────────────
        args_text = self._to_text(args)

        # Secrets in args
        hit = self._contains_secret(args_text)
        if hit:
            detail = f"Secret in args for tool '{tool_name}': {hit}"
            self._record("SECRET_IN_TOOL_ARGS", f"tool:{tool_name}", detail, self.config.block_on_violation)
            if self.config.block_on_violation:
                return self._block_tool_response(detail)

        # Dangerous keywords in args
        kw = self._contains_forbidden_keyword(args_text)
        if kw:
            detail = f"Forbidden keyword '{kw}' in args for tool '{tool_name}'"
            self._record("DANGEROUS_KEYWORD_IN_TOOL_ARGS", f"tool:{tool_name}", detail, self.config.block_on_violation)
            if self.config.block_on_violation:
                return self._block_tool_response(detail)

        logger.debug("[PolicyGuard] Tool '%s' approved", tool_name)
        return None  # allow

    # -----------------------------------------------------------------------
    # Hook 4: after_tool_callback
    # Fires after a tool returns its result — catches secrets in outputs.
    # -----------------------------------------------------------------------

    def after_tool_callback(
        self,
        tool: BaseTool,
        args: dict[str, Any],
        tool_context: ToolContext,
        tool_response: dict,
    ) -> Optional[dict]:
        """Scrub secrets from tool results before the LLM sees them."""

        result_text = self._to_text(tool_response)
        hit = self._contains_secret(result_text)
        if hit:
            detail = f"Secret in result from tool '{tool.name}': {hit}"
            self._record("SECRET_IN_TOOL_RESULT", f"tool:{tool.name}", detail, self.config.block_on_violation)
            if self.config.block_on_violation:
                # Replace the result entirely
                return {"error": f"[PolicyGuard] Tool result redacted: {detail}", "blocked": True}

        return None  # pass through

    # -----------------------------------------------------------------------
    # Hook 5: before_agent_callback
    # Fires before a sub-agent starts — this is the A2A boundary.
    # Inspect the invocation context's input message for secrets.
    # -----------------------------------------------------------------------

    def before_agent_callback(
        self,
        callback_context: CallbackContext,
    ) -> Optional[types.Content]:
        """
        A2A boundary guard: block secrets being passed to sub-agents.

        In ADK, when Agent A calls Agent B via A2A, before_agent_callback
        fires for Agent B. The incoming message is in callback_context.
        """
        # The incoming content for this agent invocation
        invocation: InvocationContext = callback_context._invocation_context  # noqa: SLF001

        text_to_check: list[str] = []

        # Check the latest user/agent message
        if invocation.user_content:
            for part in (invocation.user_content.parts or []):
                if hasattr(part, "text") and part.text:
                    text_to_check.append(part.text)

        full_text = "\n".join(text_to_check)

        hit = self._contains_secret(full_text)
        if hit:
            agent_name = callback_context.agent_name
            detail = f"Secret in A2A message to agent '{agent_name}': {hit}"
            self._record("SECRET_IN_A2A_MESSAGE", f"a2a:{agent_name}", detail, self.config.block_on_violation)
            if self.config.block_on_violation:
                blocked_msg = f"[PolicyGuard] A2A message blocked: {detail}"
                logger.error(blocked_msg)
                return types.Content(
                    role="model",
                    parts=[types.Part(text=blocked_msg)],
                )

        return None  # allow
