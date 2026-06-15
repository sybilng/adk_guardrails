"""
test_policy_guard_plugin.py
===========================
Deterministic unit tests for PolicyGuardPlugin.
No LLM, no AI judge — pure rule-based assertions.

Run with:
    pip install pytest
    pytest test_policy_guard_plugin.py -v
"""

import pytest
from unittest.mock import MagicMock, PropertyMock

from policy_guard_plugin import PolicyGuardPlugin, PolicyConfig, Violation


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def make_plugin(**kwargs) -> PolicyGuardPlugin:
    config = PolicyConfig(
        secret_values=["hunter2", "supersecrettoken"],
        secret_env_vars=[],            # skip env in unit tests
        **{"block_on_violation": True, **kwargs},
    )
    return PolicyGuardPlugin(config=config)


def make_llm_request(text: str):
    """Build a minimal mock LlmRequest."""
    part = MagicMock()
    part.text = text
    content = MagicMock()
    content.parts = [part]
    req = MagicMock()
    req.contents = [content]
    req.system_instruction = None
    return req


def make_llm_response(text: str):
    """Build a minimal mock LlmResponse."""
    part = MagicMock()
    part.text = text
    content = MagicMock()
    content.parts = [part]
    resp = MagicMock()
    resp.content = content
    return resp


def make_tool(name: str):
    tool = MagicMock()
    tool.name = name
    return tool


def make_callback_context(agent_name: str = "sub_agent", user_text: str = ""):
    """Build a minimal mock CallbackContext for agent hooks."""
    part = MagicMock()
    part.text = user_text
    user_content = MagicMock()
    user_content.parts = [part]

    invocation = MagicMock()
    invocation.user_content = user_content

    ctx = MagicMock()
    ctx.agent_name = agent_name
    ctx._invocation_context = invocation
    return ctx


def make_tool_context():
    return MagicMock()


# ===========================================================================
# 1. Secret detection — before_model_callback (LLM prompt)
# ===========================================================================

class TestBeforeModelCallback:

    def test_clean_prompt_passes(self):
        plugin = make_plugin()
        req = make_llm_request("What is the weather in Singapore?")
        result = plugin.before_model_callback(MagicMock(), req)
        assert result is None, "Clean prompt should not be blocked"
        assert plugin.get_violations() == []

    def test_literal_secret_in_prompt_blocked(self):
        plugin = make_plugin()
        req = make_llm_request("Connect to DB with password hunter2")
        result = plugin.before_model_callback(MagicMock(), req)
        assert result is not None, "Prompt with secret should be blocked"
        v = plugin.get_violations()
        assert len(v) == 1
        assert v[0].check == "SECRET_IN_PROMPT"
        assert v[0].blocked is True

    def test_second_literal_secret_blocked(self):
        plugin = make_plugin()
        req = make_llm_request("Token is supersecrettoken, use it to call the API")
        result = plugin.before_model_callback(MagicMock(), req)
        assert result is not None

    def test_regex_bearer_token_blocked(self):
        plugin = make_plugin()
        req = make_llm_request("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig")
        result = plugin.before_model_callback(MagicMock(), req)
        assert result is not None
        assert plugin.get_violations()[0].check == "SECRET_IN_PROMPT"

    def test_regex_api_key_pattern_blocked(self):
        plugin = make_plugin()
        req = make_llm_request("Set API_KEY=sk-abcdef1234567890abcdef1234567890ab")
        result = plugin.before_model_callback(MagicMock(), req)
        assert result is not None

    def test_observe_mode_logs_but_does_not_block(self):
        """block_on_violation=False → violation recorded but no block returned."""
        plugin = make_plugin(block_on_violation=False)
        req = make_llm_request("password=hunter2 connect to prod DB")
        result = plugin.before_model_callback(MagicMock(), req)
        assert result is None, "Observe mode should not block"
        assert len(plugin.get_violations()) == 1
        assert plugin.get_violations()[0].blocked is False

    def test_multiple_calls_accumulate_violations(self):
        plugin = make_plugin()
        for _ in range(3):
            plugin.before_model_callback(MagicMock(), make_llm_request("hunter2"))
        assert len(plugin.get_violations()) == 3

    def test_clear_violations(self):
        plugin = make_plugin()
        plugin.before_model_callback(MagicMock(), make_llm_request("hunter2"))
        plugin.clear_violations()
        assert plugin.get_violations() == []


# ===========================================================================
# 2. Secret detection — after_model_callback (LLM response)
# ===========================================================================

class TestAfterModelCallback:

    def test_clean_response_passes(self):
        plugin = make_plugin()
        resp = make_llm_response("The weather is sunny today.")
        result = plugin.after_model_callback(MagicMock(), resp)
        assert result is None

    def test_secret_in_response_blocked(self):
        plugin = make_plugin()
        resp = make_llm_response("Here is the password: hunter2")
        result = plugin.after_model_callback(MagicMock(), resp)
        assert result is not None
        assert plugin.get_violations()[0].check == "SECRET_IN_LLM_RESPONSE"


# ===========================================================================
# 3. Tool call enforcement — before_tool_callback
# ===========================================================================

class TestBeforeToolCallback:

    def test_safe_select_query_passes(self):
        plugin = make_plugin()
        tool = make_tool("db_query")
        args = {"query": "SELECT id, name FROM users WHERE active = 1"}
        result = plugin.before_tool_callback(tool, args, make_tool_context())
        assert result is None

    def test_drop_table_blocked(self):
        plugin = make_plugin()
        tool = make_tool("db_query")
        args = {"query": "DROP TABLE users"}
        result = plugin.before_tool_callback(tool, args, make_tool_context())
        assert result is not None
        assert result["blocked"] is True
        assert plugin.get_violations()[0].check == "DANGEROUS_KEYWORD_IN_TOOL_ARGS"

    def test_delete_from_blocked(self):
        plugin = make_plugin()
        tool = make_tool("db_query")
        args = {"query": "DELETE FROM orders WHERE 1=1"}
        result = plugin.before_tool_callback(tool, args, make_tool_context())
        assert result is not None

    def test_sql_injection_pattern_blocked(self):
        plugin = make_plugin()
        tool = make_tool("db_query")
        args = {"query": "SELECT * FROM users WHERE name='admin'; DROP TABLE users--"}
        result = plugin.before_tool_callback(tool, args, make_tool_context())
        assert result is not None

    def test_rm_rf_shell_blocked(self):
        plugin = make_plugin()
        tool = make_tool("shell_exec")
        args = {"command": "rm -rf /var/data"}
        result = plugin.before_tool_callback(tool, args, make_tool_context())
        assert result is not None

    def test_secret_in_tool_args_blocked(self):
        plugin = make_plugin()
        tool = make_tool("http_request")
        args = {"url": "https://api.example.com", "auth": "hunter2"}
        result = plugin.before_tool_callback(tool, args, make_tool_context())
        assert result is not None
        assert plugin.get_violations()[0].check == "SECRET_IN_TOOL_ARGS"

    def test_allowlist_blocks_unknown_tool(self):
        plugin = make_plugin(allowed_tools={"db_query", "send_email"})
        tool = make_tool("file_delete")
        result = plugin.before_tool_callback(tool, {}, make_tool_context())
        assert result is not None
        assert plugin.get_violations()[0].check == "TOOL_NOT_ALLOWED"

    def test_allowlist_permits_known_tool(self):
        plugin = make_plugin(allowed_tools={"db_query", "send_email"})
        tool = make_tool("db_query")
        args = {"query": "SELECT 1"}
        result = plugin.before_tool_callback(tool, args, make_tool_context())
        assert result is None

    def test_forbidden_tool_blocked_regardless_of_allowlist(self):
        plugin = make_plugin(forbidden_tools={"dangerous_tool"})
        tool = make_tool("dangerous_tool")
        result = plugin.before_tool_callback(tool, {}, make_tool_context())
        assert result is not None
        assert plugin.get_violations()[0].check == "TOOL_FORBIDDEN"

    def test_empty_args_passes(self):
        plugin = make_plugin()
        tool = make_tool("ping")
        result = plugin.before_tool_callback(tool, {}, make_tool_context())
        assert result is None


# ===========================================================================
# 4. Tool result scrubbing — after_tool_callback
# ===========================================================================

class TestAfterToolCallback:

    def test_clean_result_passes(self):
        plugin = make_plugin()
        tool = make_tool("db_query")
        result = plugin.after_tool_callback(tool, {}, make_tool_context(), {"rows": [1, 2, 3]})
        assert result is None

    def test_secret_in_result_blocked(self):
        plugin = make_plugin()
        tool = make_tool("db_query")
        tool_response = {"user": "admin", "password": "hunter2", "role": "superuser"}
        result = plugin.after_tool_callback(tool, {}, make_tool_context(), tool_response)
        assert result is not None
        assert result["blocked"] is True
        assert plugin.get_violations()[0].check == "SECRET_IN_TOOL_RESULT"


# ===========================================================================
# 5. A2A boundary — before_agent_callback
# ===========================================================================

class TestBeforeAgentCallback:

    def test_clean_a2a_message_passes(self):
        plugin = make_plugin()
        ctx = make_callback_context(agent_name="billing_agent", user_text="Fetch invoice #1234")
        result = plugin.before_agent_callback(ctx)
        assert result is None

    def test_secret_in_a2a_message_blocked(self):
        plugin = make_plugin()
        ctx = make_callback_context(
            agent_name="billing_agent",
            user_text="Use DB password hunter2 to pull customer records"
        )
        result = plugin.before_agent_callback(ctx)
        assert result is not None
        v = plugin.get_violations()
        assert v[0].check == "SECRET_IN_A2A_MESSAGE"
        assert "billing_agent" in v[0].location

    def test_bearer_token_in_a2a_blocked(self):
        plugin = make_plugin()
        ctx = make_callback_context(
            user_text="Call inventory API with Bearer ABCDEFGHIJ1234567890ABCDEFGHIJ12"
        )
        result = plugin.before_agent_callback(ctx)
        assert result is not None


# ===========================================================================
# 6. Violation record integrity
# ===========================================================================

class TestViolationRecords:

    def test_violation_fields_populated(self):
        plugin = make_plugin()
        req = make_llm_request("hunter2 is the password")
        plugin.before_model_callback(MagicMock(), req)
        v: Violation = plugin.get_violations()[0]
        assert v.check == "SECRET_IN_PROMPT"
        assert v.location == "llm_request"
        assert "secret" in v.detail.lower()
        assert v.blocked is True

    def test_no_violations_on_clean_run(self):
        plugin = make_plugin()
        plugin.before_model_callback(MagicMock(), make_llm_request("Hello world"))
        plugin.after_model_callback(MagicMock(), make_llm_response("Hi there!"))
        plugin.before_tool_callback(make_tool("ping"), {}, make_tool_context())
        plugin.after_tool_callback(make_tool("ping"), {}, make_tool_context(), {"ok": True})
        plugin.before_agent_callback(make_callback_context(user_text="Do stuff"))
        assert plugin.get_violations() == []
