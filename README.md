# ADK Policy Guard Plugin

A deterministic, rule-based runtime guardrail plugin for [Google ADK](https://google.github.io/adk-docs/). Register it once on the `Runner` and it automatically applies to every agent, tool call, and LLM interaction in the workflow — including A2A sub-agents.

## Features

| Hook | What it guards |
|------|----------------|
| `before_model_callback` | Blocks secrets/PII leaking **into** LLM prompts |
| `after_model_callback` | Blocks secrets leaking **out of** LLM responses |
| `before_tool_callback` | Blocks dangerous tool calls (SQL injection, `rm -rf`, etc.) and enforces tool allowlists/blocklists |
| `after_tool_callback` | Redacts secrets from tool results before the LLM sees them |
| `before_agent_callback` | Blocks secrets in A2A messages passed to sub-agents |

## Design: Why ADK Plugin, Not Callbacks

The ADK Plugin system is the right architecture here — registered once on the `Runner`, it applies globally to every agent.

| | Agent Callback | Plugin (what we use) |
|---|---|---|
| Scope | Single agent only | Every agent + sub-agent in the runner |
| A2A coverage | Agent B misses Agent A's callbacks | Fires for all agents |
| Registration | Per-agent | Once on `Runner` |

Since the concern is A2A — where Agent A passes a message to Agent B — a callback on Agent A won't help. The Plugin fires **at the boundary of every agent**, which is exactly the enforcement point needed.

## How It Works

### Execution Flow

```
User Message
     │
     ▼
[before_agent_callback]  ← Hook 5: A2A boundary guard
     │
     ▼
[before_model_callback]  ← Hook 1: block secrets going INTO the LLM
     │
     ▼
   LLM Call
     │
     ▼
[after_model_callback]   ← Hook 2: block secrets echoed back by LLM
     │
     ▼
[before_tool_callback]   ← Hook 3: block DROP TABLE, secret in args, banned tools
     │
     ▼
   Tool Execution
     │
     ▼
[after_tool_callback]    ← Hook 4: redact secrets from tool results before LLM sees them
```

### Architecture

```
┌─────────────────────────────────────────────────┐
│                  Your Agent                      │
│                                                  │
│  Decision  ──►  [ POLICY INTERCEPTOR ]  ──►  Tool Execution
│                        │                         │
│                   Block / Allow                  │
│                   Log / Alert                    │
└─────────────────────────────────────────────────┘
          ▲                              ▲
    LLM Prompt                    A2A Message
    Scanner                       Boundary Check
```

## Installation

```bash
pip install google-adk
```

Copy `policy_guard_plugin.py` into your project.

## Quick Start

```python
from policy_guard_plugin import PolicyGuardPlugin, PolicyConfig
from google.adk.runners import Runner

guard = PolicyGuardPlugin(config=PolicyConfig(
    secret_env_vars=["DB_PASSWORD", "API_KEY"],
    secret_values=["my-hardcoded-token"],
    allowed_tools={"query_database", "send_report"},
    block_on_violation=True,
))

runner = Runner(
    agent=root_agent,
    app_name="my_app",
    session_service=session_service,
    plugins=[guard],  # one registration covers all agents
)
```

## Configuration

`PolicyConfig` accepts the following options:

### Secret Detection

| Field | Type | Description |
|-------|------|-------------|
| `secret_values` | `list[str]` | Literal secret strings that must never appear in any payload |
| `secret_env_vars` | `list[str]` | Env var names whose values are treated as secrets (read at startup) |
| `secret_patterns` | `list[str]` | Regex patterns to detect credentials (defaults cover common shapes) |

Default patterns detect: `password=`, `secret=`, `api_key=`, Bearer tokens, AWS-style keys, OpenAI `sk-` keys, GitHub PATs, and PEM private keys.

### Tool Policy

| Field | Type | Description |
|-------|------|-------------|
| `allowed_tools` | `set[str]` | If non-empty, only these tool names are permitted |
| `forbidden_tools` | `set[str]` | Tool names that are always blocked |
| `forbidden_keywords` | `list[str]` | SQL/shell keywords blocked in tool arguments |

Default forbidden keywords: `DROP TABLE`, `DROP DATABASE`, `TRUNCATE`, `DELETE FROM`, `ALTER TABLE`, `GRANT ALL`, `REVOKE`, `rm -rf`, `sudo`, `shutdown`, `mkfs`, and SQL injection patterns.

### Behaviour

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `block_on_violation` | `bool` | `True` | Halt execution on violation; set `False` to log only |
| `debug_log_payloads` | `bool` | `False` | Log full payloads for debugging (never use in production) |

## Inspecting Violations

```python
violations = guard.get_violations()
for v in violations:
    print(v.check)     # e.g. SECRET_IN_PROMPT
    print(v.location)  # e.g. llm_request, tool:db_query, a2a:sub_agent
    print(v.detail)    # human-readable description
    print(v.blocked)   # True if execution was halted

guard.clear_violations()  # reset between sessions
```

### Violation check names

- `SECRET_IN_PROMPT` — secret found in LLM input
- `SECRET_IN_LLM_RESPONSE` — secret found in LLM output
- `SECRET_IN_TOOL_ARGS` — secret found in tool call arguments
- `SECRET_IN_TOOL_RESULT` — secret found in tool result
- `SECRET_IN_A2A_MESSAGE` — secret found in A2A message to sub-agent
- `DANGEROUS_KEYWORD_IN_TOOL_ARGS` — forbidden SQL/shell keyword in tool args
- `TOOL_NOT_ALLOWED` — tool not in `allowed_tools` allowlist
- `TOOL_FORBIDDEN` — tool explicitly listed in `forbidden_tools`

## Running the Example

```bash
export GOOGLE_API_KEY=...
export DB_PASSWORD=my_real_db_password

python example_usage.py
```

The example runs three queries against a multi-agent setup (root agent + sub-agent via A2A):
1. A clean query that succeeds
2. A query containing a secret — blocked
3. A query with a `DELETE FROM` statement — blocked

## Running Tests

```bash
pip install pytest
pytest test_policy_guard_plugin.py -v
```

Tests are fully deterministic — no LLM or network calls required.

## Known ADK Caveat

Plugin callbacks (`before_agent_callback`, `before_model_callback`, `after_model_callback`) may not fire when registered on `InMemoryRunner` in some ADK versions — only `before_tool_callback` and `after_tool_callback` reliably fire when passed directly to an agent. Workaround: pass callbacks both to the **agent** and the **plugin** on the runner, or use ADK >= 1.7.0 where the Plugin system is stable.

The test suite mocks all ADK types and runs independently of this issue.

## Project Structure

```
policy_guard_plugin.py      # Plugin implementation
example_usage.py            # Multi-agent usage example
test_policy_guard_plugin.py # Unit tests
```
