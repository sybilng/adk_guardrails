"""
example_usage.py
================
Shows how to wire PolicyGuardPlugin into a real Google ADK multi-agent
setup (root agent + sub-agent via A2A).

Install deps:
    pip install google-adk

Set your key:
    export GOOGLE_API_KEY=...
    export DB_PASSWORD=my_real_db_password   # plugin reads this at startup
"""

import asyncio
import os

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types

from policy_guard_plugin import PolicyGuardPlugin, PolicyConfig


# ---------------------------------------------------------------------------
# Define some tools
# ---------------------------------------------------------------------------

def query_database(sql: str) -> dict:
    """Pretend database tool."""
    return {"rows": [{"id": 1, "name": "Alice"}]}


def send_report(recipient: str, body: str) -> dict:
    """Pretend email tool."""
    return {"sent": True, "to": recipient}


# ---------------------------------------------------------------------------
# Define agents
# ---------------------------------------------------------------------------

data_agent = LlmAgent(
    name="data_agent",
    model="gemini-2.0-flash",
    instruction="You query the database and return structured data.",
    tools=[FunctionTool(query_database)],
)

reporting_agent = LlmAgent(
    name="reporting_agent",
    model="gemini-2.0-flash",
    instruction="You format data into reports and send them.",
    tools=[FunctionTool(send_report)],
    sub_agents=[data_agent],   # A2A: reporting_agent delegates to data_agent
)


# ---------------------------------------------------------------------------
# Configure the guard
# ---------------------------------------------------------------------------

guard = PolicyGuardPlugin(config=PolicyConfig(
    # Runtime secrets — loaded from env so they are NEVER hardcoded
    secret_env_vars=["DB_PASSWORD", "API_KEY", "GOOGLE_API_KEY"],

    # Extra literals you know at startup
    secret_values=["hunter2"],

    # Only allow these tools to run
    allowed_tools={"query_database", "send_report"},

    # Dangerous SQL/shell keywords
    # (defaults already cover DROP TABLE, DELETE FROM, rm -rf, etc.)

    # In production: True. During debugging: set False to just log.
    block_on_violation=True,
))


# ---------------------------------------------------------------------------
# Wire the plugin into the Runner — ONE registration covers ALL agents
# ---------------------------------------------------------------------------

APP_NAME = "secure_reporting_app"
USER_ID = "user_1"
SESSION_ID = "session_1"

session_service = InMemorySessionService()

runner = Runner(
    agent=reporting_agent,
    app_name=APP_NAME,
    session_service=session_service,
    plugins=[guard],   # ← fires for reporting_agent AND data_agent
)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

async def main():
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )

    queries = [
        # ✅ Should succeed
        "Fetch all active users and send a report to manager@example.com",

        # ❌ Should be blocked — secret in request
        f"Connect with password {os.environ.get('DB_PASSWORD', 'hunter2')} and fetch data",

        # ❌ Should be blocked — dangerous SQL
        "Run the query: DELETE FROM users WHERE 1=1",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"USER: {query}")
        print("="*60)

        content = types.Content(role="user", parts=[types.Part(text=query)])
        async for event in runner.run_async(
            user_id=USER_ID, session_id=SESSION_ID, new_message=content
        ):
            if event.is_final_response() and event.content:
                print(f"AGENT: {event.content.parts[0].text}")

    # Print violation report
    print(f"\n{'='*60}")
    print("POLICY VIOLATION REPORT")
    print("="*60)
    violations = guard.get_violations()
    if not violations:
        print("No violations recorded.")
    for v in violations:
        status = "🔴 BLOCKED" if v.blocked else "🟡 LOGGED"
        print(f"{status} [{v.check}] @ {v.location}")
        print(f"         {v.detail}")


if __name__ == "__main__":
    asyncio.run(main())
