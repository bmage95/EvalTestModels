"""Persistent Session Runner for Google ADK

This module configures a stateful agent with a database-backed session service
and provides a simple CLI loop to test persistence locally.

Based on: "Building Persistent Sessions with Google ADK" guide.
"""
import asyncio
import os
from datetime import datetime
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from google.adk.sessions import DatabaseSessionService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.llm_agent import Agent
from google.genai import types


# Load environment variables from .env (located in my_agent/.env)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


APP_NAME = os.getenv("APP_NAME", "personal_assistant")
USER_ID = os.getenv("USER_ID", "user_local")
DB_URL = os.getenv("DB_URL", "sqlite+aiosqlite:///./session_data.db")


def initial_state() -> Dict[str, Any]:
    """Default session state when creating a new session."""
    return {
        "preferences": {
            "favorite_color": "blue",
            "favorite_food": "pizza",
        }
    }


async def latest_preferences(session_service: DatabaseSessionService) -> Optional[Dict[str, Any]]:
    """Fetch preferences from the most recently listed session (if any)."""
    existing = await session_service.list_sessions(app_name=APP_NAME, user_id=USER_ID)
    sessions = getattr(existing, "sessions", []) if existing else []
    if not sessions:
        return None
    # Use the first listed session (ADK lists latest first); fall back defensively.
    candidate = sessions[0]
    try:
        sess = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=candidate.id)
        if sess and "preferences" in (sess.state or {}):
            return dict(sess.state.get("preferences", {}))
    except Exception:
        return None
    return None


async def sync_preferences_from_latest(
    session_service: DatabaseSessionService, session_id: str
) -> None:
    """Merge preferences from the latest session into the active session (missing keys only)."""
    seed = await latest_preferences(session_service)
    if not seed:
        return
    try:
        sess = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
        if not sess or sess.state is None:
            return
        prefs = sess.state.get("preferences", {}) or {}
        changed = False
        for k, v in seed.items():
            if k not in prefs:
                prefs[k] = v
                changed = True
        if changed:
            sess.state["preferences"] = prefs
            await session_service.update_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id,
                state=sess.state,
            )
            print("Synced preferences from latest session into this session.")
    except Exception as exc:  # noqa: BLE001
        print(f"Could not sync preferences from latest session: {exc}")


def get_user_state(tool_context: ToolContext) -> Dict[str, Any]:
    """Retrieve all user preferences from session state."""
    preferences = tool_context.state.get("preferences", {})
    return {
        "status": "success",
        "preferences": preferences,
        "retrieved_at": datetime.now().isoformat(),
    }


def update_preference(
    preference_key: str,
    preference_value: str,
    tool_context: ToolContext,
) -> Dict[str, Any]:
    """Update or add a user preference to the session state."""
    preferences = tool_context.state.get("preferences", {})
    preferences[preference_key] = preference_value
    tool_context.state["preferences"] = preferences
    return {
        "status": "success",
        "message": f"Updated {preference_key} to {preference_value}",
        "updated_preferences": preferences,
        "updated_at": datetime.now().isoformat(),
    }


def create_agent(app_name: str) -> Agent:
    """Create and configure the agent with tools and instructions."""
    return Agent(
        name=app_name,
        model="gemini-2.5-flash",
        instruction=(
            "You are a helpful assistant with access to persistent session state.\n\n"
            "GUIDELINES:\n"
            "- Use `get_user_state` to read preferences.\n"
            "- Use `update_preference(preference_key, preference_value)` to save new preferences.\n"
            "- Personalize responses using stored preferences.\n"
        ),
        tools=[get_user_state, update_preference],
    )


def create_runner(app_name: str, session_service: DatabaseSessionService, agent: Agent) -> Runner:
    """Create a runner to orchestrate the agent and session service."""
    return Runner(app_name=app_name, session_service=session_service, agent=agent)


def process_agent_event(event) -> Optional[str]:
    """Extract final model text from runner events."""
    if hasattr(event, "is_final_response") and event.is_final_response():
        if event.content and event.content.parts:
            return event.content.parts[0].text
    return None


async def ainvoke_message(runner: Runner, user_id: str, session_id: str, message_text: str):
    """Send a message to the agent and stream the response."""
    message = types.Content(role="user", parts=[types.Part(text=message_text)])
    print("\nAssistant: ", end="", flush=True)
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=message):
        response = process_agent_event(event)
        if response:
            print(response, end="", flush=True)


async def main():
    # Configure database-backed session service
    session_service = DatabaseSessionService(db_url=DB_URL)

    # Build agent and runner
    agent = create_agent(APP_NAME)
    runner = create_runner(APP_NAME, session_service, agent)

    # Reuse existing session or create a new one based on user choice
    existing = await session_service.list_sessions(app_name=APP_NAME, user_id=USER_ID)
    sessions = getattr(existing, "sessions", []) if existing else []

    session_id: str
    if sessions:
        print("Existing sessions:")
        for idx, s in enumerate(sessions):
            print(f"  [{idx}] {s.id}")
        choice = input("Select a session index or 'n' for new [0]: ").strip().lower()
        if choice in ("n", "new", "c", "create"):
            seed_prefs = await latest_preferences(session_service)
            seed_state = initial_state()
            if seed_prefs:
                seed_state["preferences"].update(seed_prefs)
            new_session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, state=seed_state)
            session_id = new_session.id
            print(f"Created new session (seeded from latest): {session_id}")
        else:
            try:
                pick = int(choice) if choice else 0
            except ValueError:
                pick = 0
            pick = max(0, min(pick, len(sessions) - 1))
            session_id = sessions[pick].id
            print(f"Using existing session: {session_id}")
    else:
        seed_state = initial_state()
        new_session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, state=seed_state)
        session_id = new_session.id
        print(f"Created new session: {session_id}")

    # Ensure the active session also sees preferences from the latest session
    await sync_preferences_from_latest(session_service, session_id)

    # Show current preferences
    current = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
    if current:
        print(f"Current preferences: {current.state.get('preferences', {})}")

    # Interactive loop
    try:
        while True:
            try:
                user_input = input("You: ")
            except EOFError:
                print("\nEOF received. Exiting and attempting to persist session...")
                try:
                    final = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
                    if final:
                        print(f"Saved. Final preferences: {final.state.get('preferences', {})}")
                except asyncio.CancelledError:
                    print("Session fetch cancelled during shutdown; state already persisted by runner.")
                except Exception as exc:  # noqa: BLE001
                    print(f"Could not fetch final session state: {exc}")
                break

            if user_input.lower() in ["exit", "quit"]:
                final = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
                if final:
                    print(f"\nSaved. Final preferences: {final.state.get('preferences', {})}")
                break
            if not user_input.strip():
                continue
            await ainvoke_message(runner, USER_ID, session_id, user_input)
            print("")
    except KeyboardInterrupt:
        print("\nInterrupted. Session persisted.")


if __name__ == "__main__":
    asyncio.run(main())
