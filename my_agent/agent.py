import asyncio
import os
from datetime import datetime
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from google.adk.agents.llm_agent import Agent
from google.adk.tools.tool_context import ToolContext
from google.adk.sessions import DatabaseSessionService


# Load environment for cross-session DB access
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
APP_NAME = os.getenv("APP_NAME", "personal_assistant")
# Default USER_ID, but we will override from ToolContext when available (e.g., ADK Web UI input)
USER_ID = os.getenv("USER_ID", "user_local")
DB_URL = os.getenv("DB_URL", "sqlite+aiosqlite:///./session_data.db")


# Global session service for cross-session context fetching
_session_service: Optional[DatabaseSessionService] = None


def get_session_service() -> DatabaseSessionService:
    """Lazy-load the session service for cross-session queries."""
    global _session_service
    if _session_service is None:
        _session_service = DatabaseSessionService(db_url=DB_URL, echo=False)
    return _session_service


async def _aggregate_cross_session_state(
    current_session_id: str,
    user_id: str,
) -> Dict[str, Any]:
    """Query all sessions for this user and aggregate preferences.
    
    Returns preferences from the current session plus context from recent sessions.
    """
    service = get_session_service()
    try:
        # List all sessions for this user
        result = await service.list_sessions(app_name=APP_NAME, user_id=user_id)
        sessions = getattr(result, "sessions", []) if result else []
        
        # Aggregate preferences from all sessions (current first, then recent)
        aggregated = {}
        session_sources = {}  # Track which session each pref came from
        
        for session_meta in sessions:
            try:
                session = await service.get_session(
                    app_name=APP_NAME,
                    user_id=user_id,
                    session_id=session_meta.id
                )
                if session and session.state:
                    prefs = session.state.get("preferences", {}) or {}
                    for key, val in prefs.items():
                        if key not in aggregated:
                            aggregated[key] = val
                            session_sources[key] = session_meta.id
            except Exception:
                # Skip sessions that fail to load
                continue
        
        return {
            "aggregated_preferences": aggregated,
            "session_sources": session_sources,  # Which session each pref came from
            "total_sessions": len(sessions),
        }
    except Exception as e:
        return {
            "error": f"Could not aggregate cross-session state: {str(e)}",
            "fallback_preferences": {},
        }


def get_user_state(tool_context: ToolContext) -> Dict[str, Any]:
    """Retrieve user preferences from current session + cross-session context.
    
    Fetches from the database across all sessions for this user and returns
    an aggregated view. The model can call this to decide whether to pull
    preferences without modifying the system prompt per request.
    """
    current_prefs = tool_context.state.get("preferences", {})
    session_id = getattr(tool_context, "session_id", "unknown")
    # Prefer the user_id from ToolContext (ADK Web UI) over env default
    user_id = getattr(tool_context, "user_id", USER_ID)
    
    # Try to run the async aggregation
    cross_session_data = {"aggregated_preferences": {}, "session_sources": {}, "total_sessions": 0}
    
    try:
        # Try to get the current event loop (ADK runs in async context)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an async context but this is a sync function.
            # Try using asyncio.run_coroutine_threadsafe as fallback
            # But for simplicity, we'll just return current prefs and note the limitation
            import concurrent.futures
            
            # Create a new thread to run the async aggregation
            def run_aggregation():
                try:
                    return asyncio.run(_aggregate_cross_session_state(session_id, user_id))
                except Exception as e:
                    return {"error": str(e), "aggregated_preferences": {}, "session_sources": {}, "total_sessions": 0}
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_aggregation)
                cross_session_data = future.result(timeout=5)
        else:
            # No running loop, safe to use asyncio.run
            cross_session_data = asyncio.run(_aggregate_cross_session_state(session_id, user_id))
    except Exception as e:
        # Aggregation failed, fall back to current session
        cross_session_data = {
            "error": f"Could not fetch cross-session data: {str(e)}",
            "aggregated_preferences": current_prefs,
            "session_sources": {k: session_id for k in current_prefs},
            "total_sessions": 1,
        }
    
    return {
        "status": "success",
        "current_session_preferences": current_prefs,
        "cross_session_context": cross_session_data,
        "retrieved_at": datetime.now().isoformat(),
        "session_id": session_id,
        "user_id": user_id,
    }


def update_preference(
    preference_key: str,
    preference_value: str,
    tool_context: ToolContext,
) -> Dict[str, Any]:
    """Update or add a user preference in session state.

    The `ToolContext` is injected by ADK and provides access to `state`.
    """
    preferences = tool_context.state.get("preferences", {})
    preferences[preference_key] = preference_value
    tool_context.state["preferences"] = preferences

    return {
        "status": "success",
        "message": f"Updated {preference_key} to {preference_value}",
        "updated_preferences": preferences,
        "updated_at": datetime.now().isoformat(),
    }


root_agent = Agent(
    model="gemini-2.5-flash",
    name="root_agent",
    description="A helpful assistant that remembers user preferences across sessions.",
    instruction=(
        "You are a helpful personal assistant with access to user preferences "
        "stored across multiple sessions in a database.\n\n"
        "TOOL USAGE GUIDELINES:\n"
        "- Use `get_user_state` to fetch preferences from the current session AND "
        "aggregate preferences from all previous sessions. This tool pulls directly "
        "from the database, so you have access to cross-session context on demand.\n"
        "- When the user shares a new preference (e.g., 'my favorite food is sushi'), "
        "call `update_preference` with `preference_key` and `preference_value` to store it.\n"
        "- The response includes:\n"
        "  * 'current_session_preferences': preferences in this session only\n"
        "  * 'cross_session_context': aggregated preferences from all sessions + which session each came from\n"
        "- Personalize answers using both current and cross-session preferences.\n"
        "- If information is not present in state, say you don't know yet and offer to save it."
    ),
    tools=[get_user_state, update_preference],
)
