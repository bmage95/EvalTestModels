# Cross-Session Context Fetching

## Overview
Enhanced the `get_user_state` tool to fetch preferences **directly from the database** across all sessions. The model can now explicitly call this tool to pull context on demand, without requiring system prompt modifications for each request.

## How It Works

### Function-Calling Pattern
- **Tool**: `get_user_state(tool_context)` 
- **When Model Uses It**: When asked about preferences, the model proactively calls this tool
- **Returns**: Both current session preferences AND aggregated context from all sessions
- **Benefit**: No wrapper needed; DB context is pulled only when needed

### Cross-Session Aggregation
The enhanced `get_user_state` function:

1. **Fetches Current Session**: Returns preferences stored in the active session
2. **Queries Database**: Retrieves all sessions for the user via `DatabaseSessionService`
3. **Aggregates Preferences**: Merges preferences from all sessions (respecting first-seen priority)
4. **Tracks Sources**: Notes which session each preference came from
5. **Returns Enriched Context**: Model sees both current + aggregated state

```python
return {
    "status": "success",
    "current_session_preferences": current_prefs,  # This session only
    "cross_session_context": {
        "aggregated_preferences": {...},  # All prefs from all sessions
        "session_sources": {...},         # Which session each pref came from
        "total_sessions": N,              # Total sessions for this user
    },
    "retrieved_at": "2026-01-08T...",
    "session_id": "..."
}
```

### Event Loop Handling
- If running in async context (ADK Runner): creates a task for async aggregation
- If running sync (CLI fallback): uses `asyncio.run()` to fetch cross-session data
- Gracefully degrades if aggregation fails: returns current session only

## Usage Example

**User**: "What are my preferences?"

**Agent Process**:
1. Recognizes the question about preferences
2. Calls `get_user_state` tool
3. Receives aggregated preferences from all sessions
4. Responds with full context: "Your favorite color is blue, favorite food is pizza, and you are 25 years old."

**Key Insight**: The model decides when to fetch context. The system prompt guides it, but the decision is dynamic—it only pulls from DB when needed.

## Advantages

| Approach | Pros | Cons |
|----------|------|------|
| **Prompt Stuffing** | Simple, stateless | Token overhead, no dynamic decisions |
| **Function-Calling** (Current) | Dynamic, DB-driven, low token cost | Requires tool implementation |
| **Wrapper/Middleware** | Automatic injection | Couples application logic |
| **Fine-tuning** | Baked-in knowledge | Expensive, static |

This implementation uses **function-calling** because:
- ✅ Model explicitly decides when to fetch
- ✅ Minimal token overhead (only fetches when asked)
- ✅ Cross-session awareness via database queries
- ✅ No prompt engineering per request
- ✅ Works with ADK's tool context pattern

## Implementation Details

**File**: `my_agent/agent.py`

Key functions:
- `get_session_service()`: Lazy-loads `DatabaseSessionService` with configured DB_URL
- `_aggregate_cross_session_state()`: Async function that queries all sessions and aggregates preferences
- `get_user_state()`: Tool function that wraps aggregation and handles event loop edge cases

**Updated Instructions**: Guide the model to call `get_user_state` when asked about preferences, and explain the structure of the response (current vs. aggregated).

## Testing

Run the persistent CLI to observe:

```bash
.venv\Scripts\python.exe my_agent/persistent_runner.py
```

1. **Session 1**: Ask "What are my preferences?" → Returns current session state
2. **Session 2**: Add a new preference (e.g., "my age is 25")
3. **Session 3**: Ask again → Returns all prefs from all sessions (blue, pizza, 25)

The aggregation happens transparently when the model calls the tool.

## Future Enhancements

- **Temporal Context**: Include session timestamps to understand preference evolution
- **Confidence Scoring**: Mark which preferences are most frequently referenced
- **Session Filtering**: Let the model call `get_user_state(session_id="...")` to fetch specific session context
- **Preference Conflicts**: Detect and surface when preferences from different sessions conflict
- **Semantic Search**: Index preferences in vector DB to support semantic similarity queries

## No Wrapper Required ✅

This solution avoids application-level wrappers by:
- Letting the agent call the tool directly
- Storing all context in the database
- Using ADK's `ToolContext` for state access
- Letting the model make the decision to fetch
