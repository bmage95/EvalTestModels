"""StatefulTest app entrypoint that re-exports the Python root_agent runner.
This lets ADK Web load the same persistent agent when the UI selects app name 'StatefulTest'.
"""
from ..agent import root_agent, get_user_state, update_preference
from ..web_adapter import get_runner, runner

__all__ = [
    "root_agent",
    "get_user_state",
    "update_preference",
    "get_runner",
    "runner",
]
