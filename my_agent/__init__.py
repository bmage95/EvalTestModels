"""Package init for ADK Web.

Exports a module-level `runner` (and `get_runner`) from `web_adapter` so
`adk web .\my_agent\` picks up the persistent DatabaseSessionService-backed
runner without needing the `--runner` flag.
"""

from . import agent  # noqa: F401
from .web_adapter import get_runner, runner  # noqa: F401

__all__ = ["agent", "get_runner", "runner"]
