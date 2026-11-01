"""Compatibility front doors for other clients' wire protocols.

Currently one: :mod:`arena_api.compat.anthropic_server` serves the Anthropic
Messages API on loopback so Claude Code can run on Arena models.
"""

from __future__ import annotations

from .anthropic_server import (
    ArenaEngine,
    MessagesRequest,
    create_server,
    main,
    translate_request,
)

__all__ = [
    "ArenaEngine",
    "MessagesRequest",
    "create_server",
    "main",
    "translate_request",
]
