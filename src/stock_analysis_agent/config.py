"""Runtime configuration.

Reads the Anthropic API key, model name, and log level from environment
variables. Deliberately minimal — no pydantic-settings, no .env loader.
The user sets variables in their shell or via a `.env` file consumed
by their own tooling.
"""

from __future__ import annotations

import os
from typing import Final

# --- LLM --------------------------------------------------------------------

# Required: the Anthropic API key. The LLM factory will fail loudly if absent.
ANTHROPIC_API_KEY: Final[str | None] = os.environ.get("ANTHROPIC_API_KEY")

# Optional override; defaults to the latest Opus.
DEFAULT_MODEL: Final[str] = "claude-opus-4-8"
MODEL_NAME: Final[str] = os.environ.get("CLAUDE_MODEL", DEFAULT_MODEL)

# Generous cap so reasoning-heavy agentic loops don't get truncated mid-tool.
LLM_MAX_TOKENS: Final[int] = 16_000

# --- Logging ----------------------------------------------------------------

DEFAULT_LOG_LEVEL: Final[str] = "INFO"
LOG_LEVEL: Final[str] = os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
