"""LLM factory.

A single `get_llm()` entry point so the rest of the codebase never has to
think about model selection, thinking config, or auth. To swap the model
for an eval, you change one place (or set `CLAUDE_MODEL` in the env).
"""

from __future__ import annotations

from functools import lru_cache

from langchain_anthropic import ChatAnthropic

from .config import ANTHROPIC_API_KEY, LLM_MAX_TOKENS, MODEL_NAME


@lru_cache(maxsize=1)
def get_llm() -> ChatAnthropic:
    """Return a process-wide singleton `ChatAnthropic`.

    Configuration:
      * model: from `config.MODEL_NAME` (default `claude-opus-4-8`).
      * max_tokens: `config.LLM_MAX_TOKENS` — generous cap for reasoning.
      * thinking: adaptive — Claude decides when and how much to reason.
      * temperature: 1.0 — Anthropic's recommendation when thinking is on.

    Raises:
        RuntimeError: if `ANTHROPIC_API_KEY` is not set.
    """
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it in your shell or add it to a .env file. "
            "See .env.example."
        )

    return ChatAnthropic(
        model=MODEL_NAME,
        max_tokens=LLM_MAX_TOKENS,
        thinking={"type": "adaptive"},
        temperature=1.0,
    )


__all__ = ["get_llm"]
