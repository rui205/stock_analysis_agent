"""Configuration package for stock_analysis_agent.

Exposes the LLM settings dataclass and a module-level accessor
that loads from environment variables on first access.
"""
from stock_analysis_agent.conf.settings import (
    LLMSettings,
    load_llm_settings,
)

__all__ = [
    "LLMSettings",
    "load_llm_settings",
]
