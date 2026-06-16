"""Tests for stock_analysis_agent.agents.base.BaseAgent."""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from stock_analysis_agent.agents.base import BaseAgent


class _NoopAgent(BaseAgent):
    """Minimal concrete subclass for testing base config behavior."""

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)


def test_base_agent_uses_default_system_prompt() -> None:
    """Spec test 1: BaseAgent() with no args must succeed and use the
    default system prompt constant."""
    agent = _NoopAgent()
    assert agent.system_prompt_value == "You are a helpful assistant."


def test_base_agent_accepts_custom_system_prompt() -> None:
    """A custom system_prompt must override the default."""
    agent = _NoopAgent(system_prompt="You are a finance expert.")
    assert agent.system_prompt_value == "You are a finance expert."


def test_base_agent_stores_model_config() -> None:
    """Model, temperature, max_tokens, max_retries, name must be stored."""
    agent = _NoopAgent(
        model="claude-opus-4-8",
        temperature=0.7,
        max_tokens=8192,
        max_retries=5,
        name="custom-name",
    )
    assert agent.model == "claude-opus-4-8"
    assert agent.temperature == 0.7
    assert agent.max_tokens == 8192
    assert agent.max_retries == 5
    assert agent.name == "custom-name"


def test_base_agent_name_defaults_to_class_name() -> None:
    """When name is not provided, the agent's name should default to the
    concrete subclass's __name__ (e.g. '_NoopAgent')."""
    agent = _NoopAgent()
    assert agent.name == "_NoopAgent"
