"""Tests for StockAnalysisAgent: tools and prompt contract.

The agent is intentionally schema-agnostic — there is no built-in default
prompt, so these tests focus on what the agent *does* guarantee:

- the right tools are exposed (``load_skill`` and ``read_file`` are
  always present; ``run_command`` is opt-in via ``include_shell_tool``)
- the caller's ``system_prompt`` is plumbed through verbatim
- a missing / empty ``system_prompt`` is rejected loudly

Note: ``get_stock_snapshot`` and ``web_search`` are no longer wired into
the agent (snapshot's ``market_data`` module is gone; web-search logic
moves back into ``agent.deepsearch``); tests covering their exposure
have been removed along with the related constructor parameters.
"""
from __future__ import annotations

import pytest

from stock_analysis_agent.agent.stock_analysis import StockAnalysisAgent


_TEST_PROMPT = "you are a test prompt for {symbol}"


def _agent(**overrides) -> StockAnalysisAgent:
    """Build a StockAnalysisAgent with the minimum required kwargs.

    Returns an agent instance for tests that don't care about the prompt
    contents. Tests that *do* care should pass ``system_prompt=`` directly.
    """
    return StockAnalysisAgent(system_prompt=_TEST_PROMPT, **overrides)


# ---------------------------------------------------------------------------
# tool exposure
# ---------------------------------------------------------------------------


def test_default_tools_are_load_skill_and_read_file() -> None:
    """The agent exposes ``load_skill`` and ``read_file`` by default.

    These are the sub-agent's data-discovery surface:
      - ``load_skill`` — load SKILL.md instructions
      - ``read_file`` — read arbitrary UTF-8 files under the project root

    Directory listing is done via ``run_command(command='ls', argv=[...])``
    when shell is enabled; no dedicated ``list_dir`` tool exists.

    ``run_command`` is opt-in (see ``test_run_command_omitted_by_default``);
    ``get_stock_snapshot`` / ``web_search`` are no longer wired.
    """
    tool_names = {t.name for t in _agent().tools}
    assert "load_skill" in tool_names
    assert "read_file" in tool_names
    assert "list_dir" not in tool_names, (
        "list_dir was removed — use run_command(command='ls', argv=[...])"
    )
    assert "get_stock_snapshot" not in tool_names
    assert "web_search" not in tool_names


# ---------------------------------------------------------------------------
# include_shell_tool — opt-in for CLI subprocess execution
# ---------------------------------------------------------------------------


def test_run_command_omitted_by_default() -> None:
    """``run_command`` is opt-in: off by default — never exposed unless asked.

    The shell tool is a privilege escalation (lets the agent invoke
    arbitrary CLI programs), so the safe default is to leave it out of
    the tool list. Tests that want it must pass ``include_shell_tool=True``.
    """
    assert "run_command" not in {t.name for t in _agent().tools}


def test_include_shell_tool_adds_run_command_to_tools() -> None:
    """When ``include_shell_tool=True``, ``run_command`` joins the tool list."""
    a = _agent(include_shell_tool=True)
    tool_names = {t.name for t in a.tools}
    assert "run_command" in tool_names
    # The other defaults are still present.
    assert "load_skill" in tool_names
    assert "read_file" in tool_names
    assert "list_dir" not in tool_names, (
        "list_dir was removed — use run_command(command='ls', argv=[...])"
    )


def test_include_shell_tool_property() -> None:
    """The ``include_shell_tool`` attribute reflects the constructor flag."""
    assert _agent().include_shell_tool is False
    assert _agent(include_shell_tool=True).include_shell_tool is True


# ---------------------------------------------------------------------------
# system_prompt — required, plumbed through verbatim, no default
# ---------------------------------------------------------------------------


def test_system_prompt_is_plumbed_through_verbatim() -> None:
    """The agent must pass ``system_prompt`` to the LLM as-is, with no mutation."""
    agent = StockAnalysisAgent(system_prompt="hello world {symbol}")
    assert agent.system_prompt_value == "hello world {symbol}"


def test_system_prompt_is_required() -> None:
    """``system_prompt`` has no default — the caller must own the schema."""
    with pytest.raises(TypeError):
        StockAnalysisAgent()  # type: ignore[call-arg]


def test_empty_system_prompt_is_rejected() -> None:
    """An empty string would silently send a blank instruction to the LLM."""
    with pytest.raises(ValueError, match="system_prompt"):
        StockAnalysisAgent(system_prompt="")


# ---------------------------------------------------------------------------
# recursion_limit — caps ReAct-style iteration depth
# ---------------------------------------------------------------------------


def test_default_recursion_limit_is_fifty() -> None:
    """Default is 50 — large enough for the bundled stock-analyst workflow
    (~8 tool calls plus intermediate LLM decisions). Subclasses can override."""
    assert _agent().recursion_limit == 50


def test_custom_recursion_limit_is_stored() -> None:
    """Caller may override the default; the value is exposed verbatim."""
    agent = StockAnalysisAgent(
        system_prompt=_TEST_PROMPT,
        recursion_limit=12,
    )
    assert agent.recursion_limit == 12


def test_recursion_limit_propagates_to_resolved_config() -> None:
    """``stream`` / ``astream`` must surface the default to the graph
    even when the caller passes ``config=None``."""
    agent = _agent()
    assert agent._resolve_config(None) == {"recursion_limit": 50}
