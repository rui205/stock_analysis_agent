"""Tests for stock_analysis_agent.agent.deepresearch.DeepResearchAgent."""
from __future__ import annotations

from pathlib import Path

import pytest

from stock_analysis_agent.agent.deepresearch import (
    DeepResearchAgent,
    render_research_prompt,
)


def test_default_construction_uses_module_constants(tmp_path: Path) -> None:
    """No-arg construction uses DEFAULT_SYSTEM_PROMPT and max_retries=3."""
    from stock_analysis_agent.agent.deepresearch import DEFAULT_SYSTEM_PROMPT

    agent = DeepResearchAgent(cache_dir=tmp_path, cache_ttl=None)
    assert agent.system_prompt_value == DEFAULT_SYSTEM_PROMPT
    assert agent.max_retries == 3
    assert agent.cache_dir == tmp_path.resolve()


def test_custom_system_prompt_overrides_default(tmp_path: Path) -> None:
    agent = DeepResearchAgent(
        system_prompt="custom prompt", cache_dir=tmp_path, cache_ttl=None
    )
    assert agent.system_prompt_value == "custom prompt"


def test_kwargs_pass_through_to_base_agent(tmp_path: Path) -> None:
    """model, temperature, name flow through to BaseAgent properties."""
    agent = DeepResearchAgent(
        model="claude-opus-4-8",
        temperature=0.7,
        name="custom",
        cache_dir=tmp_path,
        cache_ttl=None,
    )
    assert agent.model == "claude-opus-4-8"
    assert agent.temperature == 0.7
    assert agent.name == "custom"


def test_cache_dir_expands_tilde(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`~` in a str cache_dir is expanded via Path.expanduser()."""
    monkeypatch.setenv("HOME", str(tmp_path))
    agent = DeepResearchAgent(cache_dir="~/my-cache", cache_ttl=None)
    assert agent.cache_dir == (tmp_path / "my-cache").resolve()


def test_cache_ttl_none_disables_expiration(tmp_path: Path) -> None:
    """cache_ttl=None means cache entries never expire."""
    agent = DeepResearchAgent(cache_dir=tmp_path, cache_ttl=None)
    agent._cache.set(site="https://a.test", query="q", text="cached")
    assert agent._cache.get(site="https://a.test", query="q") == "cached"
    assert agent.cache_ttl is None  # use public property


# ---------------------------------------------------------------------------
# tool exposure — load_skill / read_file always on; run_command opt-in
# ---------------------------------------------------------------------------


def test_default_tools_include_load_skill_read_file_web_search(tmp_path: Path) -> None:
    """Default tool set is load_skill + read_file + web_search (no run_command).

    These are the deep-research data-discovery surface:
      - ``load_skill`` — read a project-level SKILL.md (mx-* skills)
      - ``read_file`` — read the skill's output files / reference docs
      - ``web_search`` — the configured external search fan-out

    ``run_command`` is opt-in (see ``test_run_command_omitted_by_default``).
    """
    agent = DeepResearchAgent(cache_dir=tmp_path, cache_ttl=None)
    tool_names = {t.name for t in agent.tools}
    assert "load_skill" in tool_names
    assert "read_file" in tool_names
    assert "web_search" in tool_names
    assert "run_command" not in tool_names


def test_run_command_omitted_by_default(tmp_path: Path) -> None:
    """``run_command`` is opt-in: off by default — never exposed unless asked.

    The shell tool is a privilege escalation (lets the agent invoke
    arbitrary CLI programs, e.g. the mx-* skill scripts), so the safe
    default is to leave it out of the tool list.
    """
    agent = DeepResearchAgent(cache_dir=tmp_path, cache_ttl=None)
    assert "run_command" not in {t.name for t in agent.tools}


def test_include_shell_tool_adds_run_command(tmp_path: Path) -> None:
    """When ``include_shell_tool=True``, ``run_command`` joins the tool list."""
    agent = DeepResearchAgent(
        cache_dir=tmp_path, cache_ttl=None, include_shell_tool=True
    )
    tool_names = {t.name for t in agent.tools}
    assert "run_command" in tool_names
    # The other defaults are still present.
    assert "load_skill" in tool_names
    assert "read_file" in tool_names
    assert "web_search" in tool_names


def test_include_shell_tool_property(tmp_path: Path) -> None:
    """The ``include_shell_tool`` attribute reflects the constructor flag."""
    assert DeepResearchAgent(cache_dir=tmp_path, cache_ttl=None).include_shell_tool is False
    assert (
        DeepResearchAgent(
            cache_dir=tmp_path, cache_ttl=None, include_shell_tool=True
        ).include_shell_tool
        is True
    )


# ---------------------------------------------------------------------------
# symbol / dimensions injection
# ---------------------------------------------------------------------------


def test_render_research_prompt_injects_symbol_and_dimensions() -> None:
    template = "股票:<!-- STOCK --> 维度:<!-- DIMENSIONS -->"
    out = render_research_prompt(
        template, symbol="02319.HK", dimensions=["基本面", "估值"]
    )
    assert out == "股票:02319.HK 维度:基本面、估值"


def test_render_research_prompt_empty_values_render_empty() -> None:
    template = "股票:<!-- STOCK --> 维度:<!-- DIMENSIONS -->"
    out = render_research_prompt(template, symbol=None, dimensions=[])
    assert out == "股票: 维度:"


def test_symbol_and_dimensions_injected_when_system_prompt_given(tmp_path: Path) -> None:
    agent = DeepResearchAgent(
        system_prompt="<!-- STOCK --> / <!-- DIMENSIONS -->",
        symbol="600519.SH",
        dimensions=["消息面"],
        cache_dir=tmp_path,
        cache_ttl=None,
    )
    assert agent.system_prompt_value == "600519.SH / 消息面"


def test_symbol_and_dimensions_load_md_when_no_system_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import stock_analysis_agent.agent.deepresearch as dr

    monkeypatch.setattr(dr, "_PROMPT_FILE", tmp_path / "prompt.md")
    (tmp_path / "prompt.md").write_text(
        "研究:<!-- STOCK --> / <!-- DIMENSIONS -->", encoding="utf-8"
    )
    agent = DeepResearchAgent(
        symbol="AAPL",
        dimensions=["估值"],
        cache_dir=tmp_path,
        cache_ttl=None,
    )
    assert agent.system_prompt_value == "研究:AAPL / 估值"