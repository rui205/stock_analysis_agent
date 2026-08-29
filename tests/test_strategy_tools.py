"""Tests for stock_analysis_agent.tools.strategy.

Covers the static parts — ``_list_strategy_names``,
``_parse_strategy_frontmatter``, and ``load_strategy``. The dynamic
``run_analyze_stock`` tool (which embeds a subagent) is tested with
``monkeypatch`` against :class:`MagicMock`.

As of the "no-schema pass-through" refactor, ``run_analyze_stock``
returns the sub-agent's Markdown output verbatim — no JSON parsing,
no schema validation, no remapping.
"""
from __future__ import annotations

from pathlib import Path
import textwrap
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from stock_analysis_agent.tools.strategy import (
    _list_strategy_names,
    _parse_strategy_frontmatter,
    load_strategy,
    run_analyze_stock,
    run_deepresearch,
    run_technical_capital,
)


@pytest.fixture(autouse=True)
def _isolate_subagent_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point the sub-agent report cache at a per-test tmp dir.

    Prevents cross-test cache hits and keeps tests from writing to the
    user's real ``~/.cache``.
    """
    import stock_analysis_agent.tools.strategy as mod
    from stock_analysis_agent.memory.file_cache import _FileCache

    monkeypatch.setattr(
        mod, "_subagent_cache", _FileCache(tmp_path, ttl_seconds=60.0)
    )


class TestListStrategyNames:
    def test_returns_alphabetical_stems(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "zeta.md").write_text("---\nname: z\n---\n", encoding="utf-8")
        (tmp_path / "alpha.md").write_text("---\nname: a\n---\n", encoding="utf-8")
        (tmp_path / "ignore.txt").write_text("not markdown", encoding="utf-8")
        monkeypatch.setattr(
            "stock_analysis_agent.tools.strategy._STRATEGIES_DIR", tmp_path
        )
        assert _list_strategy_names() == ("alpha", "zeta")

    def test_empty_dir_returns_empty_tuple(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "stock_analysis_agent.tools.strategy._STRATEGIES_DIR", tmp_path
        )
        assert _list_strategy_names() == ()


class TestParseStrategyFrontmatter:
    def test_single_line_fields(self) -> None:
        text = textwrap.dedent("""\
            ---
            name: foo
            version: "2"
            description: short description
            ---

            body
        """)
        fm = _parse_strategy_frontmatter(text)
        assert fm == {"name": "foo", "version": "2", "description": "short description"}

    def test_multiline_description_block(self) -> None:
        text = textwrap.dedent("""\
            ---
            name: foo
            description: |
              first line
              second line
            ---

            body
        """)
        fm = _parse_strategy_frontmatter(text)
        assert fm["name"] == "foo"
        assert "first line second line" in fm["description"]

    def test_missing_frontmatter_returns_empty(self) -> None:
        assert _parse_strategy_frontmatter("no fence here\n") == {}

    def test_unknown_keys_are_ignored(self) -> None:
        text = "---\nname: x\nfoo: bar\n---\nbody\n"
        fm = _parse_strategy_frontmatter(text)
        assert fm == {"name": "x"}


class TestLoadStrategyTool:
    def test_returns_full_file_text(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "value.md").write_text(
            "---\nname: value\n---\n\n# Value\n", encoding="utf-8"
        )
        monkeypatch.setattr(
            "stock_analysis_agent.tools.strategy._STRATEGIES_DIR", tmp_path
        )
        out = load_strategy.invoke({"name": "value"})
        assert "name: value" in out
        assert "# Value" in out

    def test_unknown_name_raises_with_available_list(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "alpha.md").write_text("---\nname: alpha\n---\n", encoding="utf-8")
        monkeypatch.setattr(
            "stock_analysis_agent.tools.strategy._STRATEGIES_DIR", tmp_path
        )
        with pytest.raises(FileNotFoundError) as exc:
            load_strategy.invoke({"name": "missing"})
        assert "missing" in str(exc.value)
        assert "alpha" in str(exc.value)


class TestRunAnalyzeStockTool:
    def test_success_returns_markdown_verbatim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Sub-agent streams Markdown chunks; the wrapper just concatenates
        # them and returns verbatim — no parsing, no rewriting.
        md_streamed = "# StockAnalysis — 600519.SH\n\n买入。\n"
        fake_events = [
            {"event": "on_chat_model_stream", "data": {"chunk": AIMessage(content="# StockAnalysis — 600519.SH\n\n")}},
            {"event": "on_chat_model_stream", "data": {"chunk": AIMessage(content="买入。\n")}},
        ]
        fake_sub = MagicMock()
        fake_sub.stream.return_value = iter(fake_events)
        fake_cls = MagicMock(return_value=fake_sub)
        import stock_analysis_agent.tools.strategy as mod
        monkeypatch.setattr(mod, "StockAnalysisAgent", fake_cls)
        out = run_analyze_stock.invoke({"symbol": "600519.SH"})
        # Verbatim: nothing added, nothing removed, no JSON validation.
        assert out == md_streamed
        fake_cls.assert_called_once()
        kwargs = fake_cls.call_args.kwargs
        assert kwargs["include_shell_tool"] is True

    def test_tool_failure_returns_error_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from stock_analysis_agent.agent.exceptions import ToolExecutionError
        fake_sub = MagicMock()
        fake_sub.stream.side_effect = ToolExecutionError("simulated")
        fake_cls = MagicMock(return_value=fake_sub)
        import stock_analysis_agent.tools.strategy as mod
        monkeypatch.setattr(mod, "StockAnalysisAgent", fake_cls)
        out = run_analyze_stock.invoke({"symbol": "000001.SZ"})
        assert out.startswith("[ERROR]")
        assert "analyze_stock" in out
        assert "simulated" in out

    def test_markdown_with_embedded_curly_braces_passes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Embedded JSON-ish text is NOT parsed by the wrapper — it just
        # streams through as markdown.
        body = "raw analysis content {verdict: buy} more text"
        fake_events = [
            {"event": "on_chat_model_stream", "data": {"chunk": AIMessage(content=body)}},
        ]
        fake_sub = MagicMock()
        fake_sub.stream.return_value = iter(fake_events)
        fake_cls = MagicMock(return_value=fake_sub)
        import stock_analysis_agent.tools.strategy as mod
        monkeypatch.setattr(mod, "StockAnalysisAgent", fake_cls)
        out = run_analyze_stock.invoke({"symbol": "000001.SZ"})
        assert out == body
        assert out.startswith("raw")


class TestRunAnalyzeStockShellDefault:
    """The embedded sub-agent always runs with ``run_command``.

    Its ``stock-analysis`` workflow needs shell access to execute the
    mx-* skill scripts and publish the report to Feishu — the sub-agent
    no longer inherits the orchestrator's opt-in flag.
    """

    def test_subagent_runs_with_shell_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import stock_analysis_agent.tools.strategy as mod

        fake_sub = MagicMock()
        fake_sub.stream.return_value = iter([])
        fake_cls = MagicMock(return_value=fake_sub)
        monkeypatch.setattr(mod, "StockAnalysisAgent", fake_cls)
        run_analyze_stock.invoke({"symbol": "06049.HK"})
        fake_cls.assert_called_once()
        assert fake_cls.call_args.kwargs["include_shell_tool"] is True


class TestRunAnalyzeStockRecursionBudget:
    """Regression: shell-enabled sub-agent runs exhaust a 50-step budget.

    With ``run_command`` wired in, the bundled ``stock-analysis``
    workflow performs many rounds (each data fetch = ``run_command`` +
    ``read_file`` ≈ 4 graph steps, plus skill loads and reasoning). A
    real run collected seven mx-* data files and died before the
    screener step — the sub-agent hit its recursion limit mid-workflow,
    the resulting ``GraphRecursionError`` escaped ``run_analyze_stock``
    (it only caught ``ToolExecutionError``), and the orchestrator's
    middleware aborted the whole pipeline with
    ``Tool 'run_analyze_stock' failed: ...`` (exit code 3).
    """

    def test_subagent_budget_matches_standalone_cli_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sub-agent gets the same budget ``script.analyze_stock``
        uses for the identical workflow (``--recursion-limit`` default
        100), not the constructor default of 50."""
        import stock_analysis_agent.tools.strategy as mod

        fake_sub = MagicMock()
        fake_sub.stream.return_value = iter([])
        fake_cls = MagicMock(return_value=fake_sub)
        monkeypatch.setattr(mod, "StockAnalysisAgent", fake_cls)
        run_analyze_stock.invoke({"symbol": "06049.HK"})
        fake_cls.assert_called_once()
        assert fake_cls.call_args.kwargs["recursion_limit"] == 100

    def test_recursion_exhaustion_returns_error_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A sub-agent that dies on recursion exhaustion degrades to the
        soft ``[ERROR]`` contract instead of escaping and aborting the
        orchestrator run."""
        import stock_analysis_agent.tools.strategy as mod

        fake_sub = MagicMock()
        fake_sub.stream.side_effect = RecursionError("simulated budget exhaustion")
        fake_cls = MagicMock(return_value=fake_sub)
        monkeypatch.setattr(mod, "StockAnalysisAgent", fake_cls)
        out = run_analyze_stock.invoke({"symbol": "06049.HK"})
        assert out.startswith("[ERROR]")
        assert "analyze_stock" in out
        assert "simulated budget exhaustion" in out

    def test_graph_recursion_error_returns_error_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same contract for langgraph's concrete exception type — the
        one actually raised when the graph runs out of steps."""
        langgraph_errors = pytest.importorskip("langgraph.errors")
        import stock_analysis_agent.tools.strategy as mod

        fake_sub = MagicMock()
        fake_sub.stream.side_effect = langgraph_errors.GraphRecursionError(
            "Recursion limit of 50 reached without hitting a stop condition."
        )
        fake_cls = MagicMock(return_value=fake_sub)
        monkeypatch.setattr(mod, "StockAnalysisAgent", fake_cls)
        out = run_analyze_stock.invoke({"symbol": "06049.HK"})
        assert out.startswith("[ERROR]")
        assert "Recursion limit of 50" in out


class TestRunDeepResearchTool:
    def test_success_returns_markdown_verbatim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        md_streamed = "# DeepResearch — 600519.SH\n\nROE 三年均值 12%。\n"
        fake_events = [
            {"event": "on_chat_model_stream", "data": {"chunk": AIMessage(content="# DeepResearch — 600519.SH\n\n")}},
            {"event": "on_chat_model_stream", "data": {"chunk": AIMessage(content="ROE 三年均值 12%。\n")}},
        ]
        fake_sub = MagicMock()
        fake_sub.stream.return_value = iter(fake_events)
        fake_cls = MagicMock(return_value=fake_sub)
        import stock_analysis_agent.agent.deepresearch as dr
        monkeypatch.setattr(dr, "DeepResearchAgent", fake_cls)
        out = run_deepresearch.invoke({"symbol": "600519.SH", "dimensions": ["盈利质量-ROE"]})
        assert out == md_streamed
        fake_cls.assert_called_once()
        kwargs = fake_cls.call_args.kwargs
        assert kwargs["symbol"] == "600519.SH"
        assert kwargs["dimensions"] == ["盈利质量-ROE"]
        assert kwargs["include_shell_tool"] is True
        assert kwargs["recursion_limit"] == 100

    def test_tool_failure_returns_error_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from stock_analysis_agent.agent.exceptions import ToolExecutionError
        fake_sub = MagicMock()
        fake_sub.stream.side_effect = ToolExecutionError("simulated")
        fake_cls = MagicMock(return_value=fake_sub)
        import stock_analysis_agent.agent.deepresearch as dr
        monkeypatch.setattr(dr, "DeepResearchAgent", fake_cls)
        out = run_deepresearch.invoke({"symbol": "000001.SZ", "dimensions": ["财务稳健"]})
        assert out.startswith("[ERROR]")
        assert "deepresearch" in out
        assert "simulated" in out

    def test_recursion_exhaustion_returns_error_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_sub = MagicMock()
        fake_sub.stream.side_effect = RecursionError("simulated budget exhaustion")
        fake_cls = MagicMock(return_value=fake_sub)
        import stock_analysis_agent.agent.deepresearch as dr
        monkeypatch.setattr(dr, "DeepResearchAgent", fake_cls)
        out = run_deepresearch.invoke({"symbol": "06049.HK", "dimensions": ["估值"]})
        assert out.startswith("[ERROR]")
        assert "deepresearch" in out


class TestRunDeepResearchShellDefault:
    """The embedded deep-research sub-agent always runs with ``run_command``."""

    def test_subagent_runs_with_shell_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import stock_analysis_agent.agent.deepresearch as dr

        fake_sub = MagicMock()
        fake_sub.stream.return_value = iter([])
        fake_cls = MagicMock(return_value=fake_sub)
        monkeypatch.setattr(dr, "DeepResearchAgent", fake_cls)
        run_deepresearch.invoke({"symbol": "06049.HK", "dimensions": ["基本面"]})
        fake_cls.assert_called_once()
        assert fake_cls.call_args.kwargs["include_shell_tool"] is True


# ---------------------------------------------------------------------------
# run_technical_capital — the technical + capital-flow sub-agent
# ---------------------------------------------------------------------------


class TestRunTechnicalCapitalTool:
    """The technical + capital-flow sub-agent is a ``StockAnalysisAgent``
    driven by the ``technical-capital`` persona (not ``DeepResearchAgent``)."""

    def test_success_returns_markdown_verbatim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        md_streamed = "# TechnicalCapital — 600519.SH\n\n站上20日线,主力净流入。\n"
        fake_events = [
            {"event": "on_chat_model_stream", "data": {"chunk": AIMessage(content="# TechnicalCapital — 600519.SH\n\n")}},
            {"event": "on_chat_model_stream", "data": {"chunk": AIMessage(content="站上20日线,主力净流入。\n")}},
        ]
        fake_sub = MagicMock()
        fake_sub.stream.return_value = iter(fake_events)
        fake_cls = MagicMock(return_value=fake_sub)
        import stock_analysis_agent.tools.strategy as mod
        monkeypatch.setattr(mod, "StockAnalysisAgent", fake_cls)
        out = run_technical_capital.invoke({"symbol": "600519.SH"})
        assert out == md_streamed
        fake_cls.assert_called_once()
        kwargs = fake_cls.call_args.kwargs
        assert kwargs["include_shell_tool"] is True
        assert kwargs["recursion_limit"] == 100

    def test_tool_failure_returns_error_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from stock_analysis_agent.agent.exceptions import ToolExecutionError
        fake_sub = MagicMock()
        fake_sub.stream.side_effect = ToolExecutionError("simulated")
        fake_cls = MagicMock(return_value=fake_sub)
        import stock_analysis_agent.tools.strategy as mod
        monkeypatch.setattr(mod, "StockAnalysisAgent", fake_cls)
        out = run_technical_capital.invoke({"symbol": "000001.SZ"})
        assert out.startswith("[ERROR]")
        assert "technical_capital" in out
        assert "simulated" in out

    def test_recursion_exhaustion_returns_error_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_sub = MagicMock()
        fake_sub.stream.side_effect = RecursionError("simulated budget exhaustion")
        fake_cls = MagicMock(return_value=fake_sub)
        import stock_analysis_agent.tools.strategy as mod
        monkeypatch.setattr(mod, "StockAnalysisAgent", fake_cls)
        out = run_technical_capital.invoke({"symbol": "06049.HK"})
        assert out.startswith("[ERROR]")
        assert "technical_capital" in out


def test_run_technical_capital_returns_cached_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """technical-capital reports are cached too, keyed by symbol."""
    import stock_analysis_agent.tools.strategy as st

    query = st._cache_query("600887.SH", dimensions=None, shell=True)
    st._subagent_cache.set(site="technical_capital", query=query, text="cached technical")

    class _ShouldNotRun:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("sub-agent must not be constructed on cache hit")

    monkeypatch.setattr(st, "StockAnalysisAgent", _ShouldNotRun)
    out = run_technical_capital.invoke({"symbol": "600887.SH"})
    assert out == "cached technical"


# ---------------------------------------------------------------------------
# sub-agent report cache
# ---------------------------------------------------------------------------


def test_cache_query_includes_shell_and_dimensions() -> None:
    import stock_analysis_agent.tools.strategy as st

    assert st._cache_query("600887.SH", dimensions=None, shell=False) == "False|600887.SH"
    assert (
        st._cache_query("600887.SH", dimensions=("基本面", "财务"), shell=True)
        == "True|600887.SH|基本面、财务"
    )


def test_run_analyze_stock_returns_cached_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cached sub-agent report short-circuits the LLM run entirely."""
    import stock_analysis_agent.tools.strategy as st

    query = st._cache_query("600887.SH", dimensions=None, shell=True)
    st._subagent_cache.set(site="analyze_stock", query=query, text="cached report")

    class _ShouldNotRun:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("sub-agent must not be constructed on cache hit")

    monkeypatch.setattr(st, "StockAnalysisAgent", _ShouldNotRun)
    assert run_analyze_stock.invoke({"symbol": "600887.SH"}) == "cached report"


def test_run_deepresearch_returns_cached_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """deepresearch reports are cached too, keyed by symbol + dimensions."""
    import stock_analysis_agent.agent.deepresearch as dr
    import stock_analysis_agent.tools.strategy as st

    query = st._cache_query("600887.SH", dimensions=("基本面",), shell=True)
    st._subagent_cache.set(site="deepresearch", query=query, text="cached deepresearch")

    class _ShouldNotRun:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("sub-agent must not be constructed on cache hit")

    monkeypatch.setattr(dr, "DeepResearchAgent", _ShouldNotRun)
    out = run_deepresearch.invoke({"symbol": "600887.SH", "dimensions": ["基本面"]})
    assert out == "cached deepresearch"
