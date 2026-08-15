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
        assert kwargs["symbol"] == "600519.SH"
        assert kwargs["include_shell_tool"] is False

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


class TestRunAnalyzeStockShellPropagation:
    """The embedded sub-agent must inherit the orchestrator's shell opt-in.

    Without ``run_command`` the sub-agent cannot execute the mx-* skill
    scripts its ``stock-analysis`` workflow depends on and degrades to
    an LLM-knowledge-only report — the root cause of the "关键数据缺失"
    strategy-match verdicts.
    """

    def _run_with_fake_subagent(
        self, monkeypatch: pytest.MonkeyPatch, shell_enabled: bool
    ) -> MagicMock:
        """Invoke the tool with a stubbed sub-agent; return the class mock."""
        import stock_analysis_agent.tools.strategy as mod

        fake_sub = MagicMock()
        fake_sub.stream.return_value = iter([])
        fake_cls = MagicMock(return_value=fake_sub)
        monkeypatch.setattr(mod, "StockAnalysisAgent", fake_cls)
        monkeypatch.setattr(mod, "_subagent_include_shell_tool", shell_enabled)
        run_analyze_stock.invoke({"symbol": "06049.HK"})
        fake_cls.assert_called_once()
        return fake_cls

    def test_subagent_gets_shell_tool_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_cls = self._run_with_fake_subagent(monkeypatch, shell_enabled=True)
        assert fake_cls.call_args.kwargs["include_shell_tool"] is True

    def test_subagent_shell_tool_defaults_to_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_cls = self._run_with_fake_subagent(monkeypatch, shell_enabled=False)
        assert fake_cls.call_args.kwargs["include_shell_tool"] is False

    def test_setter_toggles_module_flag(self) -> None:
        import stock_analysis_agent.tools.strategy as mod

        original = mod._subagent_include_shell_tool
        try:
            mod.set_subagent_include_shell_tool(True)
            assert mod._subagent_include_shell_tool is True
            mod.set_subagent_include_shell_tool(False)
            assert mod._subagent_include_shell_tool is False
        finally:
            mod._subagent_include_shell_tool = original


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
