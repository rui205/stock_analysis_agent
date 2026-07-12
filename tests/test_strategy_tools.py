"""Tests for stock_analysis_agent.tools.strategy.

Covers the static parts only — ``_list_strategy_names``,
``_parse_strategy_frontmatter``, and ``load_strategy``. The dynamic
``run_analyze_stock`` tool (which embeds a subagent) is tested in
Task 4 with ``monkeypatch``.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from stock_analysis_agent.agent.analysis_schema import StockAnalysis
from stock_analysis_agent.tools.strategy import (
    _list_strategy_names,
    _parse_strategy_frontmatter,
    _render_analysis_summary,
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


def _stub_analysis_dict() -> dict:
    return {
        "symbol": "600519.SH",
        "company_profile": "A leading baijiu producer...",
        "verdict": {
            "decision": "buy",
            "decision_label": "买入",
            "confidence": "high",
            "summary": "quality at reasonable price",
        },
        "price_plan": {
            "current_price": 1500.0,
            "entry_zone": [1400.0, 1480.0],
            "add_zone": [1350.0, 1400.0],
            "target_price": 1750.0,
            "stop_loss": 1320.0,
            "expected_return": "+10% ~ +17%",
            "risk_reward_ratio": "2.5:1",
            "time_horizon": "3-6 个月",
        },
        "scores": {
            "fundamental": 8.0, "technical": 7.0,
            "news_catalyst": 6.5, "peer_positioning": 8.5,
            "weighted_total": 7.6,
        },
        "fundamental_analysis": {"highlights": ["stable margin"], "concerns": ["slowing growth"]},
        "technical_analysis": {"highlights": [], "concerns": []},
        "news_catalysts": ["new product launch"],
        "peer_compare": "leading in segment",
        "risks": [{"type": "行业", "description": "macro slowdown", "severity": "medium"}],
        "action_plan": {
            "position_size": "5-10%",
            "execution": ["scale in entry zone"],
            "review_triggers": ["Q3 earnings miss"],
        },
        "reasoning_chain": "...",
    }


class TestRenderAnalysisSummary:
    def test_includes_verdict_and_score(self) -> None:
        a = StockAnalysis.model_validate(_stub_analysis_dict())
        md = _render_analysis_summary("600519.SH", a)
        assert "600519.SH" in md
        assert "buy" in md.lower() or "买入" in md
        assert "7.6" in md
        assert "1500" in md
        assert "macro slowdown" in md


class TestRunAnalyzeStockTool:
    def test_success_returns_summary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        body = json.dumps(_stub_analysis_dict(), ensure_ascii=False)
        fake_events = [
            {"event": "on_chat_model_stream", "data": {"chunk": AIMessage(content=body)}},
        ]
        fake_sub = MagicMock()
        fake_sub.stream.return_value = iter(fake_events)
        fake_cls = MagicMock(return_value=fake_sub)
        import stock_analysis_agent.tools.strategy as mod
        monkeypatch.setattr(mod, "StockAnalysisAgent", fake_cls)
        out = run_analyze_stock.invoke({"symbol": "600519.SH"})
        assert "600519.SH" in out
        assert "买入" in out or "buy" in out.lower()
        fake_cls.assert_called_once()
        kwargs = fake_cls.call_args.kwargs
        assert kwargs["symbol"] == "600519.SH"
        assert kwargs["include_peers"] is True
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

    def test_bad_json_returns_error_with_raw_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_events = [
            {"event": "on_chat_model_stream", "data": {"chunk": AIMessage(content="not json at all")}},
        ]
        fake_sub = MagicMock()
        fake_sub.stream.return_value = iter(fake_events)
        fake_cls = MagicMock(return_value=fake_sub)
        import stock_analysis_agent.tools.strategy as mod
        monkeypatch.setattr(mod, "StockAnalysisAgent", fake_cls)
        out = run_analyze_stock.invoke({"symbol": "000001.SZ"})
        assert out.startswith("[ERROR]")
        assert "JSON" in out
        assert "not json at all" in out
