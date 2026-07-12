"""Tests for evaluate_strategy CLI helpers + StrategyMatchAgent wiring.

Layered test strategy (matches tests/test_analyze_stock.py style):

* Pure helpers: ``build_output_path``, ``render_local_markdown``,
  ``_format_strategy_index``, ``_strip_code_fence``,
  ``_extract_json_object``.
* CLI argparse: ``_build_parser`` defaults and required flag behaviour.
* Startup validation: ``_validate_strategy`` exits 4 on unknown name.
* ``StrategyMatchAgent``: construction accepts system prompt; rejects
  empty system prompt; tool list reflects ``include_shell_tool``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from stock_analysis_agent.agent.strategy_match import StrategyMatchAgent
from stock_analysis_agent.agent.strategy_match_schema import (
    StrategyCriterionMatch,
    StrategyMatchReport,
)
from stock_analysis_agent.script import evaluate_strategy as es
from stock_analysis_agent.script.evaluate_strategy import (
    _build_parser,
    _extract_json_object,
    _format_strategy_index,
    _publish_to_feishu,
    _strip_code_fence,
    _validate_strategy,
    build_output_path,
    main,
    render_local_markdown,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _valid_report() -> StrategyMatchReport:
    return StrategyMatchReport(
        symbol="600519.SH",
        strategy_name="value-investing",
        strategy_version="1",
        overall_fit="buy",
        fit_score=8.5,
        summary="quality at reasonable price, fits value-investing profile",
        criterion_matches=[
            StrategyCriterionMatch(
                criterion="PE < 15",
                match_level="fit",
                evidence="PE-TTM 12.3",
                reasoning="well below threshold",
            )
        ],
        raw_analysis_excerpt="verdict=buy score=8.0 risks=macro slowdown",
        action_recommendation="build 5% position in entry zone",
        confidence="high",
    )


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


class TestBuildOutputPath:
    def test_includes_symbol_and_timestamp(self, tmp_path: Path) -> None:
        p = build_output_path("600519.SH", tmp_path, now_epoch=1_700_000_000)
        assert p.parent == tmp_path
        assert p.name.startswith("strategy-match-600519_SH-")
        assert p.name.endswith(".md")
        assert "1700000000" in p.name

    def test_replaces_slashes_in_symbol(self, tmp_path: Path) -> None:
        p = build_output_path("foo/bar", tmp_path, now_epoch=1)
        assert "/" not in p.name


class TestRenderLocalMarkdown:
    def test_includes_all_sections(self) -> None:
        report = _valid_report()
        md = render_local_markdown(report, "2026-07-12")
        assert "[600519.SH] 策略匹配报告 · 2026-07-12" in md
        assert "value-investing" in md
        assert "buy" in md
        assert "PE < 15" in md
        assert "fit" in md
        assert "build 5% position" in md
        assert "不构成投资建议" in md


class TestFormatStrategyIndex:
    def test_renders_one_bullet_per_strategy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The CLI resolves strategies from a hardcoded path under
        # `src/<pkg>/conf/strategies/`. Drop two synthetic files there,
        # mock the catalog to point at them, then clean up.
        strategies_dir = Path(__file__).resolve().parents[1] / "src" / "stock_analysis_agent" / "conf" / "strategies"
        # Fallback for repo layouts where tests/ and src/ share the same parents[1].
        if not strategies_dir.is_dir():
            strategies_dir = (
                Path(__file__).resolve().parents[1]
                / "stock_analysis_agent"
                / "conf"
                / "strategies"
            )
        (strategies_dir / "alpha.md").write_text(
            "---\nname: alpha\ndescription: alpha desc\n---\nbody\n",
            encoding="utf-8",
        )
        (strategies_dir / "beta.md").write_text(
            "---\nname: beta\ndescription: beta desc\n---\nbody\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(es, "_list_strategy_names", lambda: ("alpha", "beta"))
        try:
            out = _format_strategy_index()
            assert "- `alpha` — alpha desc" in out
            assert "- `beta` — beta desc" in out
        finally:
            (strategies_dir / "alpha.md").unlink(missing_ok=True)
            (strategies_dir / "beta.md").unlink(missing_ok=True)

    def test_empty_when_no_strategies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(es, "_list_strategy_names", lambda: ())
        assert "(no strategies available)" in _format_strategy_index()


class TestStripCodeFence:
    def test_strips_fences(self) -> None:
        assert _strip_code_fence("```json\n{}\n```") == "{}"

    def test_passthrough_when_no_fence(self) -> None:
        assert _strip_code_fence("{}") == "{}"


class TestExtractJsonObject:
    def test_returns_longest_object(self) -> None:
        text = '{"short": 1} {"longer": true, "nested": {"x": 1}}'
        assert json.loads(_extract_json_object(text)) == {"longer": True, "nested": {"x": 1}}

    def test_raises_when_no_object(self) -> None:
        with pytest.raises(ValueError):
            _extract_json_object("no json here")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_required_strategy(self) -> None:
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["600519.SH"])

    def test_defaults(self) -> None:
        args = _build_parser().parse_args(["600519.SH", "--strategy", "x"])
        assert args.symbol == "600519.SH"
        assert args.strategy == "x"
        assert args.delivery == "both"
        assert args.recursion_limit == 80
        assert args.include_shell_tool is False

    def test_delivery_choices(self) -> None:
        args = _build_parser().parse_args(
            ["600519.SH", "--strategy", "x", "--delivery", "local"]
        )
        assert args.delivery == "local"
        args = _build_parser().parse_args(
            ["600519.SH", "--strategy", "x", "--delivery", "feishu"]
        )
        assert args.delivery == "feishu"


class TestValidateStrategy:
    def test_unknown_strategy_system_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(es, "_list_strategy_names", lambda: ("alpha",))
        with pytest.raises(SystemExit):
            _validate_strategy("missing")

    def test_known_strategy_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(es, "_list_strategy_names", lambda: ("alpha",))
        _validate_strategy("alpha")  # no exception


class TestRunExitCode4OnUnknownStrategy:
    def test_main_returns_bad_strategy_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(es, "_list_strategy_names", lambda: ("alpha",))
        code = main(["600519.SH", "--strategy", "missing"])
        assert code == es.EXIT_BAD_STRATEGY


# ---------------------------------------------------------------------------
# Feishu publish helper
# ---------------------------------------------------------------------------


class TestPublishToFeishu:
    def test_returns_none_when_lark_cli_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import shutil

        monkeypatch.setattr(shutil, "which", lambda name: None)
        assert _publish_to_feishu(_valid_report()) is None

    def test_returns_url_on_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import shutil
        import subprocess

        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/lark-cli")

        class _FakeProc:
            returncode = 0
            stdout = "Creating doc...\nhttps://example.feishu.cn/docx/abc123\n"
            stderr = ""

        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: _FakeProc(),
        )
        url = _publish_to_feishu(_valid_report())
        assert url == "https://example.feishu.cn/docx/abc123"

    def test_returns_none_on_nonzero_exit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import shutil
        import subprocess

        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/lark-cli")

        class _FakeProc:
            returncode = 1
            stdout = ""
            stderr = "auth failed"

        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _FakeProc())
        assert _publish_to_feishu(_valid_report()) is None

    def test_returns_none_on_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import shutil
        import subprocess

        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/lark-cli")

        def _raise(*a: Any, **kw: Any) -> None:
            raise subprocess.TimeoutExpired(cmd="lark-cli", timeout=60)

        monkeypatch.setattr(subprocess, "run", _raise)
        assert _publish_to_feishu(_valid_report()) is None


# ---------------------------------------------------------------------------
# StrategyMatchAgent
# ---------------------------------------------------------------------------


class TestStrategyMatchAgent:
    def test_rejects_empty_system_prompt(self) -> None:
        with pytest.raises(ValueError, match="system_prompt"):
            StrategyMatchAgent(system_prompt="")

    def test_default_tools_include_strategy_tools(self) -> None:
        agent = StrategyMatchAgent(system_prompt="hello")
        names = sorted(t.name for t in agent.tools)
        assert "load_strategy" in names
        assert "run_analyze_stock" in names
        assert "load_skill" in names
        assert "read_file" in names
        assert "run_command" not in names

    def test_shell_tool_opt_in(self) -> None:
        agent = StrategyMatchAgent(system_prompt="hello", include_shell_tool=True)
        names = sorted(t.name for t in agent.tools)
        assert "run_command" in names

    def test_recursion_limit_default_is_80(self) -> None:
        agent = StrategyMatchAgent(system_prompt="hello")
        assert agent.recursion_limit == 80