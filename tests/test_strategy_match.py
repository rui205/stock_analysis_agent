"""Tests for evaluate_strategy CLI helpers + StrategyMatchAgent wiring.

Layered test strategy:

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
    DataSourceBreakdown,
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
        data_sources=DataSourceBreakdown(
            stock_analysis="verdict=buy score=8.0 risks=macro slowdown",
            deepresearch="",
        ),
        judgment_rationale="估值分位低且盈利质量达标,故 buy",
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
        assert "## 数据来源" in md
        assert "### 来自 stock_analysis" in md
        assert "### 来自 deepresearch" in md
        assert "未调用 deepresearch" in md
        assert "## 判断理论" in md

    def test_escapes_pipe_and_newline_in_table_cells(self) -> None:
        report = _valid_report()
        report.criterion_matches[0].criterion = "PE < 15 | ROE > 15\nsecond line"
        md = render_local_markdown(report, "2026-07-12")
        assert "PE < 15 \\| ROE > 15 second line" in md


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
        assert args.recursion_limit == 120
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
    def test_unknown_strategy_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(es, "_list_strategy_names", lambda: ("alpha",))
        with pytest.raises(es.UnknownStrategyError):
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
    _MARKDOWN = "# [600519.SH] 策略匹配报告\n\nbody"

    def test_returns_none_when_lark_cli_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import shutil

        monkeypatch.setattr(shutil, "which", lambda name: None)
        assert _publish_to_feishu(self._MARKDOWN) is None

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

        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _FakeProc())
        url = _publish_to_feishu(self._MARKDOWN)
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
        assert _publish_to_feishu(self._MARKDOWN) is None

    def test_returns_none_after_repeated_timeouts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import shutil
        import subprocess
        import time

        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/lark-cli")
        monkeypatch.setattr(time, "sleep", lambda *a: None)

        calls = {"n": 0}

        def _raise(*a: Any, **kw: Any) -> None:
            calls["n"] += 1
            raise subprocess.TimeoutExpired(cmd="lark-cli", timeout=60)

        monkeypatch.setattr(subprocess, "run", _raise)
        assert _publish_to_feishu(self._MARKDOWN) is None
        assert calls["n"] == 3  # 1 initial + 2 retries

    def test_retries_on_timeout_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import shutil
        import subprocess
        import time

        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/lark-cli")
        monkeypatch.setattr(time, "sleep", lambda *a: None)

        class _FakeProc:
            returncode = 0
            stdout = "https://example.feishu.cn/docx/abc123"
            stderr = ""

        calls = {"n": 0}

        def _run(*a: Any, **kw: Any) -> Any:
            calls["n"] += 1
            if calls["n"] == 1:
                raise subprocess.TimeoutExpired(cmd="lark-cli", timeout=60)
            return _FakeProc()

        monkeypatch.setattr(subprocess, "run", _run)
        url = _publish_to_feishu(self._MARKDOWN)
        assert url == "https://example.feishu.cn/docx/abc123"
        assert calls["n"] == 2


# ---------------------------------------------------------------------------
# StrategyMatchAgent
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_subagent_shell_flag():
    """Snapshot/restore the ``tools.strategy`` sub-agent shell flag.

    ``StrategyMatchAgent.__init__`` writes the module-level provider
    read by ``run_analyze_stock``; without restoration a test that
    constructs an agent with ``include_shell_tool=True`` would leak
    state into later test modules.
    """
    import stock_analysis_agent.tools.strategy as strategy_tools

    original = strategy_tools._subagent_include_shell_tool
    yield
    strategy_tools._subagent_include_shell_tool = original


class TestStrategyMatchAgent:
    def test_rejects_empty_system_prompt(self) -> None:
        with pytest.raises(ValueError, match="system_prompt"):
            StrategyMatchAgent(system_prompt="")

    def test_default_tools_are_orchestration_only(self) -> None:
        """The orchestrator owns strategy + sub-agent + skill only — not
        the sub-agent's data-discovery surface (``read_file``).
        Sharing the latter would let the orchestrator bypass the
        sub-agent and produce raw reports itself.
        """
        agent = StrategyMatchAgent(system_prompt="hello")
        names = set(t.name for t in agent.tools)
        # Workflow glue (must-have):
        assert "load_strategy" in names
        assert "run_analyze_stock" in names
        assert "run_deepresearch" in names
        assert "load_skill" in names
        # Sub-agent's data-discovery surface must NOT leak here:
        assert "read_file" not in names
        # list_dir was removed entirely from the project.
        assert "list_dir" not in names
        # Shell is opt-in:
        assert "run_command" not in names

    def test_shell_tool_opt_in(self) -> None:
        agent = StrategyMatchAgent(system_prompt="hello", include_shell_tool=True)
        names = sorted(t.name for t in agent.tools)
        assert "run_command" in names

    def test_recursion_limit_default_is_120(self) -> None:
        agent = StrategyMatchAgent(system_prompt="hello")
        assert agent.recursion_limit == 120

    def test_thinking_budget_default_is_8192(self) -> None:
        agent = StrategyMatchAgent(system_prompt="hello")
        assert agent.thinking_budget_tokens == 8192

    def test_shell_flag_propagates_to_subagent_provider_when_enabled(self) -> None:
        """``include_shell_tool=True`` must reach the module-level flag that
        ``run_analyze_stock`` reads — the sub-agent's stock-analysis
        workflow needs ``run_command`` to execute the mx-* skill scripts.
        """
        import stock_analysis_agent.tools.strategy as strategy_tools

        StrategyMatchAgent(system_prompt="hello", include_shell_tool=True)
        assert strategy_tools._subagent_include_shell_tool is True

    def test_shell_flag_propagates_to_subagent_provider_by_default(self) -> None:
        """The default constructor resets the provider to ``False`` so a
        shell-enabled agent constructed earlier cannot leak its flag.
        """
        import stock_analysis_agent.tools.strategy as strategy_tools

        strategy_tools._subagent_include_shell_tool = True
        StrategyMatchAgent(system_prompt="hello")
        assert strategy_tools._subagent_include_shell_tool is False


class TestRunFeishuOnlyDegradesToLocal:
    def test_writes_local_markdown_when_publish_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--delivery feishu`` + failed publish must still produce an
        artifact — degrade to local markdown per SKILL.md, not just log."""
        from types import SimpleNamespace

        report_json = _valid_report().model_dump_json()
        fake_events = [
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": SimpleNamespace(content=report_json)},
            }
        ]

        class _FakeAgent:
            def __init__(self, **kwargs: Any) -> None:
                pass

            def stream(self, messages: Any) -> Any:
                return iter(fake_events)

        monkeypatch.setattr(es, "_validate_strategy", lambda name: None)
        monkeypatch.setattr(es, "_load_system_prompt", lambda **kw: "dummy")
        monkeypatch.setattr(es, "StrategyMatchAgent", _FakeAgent)
        monkeypatch.setattr(es, "_publish_to_feishu", lambda markdown: None)

        args = SimpleNamespace(
            symbol="600519.SH",
            strategy="value-investing",
            delivery="feishu",
            include_shell_tool=False,
            recursion_limit=80,
            output_dir=tmp_path,
            verbose=False,
        )
        assert es.run(args) == es.EXIT_OK
        files = list(tmp_path.glob("strategy-match-*.md"))
        assert len(files) == 1