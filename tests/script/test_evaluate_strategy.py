"""Tests for the Feishu delivery path of ``script.evaluate_strategy``.

Regression background: ``lark-cli docs +create`` prints a JSON envelope
(``{"ok": ..., "data": {"document": {"url": ...}}}``), pretty-printed —
so the last non-empty stdout line is ``}``. The old "last line is the
URL" heuristic logged ``published to Feishu: }`` and lost the link.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from stock_analysis_agent.agent.strategy_match_schema import (
    DataSourceBreakdown,
    StrategyCriterionMatch,
    StrategyMatchReport,
)
from stock_analysis_agent.script.evaluate_strategy import (
    _build_parser,
    _extract_feishu_doc_url,
    _handle_parse_failure,
    _publish_to_feishu,
    _write_raw_output,
    render_local_markdown,
    ReportParseError,
)


def _make_report() -> StrategyMatchReport:
    """Build a minimal valid :class:`StrategyMatchReport`."""
    return StrategyMatchReport(
        symbol="06049.HK",
        strategy_name="value-investing",
        strategy_version="1",
        overall_fit="hold",
        fit_score=6.0,
        summary="测试摘要",
        criterion_matches=[
            StrategyCriterionMatch(
                criterion="毛利率 ≥ 30%",
                match_level="partial",
                evidence="毛利率 17.43%",
                reasoning="低于门槛",
            )
        ],
        data_sources=DataSourceBreakdown(
            stock_analysis="verdict: hold",
            deepresearch="",
        ),
        judgment_rationale="毛利率低于门槛,给 partial,综合 hold",
        action_recommendation="观察",
        confidence="medium",
    )


class TestExtractFeishuDocUrl:
    def test_extracts_url_from_json_envelope(self) -> None:
        stdout = json.dumps(
            {
                "ok": True,
                "identity": "user",
                "data": {
                    "document": {
                        "document_id": "doxcnABC",
                        "url": "https://x.feishu.cn/docx/doxcnABC",
                    }
                },
            }
        )
        assert _extract_feishu_doc_url(stdout) == "https://x.feishu.cn/docx/doxcnABC"

    def test_extracts_url_from_pretty_printed_envelope(self) -> None:
        # Regression: pretty-printed output ends with a ``}`` line — the
        # old last-line heuristic returned that brace as the "URL".
        stdout = json.dumps(
            {
                "ok": True,
                "data": {
                    "document": {"url": "https://x.feishu.cn/docx/doxcnPRETTY"}
                },
            },
            indent=2,
        )
        assert stdout.rstrip().endswith("}")
        assert _extract_feishu_doc_url(stdout) == "https://x.feishu.cn/docx/doxcnPRETTY"

    def test_missing_url_field_returns_none(self) -> None:
        stdout = json.dumps({"ok": True, "data": {"document": {"document_id": "d"}}})
        assert _extract_feishu_doc_url(stdout) is None

    def test_falls_back_to_url_scan_for_non_json(self) -> None:
        stdout = "created https://y.feishu.cn/docx/doxcnXYZ successfully\n"
        assert _extract_feishu_doc_url(stdout) == "https://y.feishu.cn/docx/doxcnXYZ"

    def test_returns_none_when_no_url_found(self) -> None:
        assert _extract_feishu_doc_url("no url here") is None
        assert _extract_feishu_doc_url("") is None


class TestPublishToFeishu:
    def _patch_lark_cli(
        self, monkeypatch: pytest.MonkeyPatch, stdout: str, returncode: int = 0
    ) -> dict[str, Any]:
        """Stub ``lark-cli`` presence + execution; capture the argv."""
        captured: dict[str, Any] = {}

        def fake_run(argv: list[str], **kwargs: Any) -> MagicMock:
            captured["argv"] = argv
            proc = MagicMock()
            proc.returncode = returncode
            proc.stdout = stdout
            proc.stderr = ""
            return proc

        monkeypatch.setattr("shutil.which", lambda _name: "/usr/local/bin/lark-cli")
        monkeypatch.setattr(
            "stock_analysis_agent.script.evaluate_strategy.subprocess.run", fake_run
        )
        return captured

    def test_returns_url_from_json_envelope(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        envelope = json.dumps(
            {"ok": True, "data": {"document": {"url": "https://z.feishu.cn/docx/doxcn1"}}},
            indent=2,
        )
        self._patch_lark_cli(monkeypatch, stdout=envelope)
        assert _publish_to_feishu(_make_report()) == "https://z.feishu.cn/docx/doxcn1"

    def test_invocation_uses_markdown_doc_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The rendered content is Markdown; without ``--doc-format
        # markdown`` lark-cli parses it as XML and produces a garbled doc.
        captured = self._patch_lark_cli(
            monkeypatch,
            stdout=json.dumps(
                {"ok": True, "data": {"document": {"url": "https://z.feishu.cn/docx/d"}}}
            ),
        )
        _publish_to_feishu(_make_report())
        argv = captured["argv"]
        assert argv[0] == "lark-cli"
        assert "--doc-format" in argv
        assert argv[argv.index("--doc-format") + 1] == "markdown"
        assert "--api-version" in argv
        assert argv[argv.index("--api-version") + 1] == "v2"

    def test_unparseable_stdout_without_url_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # lark-cli exited 0 but printed nothing usable — degrade to the
        # local-markdown fallback instead of logging a bogus URL.
        self._patch_lark_cli(monkeypatch, stdout="")
        assert _publish_to_feishu(_make_report()) is None


class TestRenderLocalMarkdown:
    def test_renders_report_link_when_url_present(self) -> None:
        report = _make_report()
        report.data_sources.stock_analysis_url = "https://x.feishu.cn/docx/doxcnLINK"
        md = render_local_markdown(report, "2026-08-29")
        assert "🔗 完整报告" in md
        assert (
            "[https://x.feishu.cn/docx/doxcnLINK](https://x.feishu.cn/docx/doxcnLINK)"
            in md
        )

    def test_omits_report_link_when_url_absent(self) -> None:
        report = _make_report()
        md = render_local_markdown(report, "2026-08-29")
        assert "🔗 完整报告" not in md


class TestModelArg:
    """The ``--model`` flag selects a model.json entry; default is 千问."""

    def test_parser_model_defaults_to_none(self) -> None:
        args = _build_parser().parse_args(
            ["600887.SH", "--strategy", "value-investing"]
        )
        assert args.model is None

    def test_parser_accepts_model(self) -> None:
        args = _build_parser().parse_args(
            ["600887.SH", "--strategy", "value-investing", "--model", "deepseek-v4-pro"]
        )
        assert args.model == "deepseek-v4-pro"

    def test_run_sets_model_env_when_provided(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        import os
        from types import SimpleNamespace

        import stock_analysis_agent.script.evaluate_strategy as mod

        monkeypatch.setattr(mod, "_validate_strategy", lambda name: None)
        monkeypatch.setattr(mod, "_run_agent_and_parse", lambda args: _make_report())
        monkeypatch.setattr(mod, "_publish_to_feishu", lambda md: None)
        monkeypatch.delenv("MODEL", raising=False)

        args = SimpleNamespace(
            symbol="600887.SH",
            strategy="value-investing",
            delivery="local",
            include_shell_tool=False,
            recursion_limit=80,
            output_dir=tmp_path,
            model="deepseek-v4-pro",
            verbose=False,
        )
        assert mod.run(args) == mod.EXIT_OK
        assert os.environ["MODEL"] == "deepseek-v4-pro"

    def test_run_does_not_set_model_env_when_omitted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        import os
        from types import SimpleNamespace

        import stock_analysis_agent.script.evaluate_strategy as mod

        monkeypatch.setattr(mod, "_validate_strategy", lambda name: None)
        monkeypatch.setattr(mod, "_run_agent_and_parse", lambda args: _make_report())
        monkeypatch.setattr(mod, "_publish_to_feishu", lambda md: None)
        monkeypatch.delenv("MODEL", raising=False)

        args = SimpleNamespace(
            symbol="600887.SH",
            strategy="value-investing",
            delivery="local",
            include_shell_tool=False,
            recursion_limit=80,
            output_dir=tmp_path,
            model=None,
            verbose=False,
        )
        assert mod.run(args) == mod.EXIT_OK
        # Omitted → no MODEL override → settings falls back to 千问.
        assert "MODEL" not in os.environ


class TestRunAgentAndParseFailure:
    """Parsing failures surface raw output instead of silently discarding."""

    def test_raises_report_parse_error_with_raw_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from types import SimpleNamespace

        import stock_analysis_agent.script.evaluate_strategy as es

        raw = "agent output with no json object"
        fake_events = [
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": SimpleNamespace(content=raw)},
            }
        ]

        class _FakeAgent:
            def __init__(self, **kwargs: Any) -> None:
                pass

            def stream(self, messages: Any) -> Any:
                return iter(fake_events)

        monkeypatch.setattr(es, "_load_system_prompt", lambda **kw: "dummy")
        monkeypatch.setattr(es, "StrategyMatchAgent", _FakeAgent)

        args = SimpleNamespace(
            symbol="600887.SH",
            strategy="value-investing",
            include_shell_tool=False,
            recursion_limit=80,
        )
        with pytest.raises(ReportParseError) as excinfo:
            es._run_agent_and_parse(args)
        assert excinfo.value.last_text == raw


class TestHandleParseFailure:
    """Interactive / non-interactive recovery from a failed report parse."""

    def _make_args(self, tmp_path):
        from types import SimpleNamespace

        return SimpleNamespace(
            symbol="600887.SH", strategy="value-investing", output_dir=tmp_path
        )

    def test_interactive_yes_saves_and_returns_ok(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        import stock_analysis_agent.script.evaluate_strategy as es

        monkeypatch.setattr(es.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda prompt: "y")
        args = self._make_args(tmp_path)

        assert (
            _handle_parse_failure(ReportParseError("raw agent output {"), args)
            == es.EXIT_OK
        )
        files = list(tmp_path.glob("strategy-match-*-raw.md"))
        assert len(files) == 1
        assert "raw agent output" in files[0].read_text(encoding="utf-8")

    def test_interactive_no_returns_parse(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        import stock_analysis_agent.script.evaluate_strategy as es

        monkeypatch.setattr(es.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda prompt: "n")
        args = self._make_args(tmp_path)

        assert (
            _handle_parse_failure(ReportParseError("raw {"), args) == es.EXIT_PARSE
        )
        assert list(tmp_path.glob("strategy-match-*-raw.md")) == []

    def test_non_interactive_saves_and_returns_parse(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        import stock_analysis_agent.script.evaluate_strategy as es

        monkeypatch.setattr(es.sys.stdin, "isatty", lambda: False)
        args = self._make_args(tmp_path)

        assert (
            _handle_parse_failure(ReportParseError("raw {"), args) == es.EXIT_PARSE
        )
        files = list(tmp_path.glob("strategy-match-*-raw.md"))
        assert len(files) == 1


class TestWriteRawOutput:
    def test_writes_raw_markdown_file(self, tmp_path) -> None:
        path = _write_raw_output("hello raw", "600887.SH", tmp_path)
        assert path.name.startswith("strategy-match-600887_SH-")
        assert path.name.endswith("-raw.md")
        assert path.read_text(encoding="utf-8") == "hello raw"
