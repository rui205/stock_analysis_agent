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
    _extract_feishu_doc_url,
    _publish_to_feishu,
    render_local_markdown,
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
