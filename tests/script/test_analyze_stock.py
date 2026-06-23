"""Tests for script/analyze_stock.py helpers and orchestration."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from stock_analysis_agent.agent.analysis_schema import StockAnalysis
from stock_analysis_agent.script.analyze_stock import (
    _build_parser,
    _strip_code_fence,
    render_markdown,
)
from stock_analysis_agent.tools.feishu_cli import FeishuCli, FeishuCliError, FeishuDocRef


# ---------------------------------------------------------------------------
# _strip_code_fence
# ---------------------------------------------------------------------------


def test_strip_code_fence_passes_through_plain_json() -> None:
    assert _strip_code_fence('{"a": 1}') == '{"a": 1}'


def test_strip_code_fence_removes_json_fenced_block() -> None:
    src = '```json\n{"a": 1}\n```'
    assert _strip_code_fence(src) == '{"a": 1}'


def test_strip_code_fence_removes_unlabelled_fenced_block() -> None:
    src = '```\n{"a": 1}\n```'
    assert _strip_code_fence(src) == '{"a": 1}'


def test_strip_code_fence_handles_leading_and_trailing_whitespace() -> None:
    src = '\n  ```json\n{"a": 1}\n```  \n'
    assert _strip_code_fence(src) == '{"a": 1}'


def test_strip_code_fence_returns_text_when_no_fence() -> None:
    assert _strip_code_fence('hello world') == 'hello world'


# ---------------------------------------------------------------------------
# render_markdown
# ---------------------------------------------------------------------------


def _sample_analysis() -> StockAnalysis:
    return StockAnalysis(
        symbol="02319.HK",
        summary="蒙牛乳业近期经营稳健,股价震荡。" * 2,
        fundamentals="乳制品行业龙头。",
        technicals="现价 12.34。",
        peer_compare="N/A",
        news="半年报发布。",
        risks="原奶价格波动。",
        recommendation="关注。",
    )


def test_render_markdown_contains_title_and_timestamp() -> None:
    md = render_markdown(_sample_analysis())
    assert "# 02319.HK 分析报告" in md
    import re
    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", md)


def test_render_markdown_contains_every_section() -> None:
    md = render_markdown(_sample_analysis())
    for header in (
        "## 总体观点", "## 基本面", "## 技术面",
        "## 同行对比", "## 近期新闻", "## 风险", "## 操作建议",
    ):
        assert header in md


def test_render_markdown_contains_field_values() -> None:
    md = render_markdown(_sample_analysis())
    assert "蒙牛乳业" in md
    assert "乳制品行业龙头。" in md
    assert "关注。" in md


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def test_parser_requires_symbol() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_defaults() -> None:
    args = _build_parser().parse_args(["02319.HK"])
    assert args.symbol == "02319.HK"
    assert args.include_peers is True
    assert args.peer_count == 2
    # title_prefix is None at parse time; run() resolves it from `symbol`.
    assert args.title_prefix is None
    assert args.verbose is False


def test_parser_no_peers_flag() -> None:
    args = _build_parser().parse_args(["02319.HK", "--no-peers"])
    assert args.include_peers is False


def test_parser_custom_title_prefix() -> None:
    args = _build_parser().parse_args(["02319.HK", "--title-prefix", "蒙牛日报"])
    assert args.title_prefix == "蒙牛日报"


# ---------------------------------------------------------------------------
# run() — orchestration
# ---------------------------------------------------------------------------


def _fake_agent_with_content(content: str) -> MagicMock:
    agent = MagicMock()
    agent.stream.return_value = iter(
        [
            {
                "event": "on_chat_model_end",
                "data": {"output": AIMessage(content=content)},
            }
        ]
    )
    return agent


def _valid_json_payload() -> str:
    return json.dumps(
        {
            "symbol": "02319.HK",
            "summary": "蒙牛乳业近期经营稳健,股价震荡。" * 2,
            "fundamentals": "乳制品行业龙头。",
            "technicals": "现价 12.34。",
            "peer_compare": "N/A",
            "news": "半年报发布。",
            "risks": "原奶价格波动。",
            "recommendation": "关注。",
        },
        ensure_ascii=False,
    )


def test_run_creates_new_doc_when_no_existing_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stock_analysis_agent.script import analyze_stock

    fake_agent = _fake_agent_with_content(_valid_json_payload())
    monkeypatch.setattr(
        analyze_stock, "StockAnalysisAgent", lambda **kwargs: fake_agent
    )

    fake_cli = MagicMock(spec=FeishuCli)
    fake_cli.list_matching_docs.return_value = []
    fake_cli.create_doc.return_value = FeishuDocRef(
        doc_id="d1", url="https://x", title="02319.HK 分析报告"
    )
    monkeypatch.setattr(analyze_stock, "FeishuCli", lambda **kwargs: fake_cli)

    args = _build_parser().parse_args(["02319.HK"])
    rc = analyze_stock.run(args)
    assert rc == 0
    fake_cli.create_doc.assert_called_once()
    fake_cli.append_to_doc.assert_not_called()


def test_run_appends_when_existing_doc_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stock_analysis_agent.script import analyze_stock

    fake_agent = _fake_agent_with_content(_valid_json_payload())
    monkeypatch.setattr(
        analyze_stock, "StockAnalysisAgent", lambda **kwargs: fake_agent
    )

    fake_cli = MagicMock(spec=FeishuCli)
    fake_cli.list_matching_docs.return_value = [
        FeishuDocRef(doc_id="existing", url="https://old", title="02319.HK 分析报告")
    ]
    monkeypatch.setattr(analyze_stock, "FeishuCli", lambda **kwargs: fake_cli)

    args = _build_parser().parse_args(["02319.HK"])
    rc = analyze_stock.run(args)
    assert rc == 0
    fake_cli.append_to_doc.assert_called_once()
    fake_cli.create_doc.assert_not_called()
    # The first existing match should be the target.
    call = fake_cli.append_to_doc.call_args
    assert call.args[0] == "existing"


def test_run_exits_2_when_agent_output_is_not_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from stock_analysis_agent.script import analyze_stock

    fake_agent = _fake_agent_with_content("not json at all")
    monkeypatch.setattr(
        analyze_stock, "StockAnalysisAgent", lambda **kwargs: fake_agent
    )

    fake_cli = MagicMock(spec=FeishuCli)
    monkeypatch.setattr(analyze_stock, "FeishuCli", lambda **kwargs: fake_cli)

    args = _build_parser().parse_args(["02319.HK"])
    rc = analyze_stock.run(args)
    assert rc == 2
    # The CLI must not be touched on JSON validation failure.
    fake_cli.list_matching_docs.assert_not_called()
    fake_cli.create_doc.assert_not_called()
    fake_cli.append_to_doc.assert_not_called()


def test_run_strips_code_fence_before_validating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stock_analysis_agent.script import analyze_stock

    fenced = "```json\n" + _valid_json_payload() + "\n```"
    fake_agent = _fake_agent_with_content(fenced)
    monkeypatch.setattr(
        analyze_stock, "StockAnalysisAgent", lambda **kwargs: fake_agent
    )

    fake_cli = MagicMock(spec=FeishuCli)
    fake_cli.list_matching_docs.return_value = []
    fake_cli.create_doc.return_value = FeishuDocRef(
        doc_id="d1", url="https://x", title="02319.HK 分析报告"
    )
    monkeypatch.setattr(analyze_stock, "FeishuCli", lambda **kwargs: fake_cli)

    args = _build_parser().parse_args(["02319.HK"])
    rc = analyze_stock.run(args)
    assert rc == 0
    fake_cli.create_doc.assert_called_once()


def test_run_exits_4_when_lark_cli_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stock_analysis_agent.script import analyze_stock

    fake_agent = _fake_agent_with_content(_valid_json_payload())
    monkeypatch.setattr(
        analyze_stock, "StockAnalysisAgent", lambda **kwargs: fake_agent
    )

    fake_cli = MagicMock(spec=FeishuCli)
    fake_cli.list_matching_docs.side_effect = FeishuCliError(
        "boom", returncode=1, stderr="perm denied"
    )
    monkeypatch.setattr(analyze_stock, "FeishuCli", lambda **kwargs: fake_cli)

    args = _build_parser().parse_args(["02319.HK"])
    rc = analyze_stock.run(args)
    assert rc == 4


def test_run_writes_temp_markdown_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The temp file path passed to FeishuCli must exist on disk and
    contain the rendered Markdown."""
    from stock_analysis_agent.script import analyze_stock

    # Redirect tempfile.gettempdir() so we can inspect it.
    monkeypatch.setattr(analyze_stock.tempfile, "gettempdir", lambda: str(tmp_path))

    fake_agent = _fake_agent_with_content(_valid_json_payload())
    monkeypatch.setattr(
        analyze_stock, "StockAnalysisAgent", lambda **kwargs: fake_agent
    )

    fake_cli = MagicMock(spec=FeishuCli)
    fake_cli.list_matching_docs.return_value = []
    captured: dict[str, Path] = {}

    def _capture_create(*, title: str, content_file: Path) -> FeishuDocRef:
        captured["file"] = content_file
        return FeishuDocRef(doc_id="d1", url="https://x", title=title)

    fake_cli.create_doc.side_effect = _capture_create
    monkeypatch.setattr(analyze_stock, "FeishuCli", lambda **kwargs: fake_cli)

    args = _build_parser().parse_args(["02319.HK"])
    rc = analyze_stock.run(args)
    assert rc == 0
    assert "file" in captured
    on_disk = Path(captured["file"])
    assert on_disk.exists()
    text = on_disk.read_text(encoding="utf-8")
    assert "02319.HK" in text
    assert "## 基本面" in text