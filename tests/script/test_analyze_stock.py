"""Tests for script/analyze_stock.py helpers and orchestration."""
from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from stock_analysis_agent.agent.analysis_schema import StockAnalysis
from stock_analysis_agent.script.analyze_stock import (
    _build_parser,
    _strip_code_fence,
    build_output_path,
    output_dir,
    render_markdown,
)


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
# output path helpers
# ---------------------------------------------------------------------------


def test_build_output_path_uses_timestamp_and_safe_symbol(tmp_path: Path) -> None:
    p = build_output_path("02319.HK", tmp_path, now_epoch=1_700_000_000)
    assert p == tmp_path / "stock-analysis-02319_HK-1700000000.md"
    assert p.suffix == ".md"


def test_build_output_path_sanitises_slash_in_symbol(tmp_path: Path) -> None:
    """Slashes in the symbol must not create subdirectories inside output/."""
    p = build_output_path("foo/bar", tmp_path, now_epoch=1)
    assert p.parent == tmp_path
    assert "/" not in p.name


def test_output_dir_resolves_to_project_root_output() -> None:
    """``output_dir()`` lives under the project root, not the user's CWD."""
    expected = Path(__file__).resolve().parents[2] / "output"
    assert output_dir() == expected


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
    assert args.output_dir is None
    assert args.verbose is False


def test_parser_no_peers_flag() -> None:
    args = _build_parser().parse_args(["02319.HK", "--no-peers"])
    assert args.include_peers is False


def test_parser_output_dir_flag() -> None:
    args = _build_parser().parse_args(["02319.HK", "--output-dir", "/tmp/custom-out"])
    assert args.output_dir == Path("/tmp/custom-out")


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


def test_run_writes_markdown_under_output_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Successful run writes a timestamped Markdown file into ``output/``."""
    from stock_analysis_agent.script import analyze_stock

    fake_agent = _fake_agent_with_content(_valid_json_payload())
    monkeypatch.setattr(
        analyze_stock, "StockAnalysisAgent", lambda **kwargs: fake_agent
    )

    args = _build_parser().parse_args(
        ["02319.HK", "--output-dir", str(tmp_path)]
    )
    rc = analyze_stock.run(args)
    assert rc == 0

    # Exactly one markdown file, named for the symbol, written into tmp_path.
    written = list(tmp_path.glob("stock-analysis-02319_HK-*.md"))
    assert len(written) == 1
    text = written[0].read_text(encoding="utf-8")
    assert "# 02319.HK 分析报告" in text
    assert "## 基本面" in text
    assert "乳制品行业龙头。" in text


def test_run_exits_2_when_agent_output_is_not_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from stock_analysis_agent.script import analyze_stock

    fake_agent = _fake_agent_with_content("not json at all")
    monkeypatch.setattr(
        analyze_stock, "StockAnalysisAgent", lambda **kwargs: fake_agent
    )

    args = _build_parser().parse_args(
        ["02319.HK", "--output-dir", str(tmp_path)]
    )
    rc = analyze_stock.run(args)
    assert rc == 2
    # No file should be written when the agent output is invalid.
    assert list(tmp_path.glob("stock-analysis-*.md")) == []


def test_run_strips_code_fence_before_validating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from stock_analysis_agent.script import analyze_stock

    fenced = "```json\n" + _valid_json_payload() + "\n```"
    fake_agent = _fake_agent_with_content(fenced)
    monkeypatch.setattr(
        analyze_stock, "StockAnalysisAgent", lambda **kwargs: fake_agent
    )

    args = _build_parser().parse_args(
        ["02319.HK", "--output-dir", str(tmp_path)]
    )
    rc = analyze_stock.run(args)
    assert rc == 0
    written = list(tmp_path.glob("stock-analysis-02319_HK-*.md"))
    assert len(written) == 1


def test_run_creates_output_dir_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The output directory must be created if it does not already exist."""
    from stock_analysis_agent.script import analyze_stock

    fake_agent = _fake_agent_with_content(_valid_json_payload())
    monkeypatch.setattr(
        analyze_stock, "StockAnalysisAgent", lambda **kwargs: fake_agent
    )

    target = tmp_path / "fresh" / "nested" / "output"
    assert not target.exists()

    args = _build_parser().parse_args(
        ["02319.HK", "--output-dir", str(target)]
    )
    rc = analyze_stock.run(args)
    assert rc == 0
    assert target.is_dir()
    assert list(target.glob("stock-analysis-02319_HK-*.md"))


def test_run_exits_3_when_agent_tools_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from stock_analysis_agent.script import analyze_stock
    from stock_analysis_agent.agent.exceptions import ToolExecutionError

    fake_agent = MagicMock()
    fake_agent.stream.side_effect = ToolExecutionError("tool blew up")
    monkeypatch.setattr(
        analyze_stock, "StockAnalysisAgent", lambda **kwargs: fake_agent
    )

    args = _build_parser().parse_args(
        ["02319.HK", "--output-dir", str(tmp_path)]
    )
    rc = analyze_stock.run(args)
    assert rc == 3
    assert list(tmp_path.glob("stock-analysis-*.md")) == []