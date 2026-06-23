"""Tests for script/analyze_stock.py helpers and orchestration."""
from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from stock_analysis_agent.agent.analysis_schema import (
    ActionPlan,
    DimensionAnalysis,
    PricePlan,
    Risk,
    Scores,
    StockAnalysis,
    Verdict,
)
from stock_analysis_agent.script.analyze_stock import (
    _PROMPT_FILE,
    _build_parser,
    _extract_json_object,
    _load_system_prompt,
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
# _extract_json_object
# ---------------------------------------------------------------------------


def test_extract_json_object_passes_through_clean_json() -> None:
    src = '{"symbol": "02319.HK", "n": 1}'
    assert _extract_json_object(src) == src


def test_extract_json_object_handles_trailing_prose() -> None:
    """LLM sometimes appends explanatory text after the JSON object."""
    src = '{"symbol": "02319.HK"}\n如有需要可继续追问。'
    assert _extract_json_object(src) == '{"symbol": "02319.HK"}'


def test_extract_json_object_handles_leading_prose() -> None:
    src = '分析结果如下:\n{"symbol": "02319.HK"}'
    assert _extract_json_object(src) == '{"symbol": "02319.HK"}'


def test_extract_json_object_handles_nested_braces() -> None:
    src = '{"outer": {"inner": [1, 2, 3]}, "x": 1} trailing'
    extracted = _extract_json_object(src)
    assert json.loads(extracted) == {"outer": {"inner": [1, 2, 3]}, "x": 1}


def test_extract_json_object_ignores_braces_in_strings() -> None:
    src = '{"text": "contains { and } braces", "n": 1} noise'
    extracted = _extract_json_object(src)
    assert json.loads(extracted) == {"text": "contains { and } braces", "n": 1}


def test_extract_json_object_raises_when_no_json() -> None:
    with pytest.raises(ValueError, match="no JSON object"):
        _extract_json_object("just plain text, no braces")


def test_extract_json_object_raises_on_malformed_json() -> None:
    with pytest.raises(ValueError, match="no JSON object"):
        _extract_json_object("{not valid json")


def test_extract_json_object_picks_longest_when_multiple_objects() -> None:
    """LLM sometimes emits a sub-object first, then the full answer.

    Example: a bare ``Verdict`` object followed by the complete
    ``StockAnalysis``. The full answer is always longer than any
    sub-object, so picking by length recovers the right one.
    """
    short = '{"decision": "watch", "summary": "缺数据"}'
    long_obj = (
        '{"symbol": "02319.HK", "company_profile": "' + ("x" * 200) + '",'
        ' "verdict": {"decision": "watch"}, "price_plan": {"current_price": 1},'
        ' "scores": {"fundamental": 5, "technical": 5, "news_catalyst": 5,'
        ' "peer_positioning": 5, "weighted_total": 5}}'
    )
    text = f"先来个 teaser: {short}\n然后是完整答案: {long_obj}\n收尾。"
    extracted = _extract_json_object(text)
    assert extracted == long_obj


# ---------------------------------------------------------------------------
# render_markdown
# ---------------------------------------------------------------------------


def _sample_analysis() -> StockAnalysis:
    return StockAnalysis(
        symbol="02319.HK",
        company_profile="### 公司画像:蒙牛乳业\n\n#### 1. 公司简介\n- 乳制品行业龙头",
        verdict=Verdict(
            decision="buy_in",
            decision_label="买进",
            confidence="high",
            summary="基本面扎实,技术形态向好,建议买进。",
        ),
        price_plan=PricePlan(
            current_price=16.06,
            entry_zone=[15.5, 15.8],
            add_zone=[14.0, 14.5],
            target_price=18.5,
            stop_loss=13.5,
            expected_return="+15% ~ +25%",
            risk_reward_ratio="2.5:1",
            time_horizon="1-3 个月",
        ),
        scores=Scores(
            fundamental=7.5,
            technical=6.0,
            news_catalyst=5.5,
            peer_positioning=6.5,
            weighted_total=6.6,
        ),
        fundamental_analysis=DimensionAnalysis(
            highlights=["ROE 持续 > 15%(akshare 报)", "营收同比 +5%"],
            concerns=["原奶价格上行"],
        ),
        technical_analysis=DimensionAnalysis(
            highlights=["站上 20 日均线"],
            concerns=["成交量未放大"],
        ),
        news_catalysts=["半年报发布(2026-08, 公司公告)"],
        peer_compare="伊利 PE 略低,光明乳业规模较小",
        risks=[
            Risk(type="行业", description="原奶价格波动", severity="medium"),
        ],
        action_plan=ActionPlan(
            position_size="建议占总资金 5-10%",
            execution=["分批:首笔 50% 在 entry_zone 上沿"],
            review_triggers=["触及止损位", "基本面重大利空"],
        ),
        reasoning_chain=(
            "按 Step 4 框架,基本面 35% × 7.5 = 2.625,技术面 25% × 6.0 = 1.5,"
            "消息面 20% × 5.5 = 1.1,同行对比 20% × 6.5 = 1.3,加权 6.525 ≈ 6.6。"
            "落入 5.5~7.0 → watch 区间,但 ROE 持续 > 15% 触发加分至 buy_in。"
        ) * 3,
    )


def test_render_markdown_contains_title_and_timestamp() -> None:
    md = render_markdown(_sample_analysis())
    assert "# 02319.HK 分析报告" in md
    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", md)


def test_render_markdown_contains_every_section() -> None:
    md = render_markdown(_sample_analysis())
    for header in (
        "## 投资决策",
        "## 价位推算",
        "## 评分",
        "## 公司画像",
        "## 基本面分析",
        "## 技术面分析",
        "## 近期催化",
        "## 同行对比",
        "## 风险",
        "## 操作建议",
        "## 推理链",
    ):
        assert header in md, f"missing section: {header}"


def test_render_markdown_contains_field_values() -> None:
    md = render_markdown(_sample_analysis())
    # Verdict
    assert "买进" in md
    assert "buy_in" in md
    assert "high" in md
    # Price plan
    assert "16.06" in md
    assert "15.5 ~ 15.8" in md
    assert "2.5:1" in md
    # Company profile (from sample)
    assert "蒙牛乳业" in md
    # Risks table
    assert "行业" in md
    assert "medium" in md
    assert "原奶价格波动" in md
    # Reasoning chain
    assert "Step 4 框架" in md


def test_render_markdown_handles_empty_optional_lists() -> None:
    """Empty news_catalysts / risks / dimension lists should render, not crash."""
    a = StockAnalysis(
        symbol="02319.HK",
        company_profile="简略画像",
        verdict=Verdict(decision="watch", decision_label="观望", confidence="low", summary="观望。"),
        price_plan=PricePlan(
            current_price=10, entry_zone=[9, 9.5], add_zone=[8, 8.5],
            target_price=12, stop_loss=7,
            expected_return="+20%", risk_reward_ratio="2:1", time_horizon="3 个月",
        ),
        scores=Scores(
            fundamental=5, technical=5, news_catalyst=5, peer_positioning=5, weighted_total=5,
        ),
        fundamental_analysis=DimensionAnalysis(),
        technical_analysis=DimensionAnalysis(),
        news_catalysts=[],
        peer_compare="N/A",
        risks=[],
        action_plan=ActionPlan(position_size="5%"),
        reasoning_chain="数据有限,维持观望。",
    )
    md = render_markdown(a)
    assert "暂无" not in md  # no junk
    assert "_无_" in md  # risks + news_catalysts fall back to _无_


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
# _load_system_prompt
# ---------------------------------------------------------------------------


def test_prompt_file_exists() -> None:
    """The bundled prompt template must be on disk at the expected path.

    Guards against the wheel being mis-built without the ``prompts/`` tree.
    """
    assert _PROMPT_FILE.is_file()
    assert _PROMPT_FILE.name == "system_prompt.md"


def test_load_system_prompt_fills_all_placeholders() -> None:
    """All three template variables must be filled, no raw braces left."""
    prompt = _load_system_prompt(
        symbol="02319.HK", include_peers=True, include_web_search=True,
    )
    assert "02319.HK" in prompt
    assert "include_peers 为 True" in prompt
    assert "视需要调用 web_search" in prompt
    for placeholder in ("{symbol}", "{include_clause}", "{web_search_clause}"):
        assert placeholder not in prompt, f"unfilled placeholder: {placeholder}"


def test_load_system_prompt_reflects_include_peers_false() -> None:
    prompt = _load_system_prompt(
        symbol="02319.HK", include_peers=False, include_web_search=True,
    )
    assert "include_peers 为 False" in prompt
    assert "include_peers 为 True" not in prompt


def test_load_system_prompt_reflects_web_search_disabled() -> None:
    prompt = _load_system_prompt(
        symbol="02319.HK", include_peers=True, include_web_search=False,
    )
    assert "没有 web_search 工具" in prompt
    assert "视需要调用 web_search" not in prompt


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
    """New-schema JSON payload (matches prompts/system_prompt.md contract)."""
    return json.dumps(
        {
            "symbol": "02319.HK",
            "company_profile": "### 公司画像:蒙牛乳业\n\n乳制品行业龙头。",
            "verdict": {
                "decision": "buy_in",
                "decision_label": "买进",
                "confidence": "high",
                "summary": "基本面扎实,值得买进。",
            },
            "price_plan": {
                "current_price": 16.06,
                "entry_zone": [15.5, 15.8],
                "add_zone": [14.0, 14.5],
                "target_price": 18.5,
                "stop_loss": 13.5,
                "expected_return": "+15% ~ +25%",
                "risk_reward_ratio": "2.5:1",
                "time_horizon": "1-3 个月",
            },
            "scores": {
                "fundamental": 7.5,
                "technical": 6.0,
                "news_catalyst": 5.5,
                "peer_positioning": 6.5,
                "weighted_total": 6.6,
            },
            "fundamental_analysis": {
                "highlights": ["ROE 持续 > 15%"],
                "concerns": ["原奶价格上行"],
            },
            "technical_analysis": {
                "highlights": ["站上 20 日均线"],
                "concerns": [],
            },
            "news_catalysts": ["半年报发布"],
            "peer_compare": "N/A",
            "risks": [
                {"type": "行业", "description": "原奶价格波动", "severity": "medium"},
            ],
            "action_plan": {
                "position_size": "占总资金 5-10%",
                "execution": ["分批建仓"],
                "review_triggers": ["触及止损"],
            },
            "reasoning_chain": "按 Step 4 框架,加权 6.6,落入 buy_in 区间。" * 5,
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
    assert "## 投资决策" in text
    assert "## 价位推算" in text
    assert "乳制品行业龙头。" in text  # from company_profile


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