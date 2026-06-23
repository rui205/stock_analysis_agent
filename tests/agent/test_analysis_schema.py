"""Tests for the StockAnalysis pydantic schema (new system_prompt.md contract)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from stock_analysis_agent.agent.analysis_schema import (
    ActionPlan,
    DimensionAnalysis,
    PricePlan,
    Risk,
    Scores,
    StockAnalysis,
    Verdict,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _verdict(**overrides) -> dict:
    base = {
        "decision": "buy_in",
        "decision_label": "买进",
        "confidence": "high",
        "summary": "基本面扎实,技术形态向好,建议买进。",
    }
    base.update(overrides)
    return base


def _price_plan(**overrides) -> dict:
    base = {
        "current_price": 16.06,
        "entry_zone": [15.5, 15.8],
        "add_zone": [14.0, 14.5],
        "target_price": 18.5,
        "stop_loss": 13.5,
        "expected_return": "+15% ~ +25%",
        "risk_reward_ratio": "2.5:1",
        "time_horizon": "1-3 个月",
    }
    base.update(overrides)
    return base


def _scores(**overrides) -> dict:
    base = {
        "fundamental": 7.5,
        "technical": 6.0,
        "news_catalyst": 5.5,
        "peer_positioning": 6.5,
        "weighted_total": 6.6,
    }
    base.update(overrides)
    return base


def _dimension(**overrides) -> dict:
    return {
        "highlights": ["亮点 1,带数据来源", "亮点 2"],
        "concerns": ["隐忧 1"],
        **overrides,
    }


def _action_plan(**overrides) -> dict:
    return {
        "position_size": "建议占总资金 5-10%",
        "execution": ["分批:首笔 50% 在 entry_zone 上沿", "余下分两次加仓"],
        "review_triggers": ["触及止损位", "基本面重大利空"],
        **overrides,
    }


def _risk(**overrides) -> dict:
    return {
        "type": "行业",
        "description": "原奶价格波动",
        "severity": "medium",
        **overrides,
    }


def _full_payload(**overrides) -> dict:
    base = {
        "symbol": "02319.HK",
        "company_profile": "### 公司画像:蒙牛乳业\n\n#### 1. 公司简介\n...",
        "verdict": _verdict(),
        "price_plan": _price_plan(),
        "scores": _scores(),
        "fundamental_analysis": _dimension(),
        "technical_analysis": _dimension(),
        "news_catalysts": ["半年报发布(2026-08, 公司公告)"],
        "peer_compare": "伊利 PE 略低,光明乳业规模较小",
        "risks": [_risk()],
        "action_plan": _action_plan(),
        "reasoning_chain": "按 Step 4 框架,基本面 35% 权重给 7.5..." * 10,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class TestVerdict:
    def test_accepts_all_three_decisions(self) -> None:
        for d in ("buy_in", "watch", "no_buy"):
            Verdict(**_verdict(decision=d))

    def test_rejects_unknown_decision(self) -> None:
        with pytest.raises(ValidationError):
            Verdict(**_verdict(decision="strong_buy"))

    def test_rejects_unknown_confidence(self) -> None:
        with pytest.raises(ValidationError):
            Verdict(**_verdict(confidence="very_high"))


class TestPricePlan:
    def test_accepts_two_element_zones(self) -> None:
        pp = PricePlan(**_price_plan())
        assert pp.entry_zone == [15.5, 15.8]
        assert pp.add_zone == [14.0, 14.5]

    def test_rejects_entry_zone_with_one_element(self) -> None:
        with pytest.raises(ValidationError):
            PricePlan(**_price_plan(entry_zone=[15.5]))

    def test_rejects_entry_zone_with_three_elements(self) -> None:
        with pytest.raises(ValidationError):
            PricePlan(**_price_plan(entry_zone=[15.0, 15.5, 16.0]))


class TestScores:
    def test_accepts_boundary_values(self) -> None:
        Scores(**_scores(fundamental=0, technical=10, news_catalyst=0))
        Scores(**_scores(peer_positioning=10, weighted_total=0))

    def test_rejects_score_above_10(self) -> None:
        with pytest.raises(ValidationError):
            Scores(**_scores(fundamental=10.5))

    def test_rejects_negative_score(self) -> None:
        with pytest.raises(ValidationError):
            Scores(**_scores(technical=-0.1))


class TestRisk:
    def test_accepts_all_six_types(self) -> None:
        for t in ("行业", "政策", "财务", "估值", "流动性", "治理"):
            Risk(**_risk(type=t))

    def test_rejects_unknown_type(self) -> None:
        with pytest.raises(ValidationError):
            Risk(**_risk(type="市场"))

    def test_rejects_unknown_severity(self) -> None:
        with pytest.raises(ValidationError):
            Risk(**_risk(severity="critical"))


class TestDimensionAnalysis:
    def test_defaults_to_empty_lists(self) -> None:
        d = DimensionAnalysis()
        assert d.highlights == []
        assert d.concerns == []


class TestActionPlan:
    def test_defaults_to_empty_lists(self) -> None:
        ap = ActionPlan(position_size="占总资金 5%")
        assert ap.execution == []
        assert ap.review_triggers == []


# ---------------------------------------------------------------------------
# Top-level StockAnalysis
# ---------------------------------------------------------------------------


def test_stock_analysis_accepts_full_payload() -> None:
    a = StockAnalysis(**_full_payload())
    assert a.symbol == "02319.HK"
    assert a.verdict.decision == "buy_in"
    assert a.price_plan.current_price == 16.06
    assert a.scores.weighted_total == 6.6
    assert a.risks[0].type == "行业"


def test_stock_analysis_rejects_empty_symbol() -> None:
    with pytest.raises(ValidationError):
        StockAnalysis(**_full_payload(symbol=""))


def test_stock_analysis_rejects_missing_top_level_field() -> None:
    payload = _full_payload()
    payload.pop("reasoning_chain")
    with pytest.raises(ValidationError):
        StockAnalysis(**payload)


def test_stock_analysis_rejects_missing_nested_field() -> None:
    payload = _full_payload()
    payload["verdict"].pop("decision")
    with pytest.raises(ValidationError):
        StockAnalysis(**payload)


def test_stock_analysis_accepts_empty_optional_lists() -> None:
    """``news_catalysts`` / ``risks`` default to []; empty is legal."""
    a = StockAnalysis(
        **_full_payload(news_catalysts=[], risks=[]),
    )
    assert a.news_catalysts == []
    assert a.risks == []


def test_stock_analysis_accepts_peer_compare_n_a() -> None:
    """Per the prompt, peer_compare is the literal string 'N/A' when peers are off."""
    a = StockAnalysis(**_full_payload(peer_compare="N/A"))
    assert a.peer_compare == "N/A"
