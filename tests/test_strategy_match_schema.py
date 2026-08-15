"""Tests for the StrategyMatchReport schema."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from stock_analysis_agent.agent.strategy_match_schema import (
    StrategyCriterionMatch,
    StrategyMatchReport,
)


def _valid_criterion() -> StrategyCriterionMatch:
    return StrategyCriterionMatch(
        criterion="PE < 15",
        match_level="fit",
        evidence="current PE-TTM is 12.3",
        reasoning="well within band",
    )


def _valid_report_kwargs() -> dict:
    return {
        "symbol": "600519.SH",
        "strategy_name": "value-investing",
        "strategy_version": "1",
        "overall_fit": "buy",
        "fit_score": 8.5,
        "summary": "fits well; cheap, high quality, low debt",
        "criterion_matches": [_valid_criterion()],
        "raw_analysis_excerpt": "verdict=buy score=8.2",
        "action_recommendation": "build 5% position in entry zone",
        "confidence": "high",
    }


def _valid_report() -> StrategyMatchReport:
    return StrategyMatchReport(**_valid_report_kwargs())


def _with(**overrides) -> StrategyMatchReport:
    """Build a StrategyMatchReport with field overrides (validated on construct)."""
    return StrategyMatchReport(**{**_valid_report_kwargs(), **overrides})


class TestStrategyCriterionMatch:
    def test_minimal_valid(self) -> None:
        c = _valid_criterion()
        assert c.match_level == "fit"

    @pytest.mark.parametrize("level", ["fit", "partial", "mismatch"])
    def test_accepts_all_match_levels(self, level: str) -> None:
        c = StrategyCriterionMatch(
            criterion="x", match_level=level, evidence="e", reasoning="r"
        )
        assert c.match_level == level

    def test_rejects_unknown_match_level(self) -> None:
        with pytest.raises(ValidationError):
            StrategyCriterionMatch(
                criterion="x", match_level="maybe", evidence="e", reasoning="r"
            )


class TestStrategyMatchReport:
    def test_minimal_valid(self) -> None:
        r = _valid_report()
        assert r.symbol == "600519.SH"
        assert r.fit_score == 8.5

    @pytest.mark.parametrize("fit", ["buy", "hold", "avoid"])
    def test_accepts_all_overall_fit_values(self, fit: str) -> None:
        r = _with(overall_fit=fit)
        assert r.overall_fit == fit

    def test_rejects_unknown_overall_fit(self) -> None:
        with pytest.raises(ValidationError):
            _with(overall_fit="strong_buy")

    @pytest.mark.parametrize("score", [-0.1, -1, 10.1, 11, 100])
    def test_rejects_fit_score_out_of_range(self, score: float) -> None:
        with pytest.raises(ValidationError):
            _with(fit_score=score)

    def test_rejects_empty_criterion_matches(self) -> None:
        with pytest.raises(ValidationError):
            _with(criterion_matches=[])

    def test_rejects_empty_symbol(self) -> None:
        with pytest.raises(ValidationError):
            _with(symbol="")

    def test_rejects_empty_strategy_name(self) -> None:
        with pytest.raises(ValidationError):
            _with(strategy_name="")


class TestOverlongFields:
    """Bounded strings must not accept runaway LLM output."""

    @pytest.mark.parametrize(
        ("field", "length"),
        [("criterion", 201), ("evidence", 501), ("reasoning", 501)],
    )
    def test_criterion_match_rejects_overlong(self, field: str, length: int) -> None:
        kwargs = {
            "criterion": "x",
            "match_level": "fit",
            "evidence": "e",
            "reasoning": "r",
        }
        kwargs[field] = "x" * length
        with pytest.raises(ValidationError):
            StrategyCriterionMatch(**kwargs)

    @pytest.mark.parametrize(
        ("field", "length"),
        [("raw_analysis_excerpt", 2001), ("action_recommendation", 301)],
    )
    def test_report_rejects_overlong(self, field: str, length: int) -> None:
        with pytest.raises(ValidationError):
            _with(**{field: "x" * length})
