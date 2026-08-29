"""Pydantic schema for the strategy-match JSON contract.

The authoritative definition lives in ``skill/strategy-match/SKILL.md``;
this module mirrors it so the script layer can validate the LLM's final
``AIMessage`` content via :meth:`StrategyMatchReport.model_validate_json`
before rendering markdown or publishing to Feishu.

The enum (``overall_fit`` / ``match_level`` / ``confidence``) is a tight
``Literal`` set so a hallucinated category fails loudly at the validator
instead of silently making it into the report.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class StrategyCriterionMatch(BaseModel):
    """One strategy principle's match result.

    Attributes:
        criterion: Quoted text of the strategy principle being evaluated.
        match_level: One of ``fit`` / ``partial`` / ``mismatch``.
        evidence: Concrete fundamental data backing the judgment.
        reasoning: Why the level was chosen.
    """

    criterion: str = Field(min_length=1, max_length=200)
    match_level: Literal["fit", "partial", "mismatch"]
    evidence: str = Field(min_length=1, max_length=500)
    reasoning: str = Field(min_length=1, max_length=500)


class DataSourceBreakdown(BaseModel):
    """What each data source contributed to the report.

    Attributes:
        stock_analysis: Key info from the stock_analysis subagent
            (verdict / score / key risks, etc.).
        stock_analysis_url: The Feishu doc URL the stock_analysis
            subagent published (the ``🔗`` link it returned verbatim);
            empty string when the sub-agent did not publish (e.g.
            lark-cli not authenticated).
        deepresearch: Supplementary info from the deepresearch subagent;
            empty string when deepresearch was not called.
        technical_capital: Supplementary info from the technical +
            capital-flow subagent (trend / key levels / timing signals);
            empty string when ``run_technical_capital`` was not called.
    """

    stock_analysis: str = Field(min_length=1, max_length=2000)
    stock_analysis_url: str = Field(default="", max_length=500)
    deepresearch: str = Field(default="", max_length=2000)
    technical_capital: str = Field(default="", max_length=2000)


class StrategyMatchReport(BaseModel):
    """Structured strategy-match output returned by ``StrategyMatchAgent``.

    The system prompt in ``prompts/strategy_match_system_prompt.md``
    describes the JSON shape the LLM must emit; this model is its
    typed mirror.
    """

    symbol: str = Field(min_length=1)
    strategy_name: str = Field(min_length=1)
    strategy_version: str = Field(min_length=1)
    overall_fit: Literal["buy", "hold", "avoid"]
    fit_score: float = Field(ge=0, le=10)
    summary: str = Field(min_length=1, max_length=200)
    criterion_matches: list[StrategyCriterionMatch] = Field(min_length=1)
    data_sources: DataSourceBreakdown
    judgment_rationale: str = Field(min_length=1, max_length=1500)
    action_recommendation: str = Field(min_length=1, max_length=300)
    confidence: Literal["high", "medium", "low"]


__all__ = ["DataSourceBreakdown", "StrategyCriterionMatch", "StrategyMatchReport"]
