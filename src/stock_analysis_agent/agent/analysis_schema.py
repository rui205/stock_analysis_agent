"""Pydantic schema for the JSON contract between the LLM agent and downstream tools.

The authoritative definition of this schema lives in
``prompts/system_prompt.md`` (read by ``script.analyze_stock._load_system_prompt``).
The pydantic models here mirror that contract: the LLM is told to emit a
JSON object matching these models, and ``script.analyze_stock.run`` then
validates the final ``AIMessage`` content against :class:`StockAnalysis`
before rendering Markdown.

Why ``Literal`` types and tight ``Field`` constraints? The system prompt
is prescriptive — it gives the LLM a fixed enum of decision /
confidence / risk-type / risk-severity values, and a 0-10 scale for
scores. Strict validation here means a hallucinated category or an
out-of-range score fails loudly with ``ValidationError`` instead of
silently making it into the report.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class Verdict(BaseModel):
    """The LLM's investment decision for the stock.

    The 5-tier enum mirrors the decision vocabulary in
    ``prompts/system_prompt.md`` (强烈买入 / 买入 / 持有 / 卖出 / 强烈卖出).
    The English ``decision`` is the machine-readable enum; the Chinese
    ``decision_label`` is the user-facing label — they must agree.

    Mapping::

        strongly_buy  → 强烈买入
        buy           → 买入
        hold          → 持有
        sell          → 卖出
        strongly_sell → 强烈卖出
    """

    decision: Literal["strongly_buy", "buy", "hold", "sell", "strongly_sell"]
    decision_label: str = Field(
        min_length=1,
        description=(
            "中文标签,必须与 decision 一一对应: "
            "strongly_buy→强烈买入, buy→买入, hold→持有, "
            "sell→卖出, strongly_sell→强烈卖出"
        ),
    )
    confidence: Literal["high", "medium", "low"]
    summary: str = Field(
        min_length=1, description="一句话核心判断,30-80 字"
    )


class PricePlan(BaseModel):
    """The LLM's recommended price levels for the stock.

    The prompt requires ``entry_zone`` and ``add_zone`` to be exactly two
    floats (low, high) — the model rejects anything else.
    """

    current_price: float
    entry_zone: list[float] = Field(min_length=2, max_length=2)
    add_zone: list[float] = Field(min_length=2, max_length=2)
    target_price: float
    stop_loss: float
    expected_return: str = Field(min_length=1, description='如 "+15% ~ +25%"')
    risk_reward_ratio: str = Field(min_length=1, description='如 "2.5:1"')
    time_horizon: str = Field(min_length=1, description='如 "1-3 个月"')


class Scores(BaseModel):
    """Multi-dimensional scoring on a 0-10 scale.

    The system prompt's Step 4 fixes the dimensions and their weights
    (基本面 35 / 技术面 25 / 消息面 20 / 同行对比 20). ``weighted_total``
    is what the LLM should compute, not a derived field, so we don't
    cross-check it against the components — the LLM is the source of
    truth for the final number.
    """

    fundamental: float = Field(ge=0, le=10)
    technical: float = Field(ge=0, le=10)
    news_catalyst: float = Field(ge=0, le=10)
    peer_positioning: float = Field(ge=0, le=10)
    weighted_total: float = Field(ge=0, le=10)


class DimensionAnalysis(BaseModel):
    """A dimension (fundamental / technical) split into highlights and concerns."""

    highlights: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)


class Risk(BaseModel):
    """A single risk item."""

    type: Literal["行业", "政策", "财务", "估值", "流动性", "治理"]
    description: str = Field(min_length=1)
    severity: Literal["high", "medium", "low"]


class ActionPlan(BaseModel):
    """Concrete actions the LLM recommends."""

    position_size: str = Field(min_length=1, description='如 "建议占总资金 5-10%"')
    execution: list[str] = Field(default_factory=list)
    review_triggers: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class StockAnalysis(BaseModel):
    """Structured analysis returned by ``StockAnalysisAgent``.

    The system prompt in ``prompts/system_prompt.md`` describes the JSON
    shape the LLM must emit. This model is the typed mirror of that
    contract; the entry script ``script.analyze_stock.run`` validates
    the final ``AIMessage`` content against this model before rendering
    Markdown.
    """

    symbol: str = Field(min_length=1)
    company_profile: str = Field(
        min_length=1,
        description=(
            "七段式公司画像,按 stock-snapshot-format skill 渲染,"
            "不含同业对比表(同业对比放在 peer_compare)"
        ),
    )
    verdict: Verdict
    price_plan: PricePlan
    scores: Scores
    fundamental_analysis: DimensionAnalysis
    technical_analysis: DimensionAnalysis
    news_catalysts: list[str] = Field(default_factory=list)
    peer_compare: str = Field(
        min_length=1,
        description="2-4 句同行对比;若 include_peers 为 False 写 'N/A'",
    )
    risks: list[Risk] = Field(default_factory=list)
    action_plan: ActionPlan
    reasoning_chain: str = Field(
        min_length=1,
        description="500-1200 字完整推理,按 Step 4-5 走完",
    )


__all__ = ["StockAnalysis"]