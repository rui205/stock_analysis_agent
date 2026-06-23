"""Pydantic schema for the JSON contract between the LLM agent and the Feishu uploader."""
from __future__ import annotations

from pydantic import BaseModel, Field


class StockAnalysis(BaseModel):
    """Structured analysis returned by ``StockAnalysisAgent``.

    The system prompt in :mod:`stock_analysis_agent.agent.stock_analysis`
    asks the LLM to emit a JSON object matching this schema. The entry
    script ``script/analyze_stock.py`` validates the final AIMessage
    content against this model before rendering Markdown.
    """

    symbol: str = Field(min_length=1)
    summary: str = Field(min_length=20)
    fundamentals: str
    technicals: str
    peer_compare: str
    news: str
    risks: str
    recommendation: str


__all__ = ["StockAnalysis"]
