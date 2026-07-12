"""Reusable agents for stock_analysis_agent.

Public API:
    BaseAgent              — wrapper around langchain.agents.create_agent
    DeepSearchAgent        — concrete LLM-driven deep-research agent
    StockAnalysisAgent     — LLM-driven stock analysis (snapshot + web_search)
    StrategyMatchAgent     — LLM-driven strategy-vs-fundamentals matching
    StockAnalysis          — JSON contract returned by StockAnalysisAgent
    StrategyMatchReport    — JSON contract returned by StrategyMatchAgent
    StrategyCriterionMatch — single criterion match inside StrategyMatchReport
    ToolExecutionError     — raised when tool calls exhaust retries
"""
from __future__ import annotations

from stock_analysis_agent.agent.analysis_schema import StockAnalysis
from stock_analysis_agent.agent.base import BaseAgent
from stock_analysis_agent.agent.deepsearch import DeepSearchAgent
from stock_analysis_agent.agent.exceptions import ToolExecutionError
from stock_analysis_agent.agent.stock_analysis import StockAnalysisAgent
from stock_analysis_agent.agent.strategy_match import StrategyMatchAgent
from stock_analysis_agent.agent.strategy_match_schema import (
    StrategyCriterionMatch,
    StrategyMatchReport,
)

__all__ = [
    "BaseAgent",
    "DeepSearchAgent",
    "StockAnalysisAgent",
    "StockAnalysis",
    "StrategyCriterionMatch",
    "StrategyMatchAgent",
    "StrategyMatchReport",
    "ToolExecutionError",
]
