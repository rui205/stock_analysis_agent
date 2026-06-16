"""Reusable agents for stock_analysis_agent.

Public API:
    BaseAgent       — wrapper around langchain.agents.create_agent
    ToolExecutionError — raised when tool calls exhaust retries
"""
from __future__ import annotations

from stock_analysis_agent.agents.base import BaseAgent
from stock_analysis_agent.agents.exceptions import ToolExecutionError

__all__ = ["BaseAgent", "ToolExecutionError"]
