"""Custom exception types for stock_analysis_agent.agent."""
from __future__ import annotations


class ToolExecutionError(RuntimeError):
    """Raised when a tool call fails after exhausting retries.

    The original exception is preserved in `__cause__`.
    """
