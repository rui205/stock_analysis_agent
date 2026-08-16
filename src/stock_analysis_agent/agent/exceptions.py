"""Custom exception types for stock_analysis_agent.agent."""
from __future__ import annotations


class ToolExecutionError(RuntimeError):
    """Raised when a tool call fails after exhausting retries.

    The original exception is preserved in `__cause__`.
    """


class AgentTimeoutError(TimeoutError):
    """Raised when an agent run exceeds its configured wall-clock timeout.

    ``__cause__`` is the underlying :class:`TimeoutError` from
    :func:`asyncio.wait_for` / :class:`asyncio.timeout`.
    """
