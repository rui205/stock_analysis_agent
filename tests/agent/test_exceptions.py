"""Tests for stock_analysis_agent.agent.exceptions."""
from __future__ import annotations

import pytest

from stock_analysis_agent.agent.exceptions import ToolExecutionError


def test_tool_execution_error_is_runtime_error() -> None:
    """ToolExecutionError must inherit from RuntimeError so callers can
    catch it as a generic 'agent failed at runtime' signal."""
    err = ToolExecutionError("boom")
    assert isinstance(err, RuntimeError)


def test_tool_execution_error_preserves_cause() -> None:
    """The original exception must be reachable via __cause__ for debugging."""
    try:
        try:
            raise TimeoutError("network down")
        except TimeoutError as original:
            raise ToolExecutionError("tool failed") from original
    except ToolExecutionError as err:
        assert isinstance(err.__cause__, TimeoutError)
        assert "network down" in str(err.__cause__)


def test_tool_execution_error_message() -> None:
    """The error message should be the constructor argument verbatim."""
    assert str(ToolExecutionError("custom message")) == "custom message"
