"""Tests for the internal _ToolRetryMiddleware."""
from __future__ import annotations

import time
from typing import Any

import pytest
from langchain_core.messages import ToolCall, ToolMessage

from stock_analysis_agent.agent.middleware import _ToolRetryMiddleware
from stock_analysis_agent.agent.exceptions import ToolExecutionError


def _make_request(call_id: str = "call_1") -> Any:
    """Build a minimal ToolCallRequest-like object for unit tests."""
    from langchain.agents.middleware.types import ToolCallRequest

    return ToolCallRequest(
        tool_call=ToolCall(
            name="t", args={}, id=call_id, type="tool_call"
        ),
        tool=None,
        state=None,
        runtime=None,
    )


def test_transient_error_is_retried_then_raises() -> None:
    """Spec test 4 part 1: a transient error must be retried up to
    `max_retries` times; if all attempts fail, ToolExecutionError is raised."""
    mw = _ToolRetryMiddleware(max_retries=2, initial_delay=0.0, backoff_factor=0.0)
    calls = {"n": 0}

    def handler(req):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        raise TimeoutError("network blip")

    req = _make_request()
    with pytest.raises(ToolExecutionError) as ei:
        mw.wrap_tool_call(req, handler)

    assert calls["n"] == 3  # 1 initial + 2 retries
    assert isinstance(ei.value.__cause__, TimeoutError)


def test_business_error_is_not_retried() -> None:
    """Non-transient errors (e.g. ValueError) must NOT be retried;
    they raise ToolExecutionError immediately."""
    mw = _ToolRetryMiddleware(max_retries=5, initial_delay=0.0, backoff_factor=0.0)
    calls = {"n": 0}

    def handler(req):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        raise ValueError("bad input")

    req = _make_request()
    with pytest.raises(ToolExecutionError) as ei:
        mw.wrap_tool_call(req, handler)

    assert calls["n"] == 1
    assert isinstance(ei.value.__cause__, ValueError)


def test_successful_call_returns_handler_result() -> None:
    """When the handler succeeds, its return value is forwarded unchanged."""
    mw = _ToolRetryMiddleware(max_retries=2, initial_delay=0.0, backoff_factor=0.0)
    expected = ToolMessage(content="ok", tool_call_id="call_1")

    def handler(req):  # type: ignore[no-untyped-def]
        return expected

    result = mw.wrap_tool_call(_make_request(), handler)
    assert result is expected


def test_max_retries_zero_means_single_attempt() -> None:
    """max_retries=0 means no retries; the first failure raises."""
    mw = _ToolRetryMiddleware(max_retries=0, initial_delay=0.0, backoff_factor=0.0)
    calls = {"n": 0}

    def handler(req):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        raise TimeoutError("nope")

    with pytest.raises(ToolExecutionError):
        mw.wrap_tool_call(_make_request(), handler)

    assert calls["n"] == 1


def test_exponential_backoff_caps_at_max_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backoff sleep durations should follow min(2**attempt * factor, max_delay)."""
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    mw = _ToolRetryMiddleware(
        max_retries=4, initial_delay=1.0, backoff_factor=2.0, max_delay=3.0
    )

    def handler(req):  # type: ignore[no-untyped-def]
        raise TimeoutError("x")

    with pytest.raises(ToolExecutionError):
        mw.wrap_tool_call(_make_request(), handler)

    # First three sleeps grow exponentially: 1, 2, 4 — capped to 3 starting at attempt 2.
    # Attempts 0..3 (4 retries) → sleeps after attempts 0,1,2,3 = [1, 2, 3, 3]
    assert sleeps == [1.0, 2.0, 3.0, 3.0], f"unexpected backoff sequence: {sleeps!r}"
