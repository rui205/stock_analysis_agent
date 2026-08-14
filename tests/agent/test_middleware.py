"""Tests for the internal _ToolRetryMiddleware."""
from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import ToolCall, ToolMessage

from stock_analysis_agent.agent.middleware import _FeedbackMiddleware, _ToolRetryMiddleware
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


def test_business_error_gets_one_unexpected_retry_by_default() -> None:
    """Non-transient errors (e.g. ValueError) get a small unexpected-retry
    budget (default 1) to absorb flaky failures that are not classified
    transient; once exhausted they raise ToolExecutionError."""
    mw = _ToolRetryMiddleware(max_retries=5, initial_delay=0.0, backoff_factor=0.0)
    calls = {"n": 0}

    def handler(req):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        raise ValueError("bad input")

    req = _make_request()
    with pytest.raises(ToolExecutionError) as ei:
        mw.wrap_tool_call(req, handler)

    assert calls["n"] == 2  # 1 initial + 1 unexpected retry
    assert isinstance(ei.value.__cause__, ValueError)


def test_business_error_recovers_on_unexpected_retry() -> None:
    """A flaky non-transient failure that succeeds on the second attempt
    must return the handler result instead of aborting the run."""
    mw = _ToolRetryMiddleware(max_retries=5, initial_delay=0.0, backoff_factor=0.0)
    calls = {"n": 0}
    expected = ToolMessage(content="ok", tool_call_id="call_1")

    def handler(req):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("flaky first attempt")
        return expected

    result = mw.wrap_tool_call(_make_request(), handler)
    assert result is expected
    assert calls["n"] == 2


def test_unexpected_retries_zero_means_no_business_retry() -> None:
    """``unexpected_retries=0`` restores the old fail-fast behavior."""
    mw = _ToolRetryMiddleware(
        max_retries=5, initial_delay=0.0, backoff_factor=0.0, unexpected_retries=0
    )
    calls = {"n": 0}

    def handler(req):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        raise ValueError("bad input")

    with pytest.raises(ToolExecutionError):
        mw.wrap_tool_call(_make_request(), handler)

    assert calls["n"] == 1


def test_keyboard_interrupt_propagates_sync() -> None:
    """``KeyboardInterrupt`` must NOT be swallowed by the retry layer —
    Ctrl+C has to reach the top level instead of being wrapped into a
    ToolExecutionError."""
    mw = _ToolRetryMiddleware(max_retries=2, initial_delay=0.0, backoff_factor=0.0)

    def handler(req):  # type: ignore[no-untyped-def]
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        mw.wrap_tool_call(_make_request(), handler)


async def test_business_error_gets_one_unexpected_retry_by_default_async() -> None:
    """Async path mirrors the sync unexpected-retry budget."""
    mw = _ToolRetryMiddleware(max_retries=5, initial_delay=0.0, backoff_factor=0.0)
    calls = {"n": 0}

    async def handler(req):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        raise ValueError("bad input")

    with pytest.raises(ToolExecutionError) as ei:
        await mw.awrap_tool_call(_make_request(), handler)

    assert calls["n"] == 2  # 1 initial + 1 unexpected retry
    assert isinstance(ei.value.__cause__, ValueError)


async def test_keyboard_interrupt_propagates_async() -> None:
    """``KeyboardInterrupt`` must escape the async retry layer unwrapped."""
    mw = _ToolRetryMiddleware(max_retries=2, initial_delay=0.0, backoff_factor=0.0)

    async def handler(req):  # type: ignore[no-untyped-def]
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        await mw.awrap_tool_call(_make_request(), handler)


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


# ---------------------------------------------------------------------------
# _FeedbackMiddleware — degrade exhausted tool errors back to the LLM
# ---------------------------------------------------------------------------


def test_feedback_converts_tool_execution_error_to_error_toolmessage() -> None:
    """Exhausted ``ToolExecutionError`` becomes an error ToolMessage the
    LLM can read — [ERROR] prefix, original message, correct call id."""
    mw = _FeedbackMiddleware(failure_budget=3)

    def handler(req):  # type: ignore[no-untyped-def]
        raise ToolExecutionError("Tool 'load_skill' failed: skill 'nope' not found")

    msg = mw.wrap_tool_call(_make_request(), handler)

    assert isinstance(msg, ToolMessage)
    assert msg.status == "error"
    assert msg.tool_call_id == "call_1"
    assert msg.content.startswith("[ERROR] ")
    assert "Tool 'load_skill' failed: skill 'nope' not found" in msg.content


def test_feedback_success_resets_consecutive_counter() -> None:
    """A successful tool call resets the budget: fail→succeed→fail must
    still degrade instead of raising."""
    mw = _FeedbackMiddleware(failure_budget=1)
    req = _make_request()

    def failing(req):  # type: ignore[no-untyped-def]
        raise ToolExecutionError("boom")

    def ok(req):  # type: ignore[no-untyped-def]
        return ToolMessage(content="ok", tool_call_id="call_1")

    assert isinstance(mw.wrap_tool_call(req, failing), ToolMessage)  # count 1 ≤ 1
    assert mw.wrap_tool_call(req, ok).content == "ok"  # reset
    assert isinstance(mw.wrap_tool_call(req, failing), ToolMessage)  # count 1 again


def test_feedback_budget_exhausted_raises() -> None:
    """After ``failure_budget`` consecutive failures, the next failure
    raises a ToolExecutionError naming the budget (original as __cause__)."""
    mw = _FeedbackMiddleware(failure_budget=2)
    req = _make_request()

    def failing(req):  # type: ignore[no-untyped-def]
        raise ToolExecutionError("boom")

    assert isinstance(mw.wrap_tool_call(req, failing), ToolMessage)  # count 1
    assert isinstance(mw.wrap_tool_call(req, failing), ToolMessage)  # count 2
    with pytest.raises(ToolExecutionError) as ei:
        mw.wrap_tool_call(req, failing)  # count 3 > 2 → raise

    assert "budget" in str(ei.value)
    assert isinstance(ei.value.__cause__, ToolExecutionError)


def test_feedback_zero_budget_is_fail_fast() -> None:
    """``failure_budget=0``: the very first failure raises immediately."""
    mw = _FeedbackMiddleware(failure_budget=0)

    def failing(req):  # type: ignore[no-untyped-def]
        raise ToolExecutionError("boom")

    with pytest.raises(ToolExecutionError):
        mw.wrap_tool_call(_make_request(), failing)


def test_feedback_lets_other_exceptions_pass_through() -> None:
    """Only ``ToolExecutionError`` is degraded; anything else propagates."""
    mw = _FeedbackMiddleware(failure_budget=3)

    def handler(req):  # type: ignore[no-untyped-def]
        raise RuntimeError("not wrapped")

    with pytest.raises(RuntimeError):
        mw.wrap_tool_call(_make_request(), handler)


def test_feedback_reads_tool_call_id_from_dict_shape() -> None:
    """``request.tool_call`` may be a plain dict at runtime — the
    degraded ToolMessage must still carry the correct id."""
    from langchain.agents.middleware.types import ToolCallRequest

    mw = _FeedbackMiddleware(failure_budget=3)
    req = ToolCallRequest(
        tool_call={"name": "t", "args": {}, "id": "call_dict_1", "type": "tool_call"},
        tool=None,
        state=None,
        runtime=None,
    )

    def handler(req):  # type: ignore[no-untyped-def]
        raise ToolExecutionError("boom")

    msg = mw.wrap_tool_call(req, handler)
    assert isinstance(msg, ToolMessage)
    assert msg.tool_call_id == "call_dict_1"


async def test_feedback_converts_tool_execution_error_async() -> None:
    """Async path mirrors the sync degrade behavior."""
    mw = _FeedbackMiddleware(failure_budget=3)

    async def handler(req):  # type: ignore[no-untyped-def]
        raise ToolExecutionError("boom")

    msg = await mw.awrap_tool_call(_make_request(), handler)

    assert isinstance(msg, ToolMessage)
    assert msg.status == "error"
    assert msg.content.startswith("[ERROR] ")
    assert "boom" in msg.content


async def test_feedback_budget_exhausted_raises_async() -> None:
    """Async path mirrors the budget-exhaustion raise."""
    mw = _FeedbackMiddleware(failure_budget=0)

    async def handler(req):  # type: ignore[no-untyped-def]
        raise ToolExecutionError("boom")

    with pytest.raises(ToolExecutionError):
        await mw.awrap_tool_call(_make_request(), handler)


async def test_feedback_lets_other_exceptions_pass_through_async() -> None:
    """Async path lets non-ToolExecutionError exceptions propagate."""
    mw = _FeedbackMiddleware(failure_budget=3)

    async def handler(req):  # type: ignore[no-untyped-def]
        raise RuntimeError("not wrapped")

    with pytest.raises(RuntimeError):
        await mw.awrap_tool_call(_make_request(), handler)
