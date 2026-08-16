"""Tool-call middleware: transient-error retry and LLM error feedback.

Extracted from base.py so BaseAgent is small and focused on construction
+ streaming, while retry/backoff policy lives in its own module.
"""
from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from stock_analysis_agent.agent.exceptions import ToolExecutionError

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ModelRequest, ToolCallRequest


# Exceptions considered "transient" — retried up to max_retries.
_TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    TimeoutError,
    ConnectionError,
)


def _is_transient(exc: BaseException) -> bool:
    """A best-effort check: built-in transient types OR an `httpx` /
    `anthropic` exception whose class name contains 'Timeout' or 'Rate'."""
    if isinstance(exc, _TRANSIENT_EXCEPTIONS):
        return True
    cls_name = type(exc).__name__.lower()
    return any(token in cls_name for token in ("timeout", "ratelimit", "rate_limit"))


def _compute_backoff(
    attempt: int, initial_delay: float, backoff_factor: float, max_delay: float
) -> float:
    """Backoff for the given (0-indexed) attempt, capped at max_delay.

    Formula: `min(initial_delay * backoff_factor ** attempt, max_delay)`.
    For example, with `initial_delay=1.0, backoff_factor=2.0` the sequence
    is 1, 2, 4, 8, ...; with `initial_delay=1.0, backoff_factor=1.0` it is
    1, 1, 1, 1, ... (capped at max_delay).
    """
    return min(initial_delay * (backoff_factor ** attempt), max_delay)


def _tool_name(request: "ToolCallRequest") -> str:
    """Return the tool name from a ToolCallRequest.

    `request.tool_call` is typed as `ToolCall` but at runtime is a
    plain `dict` (TypedDict). This helper accepts both shapes so the
    middleware works regardless of the LangChain version's coercion.
    """
    tc = request.tool_call
    if isinstance(tc, dict):
        return str(tc.get("name", "<unknown>"))
    return str(getattr(tc, "name", "<unknown>"))


class _ToolRetryMiddleware(AgentMiddleware):
    """Retry tool calls on transient errors with exponential backoff.

    On the final failure, raise `ToolExecutionError` (from
    `stock_analysis_agent.agent.exceptions`) with the original
    exception preserved as `__cause__`.

    Business errors (anything not classified transient) get a small
    unexpected-retry budget (`unexpected_retries`, default 1) with a
    fixed `initial_delay` pause, to absorb flaky failures that are not
    classified transient (e.g. a subprocess hiccup). Once that budget
    is exhausted they are wrapped in `ToolExecutionError`. Set
    `unexpected_retries=0` to restore fail-fast behavior.

    `KeyboardInterrupt` / `SystemExit` are never caught — they
    propagate to the caller so Ctrl+C can abort a run.
    """

    def __init__(
        self,
        max_retries: int = 2,
        *,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 30.0,
        unexpected_retries: int = 1,
    ) -> None:
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.unexpected_retries = unexpected_retries

    def wrap_tool_call(
        self,
        request: "ToolCallRequest",
        handler: Callable[..., Any],
    ) -> Any:
        """Sync retry loop. Delegates sleeping to `time.sleep`."""
        return _retry_loop(
            request,
            handler,
            max_retries=self.max_retries,
            initial_delay=self.initial_delay,
            backoff_factor=self.backoff_factor,
            max_delay=self.max_delay,
            unexpected_retries=self.unexpected_retries,
            sleep_fn=time.sleep,
        )

    async def awrap_tool_call(
        self,
        request: "ToolCallRequest",
        handler: Callable[..., Any],
    ) -> Any:
        """Async retry loop. Delegates sleeping to `asyncio.sleep`.

        Implemented separately from `wrap_tool_call` because LangChain's
        `AgentMiddleware` requires both — leaving either as the inherited
        default raises `NotImplementedError` in the matching execution
        context (e.g. async agent streams).
        """
        return await _aretry_loop(
            request,
            handler,
            max_retries=self.max_retries,
            initial_delay=self.initial_delay,
            backoff_factor=self.backoff_factor,
            max_delay=self.max_delay,
            unexpected_retries=self.unexpected_retries,
            sleep_fn=asyncio.sleep,
        )


def _retry_loop(
    request: "ToolCallRequest",
    handler: Callable[..., Any],
    *,
    max_retries: int,
    initial_delay: float,
    backoff_factor: float,
    max_delay: float,
    unexpected_retries: int,
    sleep_fn: Callable[[float], Any],
) -> Any:
    """Shared retry loop. `sleep_fn` is `time.sleep` for sync callers.

    Transient errors are retried up to `max_retries` times with
    exponential backoff. Non-transient errors consume a separate,
    smaller `unexpected_retries` budget with a fixed `initial_delay`
    pause. Only `Exception` subclasses are caught — `KeyboardInterrupt`
    and `SystemExit` propagate to the caller.
    """
    last_exc: Exception | None = None
    total_attempts = max_retries + 1
    unexpected_left = unexpected_retries
    attempt = 0
    while attempt < total_attempts:
        try:
            return handler(request)
        except Exception as exc:  # noqa: BLE001 — retry layer by design
            last_exc = exc
            if _is_transient(exc):
                attempt += 1
                if attempt < total_attempts:
                    delay = _compute_backoff(
                        attempt - 1, initial_delay, backoff_factor, max_delay
                    )
                    if delay > 0:
                        sleep_fn(delay)
                    continue
                break
            # Non-transient: give flaky-but-unexpected failures a small
            # second chance before aborting the run.
            if unexpected_left > 0:
                unexpected_left -= 1
                if initial_delay > 0:
                    sleep_fn(initial_delay)
                continue
            raise ToolExecutionError(
                f"Tool '{_tool_name(request)}' failed: {exc}"
            ) from exc
    # Exhausted all retries on a transient error.
    assert last_exc is not None  # for type-checkers
    raise ToolExecutionError(
        f"Tool '{_tool_name(request)}' failed after "
        f"{max_retries} retries: {last_exc}"
    ) from last_exc


async def _aretry_loop(
    request: "ToolCallRequest",
    handler: Callable[..., Any],
    *,
    max_retries: int,
    initial_delay: float,
    backoff_factor: float,
    max_delay: float,
    unexpected_retries: int,
    sleep_fn: Callable[[float], Any],
) -> Any:
    """Async counterpart of `_retry_loop`. `sleep_fn` is `asyncio.sleep`.

    Duplicated rather than parameterized because the handler itself is
    async — awaiting the handler requires a coroutine context, which a
    sync function cannot provide.
    """
    last_exc: Exception | None = None
    total_attempts = max_retries + 1
    unexpected_left = unexpected_retries
    attempt = 0
    while attempt < total_attempts:
        try:
            return await handler(request)
        except Exception as exc:  # noqa: BLE001 — retry layer by design
            last_exc = exc
            if _is_transient(exc):
                attempt += 1
                if attempt < total_attempts:
                    delay = _compute_backoff(
                        attempt - 1, initial_delay, backoff_factor, max_delay
                    )
                    if delay > 0:
                        await sleep_fn(delay)
                    continue
                break
            # Non-transient: give flaky-but-unexpected failures a small
            # second chance before aborting the run.
            if unexpected_left > 0:
                unexpected_left -= 1
                if initial_delay > 0:
                    await sleep_fn(initial_delay)
                continue
            raise ToolExecutionError(
                f"Tool '{_tool_name(request)}' failed: {exc}"
            ) from exc
    # Exhausted all retries on a transient error.
    assert last_exc is not None  # for type-checkers
    raise ToolExecutionError(
        f"Tool '{_tool_name(request)}' failed after "
        f"{max_retries} retries: {last_exc}"
    ) from last_exc


_FEEDBACK_GUIDANCE: str = (
    " You may retry with corrected arguments, or finish with the "
    "information already gathered."
)


def _tool_call_id(request: "ToolCallRequest") -> str:
    """Return the tool call id from a ToolCallRequest.

    Like :func:`_tool_name`, accepts both runtime shapes of
    ``request.tool_call``: a plain dict (TypedDict) or a ``ToolCall``
    object.
    """
    tc = request.tool_call
    if isinstance(tc, dict):
        return str(tc.get("id", ""))
    return str(getattr(tc, "id", ""))


class _FeedbackMiddleware(AgentMiddleware):
    """Feed exhausted tool errors back to the LLM instead of aborting.

    Sits OUTSIDE :class:`_ToolRetryMiddleware` in the middleware list
    (first defined = outermost). When the inner retry layer raises
    ``ToolExecutionError`` (retries exhausted), this middleware converts
    it into an error ``ToolMessage`` so the LLM can see the failure and
    self-correct — retry with corrected arguments, switch tools, or
    finish with the information already gathered.

    A consecutive-failure budget guards against error loops: any
    successful tool call resets the counter; once consecutive failures
    exceed ``failure_budget``, a new ``ToolExecutionError`` naming the
    budget is raised and the run terminates. ``failure_budget=0``
    restores fail-fast behavior (the first failure raises).

    Exceptions other than ``ToolExecutionError`` pass through untouched
    (``KeyboardInterrupt`` never reaches this layer — the retry layer
    only catches ``Exception``).
    """

    def __init__(self, failure_budget: int = 3) -> None:
        self.failure_budget = failure_budget
        self._consecutive_failures = 0

    def wrap_tool_call(
        self,
        request: "ToolCallRequest",
        handler: Callable[..., Any],
    ) -> Any:
        """Sync path: degrade ``ToolExecutionError`` into an error message."""
        try:
            result = handler(request)
        except ToolExecutionError as exc:
            return self._degrade_or_raise(request, exc)
        self._consecutive_failures = 0
        return result

    async def awrap_tool_call(
        self,
        request: "ToolCallRequest",
        handler: Callable[..., Any],
    ) -> Any:
        """Async path: mirror of :meth:`wrap_tool_call`."""
        try:
            result = await handler(request)
        except ToolExecutionError as exc:
            return self._degrade_or_raise(request, exc)
        self._consecutive_failures = 0
        return result

    def _degrade_or_raise(
        self, request: "ToolCallRequest", exc: ToolExecutionError
    ) -> Any:
        """Count the failure, then feed back or terminate the run.

        Within budget: return an error ``ToolMessage`` (``[ERROR]`` prefix
        + original message + recovery guidance) addressed to the failing
        tool call. Over budget: raise a new ``ToolExecutionError`` naming
        the budget, chaining the exhausted error as ``__cause__``.
        """
        self._consecutive_failures += 1
        if self._consecutive_failures > self.failure_budget:
            raise ToolExecutionError(
                f"tool failure budget exhausted after {self.failure_budget} "
                f"consecutive tool failures; last error: {exc}"
            ) from exc
        return ToolMessage(
            content=f"[ERROR] {exc}{_FEEDBACK_GUIDANCE}",
            tool_call_id=_tool_call_id(request),
            status="error",
        )


_THINKING_BLOCK_TYPES: frozenset[str] = frozenset(("thinking", "redacted_thinking"))


def _strip_thinking_blocks(messages: list[Any]) -> list[Any]:
    """Return ``messages`` with thinking blocks removed from list-content messages.

    ``deepseek-v4-pro`` (and qwen's gateway) require ``thinking`` to be
    enabled in the request, but their Anthropic-compatible endpoints cannot
    deserialize the thinking blocks langchain re-sends in a long multi-turn
    history — the streaming merge occasionally drops the ``thinking`` field,
    producing ``400 missing field 'thinking'``. Because these gateways
    regenerate reasoning server-side each turn, past thinking blocks are safe
    to drop; stripping them keeps the re-sent request schema valid.
    """
    stripped: list[Any] = []
    for msg in messages:
        content = getattr(msg, "content", None)
        has_thinking = isinstance(content, list) and any(
            isinstance(block, dict) and block.get("type") in _THINKING_BLOCK_TYPES
            for block in content
        )
        if has_thinking:
            msg = msg.model_copy(
                update={
                    "content": [
                        block
                        for block in content
                        if not (
                            isinstance(block, dict)
                            and block.get("type") in _THINKING_BLOCK_TYPES
                        )
                    ]
                }
            )
        stripped.append(msg)
    return stripped


class _StripThinkingMiddleware(AgentMiddleware):
    """Strip thinking blocks from the message history before each model call.

    See :func:`_strip_thinking_blocks` for why. Applies on every model
    invocation; a no-op when the history has no thinking blocks (e.g. the
    first turn, or when ``thinking`` is disabled).
    """

    def wrap_model_call(
        self, request: "ModelRequest", handler: Callable[..., Any]
    ) -> Any:
        """Sync path: strip thinking blocks, then delegate to ``handler``."""
        return handler(
            request.override(messages=_strip_thinking_blocks(request.messages))
        )

    async def awrap_model_call(
        self, request: "ModelRequest", handler: Callable[..., Any]
    ) -> Any:
        """Async path: mirror of :meth:`wrap_model_call`."""
        return await handler(
            request.override(messages=_strip_thinking_blocks(request.messages))
        )
