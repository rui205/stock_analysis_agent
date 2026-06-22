"""Retry middleware for tool calls in agent streams.

Extracted from base.py so BaseAgent is small and focused on construction
+ streaming, while retry/backoff policy lives in its own module.
"""
from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Callable

from langchain.agents.middleware import AgentMiddleware

from stock_analysis_agent.agent.exceptions import ToolExecutionError

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ToolCallRequest


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

    Business errors (anything not classified transient) are wrapped
    in `ToolExecutionError` immediately without retrying.
    """

    def __init__(
        self,
        max_retries: int = 2,
        *,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 30.0,
    ) -> None:
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay

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
    sleep_fn: Callable[[float], Any],
) -> Any:
    """Shared retry loop. `sleep_fn` is `time.sleep` for sync callers."""
    last_exc: BaseException | None = None
    total_attempts = max_retries + 1
    for attempt in range(total_attempts):
        try:
            return handler(request)
        except BaseException as exc:  # noqa: BLE001 — top-level guard
            last_exc = exc
            if not _is_transient(exc):
                raise ToolExecutionError(
                    f"Tool '{_tool_name(request)}' failed: {exc}"
                ) from exc
            if attempt < max_retries:
                delay = _compute_backoff(
                    attempt, initial_delay, backoff_factor, max_delay
                )
                if delay > 0:
                    sleep_fn(delay)
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
    sleep_fn: Callable[[float], Any],
) -> Any:
    """Async counterpart of `_retry_loop`. `sleep_fn` is `asyncio.sleep`.

    Duplicated rather than parameterized because the handler itself is
    async — awaiting the handler requires a coroutine context, which a
    sync function cannot provide.
    """
    last_exc: BaseException | None = None
    total_attempts = max_retries + 1
    for attempt in range(total_attempts):
        try:
            return await handler(request)
        except BaseException as exc:  # noqa: BLE001 — top-level guard
            last_exc = exc
            if not _is_transient(exc):
                raise ToolExecutionError(
                    f"Tool '{_tool_name(request)}' failed: {exc}"
                ) from exc
            if attempt < max_retries:
                delay = _compute_backoff(
                    attempt, initial_delay, backoff_factor, max_delay
                )
                if delay > 0:
                    await sleep_fn(delay)
    # Exhausted all retries on a transient error.
    assert last_exc is not None  # for type-checkers
    raise ToolExecutionError(
        f"Tool '{_tool_name(request)}' failed after "
        f"{max_retries} retries: {last_exc}"
    ) from last_exc