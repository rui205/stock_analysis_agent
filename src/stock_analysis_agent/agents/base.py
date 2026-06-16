"""BaseAgent: a reusable wrapper around langchain.agents.create_agent."""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from stock_analysis_agent.agents.exceptions import ToolExecutionError

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ToolCallRequest

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


class BaseAgent:
    """Reusable Agent base class for stock_analysis_agent.

    Configuration is supplied via the constructor; subclasses typically
    override the defaults to pre-bake a system prompt and tool set.

    The class is stateless: each call to `stream` / `astream` receives
    the full `messages` list from the caller.
    """

    def __init__(
        self,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        tools: Sequence[BaseTool | Callable[..., Any]] = (),
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_retries: int = 2,
        name: str | None = None,
    ) -> None:
        self._system_prompt = system_prompt
        self._tools = list(tools)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._name = name if name is not None else type(self).__name__

    @property
    def system_prompt_value(self) -> str:
        """The system prompt passed at construction time."""
        return self._system_prompt

    @property
    def model(self) -> str:
        return self._model

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def max_retries(self) -> int:
        return self._max_retries

    @property
    def name(self) -> str:
        return self._name

    @property
    def tools(self) -> list[BaseTool | Callable[..., Any]]:
        return list(self._tools)

    def _build_graph(self):  # type: ignore[no-untyped-def]
        """Construct the CompiledStateGraph. Imported lazily so module
        import is cheap."""
        from langchain.agents import create_agent
        from langchain.chat_models import init_chat_model

        model = init_chat_model(
            self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        middleware = [_ToolRetryMiddleware(max_retries=self._max_retries)]
        return create_agent(
            model=model,
            tools=self._tools,
            system_prompt=self._system_prompt,
            middleware=middleware,
            name=self._name,
        )

    def stream(
        self,
        messages: list[BaseMessage],
        *,
        config: RunnableConfig | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream LangChain events from a fresh agent run.

        Uses a background thread with a private event loop to drive
        the async `astream_events` API, then yields events to the caller
        synchronously. Each call to `stream` runs in an isolated thread
        and event loop, so the base class remains stateless.

        Exceptions raised inside the async drain are captured by the
        runner thread and re-raised to the consumer on the sentinel
        boundary, so a failure in the agent graph surfaces to the
        caller instead of hanging the consumer's `event_queue.get()`.
        """
        import asyncio
        import queue
        import threading

        graph = self._build_graph()
        event_queue: queue.Queue = queue.Queue()
        sentinel = object()
        exception_holder: list[BaseException] = []

        async def _drain() -> None:
            try:
                async for event in graph.astream_events(
                    {"messages": list(messages)},
                    version="v2",
                    config=config,
                ):
                    event_queue.put(event)
            except BaseException as exc:
                exception_holder.append(exc)
            finally:
                event_queue.put(sentinel)

        def _runner() -> None:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_drain())
            finally:
                loop.close()

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        try:
            while True:
                event = event_queue.get()
                if event is sentinel:
                    if exception_holder:
                        raise exception_holder[0]
                    break
                yield event
        finally:
            thread.join()

    async def astream(
        self,
        messages: list[BaseMessage],
        *,
        config: RunnableConfig | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async stream of LangChain events from a fresh agent run.

        Builds a fresh graph on each call and consumes its
        `astream_events` generator. No internal state is retained
        between calls.
        """
        graph = self._build_graph()
        async for event in graph.astream_events(
            {"messages": list(messages)},
            version="v2",
            config=config,
        ):
            yield event


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
    `stock_analysis_agent.agents.exceptions`) with the original
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
