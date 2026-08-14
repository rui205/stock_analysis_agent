"""BaseAgent: a reusable wrapper around langchain.agents.create_agent."""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import Any, cast

from langchain.agents.middleware.types import InputAgentState
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.schema import StreamEvent
from langchain_core.tools import BaseTool

from stock_analysis_agent.agent.middleware import (
    _FeedbackMiddleware,
    _ToolRetryMiddleware,
)
from stock_analysis_agent.conf.settings import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
)

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


class BaseAgent:
    """Reusable Agent base class for stock_analysis_agent.

    Configuration is supplied via the constructor; subclasses typically
    override the defaults to pre-bake a system prompt and tool set.

    The class is stateless: each call to `stream` / `astream` receives
    the full `messages` list from the caller.

    Default model, temperature, and ``max_tokens`` are sourced from
    :mod:`stock_analysis_agent.conf.settings` (single source of truth).
    The API key is read from the ``ANTHROPIC_API_KEY`` env var at
    :meth:`_build_graph` time — never hardcoded — via
    :func:`stock_analysis_agent.conf.settings.get_settings`.
    """

    def __init__(
        self,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        tools: Sequence[BaseTool | Callable[..., Any]] = (),
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = 2,
        tool_failure_budget: int = 3,
        recursion_limit: int | None = None,
        name: str | None = None,
    ) -> None:
        self._system_prompt = system_prompt
        self._tools = list(tools)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._tool_failure_budget = tool_failure_budget
        self._recursion_limit = recursion_limit
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
    def tool_failure_budget(self) -> int:
        """Consecutive tool-failure budget for the feedback middleware.

        Tool errors fed back to the LLM may loop; after this many
        consecutive failing tool calls the run terminates. ``0`` restores
        fail-fast behavior.
        """
        return self._tool_failure_budget

    @property
    def recursion_limit(self) -> int | None:
        """Default ``recursion_limit`` injected into the graph config.

        ``None`` means no default — the LangGraph default (25) applies.
        Subclasses can set this to cap ReAct-style iteration depth.
        """
        return self._recursion_limit

    def _resolve_config(
        self, config: RunnableConfig | None
    ) -> RunnableConfig:
        """Merge the default ``recursion_limit`` into ``config`` if appropriate.

        Rules:
        - If the agent has no default (``recursion_limit is None``), return
          ``config`` unchanged (or ``{}`` if it was ``None``).
        - If the caller already passed a ``recursion_limit`` key, respect it.
        - Otherwise inject the agent's default into a shallow copy.

        Args:
            config: Caller-supplied LangChain runnable config (may be ``None``).

        Returns:
            The config to pass to ``graph.astream_events``. Never ``None``.
        """
        base: RunnableConfig = cast(RunnableConfig, dict(config) if config else {})
        if self._recursion_limit is None or "recursion_limit" in base:
            return base
        return {**base, "recursion_limit": self._recursion_limit}

    @property
    def name(self) -> str:
        return self._name

    @property
    def tools(self) -> list[BaseTool | Callable[..., Any]]:
        return list(self._tools)

    def _build_graph(self):  # type: ignore[no-untyped-def]
        """Construct the CompiledStateGraph. Imported lazily so module
        import is cheap.

        The LLM API key is sourced from ``$ANTHROPIC_API_KEY`` via
        :func:`stock_analysis_agent.conf.settings.get_settings` and
        passed explicitly to ``init_chat_model``. A missing env var
        raises :class:`MissingAPIKeyError` from that helper — by design,
        so operators see a clear error before the LangChain stack
        attempts an unauthenticated call.

        The provider is also passed explicitly (``model_provider``).
        LangChain's ``init_chat_model`` cannot infer a provider from
        the bare ``MiniMax-M3`` model id, even though MiniMax is
        reached via an Anthropic-protocol endpoint
        (``$ANTHROPIC_BASE_URL``). Declaring the provider here routes
        the call through :class:`langchain_anthropic.ChatAnthropic`,
        which the Anthropic SDK drives against the configured base URL.
        """
        from langchain.agents import create_agent
        from langchain.chat_models import init_chat_model

        from stock_analysis_agent.conf.settings import get_settings

        settings = get_settings()
        model = init_chat_model(
            self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            api_key=settings.api_key,
            model_provider=settings.provider,
        )
        middleware = [
            # First defined = outermost: feedback must wrap the retry
            # layer so it sees the retry layer's exhausted
            # ToolExecutionError and can degrade it into a ToolMessage.
            _FeedbackMiddleware(failure_budget=self._tool_failure_budget),
            _ToolRetryMiddleware(max_retries=self._max_retries),
        ]
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
    ) -> Iterator[StreamEvent]:
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
        resolved_config = self._resolve_config(config)
        event_queue: queue.Queue = queue.Queue()
        sentinel = object()
        exception_holder: list[BaseException] = []

        async def _drain() -> None:
            try:
                async for event in graph.astream_events(
                    cast(InputAgentState, {"messages": list(messages)}),
                    version="v2",
                    config=resolved_config,
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
    ) -> AsyncIterator[StreamEvent]:
        """Async stream of LangChain events from a fresh agent run.

        Builds a fresh graph on each call and consumes its
        `astream_events` generator. No internal state is retained
        between calls.
        """
        graph = self._build_graph()
        resolved_config = self._resolve_config(config)
        async for event in graph.astream_events(
            cast(InputAgentState, {"messages": list(messages)}),
            version="v2",
            config=resolved_config,
        ):
            yield event