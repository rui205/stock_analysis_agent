"""BaseAgent: a reusable wrapper around langchain.agents.create_agent."""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import Any, cast

from langchain.agents.middleware.types import InputAgentState
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.schema import StreamEvent
from langchain_core.tools import BaseTool

from stock_analysis_agent.agent.exceptions import AgentTimeoutError
from stock_analysis_agent.agent.middleware import (
    _FeedbackMiddleware,
    _StripThinkingMiddleware,
    _ToolRetryMiddleware,
)
from stock_analysis_agent.conf.settings import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
)

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

logger = logging.getLogger(__name__)


def _usage_from_event(event: StreamEvent) -> tuple[int, int] | None:
    """Best-effort ``(input_tokens, output_tokens)`` from a chat-model-end event.

    Usage lives in a few different shapes across LangChain versions and
    providers; probe the common ones and return ``None`` when it can't be
    located (callers then just skip the summary log).
    """
    output = (event.get("data") or {}).get("output")
    if output is None:
        return None

    usage: Any = None
    if isinstance(output, dict):
        usage = (output.get("llm_output") or {}).get("usage")
    else:
        try:
            generations = getattr(output, "generations", None)
            message = generations[0][0].message if generations else None
            usage = getattr(message, "usage_metadata", None)
            if not isinstance(usage, dict):
                response_meta = getattr(message, "response_metadata", None)
                if isinstance(response_meta, dict):
                    usage = response_meta.get("usage")
        except (AttributeError, IndexError, TypeError):
            usage = None

    if not isinstance(usage, dict):
        return None
    return (
        int(usage.get("input_tokens", 0) or 0),
        int(usage.get("output_tokens", 0) or 0),
    )


class BaseAgent:
    """Reusable Agent base class for stock_analysis_agent.

    Configuration is supplied via the constructor; subclasses typically
    override the defaults to pre-bake a system prompt and tool set.

    The class is stateless: each call to `stream` / `astream` receives
    the full `messages` list from the caller.

    Default model, temperature, and ``max_tokens`` are sourced from
    :mod:`stock_analysis_agent.conf.settings` (single source of truth).
    The API key, model, and endpoint are resolved at :meth:`_build_graph`
    time — never hardcoded — via
    :func:`stock_analysis_agent.conf.settings.get_settings`, which honors
    the ``select_source`` config to switch between the qwen and deepseek
    models.
    """

    def __init__(
        self,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        tools: Sequence[BaseTool | Callable[..., Any]] = (),
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        thinking_budget_tokens: int | None = None,
        max_retries: int = 2,
        tool_failure_budget: int = 3,
        recursion_limit: int | None = None,
        timeout: float | None = None,
        name: str | None = None,
    ) -> None:
        self._system_prompt = system_prompt
        self._tools = list(tools)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._thinking_budget_tokens = thinking_budget_tokens
        self._max_retries = max_retries
        self._tool_failure_budget = tool_failure_budget
        self._recursion_limit = recursion_limit
        self._timeout = timeout
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
    def thinking_budget_tokens(self) -> int | None:
        """Extended-thinking ("think") budget in tokens, or ``None`` to disable.

        When set, :meth:`_build_graph` passes
        ``thinking={"type": "enabled", "budget_tokens": N}`` to
        ``init_chat_model`` so the model emits a hidden reasoning pass
        before answering.
        """
        return self._thinking_budget_tokens

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

    @property
    def timeout(self) -> float | None:
        """Wall-clock timeout (seconds) applied to a whole run.

        ``None`` (default) disables the guard. When set, ``stream`` /
        ``astream`` raise :class:`AgentTimeoutError` if the run exceeds it.
        """
        return self._timeout

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

        The LLM API key, model, and endpoint are sourced from
        :func:`stock_analysis_agent.conf.settings.get_settings` and passed
        explicitly to ``init_chat_model``. The model source is chosen by
        ``select_source`` (see :mod:`stock_analysis_agent.conf.settings`):
        ``qwen`` uses ``$ANTHROPIC_API_KEY`` / ``$ANTHROPIC_BASE_URL``,
        ``deepseek`` uses ``$DEEPSEEK_API_KEY`` / ``$DEEPSEEK_BASE_URL``.
        A missing env var raises :class:`MissingAPIKeyError` from that
        helper — by design, so operators see a clear error before the
        LangChain stack attempts an unauthenticated call.

        The provider is also passed explicitly (``model_provider``).
        LangChain's ``init_chat_model`` cannot infer a provider from
        the bare model id, even though the endpoint speaks the Anthropic
        protocol. Declaring ``model_provider="anthropic"`` routes the
        call through :class:`langchain_anthropic.ChatAnthropic`, which
        the Anthropic SDK drives against the configured ``base_url``.
        """
        from langchain.agents import create_agent
        from langchain.chat_models import init_chat_model

        from stock_analysis_agent.conf.settings import get_settings

        settings = get_settings()
        # The model id is source-aware: when the agent was built with the
        # default model (no per-agent override), defer to ``settings.model``
        # so ``select_source`` can switch between qwen and deepseek. An
        # explicit per-agent ``model=`` still wins.
        model_id = self._model if self._model != DEFAULT_MODEL else settings.model
        thinking = (
            {"type": "enabled", "budget_tokens": self._thinking_budget_tokens}
            if self._thinking_budget_tokens is not None
            else None
        )
        model = init_chat_model(
            model_id,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            api_key=settings.api_key,
            base_url=settings.base_url,
            model_provider=settings.provider,
            thinking=thinking,
        )
        middleware = [
            # Strips thinking blocks from the re-sent history so the
            # deepseek/qwen gateways never receive a malformed thinking
            # block (400 "missing field 'thinking'"). Orthogonal to the
            # tool-call middlewares below — it hooks wrap_model_call only.
            _StripThinkingMiddleware(),
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
            input_tokens = 0
            output_tokens = 0
            try:
                async def _consume() -> None:
                    nonlocal input_tokens, output_tokens
                    async for event in graph.astream_events(
                        cast(InputAgentState, {"messages": list(messages)}),
                        version="v2",
                        config=resolved_config,
                    ):
                        if event.get("event") == "on_chat_model_end":
                            usage = _usage_from_event(event)
                            if usage is not None:
                                input_tokens += usage[0]
                                output_tokens += usage[1]
                        event_queue.put(event)

                if self._timeout is not None:
                    await asyncio.wait_for(_consume(), timeout=self._timeout)
                else:
                    await _consume()
                if input_tokens or output_tokens:
                    logger.info(
                        "agent %s token usage: input=%d output=%d",
                        self._name,
                        input_tokens,
                        output_tokens,
                    )
            except TimeoutError as exc:
                if self._timeout is None:
                    exception_holder.append(exc)
                else:
                    timeout_err = AgentTimeoutError(
                        f"agent {self._name} exceeded {self._timeout}s timeout"
                    )
                    timeout_err.__cause__ = exc
                    exception_holder.append(timeout_err)
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
        between calls. An optional ``timeout`` (see constructor) raises
        :class:`AgentTimeoutError` when the run exceeds it.
        """
        import asyncio

        graph = self._build_graph()
        resolved_config = self._resolve_config(config)
        input_tokens = 0
        output_tokens = 0

        async def _iter() -> AsyncIterator[StreamEvent]:
            nonlocal input_tokens, output_tokens
            async for event in graph.astream_events(
                cast(InputAgentState, {"messages": list(messages)}),
                version="v2",
                config=resolved_config,
            ):
                if event.get("event") == "on_chat_model_end":
                    usage = _usage_from_event(event)
                    if usage is not None:
                        input_tokens += usage[0]
                        output_tokens += usage[1]
                yield event

        try:
            if self._timeout is None:
                async for event in _iter():
                    yield event
            else:
                async with asyncio.timeout(self._timeout):
                    async for event in _iter():
                        yield event
        except TimeoutError as exc:
            if self._timeout is None:
                raise
            raise AgentTimeoutError(
                f"agent {self._name} exceeded {self._timeout}s timeout"
            ) from exc
        finally:
            if input_tokens or output_tokens:
                logger.info(
                    "agent %s token usage: input=%d output=%d",
                    self._name,
                    input_tokens,
                    output_tokens,
                )