"""BaseAgent: a reusable wrapper around langchain.agents.create_agent."""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

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
        """
        import asyncio
        import queue
        import threading

        graph = self._build_graph()
        event_queue: queue.Queue = queue.Queue()
        sentinel = object()

        async def _drain() -> None:
            async for event in graph.astream_events(
                {"messages": list(messages)},
                version="v2",
                config=config,
            ):
                event_queue.put(event)
            event_queue.put(sentinel)

        def _runner() -> None:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_drain())
            finally:
                loop.close()

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        while True:
            event = event_queue.get()
            if event is sentinel:
                break
            yield event
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


class _ToolRetryMiddleware(AgentMiddleware):
    """Placeholder. Real implementation in Task 8."""

    def __init__(self, max_retries: int = 2) -> None:
        self.max_retries = max_retries
