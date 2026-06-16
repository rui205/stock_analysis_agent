"""BaseAgent: a reusable wrapper around langchain.agents.create_agent."""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import Any

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

    def stream(
        self,
        messages: list[BaseMessage],
        *,
        config: RunnableConfig | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream LangChain events from a fresh agent run.

        Implemented in Task 5; declared here so subclasses and tests
        can reference the method.
        """
        raise NotImplementedError

    async def astream(
        self,
        messages: list[BaseMessage],
        *,
        config: RunnableConfig | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async stream of LangChain events from a fresh agent run.

        Implemented in Task 7; declared here for type completeness.
        """
        raise NotImplementedError
        yield  # pragma: no cover  (makes this a generator for type checkers)
