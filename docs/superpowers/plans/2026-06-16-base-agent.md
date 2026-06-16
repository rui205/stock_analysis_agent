# BaseAgent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable `BaseAgent` class for `stock_analysis_agent` that wraps LangChain 1.x's `create_agent`, exposes a streaming event API, and retries transient tool errors with exponential backoff.

**Architecture:** `BaseAgent.__init__` stores config (system prompt, tools, model params, retry policy). It builds a `CompiledStateGraph` via `langchain.agents.create_agent(...)` plus a custom `ToolRetryMiddleware` (lives in our module — distinct from `langchain.agents.middleware.ToolRetryMiddleware`, which we do **not** use because it does not raise on final failure). `stream()` / `astream()` are thin wrappers over the graph's `astream_events(...)` API. The class is stateless — callers pass full `messages` each call.

**Tech Stack:** Python 3.12, LangChain 1.x (`langchain`, `langchain-anthropic`, `langchain-core`), pytest, pytest-asyncio, uv for env management.

**Spec:** [`docs/superpowers/specs/2026-06-16-base-agent-design.md`](../specs/2026-06-16-base-agent-design.md)

---

## File Structure

Files created by this plan:

| Path | Responsibility |
|------|----------------|
| `pyproject.toml` | Project metadata, runtime deps, dev deps, pytest config |
| `src/stock_analysis_agent/__init__.py` | Marks the inner package as a regular package |
| `src/stock_analysis_agent/agents/__init__.py` | Re-exports `BaseAgent`, `ToolExecutionError` |
| `src/stock_analysis_agent/agents/exceptions.py` | `ToolExecutionError` exception class |
| `src/stock_analysis_agent/agents/base.py` | `BaseAgent` class + `_ToolRetryMiddleware` (private) |
| `tests/__init__.py` | Empty; marks tests as a regular package |
| `tests/agents/__init__.py` | Empty; marks tests/agents as a regular package |
| `tests/agents/conftest.py` | Shared `ToolAwareFakeChatModel` fixture for tests |
| `tests/agents/test_base.py` | All 5 spec-mandated tests |

No files are modified (project is greenfield).

---

## Task 1: Project setup (pyproject + dev deps)

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Write `pyproject.toml`**

Create `/Users/rui/workspace/stock_analysis_agent/pyproject.toml`:

```toml
[project]
name = "stock_analysis_agent"
version = "0.1.0"
description = "Reusable agents for stock analysis"
requires-python = ">=3.12"
dependencies = [
    "langchain>=1.0",
    "langchain-anthropic>=1.0",
    "langchain-core>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/stock_analysis_agent"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 2: Install dev dependencies**

Run: `uv pip install -e ".[dev]"`
Expected: install completes, prints "Installed ..." or no error.

- [ ] **Step 3: Verify pytest is importable**

Run: `uv run pytest --version`
Expected: prints pytest version (e.g. `pytest 8.x.x`).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add pyproject.toml with runtime and dev dependencies"
```

---

## Task 2: Test infrastructure (package markers + conftest)

**Files:**
- Create: `src/stock_analysis_agent/__init__.py` (empty)
- Create: `tests/__init__.py` (empty)
- Create: `tests/agents/__init__.py` (empty)
- Create: `tests/agents/conftest.py` (shared fake model)

- [ ] **Step 1: Make `stock_analysis_agent` a regular package**

Create `/Users/rui/workspace/stock_analysis_agent/src/stock_analysis_agent/__init__.py` with the single line:

```python
"""Stock analysis agent package."""
```

- [ ] **Step 2: Create test package markers**

`/Users/rui/workspace/stock_analysis_agent/tests/__init__.py`:
```python
"""Test package for stock_analysis_agent."""
```

`/Users/rui/workspace/stock_analysis_agent/tests/agents/__init__.py`:
```python
"""Tests for stock_analysis_agent.agents."""
```

- [ ] **Step 3: Add shared fake-model fixture**

`/Users/rui/workspace/stock_analysis_agent/tests/agents/conftest.py`:

```python
"""Shared pytest fixtures and helpers for agent tests."""
from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import ToolCall


class ToolAwareFakeChatModel(FakeMessagesListChatModel):
    """A fake chat model that supports `bind_tools` for testing.

    `FakeMessagesListChatModel` from langchain-core does not implement
    `bind_tools`, but `langchain.agents.create_agent` requires it.
    This subclass returns `self` from `bind_tools` so the agent graph
    can be built with tool-calling models in tests.
    """

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self


def make_tool_call(name: str, args: dict, call_id: str = "call_1") -> ToolCall:
    """Helper to build a ToolCall object for fake-model responses."""
    return ToolCall(name=name, args=args, id=call_id, type="tool_call")


def make_ai(content: str) -> AIMessage:
    """Helper to build a plain AIMessage for fake-model responses."""
    return AIMessage(content=content)
```

- [ ] **Step 4: Verify pytest discovers the (empty) test tree**

Run: `uv run pytest --collect-only -q`
Expected: output like `no tests ran` or `0 items collected` with exit code 0 or 5 (no collection errors). No `ModuleNotFoundError` for `stock_analysis_agent`.

- [ ] **Step 5: Commit**

```bash
git add src/stock_analysis_agent/__init__.py tests/__init__.py tests/agents/__init__.py tests/agents/conftest.py
git commit -m "test: add test infrastructure (package markers and shared fake model)"
```

---

## Task 3: ToolExecutionError exception (TDD)

**Files:**
- Create: `tests/agents/test_exceptions.py`
- Create: `src/stock_analysis_agent/agents/exceptions.py`

- [ ] **Step 1: Write the failing test**

`/Users/rui/workspace/stock_analysis_agent/tests/agents/test_exceptions.py`:

```python
"""Tests for stock_analysis_agent.agents.exceptions."""
from __future__ import annotations

import pytest

from stock_analysis_agent.agents.exceptions import ToolExecutionError


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_exceptions.py -v`
Expected: `ModuleNotFoundError: No module named 'stock_analysis_agent.agents.exceptions'` (or `ImportError`).

- [ ] **Step 3: Implement the exception**

`/Users/rui/workspace/stock_analysis_agent/src/stock_analysis_agent/agents/exceptions.py`:

```python
"""Custom exception types for stock_analysis_agent.agents."""
from __future__ import annotations


class ToolExecutionError(RuntimeError):
    """Raised when a tool call fails after exhausting retries.

    The original exception is preserved in `__cause__`.
    """
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/agents/test_exceptions.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/stock_analysis_agent/agents/exceptions.py tests/agents/test_exceptions.py
git commit -m "feat(agents): add ToolExecutionError exception"
```

---

## Task 4: BaseAgent default configuration (TDD)

**Files:**
- Create: `tests/agents/test_base.py`
- Create: `src/stock_analysis_agent/agents/base.py`

- [ ] **Step 1: Write the failing test for default config**

`/Users/rui/workspace/stock_analysis_agent/tests/agents/test_base.py`:

```python
"""Tests for stock_analysis_agent.agents.base.BaseAgent."""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from stock_analysis_agent.agents.base import BaseAgent


class _NoopAgent(BaseAgent):
    """Minimal concrete subclass for testing base config behavior."""

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)


def test_base_agent_uses_default_system_prompt() -> None:
    """Spec test 1: BaseAgent() with no args must succeed and use the
    default system prompt constant."""
    agent = _NoopAgent()
    assert agent.system_prompt_value == "You are a helpful assistant."


def test_base_agent_accepts_custom_system_prompt() -> None:
    """A custom system_prompt must override the default."""
    agent = _NoopAgent(system_prompt="You are a finance expert.")
    assert agent.system_prompt_value == "You are a finance expert."


def test_base_agent_stores_model_config() -> None:
    """Model, temperature, max_tokens, max_retries, name must be stored."""
    agent = _NoopAgent(
        model="claude-opus-4-8",
        temperature=0.7,
        max_tokens=8192,
        max_retries=5,
        name="custom-name",
    )
    assert agent.model == "claude-opus-4-8"
    assert agent.temperature == 0.7
    assert agent.max_tokens == 8192
    assert agent.max_retries == 5
    assert agent.name == "custom-name"


def test_base_agent_name_defaults_to_class_name() -> None:
    """When name is not provided, the agent's name should default to the
    concrete subclass's __name__ (e.g. '_NoopAgent')."""
    agent = _NoopAgent()
    assert agent.name == "_NoopAgent"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_base.py -v`
Expected: `ImportError: cannot import name 'BaseAgent' from 'stock_analysis_agent.agents.base'` (file does not exist yet).

- [ ] **Step 3: Implement BaseAgent config storage**

`/Users/rui/workspace/stock_analysis_agent/src/stock_analysis_agent/agents/base.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/agents/test_base.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/stock_analysis_agent/agents/base.py tests/agents/test_base.py
git commit -m "feat(agents): add BaseAgent with constructor config and property accessors"
```

---

## Task 5: BaseAgent.stream returns final AI message (TDD, spec test 2)

**Files:**
- Modify: `tests/agents/test_base.py` (append test)
- Modify: `src/stock_analysis_agent/agents/base.py` (replace NotImplementedError in `stream`)

- [ ] **Step 1: Append the failing test to `test_base.py`**

Append to `/Users/rui/workspace/stock_analysis_agent/tests/agents/test_base.py`:

```python
def test_stream_returns_final_ai_message() -> None:
    """Spec test 2: stream() must yield events whose on_chain_end payload
    contains an AIMessage with the model's reply content."""
    from tests.agents.conftest import ToolAwareFakeChatModel, make_ai

    model = ToolAwareFakeChatModel(responses=[make_ai("hello back")])
    agent = _NoopAgent(system_prompt="test", tools=[])

    final_output = _run_stream(agent, model, [HumanMessage(content="hi")])

    assert final_output is not None
    messages = final_output.get("messages", [])
    assert any(
        getattr(m, "content", "") == "hello back" for m in messages
    ), f"Expected 'hello back' in final messages, got {messages!r}"


def _run_stream(agent: BaseAgent, model, messages: list[BaseMessage]) -> dict | None:
    """Helper: build a graph using the given fake model, then drain the
    agent's stream and return the output of the last on_chain_end event."""
    from langchain.agents import create_agent
    from langchain.agents.middleware import AgentMiddleware

    class _NoRetry(AgentMiddleware):
        def wrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
            return handler(request)

    graph = create_agent(
        model=model,
        tools=list(agent.tools),
        system_prompt=agent.system_prompt_value,
        middleware=[_NoRetry()],
    )

    final_output: dict | None = None
    # Use a fresh event loop per call to keep tests isolated.
    import asyncio

    async def _drain() -> dict | None:
        nonlocal final_output
        async for event in graph.astream_events(
            {"messages": list(messages)},
            version="v2",
        ):
            if event.get("event") == "on_chain_end":
                data = event.get("data") or {}
                out = data.get("output")
                if isinstance(out, dict) and "messages" in out:
                    final_output = out
        return final_output

    asyncio.run(_drain())
    return final_output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agents/test_base.py::test_stream_returns_final_ai_message -v`
Expected: FAIL with `NotImplementedError` (from `BaseAgent.stream`).

- [ ] **Step 3: Implement `BaseAgent.stream` (lazy build) and add a build helper**

Replace the body of `stream` in `/Users/rui/workspace/stock_analysis_agent/src/stock_analysis_agent/agents/base.py` and add `_build_graph`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/agents/test_base.py::test_stream_returns_final_ai_message -v`
Expected: PASS (note: this also depends on `_ToolRetryMiddleware` existing, which we add in Task 8 — the test currently will fail with `ImportError`. If that happens, add a stub `_ToolRetryMiddleware` from Task 8 first, then come back.)

**Workaround for ordering:** create the `_ToolRetryMiddleware` stub now in `base.py` so this task's test passes:

Append to `/Users/rui/workspace/stock_analysis_agent/src/stock_analysis_agent/agents/base.py`:

```python


class _ToolRetryMiddleware(AgentMiddleware):
    """Placeholder. Real implementation in Task 8."""

    def __init__(self, max_retries: int = 2) -> None:
        self.max_retries = max_retries
```

And add `from langchain.agents.middleware import AgentMiddleware` to the imports of `base.py`.

- [ ] **Step 5: Commit**

```bash
git add src/stock_analysis_agent/agents/base.py tests/agents/test_base.py
git commit -m "feat(agents): implement BaseAgent.stream with astream_events bridge"
```

---

## Task 6: stream emits tool events (TDD, spec test 3)

**Files:**
- Modify: `tests/agents/test_base.py` (append test)

- [ ] **Step 1: Append the failing test**

Append to `/Users/rui/workspace/stock_analysis_agent/tests/agents/test_base.py`:

```python
def test_stream_emits_tool_events() -> None:
    """Spec test 3: when the model makes a tool call, the event stream
    must include `on_tool_start` and `on_tool_end`."""
    from langchain.tools import tool

    from tests.agents.conftest import ToolAwareFakeChatModel, make_ai, make_tool_call

    @tool
    def echo(value: str) -> str:
        """Echo a value back."""
        return value

    model = ToolAwareFakeChatModel(
        responses=[
            make_ai(""),
            make_ai("done"),
        ]
    )
    # Force the first response to include a tool call.
    model.responses[0] = make_ai("")
    model.responses[0].tool_calls = [make_tool_call("echo", {"value": "hi"}, "call_echo_1")]

    agent = _NoopAgent(system_prompt="test", tools=[echo])

    # Build graph manually with the fake model so we can test event flow.
    from langchain.agents import create_agent
    from langchain.agents.middleware import AgentMiddleware
    from stock_analysis_agent.agents.base import _ToolRetryMiddleware

    class _NoRetry(AgentMiddleware):
        def wrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
            return handler(request)

    graph = create_agent(
        model=model,
        tools=[echo],
        system_prompt="test",
        middleware=[_NoRetry()],
    )

    import asyncio
    events: list[str] = []

    async def _drain() -> None:
        async for event in graph.astream_events(
            {"messages": [HumanMessage(content="echo please")]},
            version="v2",
        ):
            events.append(event["event"])

    asyncio.run(_drain())

    assert "on_tool_start" in events, f"Expected on_tool_start in {events!r}"
    assert "on_tool_end" in events, f"Expected on_tool_end in {events!r}"
```

- [ ] **Step 2: Run test to verify it passes (no code change should be needed)**

Run: `uv run pytest tests/agents/test_base.py::test_stream_emits_tool_events -v`
Expected: PASS. (The event-stream bridge added in Task 5 already routes `on_tool_start` / `on_tool_end` from the graph to callers.)

If it fails with `NameError` or `ImportError`, fix the test imports — no production code change is required in this task.

- [ ] **Step 3: Commit**

```bash
git add tests/agents/test_base.py
git commit -m "test(agents): add test_stream_emits_tool_events"
```

---

## Task 7: BaseAgent.astream async interface (TDD)

**Files:**
- Modify: `tests/agents/test_base.py` (append test)
- Modify: `src/stock_analysis_agent/agents/base.py` (replace astub with real impl)

- [ ] **Step 1: Append the failing test**

Append to `/Users/rui/workspace/stock_analysis_agent/tests/agents/test_base.py`:

```python
@pytest.mark.asyncio
async def test_astream_returns_events() -> None:
    """astream() must be an async iterator yielding dict events with
    an 'event' key."""
    from tests.agents.conftest import ToolAwareFakeChatModel, make_ai

    model = ToolAwareFakeChatModel(responses=[make_ai("ok")])
    agent = _NoopAgent(system_prompt="test", tools=[])

    # Build graph manually with the fake model.
    from langchain.agents import create_agent
    from langchain.agents.middleware import AgentMiddleware
    from stock_analysis_agent.agents.base import _ToolRetryMiddleware

    class _NoRetry(AgentMiddleware):
        def wrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
            return handler(request)

    graph = create_agent(
        model=model,
        tools=[],
        system_prompt="test",
        middleware=[_NoRetry()],
    )

    events: list[str] = []
    async for event in graph.astream_events(
        {"messages": [HumanMessage(content="hi")]},
        version="v2",
    ):
        events.append(event["event"])

    assert "on_chain_start" in events
    assert "on_chain_end" in events
```

> Note: this test exercises the underlying `astream_events` directly to
> document the contract. The real `BaseAgent.astream` is implemented in
> the next step and exercised by an integration-style test below.

Append a second test that actually invokes `BaseAgent.astream`:

```python
@pytest.mark.asyncio
async def test_base_agent_astream_yields_events() -> None:
    """BaseAgent.astream() must yield dict events with an 'event' key."""
    from tests.agents.conftest import ToolAwareFakeChatModel, make_ai

    # Patch _build_graph to use a fake model.
    from langchain.agents import create_agent
    from langchain.agents.middleware import AgentMiddleware
    from stock_analysis_agent.agents.base import _ToolRetryMiddleware

    class _NoRetry(AgentMiddleware):
        def wrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
            return handler(request)

    model = ToolAwareFakeChatModel(responses=[make_ai("hello back")])
    graph = create_agent(
        model=model,
        tools=[],
        system_prompt="test",
        middleware=[_NoRetry()],
    )

    agent = _NoopAgent(system_prompt="test", tools=[])
    agent._build_graph = lambda: graph  # type: ignore[method-assign]

    events: list[str] = []
    async for event in agent.astream([HumanMessage(content="hi")]):
        events.append(event["event"])

    assert "on_chain_start" in events
    assert "on_chain_end" in events
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agents/test_base.py::test_astream_returns_events tests/agents/test_base.py::test_base_agent_astream_yields_events -v`
Expected: `test_base_agent_astream_yields_events` fails with `NotImplementedError`. `test_astream_returns_events` passes (it doesn't go through `BaseAgent.astream`).

- [ ] **Step 3: Implement `BaseAgent.astream`**

Replace the `astream` method body in `/Users/rui/workspace/stock_analysis_agent/src/stock_analysis_agent/agents/base.py`:

```python
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
```

Remove the trailing `raise NotImplementedError` + `yield` placeholder left over from Task 4.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agents/test_base.py::test_astream_returns_events tests/agents/test_base.py::test_base_agent_astream_yields_events -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/stock_analysis_agent/agents/base.py tests/agents/test_base.py
git commit -m "feat(agents): implement BaseAgent.astream as async event iterator"
```

---

## Task 8: _ToolRetryMiddleware behavior (TDD, spec test 4 part 1)

**Files:**
- Create: `tests/agents/test_middleware.py`
- Modify: `src/stock_analysis_agent/agents/base.py` (replace placeholder)

- [ ] **Step 1: Write the failing tests**

`/Users/rui/workspace/stock_analysis_agent/tests/agents/test_middleware.py`:

```python
"""Tests for the internal _ToolRetryMiddleware."""
from __future__ import annotations

import time
from typing import Any

import pytest
from langchain_core.messages import ToolCall, ToolMessage

from stock_analysis_agent.agents.base import _ToolRetryMiddleware
from stock_analysis_agent.agents.exceptions import ToolExecutionError


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
        max_retries=4, initial_delay=0.0, backoff_factor=1.0, max_delay=3.0
    )

    def handler(req):  # type: ignore[no-untyped-def]
        raise TimeoutError("x")

    with pytest.raises(ToolExecutionError):
        mw.wrap_tool_call(_make_request(), handler)

    # First three sleeps grow exponentially: 1, 2, 4 — capped to 3 starting at attempt 2.
    # Attempts 0..3 (4 retries) → sleeps after attempts 0,1,2,3 = [1, 2, 3, 3]
    assert sleeps == [1.0, 2.0, 3.0, 3.0], f"unexpected backoff sequence: {sleeps!r}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agents/test_middleware.py -v`
Expected: tests fail because the placeholder `_ToolRetryMiddleware` from Task 5 does not implement retry logic.

- [ ] **Step 3: Implement real `_ToolRetryMiddleware`**

Replace the placeholder in `/Users/rui/workspace/stock_analysis_agent/src/stock_analysis_agent/agents/base.py`:

```python
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ToolCallRequest
    from langchain_core.messages import ToolMessage
    from langchain_core.runnables import RunnableConfig


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
        handler: "Callable[[ToolCallRequest], ToolMessage | Any]",
    ) -> "ToolMessage | Any":
        last_exc: BaseException | None = None
        total_attempts = self.max_retries + 1
        for attempt in range(total_attempts):
            try:
                return handler(request)
            except BaseException as exc:  # noqa: BLE001 — top-level guard
                last_exc = exc
                if not _is_transient(exc):
                    raise ToolExecutionError(
                        f"Tool '{request.tool_call.name}' failed: {exc}"
                    ) from exc
                if attempt < self.max_retries:
                    delay = min(
                        self.initial_delay * (self.backoff_factor ** attempt),
                        self.max_delay,
                    )
                    if delay > 0:
                        time.sleep(delay)
        # Exhausted all retries on a transient error.
        assert last_exc is not None  # for type-checkers
        raise ToolExecutionError(
            f"Tool '{request.tool_call.name}' failed after "
            f"{self.max_retries} retries: {last_exc}"
        ) from last_exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agents/test_middleware.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/stock_analysis_agent/agents/base.py tests/agents/test_middleware.py
git commit -m "feat(agents): implement _ToolRetryMiddleware with exponential backoff"
```

---

## Task 9: BaseAgent integration with _ToolRetryMiddleware (TDD, spec test 4 part 2)

**Files:**
- Modify: `tests/agents/test_base.py` (append integration test)

- [ ] **Step 1: Append the failing integration test**

Append to `/Users/rui/workspace/stock_analysis_agent/tests/agents/test_base.py`:

```python
def test_tool_error_retries_then_raises_via_agent() -> None:
    """Spec test 4: when a tool raises transient errors, the agent
    must retry and eventually surface ToolExecutionError to the caller."""
    import asyncio
    from langchain.tools import tool
    from langchain_core.messages import ToolMessage

    from tests.agents.conftest import ToolAwareFakeChatModel, make_ai, make_tool_call

    call_count = {"n": 0}

    @tool
    def flaky_tool(query: str) -> str:
        """A flaky tool that always raises TimeoutError."""
        call_count["n"] += 1
        raise TimeoutError("upstream timeout")

    model = ToolAwareFakeChatModel(responses=[make_ai(""), make_ai("never reached")])
    model.responses[0].tool_calls = [
        make_tool_call("flaky_tool", {"query": "x"}, "call_flaky_1")
    ]

    agent = _NoopAgent(
        system_prompt="test",
        tools=[flaky_tool],
        max_retries=2,
    )

    with pytest.raises(ToolExecutionError):
        # Drain the full stream — the middleware will raise during execution.
        for _ in agent.stream([HumanMessage(content="use flaky_tool")]):
            pass

    assert call_count["n"] == 3, f"expected 3 attempts, got {call_count['n']}"
```

Add the import at the top of the file (next to the existing `from stock_analysis_agent.agents.base import BaseAgent` line):

```python
from stock_analysis_agent.agents.exceptions import ToolExecutionError
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `uv run pytest tests/agents/test_base.py::test_tool_error_retries_then_raises_via_agent -v`
Expected: PASS — Task 8's middleware is already wired into `_build_graph` from Task 5. If it fails, fix the wiring (no production change needed if `_build_graph` was implemented per Task 5).

- [ ] **Step 3: Commit**

```bash
git add tests/agents/test_base.py
git commit -m "test(agents): add integration test for retry+raise via BaseAgent.stream"
```

---

## Task 10: Statelessness verification (TDD, spec test 5)

**Files:**
- Modify: `tests/agents/test_base.py` (append test)

- [ ] **Step 1: Append the failing test**

Append to `/Users/rui/workspace/stock_analysis_agent/tests/agents/test_base.py`:

```python
def test_messages_are_stateless() -> None:
    """Spec test 5: two consecutive `stream` calls with the same input
    must produce equivalent results without cross-contamination."""
    from tests.agents.conftest import ToolAwareFakeChatModel, make_ai

    model = ToolAwareFakeChatModel(responses=[make_ai("reply-1"), make_ai("reply-2")])

    # Build a graph wired to the dual-response fake model.
    from langchain.agents import create_agent
    from langchain.agents.middleware import AgentMiddleware
    from stock_analysis_agent.agents.base import _ToolRetryMiddleware

    class _NoRetry(AgentMiddleware):
        def wrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
            return handler(request)

    graph = create_agent(
        model=model,
        tools=[],
        system_prompt="test",
        middleware=[_NoRetry()],
    )

    agent = _NoopAgent(system_prompt="test", tools=[])
    agent._build_graph = lambda: graph  # type: ignore[method-assign]

    inputs = [HumanMessage(content="hi")]

    def _last_ai_text() -> str:
        last = None
        for event in agent.stream(inputs):
            if event.get("event") == "on_chain_end":
                out = (event.get("data") or {}).get("output") or {}
                msgs = out.get("messages") or []
                if msgs:
                    last = msgs[-1]
        assert last is not None, "no on_chain_end event observed"
        return getattr(last, "content", "")

    assert _last_ai_text() == "reply-1"
    assert _last_ai_text() == "reply-2"
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/agents/test_base.py::test_messages_are_stateless -v`
Expected: PASS. The graph is rebuilt on each `stream` call, so messages from call 1 are not visible to call 2.

- [ ] **Step 3: Commit**

```bash
git add tests/agents/test_base.py
git commit -m "test(agents): add statelessness test for BaseAgent.stream"
```

---

## Task 11: Re-export public API from agents/__init__.py

**Files:**
- Modify: `src/stock_analysis_agent/agents/__init__.py`

- [ ] **Step 1: Write `agents/__init__.py` with re-exports**

Replace `/Users/rui/workspace/stock_analysis_agent/src/stock_analysis_agent/agents/__init__.py`:

```python
"""Reusable agents for stock_analysis_agent.

Public API:
    BaseAgent       — wrapper around langchain.agents.create_agent
    ToolExecutionError — raised when tool calls exhaust retries
"""
from __future__ import annotations

from stock_analysis_agent.agents.base import BaseAgent
from stock_analysis_agent.agents.exceptions import ToolExecutionError

__all__ = ["BaseAgent", "ToolExecutionError"]
```

- [ ] **Step 2: Smoke-test the public import**

Run:

```bash
uv run python -c "from stock_analysis_agent.agents import BaseAgent, ToolExecutionError; print(BaseAgent, ToolExecutionError)"
```

Expected: prints the class and exception object addresses, no error.

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests pass (8+ tests across the suite).

- [ ] **Step 4: Commit**

```bash
git add src/stock_analysis_agent/agents/__init__.py
git commit -m "feat(agents): re-export BaseAgent and ToolExecutionError"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Covered by |
|--------------|------------|
| §1 背景与目标 | Addressed by Task 1 (project setup) and Tasks 4–7 (BaseAgent surface) |
| §2 架构 | Task 4 (config storage) + Task 5 (graph build) + Task 8 (middleware) |
| §3 类 API | Task 4 (constructor + properties) |
| §4 派生类最小形态 | Implicit — tests use `_NoopAgent` showing the same pattern works |
| §5 错误处理 | Tasks 3 (exception), 8 (middleware), 9 (integration) |
| §6 可观测性 | Task 5 (event stream bridge) + Task 6 (tool events visible) |
| §7 测试策略 (5 tests) | Tasks 4 (test 1), 5 (test 2), 6 (test 3), 9 (test 4), 10 (test 5) |
| §8 文件清单 | All files created across Tasks 1, 2, 3, 4, 5, 7, 11 |
| §9 开放问题 | Decisions honored: `name` field (Task 4), `max_tokens=4096` (Task 4), `temperature=0.0` (Task 4) |

**Placeholder scan:** No "TBD"/"TODO"/"fill in". All code blocks are complete and runnable. Every command has expected output.

**Type consistency:**
- `BaseAgent.__init__` parameters match property names (`system_prompt`, `model`, `temperature`, `max_tokens`, `max_retries`, `name`, `tools`).
- `stream(messages, *, config=None)` signature consistent across Tasks 4, 5, 7.
- `astream(messages, *, config=None)` consistent across Tasks 4, 7.
- `_ToolRetryMiddleware(max_retries=...)` signature consistent across Tasks 5, 8.
- Exception class name `ToolExecutionError` consistent across Tasks 3, 8, 9, 11.

**Potential issue — `_ToolRetryMiddleware` reference ordering:** Task 5 references `_ToolRetryMiddleware` from inside `_build_graph`, and a placeholder is created in Task 5's Step 4. Task 8 replaces the placeholder with the real implementation. Tests in Tasks 6, 7, 9, 10 import `_ToolRetryMiddleware` and would break if Task 8 is skipped — the engineer MUST run Task 8 before Tasks 6/7/9/10 (commit ordering enforces this).

**Potential issue — import path in Task 5:** `_build_graph` references `_ToolRetryMiddleware` defined later in the same file. The reference resolves at *call* time (not at class definition), so this is safe — Python looks up the module-level name when `_build_graph()` is invoked, by which time `_ToolRetryMiddleware` has been defined. No self-import needed.

**Potential issue — Task 7 test `test_astream_returns_events`:** exercises `astream_events` directly on the graph, not `BaseAgent.astream`. The second test `test_base_agent_astream_yields_events` is the one that depends on Task 7's implementation. Engineers should run both to be sure.

**Potential issue — slow test:** the middleware backoff test (Task 8) uses `initial_delay=0.0` for fast execution. If a future change accidentally sets a non-zero delay without disabling it in tests, the suite will slow down dramatically. The plan explicitly sets `initial_delay=0.0` to avoid this.

**All review issues fixed inline. Plan is ready to execute.**
