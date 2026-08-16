"""Tests for stock_analysis_agent.agent.base.BaseAgent."""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from stock_analysis_agent.agent.base import BaseAgent
from stock_analysis_agent.agent.exceptions import ToolExecutionError


class _NoopAgent(BaseAgent):
    """Minimal concrete subclass for testing base config behavior."""

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)


def test_base_agent_uses_default_system_prompt() -> None:
    """Spec test 1: BaseAgent() with no args must succeed and use the
    default system prompt constant."""
    agent = _NoopAgent()
    print(agent.system_prompt_value)
    print(f"DEBUG: system_prompt_value = {agent.system_prompt_value!r}")  
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
        max_tokens=32768,
        max_retries=5,
        name="custom-name",
    )
    assert agent.model == "claude-opus-4-8"
    assert agent.temperature == 0.7
    assert agent.max_tokens == 32768
    assert agent.max_retries == 5
    assert agent.name == "custom-name"


def test_base_agent_recursion_limit_defaults_to_none() -> None:
    """``recursion_limit`` defaults to ``None`` — LangGraph's own default
    (25) applies unless a subclass opts in."""
    assert _NoopAgent().recursion_limit is None


def test_base_agent_recursion_limit_stores_value() -> None:
    """``recursion_limit`` is stored verbatim when provided."""
    agent = _NoopAgent(recursion_limit=6)
    assert agent.recursion_limit == 6


def test_base_agent_max_tokens_default_is_32768() -> None:
    """The default ``max_tokens`` is 32768 — large enough for the full
    analysis JSON (12 fields + ~1200-word reasoning) without truncation,
    even when the model is verbose. Subclasses can override down."""
    assert _NoopAgent().max_tokens == 32768


def test_tool_failure_budget_defaults_to_three() -> None:
    """The feedback middleware budget defaults to 3 consecutive failures."""
    assert _NoopAgent().tool_failure_budget == 3


def test_tool_failure_budget_stores_value() -> None:
    """``tool_failure_budget`` is stored verbatim when provided."""
    assert _NoopAgent(tool_failure_budget=5).tool_failure_budget == 5


# ---------------------------------------------------------------------------
# settings conf wiring — model defaults + api_key binding
# ---------------------------------------------------------------------------


def test_base_agent_default_model_comes_from_settings() -> None:
    """The default model identifier must equal ``conf.DEFAULT_MODEL``
    so :mod:`conf.settings` stays the single source of truth."""
    from stock_analysis_agent.conf.settings import DEFAULT_MODEL

    assert _NoopAgent().model == DEFAULT_MODEL


def test_base_agent_default_temperature_comes_from_settings() -> None:
    """The default temperature mirrors ``conf.DEFAULT_TEMPERATURE``."""
    from stock_analysis_agent.conf.settings import DEFAULT_TEMPERATURE

    assert _NoopAgent().temperature == DEFAULT_TEMPERATURE


def test_base_agent_build_graph_passes_api_key_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_build_graph`` must source the API key from ``$ANTHROPIC_API_KEY``
    via :func:`stock_analysis_agent.conf.settings.get_settings` and pass
    it to ``init_chat_model``. We stub ``langchain.chat_models`` directly
    so the assertion runs offline — no live model call, no provider
    resolution needed for this test."""
    import langchain.chat_models as chat_models_module

    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-supplied-key")

    # The settings loader is module-level cached; clear it so the env
    # change above is honored (an earlier test may have warmed the cache).
    from stock_analysis_agent.conf import settings as settings_module
    settings_module._cached_settings.cache_clear()
    try:
        captured: dict = {}

        def _fake_init(model: str, **kwargs):  # type: ignore[no-untyped-def]
            captured["model"] = model
            captured.update(kwargs)
            return object()  # placeholder; graph build only inspects kwargs

        monkeypatch.setattr(chat_models_module, "init_chat_model", _fake_init)

        agent = _NoopAgent(model="override-model", max_tokens=512)
        agent._build_graph()  # real path; settings + chat_models are patched

        assert captured["model"] == "override-model"
        assert captured["api_key"] == "env-supplied-key"
        assert captured["max_tokens"] == 512
        assert captured["model_provider"] == "anthropic"
    finally:
        settings_module._cached_settings.cache_clear()


def test_base_agent_build_graph_passes_base_url_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_build_graph`` forwards the resolved endpoint (``base_url``) to
    ``init_chat_model`` so a non-default gateway (MiniMax / DeepSeek) is
    reached instead of the SDK's default endpoint."""
    import langchain.chat_models as chat_models_module

    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-supplied-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.minimaxi.com/anthropic")

    from stock_analysis_agent.conf import settings as settings_module

    settings_module._cached_settings.cache_clear()
    try:
        captured: dict = {}

        def _fake_init(model: str, **kwargs):  # type: ignore[no-untyped-def]
            captured["model"] = model
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(chat_models_module, "init_chat_model", _fake_init)

        agent = _NoopAgent()
        agent._build_graph()

        assert captured["base_url"] == "https://api.minimaxi.com/anthropic"
    finally:
        settings_module._cached_settings.cache_clear()


def test_base_agent_build_graph_passes_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_build_graph`` forwards the extended-thinking budget to
    ``init_chat_model`` as ``thinking={"type": "enabled", "budget_tokens": N}``
    when a budget is set, and ``None`` when thinking is disabled."""
    import langchain.chat_models as chat_models_module

    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-supplied-key")

    from stock_analysis_agent.conf import settings as settings_module

    settings_module._cached_settings.cache_clear()
    try:
        captured: dict = {}

        def _fake_init(model: str, **kwargs):  # type: ignore[no-untyped-def]
            captured["model"] = model
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(chat_models_module, "init_chat_model", _fake_init)

        _NoopAgent(thinking_budget_tokens=8192)._build_graph()
        assert captured["thinking"] == {"type": "enabled", "budget_tokens": 8192}

        _NoopAgent()._build_graph()
        assert captured["thinking"] is None
    finally:
        settings_module._cached_settings.cache_clear()


def test_base_agent_build_graph_raises_when_api_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no ``ANTHROPIC_API_KEY`` in env, ``_build_graph`` must raise
    :class:`MissingAPIKeyError` before any LLM call is attempted."""
    from stock_analysis_agent.conf.settings import MissingAPIKeyError

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    # The settings loader is module-level cached; clear it so the env
    # delete above is honored (an earlier test may have warmed the cache).
    from stock_analysis_agent.conf import settings as settings_module
    settings_module._cached_settings.cache_clear()
    try:
        agent = _NoopAgent()
        with pytest.raises(MissingAPIKeyError):
            agent._build_graph()
    finally:
        settings_module._cached_settings.cache_clear()


# ---------------------------------------------------------------------------
# _resolve_config — recursion_limit injection
# ---------------------------------------------------------------------------


class TestResolveConfig:
    """The base agent's _resolve_config helper merges the default
    ``recursion_limit`` into caller-supplied LangChain config."""

    def test_returns_empty_dict_when_no_default_and_no_caller_config(self) -> None:
        agent = _NoopAgent()
        assert agent._resolve_config(None) == {}

    def test_passes_through_caller_config_when_no_default(self) -> None:
        """Without a default, caller's config is returned as a shallow copy."""
        agent = _NoopAgent()
        caller_config = {"configurable": {"thread_id": "abc"}}
        result = agent._resolve_config(caller_config)
        assert result == {"configurable": {"thread_id": "abc"}}
        # Mutating the result must not mutate the caller's dict.
        result["x"] = 1
        assert "x" not in caller_config

    def test_injects_default_when_caller_omits_recursion_limit(self) -> None:
        agent = _NoopAgent(recursion_limit=6)
        result = agent._resolve_config({"configurable": {"thread_id": "t1"}})
        assert result["recursion_limit"] == 6
        assert result["configurable"] == {"thread_id": "t1"}

    def test_respects_caller_supplied_recursion_limit(self) -> None:
        """If the caller already set recursion_limit, the agent's default
        must not overwrite it."""
        agent = _NoopAgent(recursion_limit=6)
        result = agent._resolve_config({"recursion_limit": 100})
        assert result["recursion_limit"] == 100

    def test_default_injection_with_none_caller_config(self) -> None:
        """Injecting into a None caller config still produces a populated dict."""
        agent = _NoopAgent(recursion_limit=6)
        result = agent._resolve_config(None)
        assert result == {"recursion_limit": 6}


def test_base_agent_name_defaults_to_class_name() -> None:
    """When name is not provided, the agent's name should default to the
    concrete subclass's __name__ (e.g. '_NoopAgent')."""
    agent = _NoopAgent()
    assert agent.name == "_NoopAgent"


def test_stream_returns_final_ai_message() -> None:
    """Spec test 2: BaseAgent.stream() must yield events whose on_chain_end
    payload contains an AIMessage with the model's reply content."""
    from tests.agent.conftest import ToolAwareFakeChatModel, make_ai
    from langchain.agents import create_agent
    from langchain.agents.middleware import AgentMiddleware

    class _NoRetry(AgentMiddleware):
        def wrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
            return handler(request)

    model = ToolAwareFakeChatModel(responses=[make_ai("hello back")])
    agent = _NoopAgent(system_prompt="test", tools=[])

    # Replace the agent's graph builder with one that uses the fake model.
    graph = create_agent(
        model=model,
        tools=list(agent.tools),
        system_prompt=agent.system_prompt_value,
        middleware=[_NoRetry()],
    )
    agent._build_graph = lambda: graph  # type: ignore[method-assign]

    final_output: dict | None = None
    for event in agent.stream([HumanMessage(content="hi")]):
        if event.get("event") == "on_chain_end":
            data = event.get("data") or {}
            out = data.get("output")
            if isinstance(out, dict) and "messages" in out:
                final_output = out

    assert final_output is not None
    messages = final_output.get("messages", [])
    assert any(
        getattr(m, "content", "") == "hello back" for m in messages
    ), f"Expected 'hello back' in final messages, got {messages!r}"


def test_stream_emits_tool_events() -> None:
    """Spec test 3: when the model makes a tool call, the event stream
    must include `on_tool_start` and `on_tool_end`."""
    from langchain.tools import tool

    from tests.agent.conftest import ToolAwareFakeChatModel, make_ai, make_tool_call

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

    class _NoRetry(AgentMiddleware):
        def wrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
            return handler(request)

        async def awrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
            return await handler(request)

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


@pytest.mark.asyncio
async def test_astream_returns_events() -> None:
    """astream() must be an async iterator yielding dict events with
    an 'event' key."""
    from tests.agent.conftest import ToolAwareFakeChatModel, make_ai

    model = ToolAwareFakeChatModel(responses=[make_ai("ok")])
    agent = _NoopAgent(system_prompt="test", tools=[])

    # Build graph manually with the fake model.
    from langchain.agents import create_agent
    from langchain.agents.middleware import AgentMiddleware

    class _NoRetry(AgentMiddleware):
        def wrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
            return handler(request)
        async def awrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
            return await handler(request)

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


@pytest.mark.asyncio
async def test_base_agent_astream_yields_events() -> None:
    """BaseAgent.astream() must yield dict events with an 'event' key."""
    from tests.agent.conftest import ToolAwareFakeChatModel, make_ai

    from langchain.agents import create_agent
    from langchain.agents.middleware import AgentMiddleware

    class _NoRetry(AgentMiddleware):
        def wrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
            return handler(request)
        async def awrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
            return await handler(request)

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


def test_tool_error_retries_then_raises_via_agent() -> None:
    """Spec test 4: when a tool raises transient errors, the agent
    must retry and eventually surface ToolExecutionError to the caller."""
    import asyncio
    from langchain.agents import create_agent
    from langchain.tools import tool
    from langchain_core.messages import ToolMessage

    from tests.agent.conftest import ToolAwareFakeChatModel, make_ai, make_tool_call

    call_count = {"n": 0}

    @tool
    def flaky_tool(query: str) -> str:
        """A flaky tool that always raises TimeoutError."""
        call_count["n"] += 1
        raise TimeoutError("upstream timeout")

    first_response = make_ai("")
    first_response.tool_calls = [
        make_tool_call("flaky_tool", {"query": "x"}, "call_flaky_1")
    ]
    model = ToolAwareFakeChatModel(
        responses=[first_response, make_ai("never reached")]
    )

    agent = _NoopAgent(
        system_prompt="test",
        tools=[flaky_tool],
        max_retries=2,
    )

    # Build the graph manually with the fake model so the test does not
    # depend on a live model call. The retry middleware is the real one.
    from stock_analysis_agent.agent.middleware import _ToolRetryMiddleware

    graph = create_agent(
        model=model,
        tools=[flaky_tool],
        system_prompt="test",
        middleware=[_ToolRetryMiddleware(max_retries=2, initial_delay=0.0, backoff_factor=0.0)],
    )
    agent._build_graph = lambda: graph  # type: ignore[method-assign]

    with pytest.raises(ToolExecutionError):
        # Drain the full stream — the middleware will raise during execution.
        for _ in agent.stream([HumanMessage(content="use flaky_tool")]):
            pass

    assert call_count["n"] == 3, f"expected 3 attempts, got {call_count['n']}"


def test_messages_are_stateless() -> None:
    """Spec test 5: two consecutive `stream` calls with the same input
    must produce equivalent results without cross-contamination."""
    from tests.agent.conftest import ToolAwareFakeChatModel, make_ai

    model = ToolAwareFakeChatModel(responses=[make_ai("reply-1"), make_ai("reply-2")])

    # Build a graph wired to the dual-response fake model.
    from langchain.agents import create_agent
    from langchain.agents.middleware import AgentMiddleware

    class _NoRetry(AgentMiddleware):
        def wrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
            return handler(request)
        async def awrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
            return await handler(request)

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
        last_ai_text: str | None = None
        for event in agent.stream(inputs):
            if event.get("event") == "on_chain_end":
                data = event.get("data") or {}
                out = data.get("output")
                # The final chain-end event has a dict output with messages.
                if isinstance(out, dict) and "messages" in out:
                    messages = out.get("messages", [])
                    if messages:
                        last_ai_text = getattr(messages[-1], "content", "")
        assert last_ai_text is not None, "no on_chain_end event with messages observed"
        return last_ai_text

    assert _last_ai_text() == "reply-1"
    assert _last_ai_text() == "reply-2"


def _patch_graph_building(monkeypatch: pytest.MonkeyPatch, model) -> None:  # type: ignore[no-untyped-def]
    """Point BaseAgent._build_graph at ``model`` instead of a live LLM.

    ``_build_graph`` imports ``init_chat_model`` and ``get_settings`` at
    call time, so patching the source modules redirects the construction.
    """
    from types import SimpleNamespace

    import langchain.chat_models as chat_models_mod

    import stock_analysis_agent.conf.settings as settings_mod

    monkeypatch.setattr(chat_models_mod, "init_chat_model", lambda *a, **k: model)
    monkeypatch.setattr(
        settings_mod,
        "get_settings",
        lambda: SimpleNamespace(
            api_key="test-key",
            provider="anthropic",
            model="test-model",
            base_url=None,
        ),
    )


def test_feedback_budget_exhaustion_terminates_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tool that keeps failing burns the consecutive-failure budget,
    then the run terminates with a budget-exhaustion ToolExecutionError.

    Also pins the middleware order: feedback MUST wrap retry (first
    defined = outermost), otherwise no "budget" message would surface.
    """
    from langchain.tools import tool

    from tests.agent.conftest import ToolAwareFakeChatModel, make_ai, make_tool_call

    @tool
    def always_broken(query: str) -> str:
        """A tool that always raises TimeoutError."""
        raise TimeoutError("upstream timeout")

    tc1 = make_ai("")
    tc1.tool_calls = [make_tool_call("always_broken", {"query": "x"}, "call_broken_1")]
    tc2 = make_ai("")
    tc2.tool_calls = [make_tool_call("always_broken", {"query": "x"}, "call_broken_2")]
    model = ToolAwareFakeChatModel(responses=[tc1, tc2, make_ai("unreached")])
    _patch_graph_building(monkeypatch, model)

    agent = _NoopAgent(
        system_prompt="test",
        tools=[always_broken],
        max_retries=0,
        tool_failure_budget=1,
    )

    with pytest.raises(ToolExecutionError) as ei:
        for _ in agent.stream([HumanMessage(content="use always_broken")]):
            pass

    assert "budget" in str(ei.value)


def test_feedback_lets_llm_recover_after_tool_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tool that fails once then succeeds: the error is fed back, the
    model retries, the run completes, and the counter is reset."""
    from langchain.tools import tool
    from langchain_core.messages import ToolMessage

    from tests.agent.conftest import ToolAwareFakeChatModel, make_ai, make_tool_call

    calls = {"n": 0}

    @tool
    def flaky_once(query: str) -> str:
        """Raises TimeoutError on the first call, then succeeds."""
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("upstream timeout")
        return "tool succeeded"

    tc1 = make_ai("")
    tc1.tool_calls = [make_tool_call("flaky_once", {"query": "x"}, "call_f1")]
    tc2 = make_ai("")
    tc2.tool_calls = [make_tool_call("flaky_once", {"query": "x"}, "call_f2")]
    model = ToolAwareFakeChatModel(responses=[tc1, tc2, make_ai("done")])
    _patch_graph_building(monkeypatch, model)

    agent = _NoopAgent(
        system_prompt="test",
        tools=[flaky_once],
        max_retries=0,
        tool_failure_budget=3,
    )

    final_messages: list = []
    for event in agent.stream([HumanMessage(content="use flaky_once")]):
        if event.get("event") == "on_chain_end":
            out = (event.get("data") or {}).get("output")
            if isinstance(out, dict) and "messages" in out:
                final_messages = out["messages"]

    assert calls["n"] == 2  # 1 failed attempt (fed back) + 1 success
    error_msgs = [
        m for m in final_messages
        if isinstance(m, ToolMessage) and m.status == "error"
    ]
    assert len(error_msgs) == 1
    assert error_msgs[0].content.startswith("[ERROR] ")
    assert getattr(final_messages[-1], "content", "") == "done"


# ---------------------------------------------------------------------------
# timeout — wall-clock guard + token-usage summary
# ---------------------------------------------------------------------------


def test_agent_timeout_defaults_to_none() -> None:
    """``timeout`` defaults to ``None`` (no wall-clock guard)."""
    assert _NoopAgent().timeout is None


def test_agent_timeout_stores_value() -> None:
    """``timeout`` is stored verbatim when provided."""
    assert _NoopAgent(timeout=1.5).timeout == 1.5


def test_agent_timeout_raises_when_run_exceeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run that outlives ``timeout`` raises :class:`AgentTimeoutError`."""
    import asyncio

    from stock_analysis_agent.agent.exceptions import AgentTimeoutError

    class _HangingGraph:
        async def astream_events(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            await asyncio.sleep(60)
            yield {"event": "on_chain_end"}  # pragma: no cover — never reached

    agent = _NoopAgent(timeout=0.05)
    monkeypatch.setattr(agent, "_build_graph", lambda: _HangingGraph())
    with pytest.raises(AgentTimeoutError):
        list(agent.stream([HumanMessage(content="hi")]))


def test_usage_from_event_returns_none_without_output() -> None:
    """Missing usage metadata yields ``None`` rather than crashing."""
    from stock_analysis_agent.agent.base import _usage_from_event

    assert _usage_from_event({"event": "on_chat_model_end", "data": {}}) is None


def test_usage_from_event_extracts_llm_output_usage() -> None:
    """The ``llm_output.usage`` dict shape is extracted as token counts."""
    from stock_analysis_agent.agent.base import _usage_from_event

    event = {
        "event": "on_chat_model_end",
        "data": {"output": {"llm_output": {"usage": {
            "input_tokens": 10, "output_tokens": 5,
        }}}},
    }
    assert _usage_from_event(event) == (10, 5)

