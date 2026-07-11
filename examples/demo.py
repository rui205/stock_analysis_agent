"""Demo: stream events from BaseAgent without a real LLM.

Wires a fake chat model into a BaseAgent subclass via monkey-patching
`_build_graph`, then drains `stream()` and prints the event stream.

Run with: `uv run python examples/demo.py`
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make tests/agents/conftest importable so we can reuse the fake model.
TESTS_DIR = Path(__file__).resolve().parent.parent / "tests" / "agents"
sys.path.insert(0, str(TESTS_DIR))

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import tool
from langchain_core.messages import HumanMessage

from conftest import ToolAwareFakeChatModel, make_ai, make_tool_call  # noqa: E402
from stock_analysis_agent.agents import BaseAgent  # noqa: E402


class _NoRetry(AgentMiddleware):
    """No-op middleware to skip retry logic for this demo."""

    def wrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
        return handler(request)

    async def awrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
        return await handler(request)


@tool
def get_weather(city: str) -> str:
    """Look up the current weather for a city."""
    return f"sunny, 22°C in {city}"


class WeatherAgent(BaseAgent):
    """A minimal concrete subclass: pre-bakes a system prompt and one tool."""

    def __init__(self) -> None:
        super().__init__(
            system_prompt="You are a weather assistant. Use the get_weather tool when asked.",
            tools=[get_weather],
            name="weather-demo",
        )


def build_demo_graph(agent: BaseAgent, model):
    """Build a `create_agent` graph wired to the given fake model."""
    return create_agent(
        model=model,
        tools=list(agent.tools),
        system_prompt=agent.system_prompt_value,
        middleware=[_NoRetry()],
        name=agent.name,
    )


def banner(text: str) -> None:
    print()
    print("=" * 60)
    print(f" {text}")
    print("=" * 60)


def extract_final_text(stream_iter) -> str | None:
    """Drain a stream, return the content of the last AIMessage in the
    final dict-shaped `on_chain_end` event (ignores earlier chain-end
    events whose output is shaped as a list of Command objects)."""
    last_text: str | None = None
    for event in stream_iter:
        if event.get("event") == "on_chain_end":
            out = (event.get("data") or {}).get("output")
            if isinstance(out, dict) and "messages" in out:
                msgs = out["messages"] or []
                if msgs:
                    last_text = getattr(msgs[-1], "content", None)
    return last_text


def main() -> None:
    agent = WeatherAgent()
    print(f"Agent name: {agent.name}")
    print(f"System prompt: {agent.system_prompt_value!r}")
    print(f"Tools: {[t.name for t in agent.tools]}")
    print(f"Model default: {agent.model}")
    print(f"max_retries: {agent.max_retries}")

    # ---- Scenario 1: simple question, no tool call -----------------------
    banner("Scenario 1: simple question (no tool call)")
    model = ToolAwareFakeChatModel(
        responses=[make_ai("The sky is blue because of Rayleigh scattering.")]
    )
    agent._build_graph = lambda: build_demo_graph(agent, model)  # type: ignore[method-assign]

    events: list[str] = []
    final: str | None = None
    for event in agent.stream([HumanMessage(content="Why is the sky blue?")]):
        evt = event.get("event", "?")
        events.append(evt)
        # Capture the final text in-line, since the same `agent` will be
        # reused for further scenarios and we want one stream per scenario.
        if evt == "on_chain_end":
            out = (event.get("data") or {}).get("output")
            if isinstance(out, dict) and "messages" in out:
                msgs = out["messages"] or []
                if msgs:
                    final = getattr(msgs[-1], "content", None)

    print(f"  events: {events}")
    print(f"  final answer: {final!r}")

    # ---- Scenario 2: tool call + final answer ----------------------------
    banner("Scenario 2: tool call")
    first = make_ai("")
    first.tool_calls = [make_tool_call("get_weather", {"city": "Beijing"}, "call_1")]
    model = ToolAwareFakeChatModel(responses=[first, make_ai("It's 22°C and sunny in Beijing.")])
    agent._build_graph = lambda: build_demo_graph(agent, model)  # type: ignore[method-assign]

    final_text: str | None = None
    for event in agent.stream([HumanMessage(content="What's the weather in Beijing?")]):
        evt = event.get("event", "?")
        data = event.get("data") or {}
        if evt == "on_tool_start":
            print(f"  on_tool_start: input={data.get('input')}")
        elif evt == "on_tool_end":
            out = data.get("output")
            content = getattr(out, "content", out)
            print(f"  on_tool_end:   output={content!r}")
        elif evt == "on_chain_end":
            # During a tool call there are multiple on_chain_end events with
            # dict output (one per node). Only the LAST one is the top-level
            # final state — capture it and print after the loop.
            out2 = data.get("output")
            if isinstance(out2, dict) and "messages" in out2:
                msgs = out2["messages"] or []
                if msgs:
                    final_text = getattr(msgs[-1], "content", None)

    print(f"  final answer: {final_text!r}")

    # ---- Scenario 3: statelessness proof --------------------------------
    banner("Scenario 3: two consecutive stream() calls are independent")
    model = ToolAwareFakeChatModel(responses=[make_ai("first"), make_ai("second")])
    agent._build_graph = lambda: build_demo_graph(agent, model)  # type: ignore[method-assign]

    call1_text = extract_final_text(
        agent.stream([HumanMessage(content="hi")])
    )
    call2_text = extract_final_text(
        agent.stream([HumanMessage(content="hi again")])
    )
    print(f"  call 1 → {call1_text!r}")
    print(f"  call 2 → {call2_text!r}")


if __name__ == "__main__":
    main()
