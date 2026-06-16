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
