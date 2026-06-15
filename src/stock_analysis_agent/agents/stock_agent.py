"""Build the LangChain agent that binds the LLM to our tools.

We use `langchain.agents.create_agent` (the modern, non-deprecated
entry point — `langgraph.prebuilt.create_react_agent` is the legacy
alias). The returned object is a compiled `StateGraph` you invoke with
`{"messages": [{"role": "user", "content": "..."}]}`.
"""

from __future__ import annotations

from functools import lru_cache

from langchain.agents import create_agent

from ..llm import get_llm
from ..tools import ALL_TOOLS
from .prompts import SYSTEM_PROMPT


@lru_cache(maxsize=1)
def build_agent():
    """Return a process-wide singleton compiled agent graph."""
    return create_agent(
        model=get_llm(),
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )


__all__ = ["build_agent"]
