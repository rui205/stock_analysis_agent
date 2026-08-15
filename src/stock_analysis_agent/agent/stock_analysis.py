"""StockAnalysisAgent: LLM-driven stock analysis with skill/file/shell tooling.

This is a **low-level reusable agent** — it owns the tool wiring
(``load_skill``, ``read_file``, opt-in ``run_command``) but does **not**
bake in any output schema or JSON contract. The caller is responsible
for supplying a ``system_prompt`` that defines the shape the LLM
should emit; this class only guarantees that whatever schema the prompt
asks for will reach the LLM, the tools will be available, and the
providers will be correctly initialized.

Directory listing is done through ``run_command(command="ls", argv=[...])``
— there is no separate ``list_dir`` tool.

Note: ``get_stock_snapshot`` has been removed (the ``market_data``
module is gone) and ``web_search`` is no longer wired into this agent
either — its provider plumbing lived here only as a side effect of the
shared module-level singletons, and ``agent.deepresearch`` owns that
logic now. The corresponding constructor parameters
(``include_peers``, ``peer_count``, ``include_web_search``,
``site_list``, ``cache_dir``, ``cache_ttl``) were removed together
with it.

Typical callers (e.g. ``script.analyze_stock``) load a prompt template
from disk and pass it in.
"""
from __future__ import annotations

from typing import Any

from stock_analysis_agent.agent.base import BaseAgent
from stock_analysis_agent.tools.read_file import read_file
from stock_analysis_agent.tools.shell import run_command
from stock_analysis_agent.tools.skill import load_skill


class StockAnalysisAgent(BaseAgent):
    """LLM-driven stock analysis agent.

    Bundles the ``load_skill``, ``read_file``, and (opt-in)
    ``run_command`` tools. The system prompt is **caller-supplied** —
    pass ``system_prompt=`` to define the output contract the LLM
    should follow. This class never infers a default prompt, so
    different callers can target different output schemas (e.g. a terse
    JSON, a structured Markdown report, a multi-section company
    profile) without subclassing.
    """

    def __init__(
        self,
        *,
        system_prompt: str,
        include_shell_tool: bool = False,
        max_retries: int = 3,
        recursion_limit: int = 50,
        **kwargs: Any,
    ) -> None:
        """Initialize the agent.

        Args:
            system_prompt: Caller-owned system prompt defining the
                output contract. Must be non-empty.
            include_shell_tool: When ``True``, also expose
                ``run_command`` to the LLM. Off by default — the shell
                tool is a privilege escalation.
            max_retries: Tool-call retry budget for transient errors.
            recursion_limit: LangGraph step budget for the agent loop.
            **kwargs: Forwarded to :class:`BaseAgent` (``model``,
                ``temperature``, ``name``, ...).

        Raises:
            ValueError: If ``system_prompt`` is empty.
        """
        if not system_prompt:
            raise ValueError("system_prompt cannot be empty")

        self._include_shell_tool = include_shell_tool

        tools = [load_skill, read_file]
        if include_shell_tool:
            tools.append(run_command)

        super().__init__(
            system_prompt=system_prompt,
            max_retries=max_retries,
            recursion_limit=recursion_limit,
            tools=tools,
            **kwargs,
        )

    @property
    def include_shell_tool(self) -> bool:
        """Whether the ``run_command`` tool is exposed to the LLM."""
        return self._include_shell_tool


__all__ = ["StockAnalysisAgent"]
