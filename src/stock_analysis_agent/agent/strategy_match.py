"""StrategyMatchAgent: top-level agent wrapping StockAnalysisAgent.

This is the entry point used by ``script.evaluate_strategy``. It owns
the tool wiring (``load_strategy``, ``run_analyze_stock``, plus the
re-used ``load_skill`` / ``read_file`` / opt-in ``run_command``) and
a caller-supplied system prompt that defines the
:class:`StrategyMatchReport` JSON contract.

Like :class:`StockAnalysisAgent`, this is a thin shell over
:class:`BaseAgent` — it does not own any business logic beyond
constructing the right tool set and pre-binding the recursion limit
to a budget that accommodates one subagent run plus the strategy
matching.
"""
from __future__ import annotations

from typing import Any

from stock_analysis_agent.agent.base import BaseAgent
from stock_analysis_agent.tools.read_file import read_file
from stock_analysis_agent.tools.shell import run_command
from stock_analysis_agent.tools.skill import load_skill
from stock_analysis_agent.tools.strategy import load_strategy, run_analyze_stock


class StrategyMatchAgent(BaseAgent):
    """LLM-driven strategy-match agent.

    Bundles ``load_strategy``, ``run_analyze_stock``, ``load_skill``,
    ``read_file``, and (opt-in) ``run_command``. The system prompt is
    **caller-supplied** — pass ``system_prompt=`` to define the
    output contract.

    Args:
        system_prompt: Full system prompt text (typically loaded from
            ``prompts/strategy_match_system_prompt.md`` with
            ``<!-- STRATEGY_INDEX -->`` and ``<!-- TOOL_INDEX -->``
            already substituted).
        include_shell_tool: When ``True``, also expose ``run_command``
            (e.g. for invoking ``lark-cli`` to publish to Feishu).
        recursion_limit: LangGraph budget. The typical run touches
            ``load_strategy`` + ``run_analyze_stock`` + 2-3 reasoning
            rounds + the final answer — ~15-20 graph nodes total.
            Default 80 gives comfortable headroom.
    """

    def __init__(
        self,
        *,
        system_prompt: str,
        include_shell_tool: bool = False,
        max_retries: int = 2,
        recursion_limit: int = 80,
        **kwargs: Any,
    ) -> None:
        if not system_prompt:
            raise ValueError("system_prompt cannot be empty")

        self._include_shell_tool = include_shell_tool

        tools: list[Any] = [load_strategy, run_analyze_stock, load_skill, read_file]
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
        return self._include_shell_tool


__all__ = ["StrategyMatchAgent"]
