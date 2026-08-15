"""StrategyMatchAgent: top-level agent wrapping StockAnalysisAgent.

This is the entry point used by ``script.evaluate_strategy``. It owns
the workflow-glue tool wiring (``load_strategy``, ``run_analyze_stock``,
``load_skill``, plus opt-in ``run_command``) and a caller-supplied
system prompt that defines the :class:`StrategyMatchReport` JSON
contract.

Tool-set split is deliberate — :class:`StrategyMatchAgent` is the
*orchestrator*, not a data-discovery layer. The sub-agent
(:class:`StockAnalysisAgent`) owns the file/shell plumbing needed for
basic research (``read_file``, ``load_skill``, opt ``run_command``);
this top-level agent does **not** inherit any of those. Sharing them
would leak the sub-agent's data-discovery surface into the
orchestrator's prompt and increase the chance of the orchestrator
trying to do sub-agent work itself.

Like :class:`StockAnalysisAgent`, this is a thin shell over
:class:`BaseAgent` — it does not own any business logic beyond
constructing the right tool set and pre-binding the recursion limit
to a budget that accommodates one subagent run plus the strategy
matching.
"""
from __future__ import annotations

from typing import Any

from stock_analysis_agent.agent.base import BaseAgent
from stock_analysis_agent.tools.shell import run_command
from stock_analysis_agent.tools.skill import load_skill
from stock_analysis_agent.tools.strategy import (
    load_strategy,
    run_analyze_stock,
    set_subagent_include_shell_tool,
)


class StrategyMatchAgent(BaseAgent):
    """LLM-driven strategy-match agent (orchestrator).

    Bundles ``load_strategy``, ``run_analyze_stock``, ``load_skill``,
    and (opt-in) ``run_command``. The system prompt is **caller-supplied**
    — pass ``system_prompt=`` to define the output contract.

    This class deliberately does NOT include ``read_file``: that is a
    sub-agent discovery primitive that :class:`StockAnalysisAgent`
    owns. Adding it here would let the orchestrator try to do
    sub-agent work, which violates the layered architecture
    (orchestration ≠ research).

    Args:
        system_prompt: Full system prompt text (typically loaded from
            ``prompts/strategy_match_system_prompt.md`` with
            ``<!-- STRATEGY_INDEX -->`` and ``<!-- TOOL_INDEX -->``
            already substituted).
        include_shell_tool: When ``True``, also expose ``run_command``
            (e.g. for invoking ``lark-cli`` to publish to Feishu).
            The flag also propagates to the embedded
            ``run_analyze_stock`` sub-agent so its ``stock-analysis``
            workflow can execute the mx-* skill data scripts — without
            it the sub-agent can only emit a degraded report.
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

        # Provider wiring (mirrors ``tools.web_search``): the embedded
        # ``run_analyze_stock`` sub-agent reads this flag at call time
        # to inherit our shell opt-in. Without ``run_command`` the
        # bundled stock-analysis workflow cannot execute its mx-* skill
        # scripts and degrades to an LLM-knowledge-only report.
        set_subagent_include_shell_tool(include_shell_tool)

        # Orchestration layer: strategy + sub-agent + skill workflow
        # discovery + opt-in shell. No file/dir primitives — those are
        # the sub-agent's surface, not ours.
        tools: list[Any] = [load_strategy, run_analyze_stock, load_skill]
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
