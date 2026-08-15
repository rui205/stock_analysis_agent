"""Strategy-related tools: load a Markdown strategy file + subagent wrapper.

The ``load_strategy`` tool reads a single ``.md`` file under
``src/<package>/conf/strategies/`` and returns its full content (YAML
frontmatter + body). The LLM uses the body as natural-language
selection principles; the frontmatter provides ``name`` / ``version``
that flow into the output report.

The dynamic ``run_analyze_stock`` tool lives in this same module so
both strategy-related tools are colocated; it depends on
:class:`StockAnalysisAgent` and is exercised separately in Task 4.

The sub-agent returns Markdown (per the bundled ``stock-analysis`` skill),
so ``run_analyze_stock`` is a thin wrapper that just forwards the
agent's final text back to the caller — no JSON parsing, no schema
validation, no remapping. The caller (the strategy-match LLM) gets
the raw report and decides what to do with it.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from stock_analysis_agent.agent.exceptions import ToolExecutionError
from stock_analysis_agent.agent.stock_analysis import StockAnalysisAgent
from stock_analysis_agent.agent.stream import collect_final_text

# Resolved at import time — points at conf/strategies/. Tests may
# monkeypatch this to a tmp dir.
_STRATEGIES_DIR = Path(__file__).resolve().parents[1] / "conf" / "strategies"

#: Frontmatter keys the strategy schema recognises. Anything else
#: (e.g. ``tags:`` in ``value-investing.md``) is dropped by
#: :func:`_parse_strategy_frontmatter` so the report sees only the
#: fields the schema binds to.
_STRATEGY_FRONTMATTER_KEYS: frozenset[str] = frozenset({"name", "version", "description"})

#: Whether the ``run_analyze_stock`` sub-agent gets the ``run_command``
#: tool. Written by :class:`StrategyMatchAgent.__init__` (mirrors the
#: module-singleton provider pattern in ``tools.web_search``); read by
#: :func:`_run_subagent_and_collect` on every tool invocation. The
#: bundled ``stock-analysis`` workflow executes its mx-* skill scripts
#: via shell — without ``run_command`` the sub-agent can only emit a
#: degraded, LLM-knowledge-only report. Default ``False`` keeps the
#: standalone tool behaviour unchanged.
_subagent_include_shell_tool: bool = False


def set_subagent_include_shell_tool(enabled: bool) -> None:
    """Set whether the ``run_analyze_stock`` sub-agent gets ``run_command``.

    Called by :class:`StrategyMatchAgent.__init__` so the embedded
    sub-agent inherits the orchestrator's shell opt-in.

    Args:
        enabled: ``True`` to wire ``run_command`` into the sub-agent
            (and advertise it in its system-prompt tool catalog);
            ``False`` to run the sub-agent without shell access.
    """
    global _subagent_include_shell_tool
    _subagent_include_shell_tool = enabled


def _list_strategy_names() -> tuple[str, ...]:
    """Return the alphabetical list of ``.md`` strategy file stems.

    Files without the ``.md`` suffix are ignored. Missing directory
    yields an empty tuple (no error).
    """
    if not _STRATEGIES_DIR.is_dir():
        return ()
    return tuple(sorted(p.stem for p in _STRATEGIES_DIR.glob("*.md")))


def _parse_strategy_frontmatter(text: str) -> dict[str, str]:
    """Extract simple ``key: value`` pairs from a YAML frontmatter block.

    Supports single-line values and ``description: |`` literal blocks
    (joined with spaces). Delegates to the shared parser; only keys
    in :data:`_STRATEGY_FRONTMATTER_KEYS` are returned (others dropped).

    Args:
        text: Full strategy Markdown text starting with the ``---`` fence.

    Returns:
        Dict of frontmatter keys. Missing/unparseable input yields ``{}``.
    """
    from stock_analysis_agent.tools._frontmatter import parse_yaml_frontmatter

    return parse_yaml_frontmatter(text, allow=_STRATEGY_FRONTMATTER_KEYS)


class LoadStrategyInput(BaseModel):
    """Input schema for the ``load_strategy`` tool."""

    name: str = Field(
        min_length=1,
        description=(
            "Strategy name — must match a `.md` file under "
            "`src/stock_analysis_agent/conf/strategies/` (without the "
            "`.md` suffix). Example: `value-investing`. Unknown names "
            "raise `FileNotFoundError` with the available list."
        ),
    )


@tool(
    "load_strategy",
    description=(
        "Load a personal stock-selection strategy from "
        "`src/stock_analysis_agent/conf/strategies/<name>.md`. Returns "
        "the full file content (YAML frontmatter + Markdown body). "
        "Use the natural-language principles in the body to drive the "
        "per-criterion strategy matching in your final JSON report. "
        "Raises `FileNotFoundError` for unknown strategy names — the "
        "error message lists the available strategies."
    ),
    args_schema=LoadStrategyInput,
)
def load_strategy(name: str) -> str:
    """Read the full Markdown content of one strategy file.

    Args:
        name: Strategy file stem (no `.md` suffix).

    Returns:
        The full UTF-8 content of the file.

    Raises:
        FileNotFoundError: ``name`` is not a known strategy.
    """
    path = _STRATEGIES_DIR / f"{name}.md"
    if not path.is_file():
        available = ", ".join(_list_strategy_names()) or "(none)"
        raise FileNotFoundError(
            f"strategy {name!r} not found at {path}; available: {available}"
        )
    return path.read_text(encoding="utf-8")


def _run_subagent_and_collect(symbol: str) -> str:
    """Inner helper — builds the subagent, runs it, returns the final text.

    The sub-agent is driven by the same ``prompts/system_prompt.md`` as
    ``script.analyze_stock``; per the bundled ``stock-analysis`` skill,
    its final text is a Markdown report. We return it verbatim — no
    parsing, no remapping.
    """
    from stock_analysis_agent.script.analyze_stock import _load_system_prompt

    # The sub-agent inherits the orchestrator's shell opt-in (written
    # by ``StrategyMatchAgent.__init__``): the bundled stock-analysis
    # workflow runs its mx-* skill scripts via ``run_command``, and
    # without it the sub-agent degrades to an LLM-knowledge-only
    # report. The prompt catalog follows the same flag so it never
    # advertises a tool the sub-agent can't actually call.
    shell_enabled = _subagent_include_shell_tool
    system_prompt = _load_system_prompt(include_shell_tool=shell_enabled)
    sub = StockAnalysisAgent(
        system_prompt=system_prompt,
        include_shell_tool=shell_enabled,
        # Shell-enabled runs execute the full mx-* workflow: each data
        # fetch costs ~4 graph steps (run_command + read_file, each
        # preceded by an LLM decision round) plus skill loads and
        # reasoning — observed runs exhaust the constructor default of
        # 50 mid-workflow. Mirror ``script.analyze_stock``'s CLI default
        # (100) for the identical workflow.
        recursion_limit=100,
    )
    events = sub.stream([HumanMessage(f"按 system prompt 的 schema 给出 {symbol} 的分析报告。")])
    return collect_final_text(events)


class RunAnalyzeStockInput(BaseModel):
    """Input schema for the ``run_analyze_stock`` tool."""

    symbol: str = Field(
        min_length=1,
        description=(
            "Stock symbol to analyze, e.g. `600519.SH`, `02319.HK`, "
            "`AAPL.US`. The tool runs the existing `StockAnalysisAgent` "
            "subagent on this symbol and returns its Markdown report verbatim."
        ),
    )


@tool(
    "run_analyze_stock",
    description=(
        "Run the existing `StockAnalysisAgent` subagent on a stock "
        "symbol and return its Markdown analysis verbatim. Returns the "
        "Markdown report on success; an `[ERROR] analyze_stock tool "
        "failed: ...` string when the sub-agent run fails (tool retries "
        "exhausted or recursion budget exceeded)."
    ),
    args_schema=RunAnalyzeStockInput,
)
def run_analyze_stock(symbol: str) -> str:
    """Synchronously run the analyze-stock subagent and forward its Markdown output.

    The sub-agent emits Markdown directly (per ``prompts/system_prompt.md`` +
    the bundled ``stock-analysis`` skill). No JSON parsing or remapping is
    performed here — the caller decides how to consume the report.

    Args:
        symbol: Stock symbol, e.g. ``"600519.SH"``.

    Returns:
        The sub-agent's final Markdown text on success, or an ``[ERROR]``-prefixed
        string when the sub-agent's tool retries are exhausted or its
        graph runs out of recursion budget mid-workflow.
    """
    try:
        return _run_subagent_and_collect(symbol)
    except (ToolExecutionError, RecursionError) as e:
        # ``RecursionError`` covers langgraph's ``GraphRecursionError``
        # (a subclass): if the sub-agent exhausts its step budget
        # mid-workflow, degrade to the soft ``[ERROR]`` contract instead
        # of letting the exception escape and abort the orchestrator run.
        return f"[ERROR] analyze_stock tool failed: {e}"


__all__ = [
    "LoadStrategyInput",
    "RunAnalyzeStockInput",
    "_list_strategy_names",
    "_parse_strategy_frontmatter",
    "load_strategy",
    "run_analyze_stock",
    "set_subagent_include_shell_tool",
]
