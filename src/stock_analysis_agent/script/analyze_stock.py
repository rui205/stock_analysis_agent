"""CLI entry: run the StockAnalysisAgent sub-agent and stream its report.

Usage::

    python -m stock_analysis_agent.script.analyze_stock 02319.HK
    python -m stock_analysis_agent.script.analyze_stock 600519.SH --include-shell-tool

The sub-agent's final text is the canonical output; per the bundled
prompt policy the LLM itself publishes the report (e.g. as a Lark
cloud document). This script parses, renders, and writes nothing.

Exit codes:
    0 — success.
    1 — unhandled exception (caught at top level).
    3 — ``ToolExecutionError`` from the agent.
"""
# pyright: reportUnusedFunction=false
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage

from stock_analysis_agent.agent.exceptions import ToolExecutionError
from stock_analysis_agent.agent.stock_analysis import StockAnalysisAgent
from stock_analysis_agent.agent.stream import chunk_text
from stock_analysis_agent.tools.prompt import (
    render_system_prompt,
    resolve_tool_names,
)
from stock_analysis_agent.tools.skill import (
    format_skill_index_markdown,
    get_skill_index,
)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_UNHANDLED = 1
EXIT_TOOL = 3


#: Tool names exposed to ``StockAnalysisAgent`` — injected into
#: ``<!-- TOOL_INDEX -->`` so the system prompt matches the wired
#: tools 1:1. The orchestrator tool ``run_analyze_stock`` is
#: deliberately excluded: this is the sub-agent, not the
#: orchestrator, and would never invoke itself.
_SUBAGENT_TOOL_NAMES: list[str] = ["load_skill", "read_file"]


def _subagent_tool_names(include_shell_tool: bool = False) -> list[str]:
    """Compute the full sub-agent tool-name list for prompt rendering.

    Args:
        include_shell_tool: Whether ``run_command`` should be
            advertised alongside the sub-agent's defaults.

    Returns:
        Sorted, deduplicated list of tool names matching what
        :class:`StockAnalysisAgent` actually wires up for this run.
    """
    return resolve_tool_names(_SUBAGENT_TOOL_NAMES, include_shell_tool)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


#: Absolute path to the bundled system prompt template. Resolved at import time
#: so ``run()`` does no extra IO just to locate the file. The file lives at
#: ``src/<package>/prompts/system_prompt.md`` — one level above this script.
_PROMPT_FILE: Path = Path(__file__).resolve().parents[1] / "prompts" / "system_prompt.md"


# ---------------------------------------------------------------------------
# Helpers — pure functions, exported for testability.
# ---------------------------------------------------------------------------


def _load_system_prompt(include_shell_tool: bool = False) -> str:
    """Load the system prompt from ``prompts/system_prompt.md``.

    The bundled template is a flat policy document with two template
    placeholders that get auto-injected at load time:

    * ``<!-- SKILL_INDEX -->`` — every skill under ``skill/*/SKILL.md``,
      rendered as ``name + one-line description`` from each file's
      YAML frontmatter. Adding a new skill is a one-line drop-in (no
      prompt edit required).
    * ``<!-- TOOL_INDEX -->`` — the @tool catalog, built by introspecting
      every self-built tool's ``args_schema`` Pydantic model plus a
      hand-curated return-shape entry from
      :data:`tools.registry._TOOL_SPECS`.

    The catalog is filtered to the **sub-agent's actual tool set**
    (:func:`_subagent_tool_names`) — same names, same order. The
    orchestrator's ``run_analyze_stock`` and the strategy-only
    ``load_strategy`` are deliberately omitted so the model doesn't
    see tools it can't actually call.

    Args:
        include_shell_tool: When ``True``, ``run_command`` is also
            advertised to the LLM (matches the constructor flag that
            controls tool wiring). Default ``False``.

    Raises:
        FileNotFoundError: if the bundled ``system_prompt.md`` is missing
            (e.g. the wheel was mis-built and excluded it).
    """
    return render_system_prompt(
        _PROMPT_FILE,
        tool_names=_subagent_tool_names(include_shell_tool),
        catalog_placeholder="<!-- SKILL_INDEX -->",
        catalog_doc=format_skill_index_markdown(get_skill_index()),
    )


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze_stock",
        description=(
            "Run an LLM agent on a stock symbol and stream its analysis "
            "report; the LLM itself publishes the report per the bundled "
            "prompt policy (e.g. as a Lark cloud document)."
        ),
    )
    parser.add_argument("symbol", help="Stock code, e.g. 02319.HK, 600519.SH, 000001.SZ")
    parser.add_argument(
        "--include-shell-tool", dest="include_shell_tool", action="store_true",
        default=False,
        help=(
            "Enable the ``run_command`` tool so the agent can shell out to CLI "
            "programs (e.g. ``lark-cli docs +create`` to publish the report as a "
            "Lark cloud document). Off by default — opt in when needed."
        ),
    )
    parser.add_argument(
        "--recursion-limit", type=int, default=100,
        help=(
            "LangGraph recursion limit for the agent loop. Each tool call "
            "consumes 2–3 graph nodes (LLM decision → tool execution → back to "
            "LLM), so the bundled stock-analyst workflow — ~8 tool calls plus "
            "intermediate decisions — needs a budget of around 30–50 steps. "
            "Default 100 (above StockAnalysisAgent's constructor default of 50) "
            "because shell-enabled runs execute the full mx-* skill workflow "
            "and exhaust smaller budgets mid-run."
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable DEBUG-level logging.",
    )
    return parser


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    """Top-level orchestration. Returns the process exit code.

    Split out from ``main`` so tests can drive it with a constructed
    ``argparse.Namespace`` and monkeypatched ``StockAnalysisAgent``.
    """
    # 1. Build agent. The system prompt is loaded from the bundled
    # ``prompts/system_prompt.md``. ``StockAnalysisAgent`` is a
    # schema-agnostic low-level agent, so the script (not the agent) owns
    # the output contract.
    system_prompt = _load_system_prompt(include_shell_tool=args.include_shell_tool)
    agent = StockAnalysisAgent(
        include_shell_tool=args.include_shell_tool,
        recursion_limit=args.recursion_limit,
        system_prompt=system_prompt,
    )

    # 2. Stream. ``agent.stream()`` returns a generator backed by a
    # daemon thread that drives the LLM in the background — iterating
    # it is the only way to actually run the agent to completion. The
    # LLM-side lark-doc output policy in the bundled prompt is the
    # canonical delivery channel now; we don't parse, render, or write
    # anything from here.
    messages: list[BaseMessage] = [HumanMessage(
        content=f"请按 system prompt 的 schema 给出 {args.symbol} 的分析报告。",
    )]
    try:
        last_text: str = ""
        # Diagnostics are only consumed by the verbose report below —
        # collecting them unconditionally would grow two lists for the
        # whole run on every event. Gate on ``args.verbose`` so normal
        # runs pay nothing.
        tool_calls: list[tuple[str, str]] = []
        event_kinds: list[str] = []
        for event in agent.stream(messages):
            kind = event.get("event", "")
            if args.verbose:
                event_kinds.append(kind)
            if kind == "on_chat_model_stream":
                # Stream chunks: data["chunk"].content may be a string or list.
                chunk = event.get("data", {}).get("chunk", {})
                last_text += chunk_text(getattr(chunk, "content", ""))
            elif kind == "on_chat_model_end" and args.verbose:
                output = event.get("data", {}).get("output", {})
                for tc in getattr(output, "tool_calls", None) or []:
                    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                    tool_args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
                    tool_calls.append((str(name), repr(tool_args)[:200]))
    except ToolExecutionError as e:
        logger.error("agent tools failed: %s", e)
        return EXIT_TOOL

    # Visibility: the LLM's final text is the canonical script-side output
    # (the agent's conversation response). When verbose, also dump the
    # event-kind histogram and the tool calls the agent issued so the
    # user can see *how* the agent got to the answer.
    if last_text:
        logger.info(last_text)
    if args.verbose:
        from collections import Counter
        logger.info(f"========== EVENT KINDS ({len(event_kinds)}) ==========")
        for k, c in Counter(event_kinds).most_common():
            logger.info(f"  {k}: {c}")
        logger.info("========== LLM TOOL CALLS ==========")
        if tool_calls:
            for tool_name, tool_args in tool_calls:
                logger.info(f"  {tool_name}({tool_args[:200]})")
        else:
            logger.info("  (none)")

    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """Parse argv, configure logging, and dispatch to :func:`run`.

    Returns the process exit code so callers (and tests) can inspect it
    without ``sys.exit`` side effects. Top-level guard in :func:`run` is
    the only place unhandled exceptions are converted to ``EXIT_UNHANDLED``.
    """
    args = _build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return run(args)
    except Exception as e:  # noqa: BLE001 — top-level guard
        logger.exception("unhandled exception: %s", e)
        return EXIT_UNHANDLED


if __name__ == "__main__":
    sys.exit(main())