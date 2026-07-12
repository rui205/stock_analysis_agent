"""Strategy-related tools: load a Markdown strategy file.

The ``load_strategy`` tool reads a single ``.md`` file under
``src/<package>/conf/strategies/`` and returns its full content (YAML
frontmatter + body). The LLM uses the body as natural-language
selection principles; the frontmatter provides ``name`` / ``version``
that flow into the output report.

The dynamic ``run_analyze_stock`` tool lives in this same module so
both strategy-related tools are colocated; it depends on
:class:`StockAnalysisAgent` and is exercised separately in Task 4.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError

from stock_analysis_agent.agent.analysis_schema import StockAnalysis
from stock_analysis_agent.agent.exceptions import ToolExecutionError
from stock_analysis_agent.agent.stock_analysis import StockAnalysisAgent

# Resolved at import time — points at conf/strategies/. Tests may
# monkeypatch this to a tmp dir.
_STRATEGIES_DIR = Path(__file__).resolve().parents[1] / "conf" / "strategies"

#: Frontmatter keys the strategy schema recognises. Anything else
#: (e.g. ``tags:`` in ``value-investing.md``) is dropped by
#: :func:`_parse_strategy_frontmatter` so the report sees only the
#: fields the schema binds to.
_STRATEGY_FRONTMATTER_KEYS: frozenset[str] = frozenset({"name", "version", "description"})


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


def _extract_final_text(events: Iterator[dict]) -> str:
    """Accumulate ``on_chat_model_stream`` text into one string."""
    last_text = ""
    for event in events:
        if event.get("event") != "on_chat_model_stream":
            continue
        chunk = event.get("data", {}).get("chunk", {})
        content = getattr(chunk, "content", "")
        if isinstance(content, str) and content:
            last_text += content
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    last_text += block.get("text", "")
    return last_text


def _strip_code_fence(text: str) -> str:
    """Strip a leading/trailing markdown code fence if present."""
    s = text.strip()
    if not s.startswith("```"):
        return s
    lines = s.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json_object(text: str) -> str:
    """Return the longest balanced JSON object in ``text``."""
    decoder = json.JSONDecoder()
    candidates: list[str] = []
    idx = 0
    while True:
        start = text.find("{", idx)
        if start < 0:
            break
        try:
            _, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        candidates.append(text[start:end])
        idx = end
    if not candidates:
        raise ValueError("no JSON object found in agent output")
    return max(candidates, key=len)


def _render_analysis_summary(symbol: str, analysis: StockAnalysis) -> str:
    """Render the key fields of ``StockAnalysis`` as a markdown summary."""
    risks_md = "\n".join(f"- [{r.type}/{r.severity}] {r.description}" for r in analysis.risks[:3])
    return (
        f"# StockAnalysis 摘要 — {symbol}\n\n"
        f"- verdict: {analysis.verdict.decision} ({analysis.verdict.decision_label}) "
        f"confidence={analysis.verdict.confidence}\n"
        f"- summary: {analysis.verdict.summary}\n"
        f"- weighted_total: {analysis.scores.weighted_total}/10 "
        f"(fundamental={analysis.scores.fundamental}, "
        f"technical={analysis.scores.technical}, "
        f"news={analysis.scores.news_catalyst}, "
        f"peer={analysis.scores.peer_positioning})\n"
        f"- current_price: {analysis.price_plan.current_price}, "
        f"target: {analysis.price_plan.target_price}, "
        f"stop_loss: {analysis.price_plan.stop_loss}\n"
        f"- 主要风险:\n{risks_md or '- (无)'}\n"
    )


class RunAnalyzeStockInput(BaseModel):
    """Input schema for the ``run_analyze_stock`` tool."""

    symbol: str = Field(
        min_length=1,
        description=(
            "Stock symbol to analyze, e.g. `600519.SH`, `02319.HK`, "
            "`AAPL.US`. The tool runs the existing `StockAnalysisAgent` "
            "subagent on this symbol and returns a structured summary."
        ),
    )


def _run_subagent_and_collect(symbol: str) -> str:
    """Inner helper — builds the subagent, runs it, returns raw final text."""
    from stock_analysis_agent.script.analyze_stock import _load_system_prompt

    system_prompt = _load_system_prompt()
    sub = StockAnalysisAgent(
        symbol=symbol,
        system_prompt=system_prompt,
        include_peers=True,
        include_web_search=True,
        include_shell_tool=False,
        recursion_limit=50,
    )
    events = sub.stream([HumanMessage(f"按 system prompt 的 schema 给出 {symbol} 的分析报告。")])
    return _extract_final_text(events)


@tool(
    "run_analyze_stock",
    description=(
        "Run the existing `StockAnalysisAgent` subagent on a stock "
        "symbol and return a structured markdown summary of its "
        "`StockAnalysis` output. Returns a markdown summary on success; "
        "an `[ERROR] analyze_stock tool failed: ...` string on "
        "`ToolExecutionError`; or `[ERROR] StockAnalysis JSON parse "
        "failed: ...` with up to 2000 chars of raw LLM output on "
        "validation failure."
    ),
    args_schema=RunAnalyzeStockInput,
)
def run_analyze_stock(symbol: str) -> str:
    """Synchronously run the analyze-stock subagent and summarise the result."""
    try:
        last_text = _run_subagent_and_collect(symbol)
    except ToolExecutionError as e:
        return f"[ERROR] analyze_stock tool failed: {e}"

    try:
        json_str = _extract_json_object(_strip_code_fence(last_text))
        analysis = StockAnalysis.model_validate_json(json_str)
    except (ValueError, ValidationError) as e:
        return (
            f"[ERROR] StockAnalysis JSON parse failed: {e}; "
            f"raw first 2000 chars:\n{last_text[:2000]}"
        )

    return _render_analysis_summary(symbol, analysis)


__all__ = [
    "LoadStrategyInput",
    "RunAnalyzeStockInput",
    "_list_strategy_names",
    "_parse_strategy_frontmatter",
    "_render_analysis_summary",
    "load_strategy",
    "run_analyze_stock",
]
