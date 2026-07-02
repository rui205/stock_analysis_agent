"""CLI entry: get_stock_snapshot + web_search agent → JSON analysis → local Markdown file.

Usage::

    python -m stock_analysis_agent.script.analyze_stock 02319.HK
    python -m stock_analysis_agent.script.analyze_stock 600519.SH --no-peers

The rendered report is written to ``<project-root>/output/<symbol>-<timestamp>.md``.

Exit codes:
    0 — success (markdown written to ``output/``).
    1 — unhandled exception (caught at top level).
    2 — agent output failed JSON / pydantic validation.
    3 — ``ToolExecutionError`` from the agent.
"""
# pyright: reportUnusedFunction=false
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage

from stock_analysis_agent.agent.analysis_schema import StockAnalysis
from stock_analysis_agent.agent.exceptions import ToolExecutionError
from stock_analysis_agent.agent.stock_analysis import StockAnalysisAgent

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_UNHANDLED = 1
EXIT_PARSE = 2
EXIT_TOOL = 3


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


#: Directory under the project root where the rendered Markdown is written.
#: Resolved as the parent of ``src/`` so the path is stable regardless of
#: the caller's CWD.
_OUTPUT_DIR_NAME = "output"


def _project_root() -> Path:
    """Return the project root directory (the directory containing ``pyproject.toml``).

    Resolved by walking up from this file: ``script/analyze_stock.py`` lives
    four levels below the root (``src/<package>/script/...``).
    """
    return Path(__file__).resolve().parents[3]


def output_dir() -> Path:
    """Return the absolute path to the ``output/`` directory at the project root."""
    return _project_root() / _OUTPUT_DIR_NAME


#: Absolute path to the bundled system prompt template. Resolved at import time
#: so ``run()`` does no extra IO just to locate the file. The file lives at
#: ``src/<package>/prompts/system_prompt.md`` — one level above this script.
_PROMPT_FILE: Path = Path(__file__).resolve().parents[1] / "prompts" / "system_prompt.md"


# ---------------------------------------------------------------------------
# Helpers — pure functions, exported for testability.
# ---------------------------------------------------------------------------


def _strip_code_fence(text: str) -> str:
    """Strip a leading/trailing markdown code fence if present.

    LLMs frequently wrap JSON in ```` ```json ... ``` ```` even when told
    not to. The schema validator downstream requires raw JSON, so we
    remove the fence lines before parsing.
    """
    s = text.strip()
    if not s.startswith("```"):
        return s
    lines = s.split("\n")
    # Drop the opening fence (e.g., "```json" or just "```").
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    # Drop the closing fence if present.
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json_object(text: str) -> str:
    """Return the longest balanced JSON object in ``text``.

    The LLM sometimes:

    * appends prose after the JSON (e.g. "如有需要可继续追问…"), or
    * emits a short summary object first, then the full answer (e.g. a
      bare ``Verdict`` followed by the complete ``StockAnalysis``), or
    * wraps the answer in code fences (handled separately by
      :func:`_strip_code_fence`).

    We greedily collect every parseable JSON object via
    :meth:`json.JSONDecoder.raw_decode` (which respects string escaping and
    nested braces) and return the **longest** one. The full ``StockAnalysis``
    is always longer than any sub-object (``Verdict``, ``PricePlan``, …),
    so this picks the answer over a teaser.

    Raises:
        ValueError: if no balanced JSON object can be found in ``text``.
    """
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


def _load_system_prompt() -> str:
    """Load the system prompt from ``prompts/system_prompt.md``.

    The bundled template is a flat policy document with no template
    placeholders, so this helper is a thin read — no substitution step.
    Runtime parameters (symbol, peer inclusion, web-search availability)
    are passed directly to :class:`StockAnalysisAgent` instead, where
    they control tool wiring rather than prompt content.

    Raises:
        FileNotFoundError: if the bundled ``system_prompt.md`` is missing
            (e.g. the wheel was mis-built and excluded it).
    """
    return _PROMPT_FILE.read_text(encoding="utf-8")


def render_markdown(a: StockAnalysis) -> str:
    """Render a :class:`StockAnalysis` to a Markdown string.

    Section order mirrors the structure of :class:`StockAnalysis`:

    1. Title + timestamp
    2. Verdict (the headline decision + confidence + one-liner)
    3. Price plan (table of current / entry / add / target / stop)
    4. Scores (compact list of 0-10 ratings)
    5. Company profile (the 七段式 text)
    6. Fundamental analysis (highlights / concerns)
    7. Technical analysis (highlights / concerns)
    8. News catalysts
    9. Peer compare
    10. Risks (table of type / description / severity)
    11. Action plan (position size, execution steps, review triggers)
    12. Reasoning chain (long form, kept as a blockquote)
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    pp = a.price_plan
    sc = a.scores

    verdict_badge = f"**{a.verdict.decision_label}** (decision={a.verdict.decision}, confidence={a.verdict.confidence})"

    price_table = "\n".join(
        [
            "| 项目 | 数值 |",
            "| --- | --- |",
            f"| 当前价 | {pp.current_price} |",
            f"| 建仓区间 | {pp.entry_zone[0]} ~ {pp.entry_zone[1]} |",
            f"| 加仓区间 | {pp.add_zone[0]} ~ {pp.add_zone[1]} |",
            f"| 目标价 | {pp.target_price} |",
            f"| 硬止损 | {pp.stop_loss} |",
            f"| 预期收益 | {pp.expected_return} |",
            f"| 风险收益比 | {pp.risk_reward_ratio} |",
            f"| 持仓周期 | {pp.time_horizon} |",
        ]
    )

    score_list = "\n".join(
        [
            f"- 基本面: {sc.fundamental}",
            f"- 技术面: {sc.technical}",
            f"- 消息面: {sc.news_catalyst}",
            f"- 同行对比: {sc.peer_positioning}",
            f"- **加权总分: {sc.weighted_total}**",
        ]
    )

    def _dim_section(heading: str, dim) -> str:
        lines = [f"## {heading}", ""]
        lines.append("**亮点**")
        if dim.highlights:
            for h in dim.highlights:
                lines.append(f"- {h}")
        else:
            lines.append("- (无)")
        lines.append("")
        lines.append("**隐忧**")
        if dim.concerns:
            for c in dim.concerns:
                lines.append(f"- {c}")
        else:
            lines.append("- (无)")
        lines.append("")
        return "\n".join(lines)

    risks_table = "\n".join(
        [
            "| 类型 | 严重度 | 描述 |",
            "| --- | --- | --- |",
            *(
                f"| {r.type} | {r.severity} | {r.description} |"
                for r in a.risks
            ),
        ]
    ) if a.risks else "_无_"

    action_lines = [
        f"- **仓位建议**: {a.action_plan.position_size}",
    ]
    if a.action_plan.execution:
        action_lines.append("- **执行步骤**:")
        action_lines.extend(f"  - {e}" for e in a.action_plan.execution)
    if a.action_plan.review_triggers:
        action_lines.append("- **复核触发条件**:")
        action_lines.extend(f"  - {t}" for t in a.action_plan.review_triggers)
    action_block = "\n".join(action_lines)

    news_block = (
        "\n".join(f"- {n}" for n in a.news_catalysts)
        if a.news_catalysts
        else "_无_"
    )

    return "\n".join(
        [
            f"# {a.symbol} 分析报告",
            "",
            f"> 生成时间: {ts}",
            "",
            "## 投资决策",
            "",
            verdict_badge,
            "",
            f"> {a.verdict.summary}",
            "",
            "## 价位推算",
            "",
            price_table,
            "",
            "## 评分",
            "",
            score_list,
            "",
            "## 公司画像",
            "",
            a.company_profile,
            "",
            _dim_section("基本面分析", a.fundamental_analysis),
            _dim_section("技术面分析", a.technical_analysis),
            "## 近期催化",
            "",
            news_block,
            "",
            "## 同行对比",
            "",
            a.peer_compare,
            "",
            "## 风险",
            "",
            risks_table,
            "",
            "## 操作建议",
            "",
            action_block,
            "",
            "## 推理链",
            "",
            f"> {a.reasoning_chain}",
            "",
        ]
    )


def build_output_path(symbol: str, output_dir: Path, now_epoch: int | None = None) -> Path:
    """Build the path to which the rendered Markdown for ``symbol`` is written.

    Files are timestamped so repeated runs do not clobber history. Pure
    function — the caller is responsible for ``mkdir(parents=True)`` and
    writing the file.
    """
    ts = now_epoch if now_epoch is not None else int(time.time())
    safe_symbol = symbol.replace(".", "_").replace("/", "_")
    return output_dir / f"stock-analysis-{safe_symbol}-{ts}.md"


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze_stock",
        description=(
            "Run an LLM agent on a stock symbol and write the rendered "
            "analysis as a Markdown file under <project-root>/output/."
        ),
    )
    parser.add_argument("symbol", help="Stock code, e.g. 02319.HK, 600519.SH, 000001.SZ")
    parser.add_argument(
        "--include-peers", dest="include_peers", action="store_true", default=True,
        help="Include top-N industry peers in the snapshot (default).",
    )
    parser.add_argument(
        "--no-peers", dest="include_peers", action="store_false",
        help="Skip peer detection.",
    )
    parser.add_argument(
        "--peer-count", type=int, default=2,
        help="How many peers to compare (default 2).",
    )
    parser.add_argument(
        "--no-web-search", dest="include_web_search", action="store_false",
        default=True,
        help="Disable the web_search tool (useful when search engines block the "
             "scraper). Analysis relies on get_stock_snapshot + LLM knowledge only.",
    )
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
        "--output-dir", type=Path, default=None,
        help=(
            "Directory to write the rendered Markdown into. Defaults to "
            "<project-root>/output/. Created if missing."
        ),
    )
    parser.add_argument(
        "--recursion-limit", type=int, default=6,
        help=(
            "LangGraph recursion limit for the agent loop. Each tool call "
            "consumes one step. Default 6 matches StockAnalysisAgent's own "
            "default — chosen so a runaway search loop fails fast instead of "
            "spinning through 30 steps before erroring. Raise this if the "
            "agent genuinely needs more than 6 round-trips."
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
    system_prompt = _load_system_prompt()
    agent = StockAnalysisAgent(
        symbol=args.symbol,
        include_peers=args.include_peers,
        peer_count=args.peer_count,
        include_web_search=args.include_web_search,
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
        tool_calls: list[tuple[str, str]] = []
        event_kinds: list[str] = []
        for event in agent.stream(messages):
            kind = event.get("event", "")
            event_kinds.append(kind)
            if kind == "on_chat_model_stream":
                # Stream chunks: data["chunk"].content may be a string or list.
                chunk = event.get("data", {}).get("chunk", {})
                content = getattr(chunk, "content", "")
                if isinstance(content, str) and content:
                    last_text += content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            last_text += block.get("text", "")
            elif kind == "on_chat_model_end":
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
        print(last_text)
    if args.verbose:
        from collections import Counter
        print(f"\n========== EVENT KINDS ({len(event_kinds)}) ==========")
        for k, c in Counter(event_kinds).most_common():
            print(f"  {k}: {c}")
        print("\n========== LLM TOOL CALLS ==========")
        if tool_calls:
            for tool_name, tool_args in tool_calls:
                print(f"  {tool_name}({tool_args[:200]})")
        else:
            print("  (none)")

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