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
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage
from pydantic import ValidationError

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


#: Prompt fragments injected into ``{web_search_clause}``. These clauses are
#: owned by the script because they are an artefact of the bundled
#: ``system_prompt.md`` template — the agent itself is schema-agnostic and
#: does not know about web_search policy.
_WEB_SEARCH_ENABLED_CLAUSE: str = "视需要调用 web_search 补充近期新闻/公告/分析师观点。"
_WEB_SEARCH_DISABLED_CLAUSE: str = (
    "**没有 web_search 工具**(搜索引擎被屏蔽/不可用)。仅依靠 "
    "get_stock_snapshot 的数据 + 你自己的训练知识,不要尝试调用 web_search,"
    "news 字段若无法补充就写 'N/A'。"
)


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


def _load_system_prompt(
    symbol: str, include_peers: bool, include_web_search: bool
) -> str:
    """Load the system prompt from ``prompts/system_prompt.md`` with the
    three template variables filled in.

    The bundled template declares ``{symbol}``, ``{include_clause}`` and
    ``{web_search_clause}``. ``include_clause`` describes whether the
    snapshot should include peer data; ``web_search_clause`` tells the
    LLM whether the ``web_search`` tool is available. Both shapes match
    the constants used by ``agent.stock_analysis``'s built-in prompt so
    the LLM-facing instruction is consistent across both code paths.

    Raises:
        FileNotFoundError: if the bundled ``system_prompt.md`` is missing
            (e.g. the wheel was mis-built and excluded it).
    """
    include_clause = "include_peers 为 True" if include_peers else "include_peers 为 False"
    web_search_clause = (
        _WEB_SEARCH_ENABLED_CLAUSE
        if include_web_search
        else _WEB_SEARCH_DISABLED_CLAUSE
    )
    template = _PROMPT_FILE.read_text(encoding="utf-8")
    return template.format(
        symbol=symbol,
        include_clause=include_clause,
        web_search_clause=web_search_clause,
    )


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
        "--output-dir", type=Path, default=None,
        help=(
            "Directory to write the rendered Markdown into. Defaults to "
            "<project-root>/output/. Created if missing."
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable DEBUG-level logging.",
    )
    return parser


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _log_event(event: dict) -> None:
    kind = event.get("event", "")
    name = event.get("name", "")
    if kind == "on_tool_start":
        logger.info("[tool-start] %s args=%s", name, event["data"].get("input"))
    elif kind == "on_tool_end":
        out = event["data"].get("output")
        snippet = repr(out)[:200] if out is not None else "None"
        logger.info("[tool-end]   %s output=%s", name, snippet)
    elif kind == "on_chat_model_start":
        logger.debug("[llm-start] %s", name)
    elif kind == "on_chat_model_end":
        msg = event["data"].get("output", {})
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.strip():
            logger.debug("[llm-text] %s", content[:200])
        tool_calls = getattr(msg, "tool_calls", []) or []
        for tc in tool_calls:
            logger.debug("[llm-tool] -> %s(%s)", tc.get("name"), tc.get("args"))


def run(args: argparse.Namespace) -> int:
    """Top-level orchestration. Returns the process exit code.

    Split out from ``main`` so tests can drive it with a constructed
    ``argparse.Namespace`` and monkeypatched ``StockAnalysisAgent``.
    """
    # 1. Build agent. The system prompt is loaded from the bundled
    # ``prompts/system_prompt.md``. ``StockAnalysisAgent`` is a
    # schema-agnostic low-level agent, so the script (not the agent) owns
    # the output contract.
    system_prompt = _load_system_prompt(
        symbol=args.symbol,
        include_peers=args.include_peers,
        include_web_search=args.include_web_search,
    )
    agent = StockAnalysisAgent(
        symbol=args.symbol,
        include_peers=args.include_peers,
        peer_count=args.peer_count,
        include_web_search=args.include_web_search,
        system_prompt=system_prompt,
    )

    # 2. Stream.
    messages = [HumanMessage(content="请按 system prompt 的 schema 给出分析报告。")]
    try:
        events = agent.stream(messages)
    except ToolExecutionError as e:
        logger.error("agent tools failed: %s", e)
        return EXIT_TOOL

    last_text: str | None = None
    for event in events:
        _log_event(event)
        if event.get("event") == "on_chat_model_end":
            msg = event["data"].get("output", {})
            content = getattr(msg, "content", None)
            if isinstance(content, str) and content.strip():
                last_text = content
            elif isinstance(content, list):
                parts = [
                    blk.get("text", "")
                    for blk in content
                    if isinstance(blk, dict) and blk.get("type") == "text"
                ]
                joined = "\n".join(p for p in parts if p)
                if joined.strip():
                    last_text = joined

    if not last_text:
        logger.error("agent emitted no final text")
        return EXIT_PARSE

    # 3. Strip code fence + extract JSON object + validate.
    cleaned = _strip_code_fence(last_text)
    try:
        json_text = _extract_json_object(cleaned)
    except ValueError as e:
        logger.error("could not locate a JSON object in agent output: %s", e)
        logger.error("raw output (first 500 chars): %s", cleaned[:500])
        return EXIT_PARSE
    try:
        analysis = StockAnalysis.model_validate_json(json_text)
    except ValidationError as e:
        logger.error("agent output is not a valid StockAnalysis JSON: %s", e)
        logger.error("raw output (first 500 chars): %s", json_text[:500])
        return EXIT_PARSE

    # 4. Render Markdown + write to <project-root>/output/.
    markdown = render_markdown(analysis)
    out_dir = args.output_dir if args.output_dir is not None else output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = build_output_path(args.symbol, out_dir)
    out_path.write_text(markdown, encoding="utf-8")
    logger.info("wrote markdown: %s", out_path)

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