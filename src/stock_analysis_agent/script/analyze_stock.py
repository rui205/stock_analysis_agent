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


def render_markdown(a: StockAnalysis) -> str:
    """Render a :class:`StockAnalysis` to a Markdown string."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return "\n".join(
        [
            f"# {a.symbol} 分析报告",
            "",
            f"> 生成时间: {ts}",
            "",
            "## 总体观点",
            a.summary,
            "",
            "## 基本面",
            a.fundamentals,
            "",
            "## 技术面",
            a.technicals,
            "",
            "## 同行对比",
            a.peer_compare,
            "",
            "## 近期新闻",
            a.news,
            "",
            "## 风险",
            a.risks,
            "",
            "## 操作建议",
            a.recommendation,
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
    # 1. Build agent.
    agent = StockAnalysisAgent(
        symbol=args.symbol,
        include_peers=args.include_peers,
        peer_count=args.peer_count,
        include_web_search=args.include_web_search,
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

    # 3. Strip code fence + validate JSON.
    cleaned = _strip_code_fence(last_text)
    try:
        analysis = StockAnalysis.model_validate_json(cleaned)
    except ValidationError as e:
        logger.error("agent output is not a valid StockAnalysis JSON: %s", e)
        logger.error("raw output (first 500 chars): %s", cleaned[:500])
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