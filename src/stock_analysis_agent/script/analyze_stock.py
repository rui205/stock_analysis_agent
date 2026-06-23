"""CLI entry: get_stock_snapshot + web_search agent → JSON analysis → Feishu doc.

Usage::

    python -m stock_analysis_agent.script.analyze_stock 02319.HK
    python -m stock_analysis_agent.script.analyze_stock 600519.SH --no-peers

Exit codes:
    0 — success (doc created or appended).
    1 — unhandled exception (caught at top level).
    2 — agent output failed JSON / pydantic validation.
    3 — ``ToolExecutionError`` from the agent.
    4 — ``FeishuCliError`` from the wrapper.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from stock_analysis_agent.agent.analysis_schema import StockAnalysis
from stock_analysis_agent.agent.exceptions import ToolExecutionError
from stock_analysis_agent.agent.stock_analysis import StockAnalysisAgent
from stock_analysis_agent.tools.feishu_cli import FeishuCli, FeishuCliError

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_UNHANDLED = 1
EXIT_PARSE = 2
EXIT_TOOL = 3
EXIT_FEISHU = 4


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
    """Render a :class:`StockAnalysis` to a Feishu-doc-ready Markdown string."""
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


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze_stock",
        description="Run an LLM agent on a stock symbol and upload the analysis to Feishu.",
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
        "--lark-bin", default=os.environ.get("LARK_CLI_BIN", "lark-cli"),
        help="Path / name of the lark-cli binary (default: $LARK_CLI_BIN or 'lark-cli').",
    )
    parser.add_argument(
        "--title-prefix", default=None,
        help="Title prefix for the Feishu doc (default: '<symbol> 分析报告').",
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
    ``argparse.Namespace`` and monkeypatched ``StockAnalysisAgent`` /
    ``FeishuCli``.
    """
    title_prefix = args.title_prefix or f"{args.symbol} 分析报告"

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

    # 4. Render Markdown + write temp file.
    markdown = render_markdown(analysis)
    tmp_dir = Path(tempfile.gettempdir())
    tmp = tmp_dir / f"stock-analysis-{args.symbol.replace('.', '_')}-{int(time.time())}.md"
    tmp.write_text(markdown, encoding="utf-8")
    logger.info("wrote temp markdown: %s", tmp)

    # 5. lark-cli (Feishu CLI) — append or create.
    # ``+search`` does not support a server-side title-prefix filter, so we
    # query by symbol and filter the returned titles client-side.
    cli = FeishuCli(binary=args.lark_bin)
    try:
        candidates = cli.list_matching_docs(args.symbol)
    except FeishuCliError as e:
        logger.error("lark-cli search failed: %s", e)
        return EXIT_FEISHU
    existing = [doc for doc in candidates if doc.title.startswith(title_prefix)]

    if existing:
        target = existing[0]
        logger.info("appending to existing doc: %s", target.url)
        try:
            cli.append_to_doc(target.doc_id, content_file=tmp)
        except FeishuCliError as e:
            logger.error("lark-cli append failed: %s", e)
            return EXIT_FEISHU
    else:
        logger.info("creating new doc with prefix: %s", title_prefix)
        try:
            ref = cli.create_doc(title=title_prefix, content_file=tmp)
        except FeishuCliError as e:
            logger.error("lark-cli create failed: %s", e)
            return EXIT_FEISHU
        logger.info("created doc: %s", ref.url)

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