"""Command-line entry point.

Two modes:

* `stock-analysis-agent "your question"` — one-shot query, prints the
  agent's final reply, exits.
* `stock-analysis-agent` — interactive REPL: read a query, invoke the
  agent, print the reply, repeat until EOF.

Run via the console script (`uv run stock-analysis-agent`) or as a
module (`uv run python -m stock_analysis_agent`).
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .agents import build_agent
from .logging import setup_logging

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_RUNTIME = 1

PROMPT = ">>> "


def _extract_final_text(result: dict) -> str:
    """Pull the last assistant message text out of an agent result.

    The agent graph returns a dict shaped like
    `{"messages": [HumanMessage, AIMessage, ToolMessage, ..., AIMessage]}`.
    The final AIMessage carries the user-facing reply.
    """
    messages = result.get("messages") or []
    for message in reversed(messages):
        # AIMessage (and its variants) expose .content as a string for
        # text-only replies, or a list of typed parts for multimodal
        # content. We only emit text.
        if getattr(message, "type", None) == "ai":
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                if parts:
                    return "\n".join(parts)
    return "(no response from agent)"


def _run_query(agent, query: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return _extract_final_text(result)


def _repl(agent) -> int:
    print("stock-analysis-agent — interactive mode. Ctrl-D to exit.")
    while True:
        try:
            query = input(PROMPT)
        except EOFError:
            print()  # newline after Ctrl-D
            return EXIT_OK
        query = query.strip()
        if not query:
            continue
        if query in {":q", ":quit", ":exit"}:
            return EXIT_OK
        try:
            reply = _run_query(agent, query)
        except KeyboardInterrupt:
            print("\n(interrupted)")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {exc}", file=sys.stderr)
            continue
        print(reply)
        print()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stock-analysis-agent",
        description=(
            "Multi-market (A股 / 美股 / 港股) stock analysis agent. "
            "Pass a question as a positional argument for one-shot mode, "
            "or run with no arguments for an interactive REPL."
        ),
    )
    parser.add_argument(
        "query",
        nargs=argparse.REMAINDER,
        help="Optional query. If omitted, the agent runs in REPL mode.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)
    query = " ".join(args.query).strip() if args.query else ""

    try:
        agent = build_agent()
    except RuntimeError as exc:
        # Most common: ANTHROPIC_API_KEY missing.
        print(f"[fatal] {exc}", file=sys.stderr)
        return EXIT_RUNTIME

    if query:
        try:
            print(_run_query(agent, query))
        except KeyboardInterrupt:
            print("\n(interrupted)", file=sys.stderr)
            return EXIT_RUNTIME
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {exc}", file=sys.stderr)
            return EXIT_RUNTIME
        return EXIT_OK

    return _repl(agent)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
