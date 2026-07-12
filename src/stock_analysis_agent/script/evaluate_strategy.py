"""CLI entry: load strategy + run strategy-match agent → local md / Feishu doc.

Usage::

    python -m stock_analysis_agent.script.evaluate_strategy 600519.SH \\
        --strategy value-investing --delivery both

Exit codes:
    0 — success
    1 — unhandled exception
    2 — agent output failed StrategyMatchReport validation
    3 — agent tool calls exhausted retries
    4 — startup validation failed (unknown strategy)
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage

from stock_analysis_agent.agent.exceptions import ToolExecutionError
from stock_analysis_agent.agent.strategy_match import StrategyMatchAgent
from stock_analysis_agent.agent.strategy_match_schema import StrategyMatchReport
from stock_analysis_agent.tools.registry import format_tool_index_markdown, get_tool_index
from stock_analysis_agent.tools.strategy import _list_strategy_names, _parse_strategy_frontmatter

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_UNHANDLED = 1
EXIT_PARSE = 2
EXIT_TOOL = 3
EXIT_BAD_STRATEGY = 4

DeliveryMode = Literal["local", "feishu", "both"]

_OUTPUT_DIR_NAME = "output"


def _project_root() -> Path:
    """Return the project root directory (the directory containing ``pyproject.toml``)."""
    return Path(__file__).resolve().parents[3]


def output_dir() -> Path:
    """Return the absolute path to the ``output/`` directory at the project root."""
    return _project_root() / _OUTPUT_DIR_NAME


_PROMPT_FILE: Path = (
    Path(__file__).resolve().parents[1] / "prompts" / "strategy_match_system_prompt.md"
)


def _format_strategy_index() -> str:
    """Render the strategy catalog as a Markdown bullet list.

    Mirrors :func:`tools.skill.format_skill_index_markdown` — one bullet
    per file in ``conf/strategies/*.md``, with the ``description`` from
    YAML frontmatter as the one-line purpose. The full body remains
    loadable via :func:`load_strategy` at agent run time.
    """
    names = _list_strategy_names()
    if not names:
        return "_(no strategies available)_\n"
    lines: list[str] = []
    strategies_dir = Path(__file__).resolve().parents[1] / "conf" / "strategies"
    for name in names:
        path = strategies_dir / f"{name}.md"
        try:
            fm = _parse_strategy_frontmatter(path.read_text(encoding="utf-8"))
        except OSError:
            fm = {}
        desc = fm.get("description", "(no description)")
        lines.append(f"- `{name}` — {desc}")
    return "\n".join(lines) + "\n"


def _load_system_prompt() -> str:
    """Load the strategy-match system prompt with both indexes injected.

    Raises:
        FileNotFoundError: if the bundled ``strategy_match_system_prompt.md``
            is missing.
    """
    template = _PROMPT_FILE.read_text(encoding="utf-8")
    strategy_doc = _format_strategy_index()
    tool_doc = format_tool_index_markdown(get_tool_index())
    return (
        template
        .replace("<!-- STRATEGY_INDEX -->", strategy_doc)
        .replace("<!-- TOOL_INDEX -->", tool_doc)
    )


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


def build_output_path(symbol: str, output_dir_path: Path, now_epoch: int | None = None) -> Path:
    """Build the path to which the rendered Markdown for ``symbol`` is written."""
    ts = now_epoch if now_epoch is not None else int(time.time())
    safe_symbol = symbol.replace(".", "_").replace("/", "_")
    return output_dir_path / f"strategy-match-{safe_symbol}-{ts}.md"


def render_local_markdown(report: StrategyMatchReport, now_iso: str) -> str:
    """Render a :class:`StrategyMatchReport` as a 7-section Markdown file."""
    rows = "\n".join(
        f"| {i} | {m.criterion} | {m.match_level} | {m.evidence} | {m.reasoning} |"
        for i, m in enumerate(report.criterion_matches, 1)
    )
    return (
        f"# [{report.symbol}] 策略匹配报告 · {now_iso}\n\n"
        f"> 策略: **{report.strategy_name}** v{report.strategy_version}\n"
        f"> 适合度: **{report.overall_fit}** "
        f"(score: {report.fit_score}/10, confidence: {report.confidence})\n\n"
        f"## 摘要\n{report.summary}\n\n"
        f"## 策略原则逐条匹配\n"
        f"| # | 原则 | 评级 | 证据 | 推理 |\n"
        f"|---|------|------|------|------|\n{rows}\n\n"
        f"## 基本面摘要(来自 subagent)\n{report.raw_analysis_excerpt}\n\n"
        f"## 行动建议\n{report.action_recommendation}\n\n"
        f"---\n*本报告由 AI 生成,不构成投资建议*\n"
    )


def _publish_to_feishu(report: StrategyMatchReport) -> str | None:
    """Best-effort publish the rendered Markdown to a Feishu cloud doc.

    Wraps a ``lark-cli docs +create`` shell call. Returns the new
    document URL on success, or ``None`` if ``lark-cli`` is missing /
    not authenticated / fails — the caller is expected to log a
    warning and fall back to the local markdown.
    """
    if shutil.which("lark-cli") is None:
        logger.warning("lark-cli not on PATH; skipping Feishu publish")
        return None

    content = render_local_markdown(report, datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"))
    title = f"[{report.symbol}] 策略匹配报告 · {report.strategy_name}"
    try:
        proc = subprocess.run(
            ["lark-cli", "docs", "+create", "--title", title, "--content", content],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning("lark-cli invocation failed: %s", e)
        return None
    if proc.returncode != 0:
        logger.warning("lark-cli returned %d: %s", proc.returncode, proc.stderr.strip()[:500])
        return None
    url = next((line.strip() for line in reversed(proc.stdout.splitlines()) if line.strip()), None)
    return url


def _validate_strategy(name: str) -> None:
    """Refuse to start if ``name`` is not a known strategy."""
    if name not in _list_strategy_names():
        available = ", ".join(_list_strategy_names()) or "(none)"
        raise SystemExit(
            f"unknown strategy {name!r}; available: {available}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_strategy",
        description=(
            "Run the strategy-match LLM agent on a stock symbol and "
            "render the report as local markdown and/or a Feishu doc."
        ),
    )
    parser.add_argument("symbol", help="Stock code, e.g. 02319.HK, 600519.SH, 000001.SZ")
    parser.add_argument(
        "--strategy", required=True,
        help="Strategy name (must match a `.md` file under conf/strategies/).",
    )
    parser.add_argument(
        "--delivery", choices=["local", "feishu", "both"], default="both",
        help="Where to deliver the rendered report (default: both).",
    )
    parser.add_argument(
        "--include-shell-tool", dest="include_shell_tool", action="store_true", default=False,
        help="Expose run_command so the agent can call lark-cli directly.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory to write local markdown. Defaults to <project-root>/output/.",
    )
    parser.add_argument(
        "--recursion-limit", type=int, default=80,
        help="LangGraph recursion limit (default 80).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG-level logging.")
    return parser


def run(args: argparse.Namespace) -> int:
    """Top-level orchestration. Returns the process exit code."""
    _validate_strategy(args.strategy)

    system_prompt = _load_system_prompt()
    agent = StrategyMatchAgent(
        system_prompt=system_prompt,
        include_shell_tool=args.include_shell_tool,
        recursion_limit=args.recursion_limit,
    )

    messages: list[BaseMessage] = [HumanMessage(
        content=f"按 system prompt 的 schema 给出 {args.symbol} 在 {args.strategy} 策略下的匹配报告。"
    )]
    last_text = ""
    try:
        for event in agent.stream(messages):
            if event.get("event") == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk", {})
                content = getattr(chunk, "content", "")
                if isinstance(content, str) and content:
                    last_text += content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            last_text += block.get("text", "")
    except ToolExecutionError as e:
        logger.error("agent tools failed: %s", e)
        return EXIT_TOOL

    try:
        json_str = _extract_json_object(_strip_code_fence(last_text))
        report = StrategyMatchReport.model_validate_json(json_str)
    except (ValueError, Exception) as e:  # noqa: BLE001
        logger.error("agent output failed StrategyMatchReport validation: %s", e)
        logger.debug("raw output: %s", last_text[:2000])
        return EXIT_PARSE

    out_dir = args.output_dir or output_dir()
    if args.delivery in ("local", "both"):
        out_dir.mkdir(parents=True, exist_ok=True)
        now_iso = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        path = build_output_path(args.symbol, out_dir)
        path.write_text(render_local_markdown(report, now_iso), encoding="utf-8")
        logger.info("wrote %s", path)

    if args.delivery in ("feishu", "both"):
        url = _publish_to_feishu(report)
        if url:
            logger.info("published to Feishu: %s", url)
        else:
            logger.warning(
                "Feishu publish failed; falling back to local markdown only. "
                "See output/%s for the rendered report.",
                build_output_path(args.symbol, out_dir).name,
            )

    logger.info("summary: %s", report.summary)
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """Parse argv, configure logging, and dispatch to :func:`run`."""
    args = _build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return run(args)
    except SystemExit as e:
        msg = str(e)
        if msg:
            logger.error(msg)
        return EXIT_BAD_STRATEGY
    except Exception as e:  # noqa: BLE001
        logger.exception("unhandled exception: %s", e)
        return EXIT_UNHANDLED


if __name__ == "__main__":
    sys.exit(main())