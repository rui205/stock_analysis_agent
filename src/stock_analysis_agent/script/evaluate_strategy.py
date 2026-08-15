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
import re
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
from stock_analysis_agent.agent.stream import collect_final_text
from stock_analysis_agent.tools.prompt import render_system_prompt, resolve_tool_names
from stock_analysis_agent.tools.strategy import (
    _STRATEGIES_DIR,
    _list_strategy_names,
    _parse_strategy_frontmatter,
)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_UNHANDLED = 1
EXIT_PARSE = 2
EXIT_TOOL = 3
EXIT_BAD_STRATEGY = 4

DeliveryMode = Literal["local", "feishu", "both"]

_OUTPUT_DIR_NAME = "output"


class UnknownStrategyError(ValueError):
    """Raised when ``--strategy`` names a strategy with no ``.md`` file."""


#: Tool names exposed to ``StrategyMatchAgent`` (the orchestrator) —
#: injected into ``<!-- TOOL_INDEX -->`` so the system prompt matches
#: the wired tools 1:1. The sub-agent's data-discovery surface
#: (``read_file``) is deliberately excluded: the
#: orchestrator's job is workflow glue, not raw research.
_ORCHESTRATOR_TOOL_NAMES: list[str] = [
    "load_skill",
    "load_strategy",
    "run_analyze_stock",
]


def _orchestrator_tool_names(include_shell_tool: bool = False) -> list[str]:
    """Compute the orchestrator's full tool-name list for prompt rendering.

    Args:
        include_shell_tool: Whether ``run_command`` should be
            advertised alongside the orchestrator's defaults.

    Returns:
        Sorted, deduplicated list of tool names matching what
        :class:`StrategyMatchAgent` actually wires up for this run.
    """
    return resolve_tool_names(_ORCHESTRATOR_TOOL_NAMES, include_shell_tool)


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
    for name in names:
        path = _STRATEGIES_DIR / f"{name}.md"
        try:
            fm = _parse_strategy_frontmatter(path.read_text(encoding="utf-8"))
        except OSError:
            fm = {}
        desc = fm.get("description", "(no description)")
        lines.append(f"- `{name}` — {desc}")
    return "\n".join(lines) + "\n"


def _load_system_prompt(include_shell_tool: bool = False) -> str:
    """Load the strategy-match system prompt with both indexes injected.

    The catalog (``<!-- TOOL_INDEX -->``) is filtered to the
    **orchestrator's actual tool set**
    (:func:`_orchestrator_tool_names`) — not the full project-wide
    registry. The sub-agent's ``read_file`` is deliberately omitted so
    the orchestrator isn't tempted to do raw research itself
    (orchestration ≠ research).

    Args:
        include_shell_tool: When ``True``, ``run_command`` is also
            advertised to the LLM (matches the constructor flag that
            controls tool wiring). Default ``False``.

    Raises:
        FileNotFoundError: if the bundled ``strategy_match_system_prompt.md``
            is missing.
    """
    return render_system_prompt(
        _PROMPT_FILE,
        tool_names=_orchestrator_tool_names(include_shell_tool),
        catalog_placeholder="<!-- STRATEGY_INDEX -->",
        catalog_doc=_format_strategy_index(),
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


def _md_cell(text: str) -> str:
    """Escape ``|`` and newlines so ``text`` stays one markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ")


def render_local_markdown(report: StrategyMatchReport, now_iso: str) -> str:
    """Render a :class:`StrategyMatchReport` as a Markdown file."""
    rows = "\n".join(
        f"| {i} | {_md_cell(m.criterion)} | {_md_cell(m.match_level)} | "
        f"{_md_cell(m.evidence)} | {_md_cell(m.reasoning)} |"
        for i, m in enumerate(report.criterion_matches, 1)
    )
    deepresearch = report.data_sources.deepresearch or "未调用 deepresearch(基本面数据已足够)"
    return (
        f"# [{report.symbol}] 策略匹配报告 · {now_iso}\n\n"
        f"> 策略: **{report.strategy_name}** v{report.strategy_version}\n"
        f"> 适合度: **{report.overall_fit}** "
        f"(score: {report.fit_score}/10, confidence: {report.confidence})\n\n"
        f"## 摘要\n{report.summary}\n\n"
        f"## 策略原则逐条匹配\n"
        f"| # | 原则 | 评级 | 证据 | 推理 |\n"
        f"|---|------|------|------|------|\n{rows}\n\n"
        f"## 数据来源\n"
        f"### 来自 stock_analysis\n{report.data_sources.stock_analysis}\n\n"
        f"### 来自 deepresearch\n{deepresearch}\n\n"
        f"## 判断理论\n{report.judgment_rationale}\n\n"
        f"## 行动建议\n{report.action_recommendation}\n\n"
        f"---\n*本报告由 AI 生成,不构成投资建议*\n"
    )


_FEISHU_DOC_URL_RE = re.compile(r"""https://[^\s"']+/docx/[A-Za-z0-9]+""")


def _extract_feishu_doc_url(stdout: str) -> str | None:
    """Extract the new document URL from ``lark-cli docs +create`` output.

    lark-cli prints a JSON envelope —
    ``{"ok": true, "data": {"document": {"url": ...}}}`` — pretty-printed
    by default, so the last stdout line is ``}``, not the URL. Parse the
    envelope first; fall back to scanning for a Feishu ``docx`` URL when
    the payload is not parseable JSON.

    Args:
        stdout: Raw stdout of the ``lark-cli docs +create`` invocation.

    Returns:
        The document URL, or ``None`` when it cannot be located.
    """
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        data = payload.get("data")
        document = data.get("document") if isinstance(data, dict) else None
        if isinstance(document, dict):
            url = document.get("url")
            if isinstance(url, str) and url:
                return url
    match = _FEISHU_DOC_URL_RE.search(stdout)
    return match.group(0) if match else None


#: Backoff (seconds) for a transient ``lark-cli`` publish failure, matching
#: the "network/rate-limit → retry twice (1s / 3s)" policy in
#: ``skill/strategy-match/SKILL.md``. The initial attempt is free; each entry
#: here is one retry.
_FEISHU_RETRY_DELAYS: tuple[float, ...] = (1.0, 3.0)


def _publish_to_feishu(markdown: str) -> str | None:
    """Best-effort publish ``markdown`` to a Feishu cloud doc.

    Wraps a ``lark-cli docs +create`` shell call. Returns the new
    document URL on success, or ``None`` if ``lark-cli`` is missing /
    not authenticated / fails after retries — the caller falls back to
    local markdown.

    The markdown starts with a single H1, which lark-cli (with
    ``--doc-format markdown``) extracts as the document title, so no
    separate ``--title`` flag is passed. Transient timeouts are retried
    with a short backoff; auth/usage errors fail fast.
    """
    if shutil.which("lark-cli") is None:
        logger.warning("lark-cli not on PATH; skipping Feishu publish")
        return None

    cmd = [
        "lark-cli", "docs", "+create",
        "--api-version", "v2",
        "--doc-format", "markdown",
        "--content", markdown,
    ]
    for attempt in range(len(_FEISHU_RETRY_DELAYS) + 1):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired as e:
            if attempt == len(_FEISHU_RETRY_DELAYS):
                logger.warning(
                    "lark-cli timed out after %d attempts: %s", attempt + 1, e
                )
                return None
            time.sleep(_FEISHU_RETRY_DELAYS[attempt])
            continue
        except OSError as e:
            logger.warning("lark-cli invocation failed: %s", e)
            return None
        if proc.returncode != 0:
            logger.warning(
                "lark-cli returned %d: %s", proc.returncode, proc.stderr.strip()[:500]
            )
            return None
        return _extract_feishu_doc_url(proc.stdout)
    return None  # unreachable — every iteration returns or retries


def _validate_strategy(name: str) -> None:
    """Refuse to start if ``name`` is not a known strategy."""
    available = _list_strategy_names()
    if name not in available:
        raise UnknownStrategyError(
            f"unknown strategy {name!r}; available: {', '.join(available) or '(none)'}"
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
        help=(
            "Expose run_command so the agent can call lark-cli directly. The "
            "flag also propagates to the analyze-stock subagent so its "
            "stock-analysis workflow can execute the mx-* skill data scripts "
            "(without it the subagent emits a degraded report)."
        ),
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

    system_prompt = _load_system_prompt(include_shell_tool=args.include_shell_tool)
    agent = StrategyMatchAgent(
        system_prompt=system_prompt,
        include_shell_tool=args.include_shell_tool,
        recursion_limit=args.recursion_limit,
    )

    messages: list[BaseMessage] = [HumanMessage(
        content=f"按 system prompt 的 schema 给出 {args.symbol} 在 {args.strategy} 策略下的匹配报告。"
    )]
    try:
        last_text = collect_final_text(agent.stream(messages))
    except ToolExecutionError as e:
        # The middleware wraps the original exception via ``raise ... from
        # exc``; surface its type so failures with empty/ambiguous messages
        # (e.g. recursion-budget exhaustion) remain diagnosable from the log.
        cause = e.__cause__
        suffix = f" (cause: {type(cause).__name__})" if cause is not None else ""
        logger.error("agent tools failed: %s%s", e, suffix)
        return EXIT_TOOL

    try:
        json_str = _extract_json_object(_strip_code_fence(last_text))
        report = StrategyMatchReport.model_validate_json(json_str)
    except ValueError as e:
        logger.error("agent output failed StrategyMatchReport validation: %s", e)
        logger.debug("raw output: %s", last_text[:2000])
        return EXIT_PARSE

    out_dir = args.output_dir or output_dir()
    now_iso = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    markdown = render_local_markdown(report, now_iso)
    path = build_output_path(args.symbol, out_dir)

    if args.delivery in ("local", "both"):
        out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
        logger.info("wrote %s", path)

    if args.delivery in ("feishu", "both"):
        url = _publish_to_feishu(markdown)
        if url:
            logger.info("published to Feishu: %s", url)
        elif args.delivery == "feishu":
            # feishu-only: degrade to local markdown (per SKILL.md) rather
            # than leaving the user with no artifact at all.
            out_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(markdown, encoding="utf-8")
            logger.warning("Feishu publish failed; wrote local markdown to %s", path)
        else:
            logger.warning("Feishu publish failed; local markdown already written to %s", path)

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
    except UnknownStrategyError as e:
        logger.error("%s", e)
        return EXIT_BAD_STRATEGY
    except Exception as e:  # noqa: BLE001
        logger.exception("unhandled exception: %s", e)
        return EXIT_UNHANDLED


if __name__ == "__main__":
    sys.exit(main())