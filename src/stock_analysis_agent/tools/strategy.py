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

from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

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
    (joined with spaces). Mirrors :func:`tools.skill._parse_frontmatter`
    but returns a free-form dict instead of a fixed schema.

    Args:
        text: Full strategy Markdown text starting with the ``---`` fence.

    Returns:
        Dict of frontmatter keys. Missing or unparseable input yields ``{}``.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    end = 1
    while end < len(lines) and lines[end].strip() != "---":
        end += 1
    fm_lines = lines[1:end]
    result: dict[str, str] = {}
    i = 0
    while i < len(fm_lines):
        line = fm_lines[i]
        if ":" not in line or line.startswith((" ", "\t")):
            i += 1
            continue
        key, _, rest = line.partition(":")
        key = key.strip()
        # Only retain the keys the strategy schema actually defines.
        if key not in _STRATEGY_FRONTMATTER_KEYS:
            i += 1
            continue
        rest = rest.strip()
        if rest == "|":
            i += 1
            block: list[str] = []
            while i < len(fm_lines):
                nxt = fm_lines[i]
                if nxt.startswith((" ", "\t")) and nxt.strip():
                    block.append(nxt.strip())
                    i += 1
                else:
                    break
            result[key] = " ".join(block)
            continue
        if len(rest) >= 2 and rest[0] == rest[-1] and rest[0] == '"':
            rest = rest[1:-1]
        result[key] = rest
        i += 1
    return result


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


__all__ = ["LoadStrategyInput", "_list_strategy_names", "_parse_strategy_frontmatter", "load_strategy"]
