"""Skill loading tool: read project-level SKILL.md files on demand.

Skills live in ``src/<package>/skill/<name>/SKILL.md`` and are meant to be
consumed by the LLM agent at runtime. The agent can call :func:`load_skill`
when it needs detailed instructions for a specific task (e.g. formatting
``get_stock_snapshot`` output as a company profile).
"""
from __future__ import annotations

from pathlib import Path
import string
from typing import Literal

from langchain.tools import tool

# Module-level constants — resolved once at import time.
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_SKILLS_DIR = _PACKAGE_ROOT / "skill"

# Known skills — used to populate the error message when an unknown skill
# is requested. The ``name`` parameter on :func:`load_skill` is a free-form
# ``str`` (not a Literal) so the LLM can request any skill whose
# ``SKILL.md`` exists on disk, including ones not baked into the agent's
# tool wiring (e.g. ``lark-doc``).
_KNOWN_SKILLS: tuple[str, ...] = ("stock-snapshot-format", "lark-doc")


def _read_skill(name: str) -> str:
    """Read the SKILL.md for ``name`` from the source tree.

    Args:
        name: Skill name (must match a directory under
            ``src/<package>/skill/<name>/``).

    Returns:
        Full Markdown content of the skill's ``SKILL.md``.

    Raises:
        FileNotFoundError: If the skill does not exist. The error message
            lists the available skills to help the LLM recover.
    """
    path = _SKILLS_DIR / name / "SKILL.md"
    if not path.is_file():
        available = ", ".join(_KNOWN_SKILLS) or "(none)"
        raise FileNotFoundError(
            f"skill {name!r} not found at {path}; available: {available}"
        )
    return path.read_text(encoding="utf-8")


@tool("load_skill")
def load_skill(
    name: str,
) -> str:
    """Load a project-level skill's full instructions as a Markdown string.

    Use this tool when you need detailed instructions for a specific task
    that the system prompt does not cover inline — typically when the
    user asks for a formatted report, company profile, or similar
    structured output.

    Bundled skills include:
        ``"stock-snapshot-format"`` — Format the nested multi-source
        JSON output of ``get_stock_snapshot`` into a standardized company
        profile with sections: 公司简介 / 主营业务 / 当前股价与估值 /
        财务概览 / 近期公告与新闻 / 治理变动 / 数据声明 /
        (可选) 同业对比. Use this whenever the user mentions
        公司画像、股票快照、stock snapshot、company profile, or asks
        for a structured summary of the snapshot data.
        ``"lark-doc"`` — Read/write 飞书云文档 (Docx v2 API). The
        instructions describe how to invoke the ``lark-cli`` shell
        command; the agent still needs a tool wrapper to actually spawn
        ``lark-cli`` (this tool only returns the documentation).

    Args:
        name: The skill name. Any skill whose ``SKILL.md`` exists under
            ``src/<package>/skill/`` is accepted.

    Returns:
        The full Markdown content of the skill's ``SKILL.md``. The LLM
        should follow these instructions to produce the formatted output
        for the user.
    """
    return _read_skill(name)


__all__ = ["load_skill", "_read_skill"]
