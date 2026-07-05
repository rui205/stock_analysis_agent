"""Skill loading tool: read project-level SKILL.md files on demand.

Skills live in ``src/<package>/skill/<name>/SKILL.md`` and are meant to be
consumed by the LLM agent at runtime. The agent can call :func:`load_skill`
when it needs detailed instructions for a specific task (e.g. formatting
``get_stock_snapshot`` output as a company profile).
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

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


class LoadSkillInput(BaseModel):
    """Input schema for the ``load_skill`` tool.

    The ``name`` argument is free-form — any directory under
    ``src/<package>/skill/<name>/`` that contains a ``SKILL.md`` is
    accepted. The schema deliberately does not use a ``Literal`` type
    so the LLM can discover and load skills that are added after the
    agent's tool wiring is built.
    """

    name: str = Field(
        description=(
            "Skill name — must match a directory under "
            "`src/stock_analysis_agent/skill/<name>/` containing a "
            "`SKILL.md` file. Examples: `lark-doc`, "
            "`stock-snapshot-format`. Unknown names raise "
            "`FileNotFoundError` with the available list."
        ),
        min_length=1,
    )


@tool(
    "load_skill",
    description=(
        "Load a project-level skill's full SKILL.md as Markdown. Use this "
        "ONLY after deciding which skill you need — the system prompt's "
        "## 我的工具 section already lists every bundled skill's name "
        "and one-line purpose. Call `load_skill(name=...)` when that "
        "summary is not enough to produce the structured output the "
        "user asked for (e.g. formatting `get_stock_snapshot` data as "
        "a company profile, or invoking 飞书云文档 via `lark-cli`). "
        "The skill name must match a directory under "
        "`src/stock_analysis_agent/skill/<name>/`; an unknown name "
        "raises FileNotFoundError and lists the available skills. To "
        "read a skill's `references/*.md` or scripts, use the separate "
        "`read_file` tool."
    ),
    args_schema=LoadSkillInput,
)
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

    Returns:
        The full Markdown content of the skill's ``SKILL.md`` as a
        ``str``. The LLM should follow these instructions to produce
        the formatted output for the user.

    Raises:
        FileNotFoundError: ``name`` does not match a bundled skill.
            The error message lists the available skills.
    """
    return _read_skill(name)


__all__ = ["LoadSkillInput", "load_skill", "_read_skill"]
