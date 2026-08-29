"""Skill loading tool: read project-level SKILL.md files on demand.

Skills live in ``src/<package>/skill/<name>/SKILL.md`` and are meant to be
consumed by the LLM agent at runtime. The agent can call :func:`load_skill`
when it needs detailed instructions for a specific task (e.g. formatting
``get_stock_snapshot`` output as a company profile).

Two-tier loading model, mirroring :mod:`stock_analysis_agent.tools.registry`:

1. **Index** — every skill's ``name`` and one-line ``description`` from
   the SKILL.md frontmatter is rendered into the system prompt via
   :func:`get_skill_index` and :func:`format_skill_index_markdown`. The
   LLM uses this catalog to decide which skill to load.
2. **Full body** — once the model picks a skill, it calls
   :func:`load_skill` which returns the entire SKILL.md.

Adding a new skill is a one-line drop-in: drop a directory at
``skill/<name>/SKILL.md`` with frontmatter and the index picks it up
automatically.
"""
from __future__ import annotations

from typing import TypedDict

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from stock_analysis_agent.tools._paths import PACKAGE_ROOT

# Module-level constants — resolved once at import time.
_SKILLS_DIR = PACKAGE_ROOT / "skill"


class SkillIndexEntry(TypedDict):
    """One skill catalog row.

    Attributes:
        name: Skill name as exposed to the LLM (matches the
            ``SKILL.md`` frontmatter ``name`` field, falling back to
            the directory name when frontmatter is missing).
        description: Purpose statement from the frontmatter
            ``description`` field. Multi-line YAML ``|`` blocks are
            preserved as embedded ``\\n`` so the caller can render
            them however it likes.
    """

    name: str
    description: str


_SKILL_FRONTMATTER_KEYS = frozenset({"name", "description"})


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Extract ``name`` and ``description`` from YAML frontmatter.

    Delegates to :func:`stock_analysis_agent.tools._frontmatter.
    parse_yaml_frontmatter`, keeping only the ``name`` and ``description``
    keys — other keys are dropped. Missing keys default to empty strings
    so callers can rely on the schema.

    Args:
        text: Full SKILL.md text starting with the ``---`` fence.

    Returns:
        Dict with ``name`` and ``description`` keys.
    """
    from stock_analysis_agent.tools._frontmatter import parse_yaml_frontmatter

    parsed = parse_yaml_frontmatter(text, allow=_SKILL_FRONTMATTER_KEYS)
    return {
        "name": parsed.get("name", ""),
        "description": parsed.get("description", ""),
    }


def list_skill_names() -> tuple[str, ...]:
    """Discover every **loadable** skill directory under ``skill/``.

    "Loadable" means: present on disk AND listed in
    :data:`_LOADABLE_SKILL_NAMES`. The allowlist controls what the
    system prompt advertises as ``## 我的工具`` — adding a skill to
    the on-disk tree does NOT auto-promote it into the agent's
    catalog. To promote a new skill, append its directory name to
    :data:`_LOADABLE_SKILL_NAMES`.

    Skills on disk that are NOT in the allowlist are still reachable
    via :func:`load_skill` (e.g. internal formatters like
    ``stock-snapshot-format``) — they're just not advertised to the
    LLM up front.

    Returns:
        Alphabetically sorted tuple of skill directory names that
        are both in the allowlist and present on disk.
    """
    if not _SKILLS_DIR.is_dir():
        return ()
    on_disk = {p.parent.name for p in _SKILLS_DIR.glob("*/SKILL.md")}
    return tuple(sorted(name for name in _LOADABLE_SKILL_NAMES if name in on_disk))


#: Allowlist of skills to advertise in the system prompt's ``## 我的工具``
#: section. Order does not matter — :func:`list_skill_names` sorts the
#: output. Edit this tuple to change which skills the agent sees in
#: its catalog; skills on disk that are absent here are still
#: loadable via :func:`load_skill` but won't appear in the index.
_LOADABLE_SKILL_NAMES: tuple[str, ...] = (
    # "announcement-search",
    "lark-doc",
    "lark-shared",
    "mx-finance-data",
    "mx-stocks-screener",
    # "news-search",
    # "report-search",
    "stock-analysis",
    "strategy-match",
    "technical-capital",
    "mx-finance-search",
    "mx-macro-data",
)


#: Known skills — populated from disk at import time so the
#: ``FileNotFoundError`` message in :func:`_read_skill` lists every
#: bundled skill (including ones not in :data:`_LOADABLE_SKILL_NAMES`)
#: without manual maintenance.
_KNOWN_SKILLS: tuple[str, ...] = tuple(
    sorted(p.parent.name for p in _SKILLS_DIR.glob("*/SKILL.md"))
) if _SKILLS_DIR.is_dir() else ()


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


def _read_skill_index_entry(name: str) -> SkillIndexEntry:
    """Read one skill's SKILL.md and extract its index entry.

    Falls back to the directory name and a placeholder description
    when the file is missing or unreadable, so the catalog still
    surfaces the skill (the LLM will see "SKILL.md missing" and skip
    it instead of thinking it doesn't exist).

    Args:
        name: Skill directory name.

    Returns:
        :class:`SkillIndexEntry` populated from frontmatter.
    """
    path = _SKILLS_DIR / name / "SKILL.md"
    if not path.is_file():
        return SkillIndexEntry(name=name, description="(SKILL.md missing)")
    try:
        text = path.read_text(encoding="utf-8")
        fm = _parse_frontmatter(text)
    except (OSError, UnicodeDecodeError):
        return SkillIndexEntry(name=name, description="(unreadable SKILL.md)")
    return SkillIndexEntry(
        name=fm.get("name") or name,
        description=fm.get("description") or "(no description in frontmatter)",
    )


def get_skill_index() -> list[SkillIndexEntry]:
    """Build the skill catalog injected into the system prompt.

    Walks every directory under ``skill/``, reads its SKILL.md
    frontmatter, and returns one entry per skill in alphabetical
    order. Mirrors :func:`stock_analysis_agent.tools.registry.get_tool_index`.

    Returns:
        Alphabetically sorted list of :class:`SkillIndexEntry`.
    """
    return [_read_skill_index_entry(name) for name in list_skill_names()]


def format_skill_index_markdown(index: list[SkillIndexEntry]) -> str:
    """Render ``index`` as a Markdown bullet list.

    Each entry renders as one bullet: ``- `<name>` — <description>``.
    Multi-line descriptions (YAML ``|`` blocks) are collapsed into a
    single paragraph by joining the lines with spaces, so the catalog
    stays compact in the system prompt. The full multi-line text is
    available via :func:`load_skill` when the LLM wants detail.

    Args:
        index: Catalog from :func:`get_skill_index`.

    Returns:
        Markdown bullet list. Empty list yields
        ``"_(no skills available)_"``.
    """
    if not index:
        return "_(no skills available)_\n"
    lines: list[str] = []
    for entry in index:
        desc = " ".join(
            line.strip() for line in entry["description"].splitlines() if line.strip()
        )
        lines.append(f"- `{entry['name']}` — {desc}")
    return "\n".join(lines) + "\n"


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
        "## 我的工具 section lists the loadable skills' name and one-line "
        "purpose, but a few internal skills (e.g. `stock-snapshot-format`) "
        "are reachable only via this tool. Call `load_skill(name=...)` when "
        "the catalog summary is not enough to produce the structured output "
        "the user asked for (e.g. formatting `get_stock_snapshot` data as "
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


__all__ = [
    "LoadSkillInput",
    "SkillIndexEntry",
    "_read_skill",
    "_read_skill_index_entry",
    "format_skill_index_markdown",
    "get_skill_index",
    "list_skill_names",
    "load_skill",
]
