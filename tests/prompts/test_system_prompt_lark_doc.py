"""Validates the system_prompt.md role / identity / stop-condition contract.

System prompt owns: role ("我是谁"), audience ("我为谁服务"), scope
("我做 / 不做"), working principles, tool/skill catalog injection
points, and the role-level stop conditions that promise three deliverables
+ disclaimer.

System prompt does NOT own: the 7-section report format, the lark-doc
publication command, the markdown fallback — those moved to the
``stock-analysis`` skill and are now pinned by
``tests/skill/test_stock_analysis_skill.py``.
"""
from __future__ import annotations

import re

from stock_analysis_agent.script.analyze_stock import _load_system_prompt


def _read_prompt() -> str:
    """Return the fully-rendered system prompt (template + injected indexes).

    Tests check the **rendered** prompt — that is what the LLM actually
    sees at runtime. The raw template (with ``<!-- SKILL_INDEX -->`` /
    ``<!-- TOOL_INDEX -->`` placeholders) is a build artifact; asserting
    on it would miss dynamic content (e.g. lark-doc is now injected via
    SKILL_INDEX, not hardcoded in the template).
    """
    return _load_system_prompt()


def _section(text: str, heading: str) -> str:
    """Return the body of a ``## <heading>`` section, stopping at the next ``## ``.

    Returns an empty string if the heading is not found.
    """
    pattern = re.compile(
        rf"^##\s+{re.escape(heading)}\s*$\n(.*?)(?=^##\s+|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    return match.group(1) if match else ""


# ---------------------------------------------------------------------------
# Tool list — lark-doc must be declared
# ---------------------------------------------------------------------------


def test_lark_doc_listed_in_my_tools_section() -> None:
    """The '## 我的工具' section must mention lark-doc as an available skill."""
    text = _read_prompt()
    section_body = _section(text, "我的工具")
    assert section_body, "missing '## 我的工具' section"
    assert "lark-doc" in section_body, (
        "lark-doc is not listed under '## 我的工具'; "
        "add it after the 5 finance skills"
    )


def test_lark_doc_tool_entry_uses_backtick_name() -> None:
    """The lark-doc tool entry must use the backticked name `lark-doc` for consistency."""
    text = _read_prompt()
    section_body = _section(text, "我的工具")
    assert "`lark-doc`" in section_body, (
        "lark-doc entry should be wrapped in backticks: `- `lark-doc` — ...`"
    )


# ---------------------------------------------------------------------------
# Stop conditions — role-level deliverables only
# ---------------------------------------------------------------------------


def test_stop_conditions_declare_three_promises_and_disclaimer() -> None:
    """The '## 我什么时候停' section must keep the four role-level promises:
    投资建议 / 估值区间 / 风险点 / 免责声明, plus a soft delegation to the
    ``stock-analysis`` skill for delivery specifics.
    """
    text = _read_prompt()
    section_body = _section(text, "我什么时候停")
    assert section_body, "missing '## 我什么时候停' section"
    for needle in ("投资建议", "估值区间", "风险点", "免责声明"):
        assert needle in section_body, f"role-level stop condition missing: {needle}"
    assert "skill" in section_body.lower(), (
        "stop section must soft-delegate delivery to the stock-analysis skill"
    )


def test_stop_conditions_do_not_leak_specific_format() -> None:
    """The stop section must NOT enumerate the 7 section names or the lark-cli
    command — those now live in the skill. This catches accidental
    back-sliding into the dual-source pattern.
    """
    text = _read_prompt()
    section_body = _section(text, "我什么时候停")
    forbidden = ("执行摘要", "多维评分", "价位计划", "数据声明与免责声明", "+create")
    leaks = [s for s in forbidden if s in section_body]
    assert not leaks, (
        f"stop section must not enumerate report format / lark-cli command; "
        f"found: {leaks}"
    )
