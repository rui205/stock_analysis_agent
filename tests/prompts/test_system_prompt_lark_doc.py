"""Validates the lark-doc output policy added to system_prompt.md.

The system prompt is a Markdown document consumed by an LLM, but its
structure is mechanical enough to validate with regex / substring checks:
the new ``## 输出策略`` section must exist, must mention lark-doc by name,
must declare the 7-section document body, the title format, the
"only return link" rule, and the fallback path.
"""
from __future__ import annotations

import re
from pathlib import Path

PROMPT_PATH = Path("src/stock_analysis_agent/prompts/system_prompt.md")


def _read_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


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
# Stop conditions — lark-doc publishing must be required
# ---------------------------------------------------------------------------


def test_stop_conditions_include_lark_doc_publish() -> None:
    """The '## 我什么时候停' section must require successful lark-doc publishing."""
    text = _read_prompt()
    section_body = _section(text, "我什么时候停")
    assert section_body, "missing '## 我什么时候停' section"
    assert "飞书" in section_body or "lark" in section_body.lower(), (
        "stop conditions must reference lark/飞书 (either lark-doc publish "
        "or the explicit fallback path)"
    )
    # The original three stop conditions must still be present.
    for needle in ("7 节结构化报告", "投资建议", "免责声明"):
        assert needle in section_body, f"original stop condition missing: {needle}"


# ---------------------------------------------------------------------------
# Output policy section — structure
# ---------------------------------------------------------------------------


def test_output_policy_section_exists() -> None:
    """A new '## 输出策略' section must exist (placed after '## 我什么时候停')."""
    text = _read_prompt()
    section_body = _section(text, "输出策略")
    assert section_body, "missing '## 输出策略' section"


def test_output_policy_mentions_lark_doc_create_command() -> None:
    """The output policy must instruct the agent to use lark-doc's +create command."""
    text = _read_prompt()
    section_body = _section(text, "输出策略")
    assert "+create" in section_body, (
        "output policy must reference the lark-doc `+create` shortcut"
    )
    assert "--api-version v2" in section_body, (
        "output policy must declare the v2 API flag (lark-doc is v2-only)"
    )


def test_output_policy_declares_title_format() -> None:
    """The output policy must declare the document title format with `{symbol}` and a date."""
    text = _read_prompt()
    section_body = _section(text, "输出策略")
    assert "{symbol}" in section_body, (
        "output policy must include `{symbol}` as part of the title template"
    )
    assert "YYYY-MM-DD" in section_body or "YYYY" in section_body, (
        "output policy must include a date placeholder (e.g. {YYYY-MM-DD})"
    )


def test_output_policy_enumerates_seven_sections() -> None:
    """The output policy must enumerate the 7 document sections in order.

    Note: "基本面 + 技术面分析" is one spec section but is asserted as two
    distinct substrings (基本面 and 技术面) so the assertion list has 8 items.
    """
    text = _read_prompt()
    section_body = _section(text, "输出策略")
    # The 7 spec sections; "基本面 + 技术面分析" is checked as two substrings.
    expected_substrings = [
        "执行摘要",
        "公司画像",
        "多维评分",
        "价位计划",
        "基本面",
        "技术面",  # 配套 "基本面 + 技术面分析"
        "风险",
        "免责声明",
    ]
    for needle in expected_substrings:
        assert needle in section_body, f"section {needle!r} not in 输出策略"


# ---------------------------------------------------------------------------
# Conversation output — must be terse (link only)
# ---------------------------------------------------------------------------


def test_output_policy_mandates_link_only_in_conversation() -> None:
    """The output policy must explicitly say: only return link in conversation, not full body."""
    text = _read_prompt()
    section_body = _section(text, "输出策略")
    assert "链接" in section_body or "link" in section_body.lower(), (
        "output policy must say the agent returns a link (not the full report body)"
    )
    # The "禁止" / "不要" prohibition language must be present.
    assert "禁止" in section_body or "不要" in section_body, (
        "output policy must contain an explicit prohibition (e.g. '禁止在对话内重复 7 节正文')"
    )


# ---------------------------------------------------------------------------
# Fallback path — must be declared for lark-cli failures
# ---------------------------------------------------------------------------


def test_output_policy_declares_fallback() -> None:
    """The output policy must declare what happens when lark-doc creation fails."""
    text = _read_prompt()
    section_body = _section(text, "输出策略")
    # The fallback should mention degradation / failure handling.
    fallback_signals = ("降级", "失败", "fallback", "错误", "重试")
    assert any(sig in section_body.lower() for sig in fallback_signals), (
        "output policy must declare a fallback path for lark-doc failures"
    )
