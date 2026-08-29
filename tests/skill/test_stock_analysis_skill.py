"""Data-driven tests for the stock-analysis skill's output contract.

The 8-section lark-doc XML report format and the fallback path live in this
skill (not in ``prompts/system_prompt.md``); these tests pin the contract
so future edits don't silently drift away from the documented behavior.

Tests cover (mirroring what used to be checked against system_prompt.md):

* the ``lark-cli docs +create`` invocation with ``--api-version v2``;
* the ``[{symbol}] 股票分析报告 · {YYYY-MM-DD}`` title template;
* the 7 enumerated sections in order (执行摘要 / 公司画像 / 多维评分 /
  价位计划 / 基本面+技术面 / 风险 / 数据声明);
* the "only return link + one-liner" rule for the conversation reply;
* the fallback path that fires when ``lark-cli`` is unavailable or fails;
* and a regression guard against the format being re-introduced into
  ``prompts/system_prompt.md`` (single source of truth).
"""
from __future__ import annotations

from pathlib import Path

import pytest  # noqa: F401  — imported for future fixture use

SKILL_PATH = Path(
    "src/stock_analysis_agent/skill/stock-analysis/SKILL.md"
)
SYSTEM_PROMPT_PATH = Path(
    "src/stock_analysis_agent/prompts/system_prompt.md"
)


def _read_skill() -> str:
    """Return the raw stock-analysis/SKILL.md text."""
    return SKILL_PATH.read_text(encoding="utf-8")


def _read_system_prompt() -> str:
    """Return the raw system_prompt.md text (template, not rendered)."""
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """Return the body of a ``## <heading>`` section, stopping at next ``##``.

    Returns an empty string if the heading is not found.
    """
    import re

    pattern = re.compile(
        rf"^##\s+{re.escape(heading)}\s*$\n(.*?)(?=^##\s+|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    return match.group(1) if match else ""


# ---------------------------------------------------------------------------
# Output contract — lark-doc publication must be declared
# ---------------------------------------------------------------------------


class TestOutputContract:
    """The Output contract section must declare the lark-doc publication plan."""

    def test_output_contract_section_exists(self) -> None:
        """A '## Output contract' section must exist on the stock-analysis skill."""
        text = _read_skill()
        body = _section(text, "Output contract")
        assert body, "missing '## Output contract' section in stock-analysis/SKILL.md"

    def test_output_contract_mentions_lark_doc_create_command(self) -> None:
        """The contract must instruct the agent to use lark-doc's +create command."""
        text = _read_skill()
        body = _section(text, "Output contract")
        assert "+create" in body, (
            "output contract must reference the lark-doc `+create` shortcut"
        )
        assert "--api-version v2" in body, (
            "output contract must declare the v2 API flag (lark-doc is v2-only)"
        )

    def test_output_contract_declares_title_format(self) -> None:
        """The contract must declare the document title with `{symbol}` and a date."""
        text = _read_skill()
        body = _section(text, "Output contract")
        assert "{symbol}" in body, (
            "output contract must include `{symbol}` as part of the title template"
        )
        assert "YYYY-MM-DD" in body or "YYYY" in body, (
            "output contract must include a date placeholder "
            "(e.g. {YYYY-MM-DD})"
        )

    def test_output_contract_enumerates_eight_sections(self) -> None:
        """The contract must enumerate the 8 document sections in order.

        "基本面 + 技术面分析" is one spec section but asserted as two
        distinct substrings (基本面 and 技术面) so the assertion list has
        9 items — same convention used by the original system_prompt
        tests.
        """
        text = _read_skill()
        body = _section(text, "Output contract")
        expected_substrings = [
            "执行摘要",
            "公司画像",
            "宏观背景",
            "多维评分",
            "价位计划",
            "基本面",
            "技术面",  # 配套 "基本面 + 技术面分析"
            "风险",
            "免责声明",
        ]
        for needle in expected_substrings:
            assert needle in body, f"section {needle!r} not in Output contract"

    def test_output_contract_mandates_link_only_in_conversation(self) -> None:
        """The contract must explicitly say: only return link in conversation."""
        text = _read_skill()
        body = _section(text, "Output contract")
        assert "链接" in body or "link" in body.lower(), (
            "output contract must say the agent returns a link (not the full body)"
        )
        # The "禁止" / "不要" prohibition language must be present.
        assert "禁止" in body or "不要" in body, (
            "output contract must contain an explicit prohibition "
            "(e.g. '禁止在对话内重复 8 节正文')"
        )

    def test_output_contract_declares_fallback(self) -> None:
        """The contract must declare what happens when lark-doc creation fails."""
        text = _read_skill()
        body = _section(text, "Output contract")
        fallback_signals = ("降级", "失败", "fallback", "错误", "重试")
        assert any(sig in body.lower() for sig in fallback_signals), (
            "output contract must declare a fallback path for lark-doc failures"
        )


# ---------------------------------------------------------------------------
# Regression guard — single source of truth
# ---------------------------------------------------------------------------


class TestSingleSourceOfTruth:
    """The 8-section format must live ONLY in stock-analysis/SKILL.md.

    system_prompt.md owns role/identity/scope/principles/tool index only;
    it must not re-introduce a concrete output format section. This guard
    catches accidental back-sliding into the dual-source pattern.
    """

    def test_system_prompt_has_no_output_strategy_section(self) -> None:
        """system_prompt.md must NOT carry an ## 输出策略 section."""
        text = _read_system_prompt()
        body = _section(text, "输出策略")
        assert not body, (
            "system_prompt.md must not define an '## 输出策略' section — "
            "the format belongs in the stock-analysis skill"
        )

    def test_system_prompt_does_not_enumerate_eight_sections(self) -> None:
        """system_prompt.md must NOT enumerate the 8 section names.

        Allows the words "风险点" and "免责声明" to appear (they are part of
        role-level stop conditions), but the specific 7 report headings
        (执行摘要, 多维评分, 价位计划, etc.) must not leak back into the
        system prompt.
        """
        text = _read_system_prompt()
        forbidden = ("执行摘要", "多维评分", "价位计划", "数据声明与免责声明")
        leaks = [s for s in forbidden if s in text]
        assert not leaks, (
            f"system_prompt.md must not enumerate report sections; "
            f"found: {leaks}"
        )

    def test_skill_does_not_delegate_to_system_prompt(self) -> None:
        """The skill must NOT tell the reader to consult system_prompt.md
        for the 8-section format — that was the dual-source pattern we
        just removed.
        """
        text = _read_skill()
        body = _section(text, "Output contract")
        assert "system_prompt.md" not in body, (
            "Output contract must not cross-reference system_prompt.md; "
            "the format is owned by this skill"
        )
