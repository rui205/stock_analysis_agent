"""Validates the strategy-match analyze-stock sub-agent prompt contract.

The ``run_analyze_stock`` sub-agent is a :class:`StockAnalysisAgent` driven
by a *dedicated* prompt (``prompts/strategy_match_stock_analysis_system_prompt.md``),
not the user-facing ``prompts/system_prompt.md``. Its output contract must
require a structured field summary in-session — overriding the stock-analysis
skill's "link only" rule — so the orchestrator has fundamentals to match
against without re-fetching everything via ``run_deepresearch``.
"""
from __future__ import annotations

import re

from stock_analysis_agent.tools.strategy import (
    _load_strategy_match_stock_analysis_prompt,
)


def _read_prompt() -> str:
    """Return the fully-rendered sub-agent prompt (template + injected indexes)."""
    return _load_strategy_match_stock_analysis_prompt()


def _section(text: str, heading: str) -> str:
    """Return the body of a ``## <heading>`` section, stopping at the next ``##``.

    Returns an empty string if the heading is not found.
    """
    pattern = re.compile(
        rf"^##\s+{re.escape(heading)}\s*$\n(.*?)(?=^##\s+|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    return match.group(1) if match else ""


class TestRenders:
    def test_placeholders_resolved(self) -> None:
        text = _read_prompt()
        assert "<!-- SKILL_INDEX -->" not in text
        assert "<!-- TOOL_INDEX -->" not in text

    def test_subagent_tool_catalog_only(self) -> None:
        """The sub-agent advertises its own tools, never the orchestrator's."""
        text = _read_prompt()
        for name in ("load_skill", "read_file", "run_command"):
            assert f"### `{name}`" in text, f"missing sub-agent tool {name!r}"
        for name in ("run_analyze_stock", "run_deepresearch", "load_strategy"):
            assert f"### `{name}`" not in text, (
                f"orchestrator-only tool {name!r} leaked into the sub-agent prompt"
            )


class TestOutputContract:
    def test_contract_section_exists(self) -> None:
        body = _section(_read_prompt(), "输出契约")
        assert body, "missing '## 输出契约' section"

    def test_contract_overrides_link_only_rule(self) -> None:
        """The contract must explicitly override the skill's 'link only' rule."""
        body = _section(_read_prompt(), "输出契约")
        assert "覆盖" in body
        assert "只回链接" in body or "只返回链接" in body

    def test_contract_requires_structured_fields(self) -> None:
        """The contract must demand the strategy-matchable fundamentals."""
        body = _section(_read_prompt(), "输出契约")
        for needle in (
            "verdict", "score", "ROE", "毛利率", "资产负债率", "股息率", "商誉",
        ):
            assert needle in body, f"structured field {needle!r} missing"

    def test_contract_marks_missing_fields_as_unknown(self) -> None:
        """Fields the workflow did not fetch must be marked, never fabricated."""
        body = _section(_read_prompt(), "输出契约")
        assert "未获取" in body
        assert "不得编造" in body
