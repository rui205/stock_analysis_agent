"""Contract tests for the strategy-match system prompt.

Pins the requirements the strategy-match prompt must uphold:
1. Deep-research fallback when stock_analysis data is insufficient.
2. A cap of at most 3 deepresearch calls, and "don't force a conclusion".
3. The new output fields ``data_sources`` and ``judgment_rationale``.
"""
from __future__ import annotations

import re
from pathlib import Path

_PROMPT_FILE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "stock_analysis_agent"
    / "prompts"
    / "strategy_match_system_prompt.md"
)


def _read_prompt() -> str:
    return _PROMPT_FILE.read_text(encoding="utf-8")


def test_prompt_has_frontmatter() -> None:
    text = _read_prompt()
    assert text.startswith("---"), "missing YAML frontmatter fence"
    assert re.search(r"^name:\s*strategy-match-analyst\s*$", text, re.MULTILINE)


def test_requires_deepresearch_fallback_when_data_insufficient() -> None:
    text = _read_prompt()
    assert "run_deepresearch" in text, "missing run_deepresearch fallback"
    assert "证据不足" in text, "missing data-insufficiency trigger"
    assert "编造" in text, "missing anti-fabrication rule"


def test_caps_deepresearch_at_three_calls() -> None:
    text = _read_prompt()
    assert "最多" in text and "3 次" in text, "missing 3-call cap"


def test_declares_data_sources_and_judgment_rationale_fields() -> None:
    text = _read_prompt()
    assert '"data_sources"' in text, "missing data_sources in JSON example"
    assert '"judgment_rationale"' in text, "missing judgment_rationale in JSON example"
