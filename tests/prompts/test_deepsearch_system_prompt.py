"""Contract tests for the deepsearch system prompt.

Pins the requirements the deepsearch agent prompt must uphold:

1. Required inputs (stock code + research dimensions).
2. The "think first" workflow — at least 3 concrete questions per dimension.
3. Evidence chain + confidence on every conclusion.
4. ``unknown`` on unsearchable questions (no fabrication).
5. The declared tool/skill surface (mx-* skills + web_search).
"""
from __future__ import annotations

import re
from pathlib import Path

_PROMPT_FILE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "stock_analysis_agent"
    / "prompts"
    / "deepsearch_system_prompt.md"
)


def _read_prompt() -> str:
    """Read the raw prompt template (no render function exists for deepsearch)."""
    return _PROMPT_FILE.read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """Return the body of a ``## <heading>`` section, stopping at the next ``## ``."""
    pattern = re.compile(
        rf"^##\s+{re.escape(heading)}\s*$\n(.*?)(?=^##\s+|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    return match.group(1) if match else ""


def test_prompt_exists_and_has_frontmatter() -> None:
    """The prompt file exists and declares a ``name``/``description`` in frontmatter."""
    text = _read_prompt()
    assert text.startswith("---"), "missing YAML frontmatter fence"
    assert re.search(r"^name:\s*deepsearch\s*$", text, re.MULTILINE), "frontmatter name must be deepsearch"
    assert re.search(r"^description:\s*\S+", text, re.MULTILINE), "frontmatter description must be non-empty"


def test_declares_required_inputs() -> None:
    """The prompt must require both a stock code and research dimensions."""
    text = _read_prompt()
    for needle in ("股票代码", "研究维度", "缺一不可"):
        assert needle in text, f"required-input contract missing: {needle}"


def test_requires_at_least_three_questions_per_dimension() -> None:
    """The '先想清楚' workflow must demand ≥3 concrete questions per dimension."""
    text = _read_prompt()
    assert "至少 3 个具体" in text or "至少 3 个" in text, (
        "prompt must require at least 3 concrete questions per dimension"
    )
    assert "阶段 0" in text, "missing the 'think first' phase"


def test_requires_evidence_chain_and_confidence() -> None:
    """Every conclusion must carry an evidence chain and a confidence level."""
    text = _read_prompt()
    assert "证据链" in text, "missing evidence-chain requirement"
    assert "置信度" in text, "missing confidence requirement"
    for level in ("high", "medium", "low"):
        assert level in text, f"confidence level missing: {level}"


def test_requires_unknown_when_not_found() -> None:
    """Unsearchable questions must yield ``unknown``, never fabricated content."""
    text = _read_prompt()
    assert "unknown" in text, "missing 'unknown' handling"
    assert "不要编造" in text, "missing anti-fabrication instruction"


def test_declares_skill_and_tool_surface() -> None:
    """The prompt must advertise the 4 mx-* skills and the web_search tool."""
    text = _read_prompt()
    for name in ("mx-finance-data", "mx-finance-search", "mx-macro-data", "mx-stocks-screener", "web_search"):
        assert name in text, f"tool/skill missing from prompt: {name}"
