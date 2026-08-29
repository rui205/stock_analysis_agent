"""Golden strategy-match regression eval.

These make real LLM calls (~minutes + tokens per case), so they are marked
``slow`` and skipped unless run explicitly with ``pytest --run-slow``.

The golden file (``golden_strategy_match.json``) holds human-verified
conclusions per (symbol, strategy). Each case asserts:
  1. ``overall_fit`` matches the expected verdict.
  2. ``fit_score`` falls in the expected range.
  3. key facts (data-traceability keywords) appear in the report.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from stock_analysis_agent.script import evaluate_strategy as es

pytestmark = pytest.mark.slow

_GOLDEN = Path(__file__).with_name("golden_strategy_match.json")


def _cases() -> list[dict[str, Any]]:
    return json.loads(_GOLDEN.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", _cases(), ids=lambda c: f"{c['strategy']}:{c['symbol']}")
def test_strategy_match_regression(case: dict[str, Any]) -> None:
    """The agent's report must match the human-verified golden conclusion."""
    args = es._build_parser().parse_args([
        case["symbol"],
        "--strategy", case["strategy"],
        "--delivery", "local",
        "--include-shell-tool",
    ])
    report = es._run_agent_and_parse(args)

    expected = case["expected"]
    assert report.overall_fit == expected["overall_fit"], report.summary
    assert (
        expected["fit_score_min"] <= report.fit_score <= expected["fit_score_max"]
    ), f"fit_score={report.fit_score} out of range"

    # Data traceability: key facts must surface somewhere in the report.
    text = json.dumps(report.model_dump(), ensure_ascii=False)
    missing = [kw for kw in expected["must_mention"] if kw not in text]
    assert not missing, f"missing key facts {missing!r} in report"
