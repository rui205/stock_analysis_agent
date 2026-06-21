"""Tests for stock_analysis_agent.agents.deepsearch.DeepSearchAgent."""
from __future__ import annotations

import pytest

from stock_analysis_agent.agents.deepsearch import _extract_text


def test_extract_text_strips_script_and_style() -> None:
    """<script> and <style> blocks must be removed entirely."""
    html = "<script>alert(1)</script><p>hello</p><style>p{}</style>"
    assert _extract_text(html) == "hello"


def test_extract_text_folds_whitespace() -> None:
    """Runs of whitespace (newlines, tabs, multiple spaces) collapse to single space."""
    html = "<p>hello   world</p>\n<p>foo\tbar</p>"
    assert _extract_text(html) == "hello world foo bar"


def test_extract_text_empty_input() -> None:
    """Empty HTML returns empty string."""
    assert _extract_text("") == ""


def test_extract_text_preserves_text_outside_tags() -> None:
    """Plain text between tags is preserved."""
    html = "before <b>middle</b> after"
    assert _extract_text(html) == "before middle after"