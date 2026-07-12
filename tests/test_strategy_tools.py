"""Tests for stock_analysis_agent.tools.strategy.

Covers the static parts only — ``_list_strategy_names``,
``_parse_strategy_frontmatter``, and ``load_strategy``. The dynamic
``run_analyze_stock`` tool (which embeds a subagent) is tested in
Task 4 with ``monkeypatch``.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from stock_analysis_agent.tools.strategy import (
    _list_strategy_names,
    _parse_strategy_frontmatter,
    load_strategy,
)


class TestListStrategyNames:
    def test_returns_alphabetical_stems(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "zeta.md").write_text("---\nname: z\n---\n", encoding="utf-8")
        (tmp_path / "alpha.md").write_text("---\nname: a\n---\n", encoding="utf-8")
        (tmp_path / "ignore.txt").write_text("not markdown", encoding="utf-8")
        monkeypatch.setattr(
            "stock_analysis_agent.tools.strategy._STRATEGIES_DIR", tmp_path
        )
        assert _list_strategy_names() == ("alpha", "zeta")

    def test_empty_dir_returns_empty_tuple(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "stock_analysis_agent.tools.strategy._STRATEGIES_DIR", tmp_path
        )
        assert _list_strategy_names() == ()


class TestParseStrategyFrontmatter:
    def test_single_line_fields(self) -> None:
        text = textwrap.dedent("""\
            ---
            name: foo
            version: "2"
            description: short description
            ---

            body
        """)
        fm = _parse_strategy_frontmatter(text)
        assert fm == {"name": "foo", "version": "2", "description": "short description"}

    def test_multiline_description_block(self) -> None:
        text = textwrap.dedent("""\
            ---
            name: foo
            description: |
              first line
              second line
            ---

            body
        """)
        fm = _parse_strategy_frontmatter(text)
        assert fm["name"] == "foo"
        assert "first line second line" in fm["description"]

    def test_missing_frontmatter_returns_empty(self) -> None:
        assert _parse_strategy_frontmatter("no fence here\n") == {}

    def test_unknown_keys_are_ignored(self) -> None:
        text = "---\nname: x\nfoo: bar\n---\nbody\n"
        fm = _parse_strategy_frontmatter(text)
        assert fm == {"name": "x"}


class TestLoadStrategyTool:
    def test_returns_full_file_text(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "value.md").write_text(
            "---\nname: value\n---\n\n# Value\n", encoding="utf-8"
        )
        monkeypatch.setattr(
            "stock_analysis_agent.tools.strategy._STRATEGIES_DIR", tmp_path
        )
        out = load_strategy.invoke({"name": "value"})
        assert "name: value" in out
        assert "# Value" in out

    def test_unknown_name_raises_with_available_list(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "alpha.md").write_text("---\nname: alpha\n---\n", encoding="utf-8")
        monkeypatch.setattr(
            "stock_analysis_agent.tools.strategy._STRATEGIES_DIR", tmp_path
        )
        with pytest.raises(FileNotFoundError) as exc:
            load_strategy.invoke({"name": "missing"})
        assert "missing" in str(exc.value)
        assert "alpha" in str(exc.value)
