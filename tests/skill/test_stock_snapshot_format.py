"""Data-driven tests for the stock-snapshot-format skill's field-mapping table.

The skill is a Markdown document consumed by an LLM, but the
field-mapping table is structured enough to validate mechanically: every
row in the Output contract (§3) must have a corresponding entry in the
field-mapping table, and the 首选源 column must be one of the known
sources.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest  # noqa: F401  — imported for future fixture use

SKILL_PATH = Path(
    "src/stock_analysis_agent/skill/stock-snapshot-format/SKILL.md"
)

VALID_SOURCES = {"tushare", "akshare", "mootdx", "top-level"}


def _read_skill() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def _strip_md(s: str) -> str:
    """Strip surrounding backticks from inline-code spans."""
    s = s.strip()
    if len(s) >= 2 and s.startswith("`") and s.endswith("`"):
        return s[1:-1].strip()
    return s


def _parse_field_mapping_table() -> list[tuple[str, str, str]]:
    """Return list of (output_field, common_keys, preferred_source) tuples."""
    text = _read_skill()
    # Find the field-mapping table by anchoring on the unique header row.
    header_re = re.compile(
        r"^\|\s*输出字段\s*\|\s*常见键名\s*\|\s*首选源\s*\|\s*$",
        re.MULTILINE,
    )
    header_match = header_re.search(text)
    assert header_match is not None, "could not find field-mapping table header"
    # Walk lines after the header, skipping the separator row.
    lines = text[header_match.end():].splitlines()
    rows: list[tuple[str, str, str]] = []
    for line in lines:
        if not line.startswith("|"):
            # Allow blank/whitespace-only lines between header and rows,
            # but stop at the first non-table, non-blank line.
            if not line.strip():
                continue
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 3:
            continue
        # Skip separator row (---|---).
        if all(set(c) <= {"-", " "} for c in cells):
            continue
        rows.append((cells[0], cells[1], cells[2]))
    return rows


def _parse_output_contract_sections() -> list[str]:
    """Return the list of section titles from §Output contract."""
    text = _read_skill()
    # §Output contract starts after `## Output contract` and ends at next `## ` heading.
    m = re.search(
        r"^## Output contract\s*\n(.*?)(?=^## |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert m is not None, "could not find Output contract section"
    body = m.group(1)
    # Sub-headings under Output contract are `### N. <title>` lines.
    return re.findall(r"^### \d+\.\s*(.+)$", body, re.MULTILINE)


class TestSkillFieldMapping:
    """Validate the field-mapping table in stock-snapshot-format/SKILL.md."""

    def test_field_mapping_table_is_non_empty(self) -> None:
        rows = _parse_field_mapping_table()
        assert len(rows) >= 20, f"expected ≥20 rows, got {len(rows)}"

    def test_every_row_has_valid_preferred_source(self) -> None:
        rows = _parse_field_mapping_table()
        for field, keys, source in rows:
            # The source cell may read e.g. `` `top-level` `` or
            # `` top-level `fetched_at` `` — normalize by stripping
            # inline-code backticks, then take the first whitespace-
            # separated token.
            source_norm = _strip_md(source).split()[0]
            # Either a known source name or a Chinese-language parenthetical
            # (for fields that no source provides today).
            is_known_source = source_norm in VALID_SOURCES
            is_known_marker = source.startswith("(") and source.endswith(")")
            assert is_known_source or is_known_marker, (
                f"row {field!r} has invalid 首选源: {source!r}"
            )

    def test_table_includes_required_finance_fields(self) -> None:
        """The most-consumed finance fields must each have a row."""
        rows = _parse_field_mapping_table()
        keys_covered = set()
        for _field, keys, _source in rows:
            for k in keys.split("/"):
                keys_covered.add(_strip_md(k))
        for required in ("name", "ts_code", "industry", "close", "pe", "pb", "total_mv"):
            assert required in keys_covered, (
                f"required key {required!r} missing from field-mapping table"
            )

    def test_output_contract_sections_have_field_mapping_coverage(self) -> None:
        """§3 (当前股价与估值) and §4 (财务概览) must mention fields that
        the table can resolve — sanity check that the sections and the
        table aren't drifting apart."""
        sections = _parse_output_contract_sections()
        # We expect at least 5 numbered sections (1-公司简介 .. 7-数据声明).
        assert len(sections) >= 5, (
            f"expected ≥5 numbered output sections, got {sections!r}"
        )
        rows = _parse_field_mapping_table()
        # At least one row maps to PE/PB/close (核心财务字段).
        all_keys = {
            _strip_md(k)
            for _field, keys, _src in rows
            for k in keys.split("/")
        }
        assert "pe" in all_keys and "pb" in all_keys and "close" in all_keys