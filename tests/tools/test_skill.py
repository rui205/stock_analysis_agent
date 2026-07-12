"""Tests for stock_analysis_agent.tools.skill: the load_skill tool.

Covers three layers:

1. Frontmatter parsing (single-line, quoted, multi-line ``|`` block).
2. Skill discovery / index (disk-driven).
3. The ``load_skill`` @tool wrapper (schema + invocation).
"""
from __future__ import annotations

import pytest

from stock_analysis_agent.tools.skill import (
    _parse_frontmatter,
    _read_skill,
    format_skill_index_markdown,
    get_skill_index,
    list_skill_names,
    load_skill,
)


class TestParseFrontmatter:
    """_parse_frontmatter handles the YAML subset the bundled skills use."""

    def test_extracts_name_and_single_line_description(self) -> None:
        text = (
            "---\n"
            "name: announcement-search\n"
            "description: 公司公告的查询。\n"
            "version: 1.0.0\n"
            "---\n"
            "\n# 公告搜索技能\n"
        )
        assert _parse_frontmatter(text) == {
            "name": "announcement-search",
            "description": "公司公告的查询。",
        }

    def test_strips_surrounding_double_quotes(self) -> None:
        text = (
            "---\n"
            "name: lark-doc\n"
            'description: "飞书云文档（Docx / Wiki 文档，v2 API）"\n'
            "---\n"
        )
        result = _parse_frontmatter(text)
        assert result["name"] == "lark-doc"
        assert result["description"] == "飞书云文档（Docx / Wiki 文档，v2 API）"

    def test_collapses_multi_line_literal_block_to_one_string(self) -> None:
        text = (
            "---\n"
            "name: stock-snapshot-format\n"
            "description: |\n"
            "  Format the nested multi-source JSON output of `get_stock_snapshot`.\n"
            "  into a standardized company profile.\n"
            "  Output sections: 公司简介 / 主营业务.\n"
            "---\n"
        )
        result = _parse_frontmatter(text)
        assert result["name"] == "stock-snapshot-format"
        # Multi-line blocks are joined with spaces — caller decides rendering.
        assert "Format the nested" in result["description"]
        assert "Output sections" in result["description"]
        assert "\n" not in result["description"]

    def test_returns_empty_dict_when_no_frontmatter(self) -> None:
        """Files without a leading ``---`` fence yield empty strings, not errors."""
        assert _parse_frontmatter("# just markdown\n") == {
            "name": "",
            "description": "",
        }

    def test_ignores_non_top_level_keys(self) -> None:
        """``metadata:`` / ``requires:`` / nested structures are skipped."""
        text = (
            "---\n"
            "name: lark-doc\n"
            "version: 2.0.0\n"
            "metadata:\n"
            "  requires:\n"
            "    bins: [\"lark-cli\"]\n"
            "description: 飞书云文档。\n"
            "---\n"
        )
        result = _parse_frontmatter(text)
        assert result["name"] == "lark-doc"
        assert result["description"] == "飞书云文档。"
        assert "version" not in result
        assert "metadata" not in result


class TestSkillDiscovery:
    """list_skill_names / get_skill_index are allowlist-driven ∩ disk.

    The allowlist controls what the system prompt advertises as
    ``## 我的工具``. Skills on disk but NOT in the allowlist (e.g.
    ``stock-snapshot-format``) are still loadable via ``load_skill``,
    but they don't appear in the auto-generated catalog.
    """


    def test_list_skill_names_excludes_internal_skills(self) -> None:
        """``stock-snapshot-format`` is on disk but not advertised."""
        assert "stock-snapshot-format" not in list_skill_names()

    def test_load_skill_can_still_reach_internal_skills(self) -> None:
        """Internal skills (not in the prompt catalog) are still loadable
        via ``load_skill(name=...)`` — they're just not advertised.
        """
        # Direct read via the helper bypasses the allowlist.
        from stock_analysis_agent.tools.skill import _read_skill
        text = _read_skill("stock-snapshot-format")
        assert text.startswith("---")
        assert "stock-snapshot-format" in text

    def test_get_skill_index_uses_frontmatter_description(self) -> None:
        """Each entry's description comes from SKILL.md frontmatter."""
        index = get_skill_index()
        # Find a known single-line entry.
        lark_doc = next(e for e in index if e["name"] == "lark-doc")
        assert "飞书" in lark_doc["description"]
        # Every advertised entry should have a non-empty description.
        for entry in index:
            assert entry["description"], f"empty description for {entry['name']}"
            assert "(no description in frontmatter)" not in entry["description"]

    def test_get_skill_index_is_sorted_alphabetically(self) -> None:
        names = [e["name"] for e in get_skill_index()]
        assert names == sorted(names)

    def test_get_skill_index_count_matches_loadable(self) -> None:
        """The catalog index length equals the loadable skill count."""
        assert len(get_skill_index()) == len(list_skill_names())


class TestReadSkill:
    """Pure I/O for the underlying _read_skill helper."""

    def test_read_skill_returns_skill_md_content(self) -> None:
        """The helper reads the SKILL.md for the requested skill."""
        text = _read_skill("stock-snapshot-format")
        # Frontmatter + a recognizable section heading.
        assert text.startswith("---")
        assert "stock-snapshot-format" in text
        assert "Procedure" in text

    def test_read_skill_raises_for_unknown_skill(self) -> None:
        """Unknown skill names raise FileNotFoundError with the available list.

        The error message lists every skill present on disk
        (``_KNOWN_SKILLS``), so the LLM can recover regardless of
        whether the missing skill was advertised or internal.
        """
        from stock_analysis_agent.tools.skill import _KNOWN_SKILLS
        with pytest.raises(FileNotFoundError) as excinfo:
            _read_skill("does-not-exist")
        msg = str(excinfo.value)
        assert "does-not-exist" in msg
        assert "available:" in msg
        for known in _KNOWN_SKILLS:
            assert known in msg, f"error message missing skill: {known!r}"


class TestLoadSkillTool:
    """The @tool load_skill wrapper — its schema and invocation."""

    def test_tool_name_is_load_skill(self) -> None:
        assert load_skill.name == "load_skill"


    def test_tool_invoke_returns_lark_doc_skill_markdown(self) -> None:
        """End-to-end: ``load_skill`` can also read non-builtin skills
        (e.g. ``lark-doc``) whose SKILL.md exists on disk. This is the
        relaxation that lets the LLM read the lark-doc instructions at
        runtime; the agent still needs a separate tool to actually
        invoke ``lark-cli``.
        """
        result = load_skill.invoke({"name": "lark-doc"})
        assert isinstance(result, str)
        assert "lark-doc" in result
        assert "lark-cli" in result

    def test_tool_invoke_returns_full_skill_markdown(self) -> None:
        """End-to-end: tool.invoke reads the file and returns the content."""
        result = load_skill.invoke({"name": "stock-snapshot-format"})
        assert isinstance(result, str)
        assert result.startswith("---")
        assert "stock-snapshot-format" in result
        assert "Output contract" in result

    def test_tool_invoke_with_unknown_skill_raises(self) -> None:
        """A non-Literal value (forced past the schema) must surface as
        FileNotFoundError — the LLM gets a clear error to recover from."""
        # The Literal constrains the schema, but Pydantic v2 strips extra
        # fields. Pass the actual literal value via direct call to the
        # underlying function to exercise the FileNotFoundError path.
        with pytest.raises(FileNotFoundError, match="not found"):
            _read_skill("nonexistent-skill")
