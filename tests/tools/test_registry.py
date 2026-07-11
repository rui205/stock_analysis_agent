"""Tests for stock_analysis_agent.tools.registry — the @tool catalog
injected into the system prompt.

Mirrors the structure of tests/tools/test_skill.py — a small unit
suite around the discovery, introspection, and Markdown-rendering
helpers, plus an end-to-end assertion that every registered tool
appears in the rendered index.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from stock_analysis_agent.tools.registry import (
    _extract_inputs,
    _TOOL_OUTPUTS,
    format_tool_index_markdown,
    get_tool_index,
    list_tools,
)
from stock_analysis_agent.tools.shell import RunCommandInput
from stock_analysis_agent.tools.read_file import ReadFileInput
from stock_analysis_agent.tools.skill import LoadSkillInput


# ---------------------------------------------------------------------------
# list_tools — every self-built @tool is registered
# ---------------------------------------------------------------------------


class TestListTools:
    """Discovery: every self-built @tool callable is surfaced."""

    def test_list_tools_includes_every_self_built_tool(self) -> None:
        """The three bundled @tools are all in the catalog.

        Adding a fourth tool requires (a) appending it to ``list_tools``
        and (b) adding a ``_TOOL_OUTPUTS`` entry — this assertion
        catches the case where someone forgot (a).

        Note: ``get_stock_snapshot`` and ``web_search`` are being prepared
        for deletion and are intentionally absent from the catalog.
        """
        names = {t.name for t in list_tools()}
        expected = {
            "load_skill",
            "read_file",
            "run_command",
        }
        assert expected.issubset(names), (
            f"missing tools: {expected - names}; "
            "update list_tools() in tools/registry.py"
        )
        assert "get_stock_snapshot" not in names, (
            "get_stock_snapshot should be absent from list_tools() — "
            "it is being prepared for deletion"
        )
        assert "web_search" not in names, (
            "web_search should be absent from list_tools() — "
            "it is being prepared for deletion"
        )

    def test_list_tools_is_sorted(self) -> None:
        """Stable alphabetical order — the index is injected into the prompt."""
        names = [t.name for t in list_tools()]
        assert names == sorted(names)
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# _extract_inputs — introspects @tool's args_schema
# ---------------------------------------------------------------------------


class TestExtractInputs:
    """Each tool's input table is built from its Pydantic ``args_schema``."""

    def test_read_file_has_one_required_path_input(self) -> None:
        from stock_analysis_agent.tools.read_file import read_file

        rows = _extract_inputs(read_file)
        assert len(rows) == 1
        assert rows[0]["name"] == "path"
        assert rows[0]["type"] == "str"
        assert rows[0]["required"] is True
        # The description text from the Pydantic Field should be present.
        assert "Project-root-relative" in rows[0]["description"]

    def test_load_skill_has_one_required_name_input(self) -> None:
        from stock_analysis_agent.tools.skill import load_skill

        rows = _extract_inputs(load_skill)
        assert len(rows) == 1
        assert rows[0]["name"] == "name"
        assert rows[0]["type"] == "str"
        assert rows[0]["required"] is True

    def test_run_command_inputs_include_command_argv(self) -> None:
        from stock_analysis_agent.tools.shell import run_command

        rows = _extract_inputs(run_command)
        names = {r["name"] for r in rows}
        assert {"command", "argv", "cwd", "timeout"}.issubset(names)
        # ``command`` and ``argv`` are required.
        required = {r["name"] for r in rows if r["required"]}
        assert {"command", "argv"}.issubset(required)
        # Optional ones are NOT required.
        for opt in ("cwd", "timeout"):
            assert opt not in required, f"{opt!r} should be optional"

    def test_input_types_render_as_short_signatures(self) -> None:
        """Pydantic types are rendered as short Python-style signatures.

        ``list[str]`` not ``array``, ``str | None`` not ``['string', 'null']``.
        """
        from stock_analysis_agent.tools.shell import run_command

        rows = _extract_inputs(run_command)
        types_by_name = {r["name"]: r["type"] for r in rows}
        assert types_by_name["argv"] == "list[str]"
        # ``cwd`` is ``str | None`` — the type renderer must collapse
        # ``anyOf: [{string}, {null}]`` to ``str | None``.
        assert types_by_name["cwd"] == "str | None"


# ---------------------------------------------------------------------------
# Output spec — every registered tool has a hand-curated entry
# ---------------------------------------------------------------------------


class TestOutputSpecs:
    """Every registered tool has a matching ``_TOOL_OUTPUTS`` entry."""

    @pytest.mark.parametrize("tool_name", [t.name for t in list_tools()])
    def test_tool_has_output_spec(self, tool_name: str) -> None:
        """No registered tool is missing its return-shape description."""
        assert tool_name in _TOOL_OUTPUTS, (
            f"tool {tool_name!r} has no _TOOL_OUTPUTS entry; "
            "add one in tools/registry.py"
        )
        spec = _TOOL_OUTPUTS[tool_name]
        assert spec["output"], f"empty output spec for {tool_name!r}"


# ---------------------------------------------------------------------------
# get_tool_index — full catalog assembly
# ---------------------------------------------------------------------------


class TestGetToolIndex:
    """The assembled catalog matches the registered tool list."""

    def test_index_is_alphabetical(self) -> None:
        names = [e["name"] for e in get_tool_index()]
        assert names == sorted(names)

    def test_each_entry_has_name_description_inputs_output(self) -> None:
        for entry in get_tool_index():
            assert isinstance(entry, dict)
            assert entry["name"]
            assert entry["description"], f"empty description for {entry['name']!r}"
            assert isinstance(entry["inputs"], list)
            assert entry["output"], f"empty output for {entry['name']!r}"

    def test_index_covers_every_listed_tool(self) -> None:
        """Catalog and discovery list stay in sync."""
        listed = {t.name for t in list_tools()}
        indexed = {e["name"] for e in get_tool_index()}
        assert listed == indexed

    def test_known_tools_have_nonempty_input_descriptions(self) -> None:
        """Field-level descriptions are present so the rendered table is useful."""
        for entry in get_tool_index():
            for inp in entry["inputs"]:
                assert inp["description"], (
                    f"empty description for input {inp['name']!r} "
                    f"of tool {entry['name']!r}"
                )


# ---------------------------------------------------------------------------
# format_tool_index_markdown — the rendered Markdown
# ---------------------------------------------------------------------------


class TestFormatToolIndexMarkdown:
    """The Markdown renderer mirrors the structured catalog."""

    def test_returns_empty_placeholder_when_no_tools(self) -> None:
        md = format_tool_index_markdown([])
        assert "no tools" in md.lower()

    def test_renders_one_section_per_tool(self) -> None:
        md = format_tool_index_markdown(get_tool_index())
        for entry in get_tool_index():
            assert f"### `{entry['name']}`" in md, (
                f"missing `### {entry['name']}` heading in rendered markdown"
            )

    def test_renders_inputs_table_with_required_column(self) -> None:
        md = format_tool_index_markdown(get_tool_index())
        # Common header.
        assert "| name | type | required | description |" in md

    def test_renders_run_command_table_with_all_four_params(self) -> None:
        md = format_tool_index_markdown(get_tool_index())
        for param in ("command", "argv", "cwd", "timeout"):
            assert f"| `{param}` |" in md, (
                f"missing `{param}` row in rendered run_command inputs table"
            )

    def test_renders_output_block_for_every_tool(self) -> None:
        md = format_tool_index_markdown(get_tool_index())
        for entry in get_tool_index():
            # Every tool gets a `**output**:` paragraph.
            assert "**output**:" in md
            # And the output description's first sentence / token survives.
            first_token = entry["output"].split(maxsplit=1)[0]
            assert first_token in md

    def test_rendered_markdown_uses_backtick_tool_name(self) -> None:
        """Convention: tool names wrapped in backticks."""
        md = format_tool_index_markdown(get_tool_index())
        assert "`read_file`" in md
        assert "`run_command`" in md
        assert "`load_skill`" in md

    def test_full_index_render_is_well_formed(self) -> None:
        """End-to-end: real catalog renders without raising."""
        md = format_tool_index_markdown(get_tool_index())
        assert isinstance(md, str)
        # Every section heading present.
        for entry in get_tool_index():
            assert entry["name"] in md


# ---------------------------------------------------------------------------
# Smoke tests — every Input schema is a Pydantic BaseModel with Field
# descriptions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_cls",
    [ReadFileInput, RunCommandInput, LoadSkillInput],
)
def test_input_schema_is_pydantic_basemodel_with_descriptions(input_cls: type[BaseModel]) -> None:
    """Each ``args_schema`` is a Pydantic model with Field descriptions.

    LangChain introspects the model for both type info and
    description text — the descriptions surface in the LLM's tool
    picker. Empty descriptions would silently degrade the LLM's
    ability to pick the right tool.
    """
    assert issubclass(input_cls, BaseModel)
    for name, field in input_cls.model_fields.items():
        assert field.description, (
            f"{input_cls.__name__}.{name} is missing a Field description; "
            "LangChain exposes this text to the LLM"
        )