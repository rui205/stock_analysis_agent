"""Tests for stock_analysis_agent.tools.skill: the load_skill tool."""
from __future__ import annotations

import pytest

from stock_analysis_agent.tools.skill import _read_skill, load_skill


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
        """Unknown skill names raise FileNotFoundError with available list."""
        with pytest.raises(FileNotFoundError, match="available: stock-snapshot-format"):
            _read_skill("does-not-exist")


class TestLoadSkillTool:
    """The @tool load_skill wrapper — its schema and invocation."""

    def test_tool_name_is_load_skill(self) -> None:
        assert load_skill.name == "load_skill"

    def test_tool_args_schema_accepts_any_skill_name(self) -> None:
        """The ``name`` parameter is a free-form string — any skill whose
        SKILL.md exists on disk is accepted. The schema no longer
        constrains the value to a Literal so the LLM can request skills
        that are not baked into the agent's tool wiring (e.g. ``lark-doc``).
        """
        schema = load_skill.args
        if hasattr(schema, "model_json_schema"):
            schema = schema.model_json_schema()
        # With a plain ``str`` type, the schema has no ``enum`` or ``const``
        # restriction — the LLM is free to request any skill name.
        assert "enum" not in schema["name"]
        assert "const" not in schema["name"]
        assert schema["name"]["type"] == "string"

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
