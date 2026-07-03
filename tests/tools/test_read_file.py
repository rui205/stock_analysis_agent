"""Tests for stock_analysis_agent.tools.read_file: the read_file tool.

``read_file`` is the generic counterpart to ``load_skill``: the LLM uses
it to load any UTF-8 file under the project root — most commonly a
skill reference file like
``src/stock_analysis_agent/skill/lark-doc/references/lark-doc-xml.md``.
"""
from __future__ import annotations

import pytest

from stock_analysis_agent.tools.read_file import _read_file, read_file


class TestReadFile:
    """Pure I/O for the underlying _read_file helper."""

    def test_read_skill_md(self) -> None:
        """A project-relative path to a known SKILL.md works."""
        text = _read_file(
            "src/stock_analysis_agent/skill/stock-snapshot-format/SKILL.md"
        )
        assert text.startswith("---")
        assert "stock-snapshot-format" in text
        assert "Procedure" in text

    def test_read_skill_reference(self) -> None:
        """The canonical lark-doc-xml reference loads under the project root."""
        text = _read_file(
            "src/stock_analysis_agent/skill/lark-doc/references/lark-doc-xml.md"
        )
        # Recognizable section from the reference file.
        assert "扩展标签" in text or "标准 HTML" in text

    def test_read_absolute_path_inside_root(self) -> None:
        """Absolute paths that resolve inside the project root are accepted."""
        abs_path = (
            "/Users/rui/workspace/stock_analysis_agent/"
            "src/stock_analysis_agent/skill/lark-doc/SKILL.md"
        )
        text = _read_file(abs_path)
        assert "lark-doc" in text
        assert "lark-cli" in text

    def test_read_rejects_path_traversal(self) -> None:
        """``..`` segments that escape the project root raise ValueError."""
        with pytest.raises(ValueError, match="outside the project root"):
            _read_file("../../../etc/passwd")

    def test_read_rejects_absolute_path_outside_root(self) -> None:
        """Absolute paths outside the project root raise ValueError."""
        with pytest.raises(ValueError, match="outside the project root"):
            _read_file("/etc/passwd")

    def test_read_missing_file_raises(self) -> None:
        """A path that resolves to nothing raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            _read_file("src/stock_analysis_agent/does-not-exist.md")

    def test_read_directory_raises(self) -> None:
        """A path that resolves to a directory raises IsADirectoryError."""
        with pytest.raises(IsADirectoryError, match="is a directory"):
            _read_file("src/stock_analysis_agent/skill/lark-doc")

    def test_read_empty_path_raises(self) -> None:
        """Empty or whitespace paths raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            _read_file("")
        with pytest.raises(ValueError, match="cannot be empty"):
            _read_file("   ")


class TestReadFileTool:
    """The @tool read_file wrapper — its schema and invocation."""

    def test_tool_name_is_read_file(self) -> None:
        assert read_file.name == "read_file"

    def test_tool_args_schema_has_string_path(self) -> None:
        """``path`` is a free-form string — no enum/const restriction."""
        schema = read_file.args
        if hasattr(schema, "model_json_schema"):
            schema = schema.model_json_schema()
        assert "enum" not in schema["path"]
        assert "const" not in schema["path"]
        assert schema["path"]["type"] == "string"

    def test_tool_invoke_returns_file_content(self) -> None:
        """End-to-end: ``read_file.invoke`` reads and returns UTF-8 text."""
        result = read_file.invoke({
            "path": "src/stock_analysis_agent/skill/stock-snapshot-format/SKILL.md",
        })
        assert isinstance(result, str)
        assert "stock-snapshot-format" in result
        assert "Procedure" in result

    def test_tool_invoke_missing_file_raises(self) -> None:
        """Errors raised by the helper surface verbatim to the LLM."""
        with pytest.raises(FileNotFoundError, match="not found"):
            read_file.invoke({"path": "src/missing.md"})