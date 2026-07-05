"""Tools for stock_analysis_agent.

Public API:
    _extract_text         — best-effort plain-text extraction from HTML/PDF
    _read_skill           — read a project-level SKILL.md
    _web_search           — @tool wrapper for cached web search
    format_tool_index_markdown — render the @tool catalog as Markdown
    get_tool_index        — catalog of every self-built @tool (used in system prompt)
    list_tools            — alphabetical list of @tool objects
    load_skill            — @tool wrapper for reading a SKILL.md
    read_file             — @tool wrapper for reading any UTF-8 file under the project root
    run_command           — @tool wrapper for running CLI subprocesses
"""
from __future__ import annotations

from stock_analysis_agent.tools.read_file import _read_file, read_file
from stock_analysis_agent.tools.registry import (
    ToolIndexEntry,
    ToolOutputSpec,
    ToolParamSpec,
    format_tool_index_markdown,
    get_tool_index,
    list_tools,
)
from stock_analysis_agent.tools.shell import run_command
from stock_analysis_agent.tools.skill import _read_skill, load_skill
from stock_analysis_agent.tools.text_extractor import _extract_text
from stock_analysis_agent.tools.web_search import _web_search

__all__ = [
    "ToolIndexEntry",
    "ToolOutputSpec",
    "ToolParamSpec",
    "_extract_text",
    "_read_file",
    "_read_skill",
    "_web_search",
    "format_tool_index_markdown",
    "get_tool_index",
    "list_tools",
    "load_skill",
    "read_file",
    "run_command",
]