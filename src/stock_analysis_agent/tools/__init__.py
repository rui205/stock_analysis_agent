"""Tools for stock_analysis_agent.

Public API:
    _extract_text — best-effort plain-text extraction from HTML/PDF
    _read_skill   — read a project-level SKILL.md
    _web_search   — cached Google search via the ``googlesearch-python`` package
    load_skill    — @tool wrapper for reading a SKILL.md
    read_file     — @tool wrapper for reading any UTF-8 file under the project root
"""
from __future__ import annotations

from stock_analysis_agent.tools.read_file import _read_file, read_file
from stock_analysis_agent.tools.skill import _read_skill, load_skill
from stock_analysis_agent.tools.text_extractor import _extract_text
from stock_analysis_agent.tools.web_search import _web_search

__all__ = [
    "_extract_text",
    "_read_file",
    "_read_skill",
    "_web_search",
    "load_skill",
    "read_file",
]
