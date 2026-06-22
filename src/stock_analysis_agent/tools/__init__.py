"""Tool functions and utilities for stock_analysis_agent.

`text_extractor` — stdlib HTML parser for stripping <script>/<style>.
`web_search`     — @tool that fans out to a configured site list.
"""
from stock_analysis_agent.tools.text_extractor import _extract_text
from stock_analysis_agent.tools.web_search import _web_search

__all__ = ["_extract_text", "_web_search"]