"""Tools for stock_analysis_agent.

Public API:
    FeishuCli      — subprocess wrapper around the external ``lark-cli`` binary
                     (Lark/Feishu CLI; class name uses the product brand)
    FeishuDocRef   — frozen reference to a Feishu document
    FeishuCliError — raised when the lark-cli call fails
"""
from __future__ import annotations

from stock_analysis_agent.tools.feishu_cli import FeishuCli, FeishuCliError, FeishuDocRef
from stock_analysis_agent.tools.text_extractor import _extract_text
from stock_analysis_agent.tools.web_search import _web_search

__all__ = [
    "FeishuCli",
    "FeishuCliError",
    "FeishuDocRef",
    "_extract_text",
    "_web_search",
]
