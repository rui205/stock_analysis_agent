"""Long-term memory and state persistence for stock_analysis_agent.

Currently exposes _FileCache, a JSON-file cache keyed by (query, site).
"""
from stock_analysis_agent.memory.file_cache import _FileCache

__all__ = ["_FileCache"]