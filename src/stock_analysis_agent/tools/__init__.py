"""Tools for stock_analysis_agent.

Public API:
    _read_skill           — read a project-level SKILL.md
    _web_search           — @tool wrapper for cached web search
    format_skill_index_markdown — render the project-skill catalog as Markdown
    format_tool_index_markdown  — render the @tool catalog as Markdown
    get_skill_index       — catalog of every project-level skill (used in system prompt)
    get_tool_index        — catalog of every self-built @tool (used in system prompt)
    list_skill_names      — alphabetical list of bundled skill directories
    list_tools            — alphabetical list of @tool objects
    load_skill            — @tool wrapper for reading a SKILL.md
    read_file             — @tool wrapper for reading any UTF-8 file under the project root
    run_command           — @tool wrapper for running CLI subprocesses
    load_strategy         — @tool wrapper for reading a strategy .md from conf/strategies/
    run_analyze_stock     — @tool wrapper that runs the StockAnalysisAgent subagent and returns its Markdown report verbatim
    run_deepresearch      — @tool wrapper that runs the DeepResearchAgent subagent and returns its Markdown report verbatim
    run_technical_capital — @tool wrapper that runs the technical + capital-flow subagent and returns its Markdown report verbatim
    _list_strategy_names  — alphabetical list of strategy file stems under conf/strategies/
    _parse_strategy_frontmatter — extract simple key: value pairs from a strategy frontmatter block
    LoadStrategyInput     — input schema for load_strategy
    RunAnalyzeStockInput  — input schema for run_analyze_stock
    RunDeepResearchInput  — input schema for run_deepresearch
    RunTechnicalCapitalInput — input schema for run_technical_capital
"""
from __future__ import annotations

from stock_analysis_agent.tools.read_file import _read_file, read_file
from stock_analysis_agent.tools.registry import (
    ToolIndexEntry,
    ToolParamSpec,
    format_tool_index_markdown,
    get_tool_index,
    list_tools,
)
from stock_analysis_agent.tools.shell import run_command
from stock_analysis_agent.tools.skill import (
    SkillIndexEntry,
    _read_skill,
    _read_skill_index_entry,
    format_skill_index_markdown,
    get_skill_index,
    list_skill_names,
    load_skill,
)
from stock_analysis_agent.tools.web_search import _web_search

# ``stock_analysis_agent.tools.strategy`` depends on
# :class:`StockAnalysisAgent`, which transitively imports
# ``stock_analysis_agent.tools.web_search``. Importing it eagerly
# here would re-enter ``stock_analysis_agent.tools.__init__`` during
# the package-level import chain
# (``stock_analysis_agent`` -> ``agent`` -> ``deepresearch`` ->
# ``tools.web_search`` evaluates ``tools.__init__`` -> ``strategy``
# -> ``agent.stock_analysis`` -> ``agent.deepresearch`` (partial) =
# ``ImportError``). Resolve the names on demand via PEP 562.
_STRATEGY_LAZY_NAMES: frozenset[str] = frozenset({
    "LoadStrategyInput",
    "RunAnalyzeStockInput",
    "RunDeepResearchInput",
    "RunTechnicalCapitalInput",
    "_list_strategy_names",
    "_parse_strategy_frontmatter",
    "load_strategy",
    "run_analyze_stock",
    "run_deepresearch",
    "run_technical_capital",
})

__all__ = [
    "SkillIndexEntry",
    "ToolIndexEntry",
    "ToolParamSpec",
    "_read_file",
    "_read_skill",
    "_read_skill_index_entry",
    "_web_search",
    "format_skill_index_markdown",
    "format_tool_index_markdown",
    "get_skill_index",
    "get_tool_index",
    "list_skill_names",
    "list_tools",
    "load_skill",
    "read_file",
    "run_command",
    "LoadStrategyInput",
    "RunAnalyzeStockInput",
    "RunDeepResearchInput",
    "RunTechnicalCapitalInput",
    "_list_strategy_names",
    "_parse_strategy_frontmatter",
    "load_strategy",
    "run_analyze_stock",
    "run_deepresearch",
    "run_technical_capital",
]

def __getattr__(name: str) -> object:
    """Lazily resolve ``stock_analysis_agent.tools.strategy`` names.

    See the comment near ``_STRATEGY_LAZY_NAMES`` for why the
    eager import is not used.
    """
    if name in _STRATEGY_LAZY_NAMES:
        from stock_analysis_agent.tools import strategy as _strategy

        value = getattr(_strategy, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
