"""Shared prompt-rendering helpers for the CLI scripts.

Both ``script.analyze_stock`` and ``script.evaluate_strategy`` render a
system prompt from a Markdown template with a catalog placeholder
(``<!-- SKILL_INDEX -->`` / ``<!-- STRATEGY_INDEX -->``) and a
``<!-- TOOL_INDEX -->`` placeholder. The tool catalog is filtered to each
agent's *actual* tool set, so the two scripts share the filtering and
substitution logic here.
"""
from __future__ import annotations

from pathlib import Path

from stock_analysis_agent.tools.registry import (
    format_tool_index_markdown,
    get_tool_index,
)


def resolve_tool_names(base: list[str], include_shell_tool: bool) -> list[str]:
    """Return the sorted, deduplicated tool names for ``base`` plus optional shell.

    Args:
        base: Default tool names for the agent.
        include_shell_tool: When ``True``, also advertise ``run_command``.

    Returns:
        Sorted, deduplicated tool-name list matching the agent's wired tools.
    """
    names = list(base)
    if include_shell_tool:
        names.append("run_command")
    return sorted(set(names))


def render_system_prompt(
    template_path: Path,
    *,
    tool_names: list[str],
    catalog_placeholder: str,
    catalog_doc: str,
) -> str:
    """Render a prompt template by substituting the catalog and tool index.

    Args:
        template_path: Path to the ``.md`` prompt template.
        tool_names: Tool names to advertise in ``<!-- TOOL_INDEX -->``.
        catalog_placeholder: Placeholder for the catalog section
            (``"<!-- SKILL_INDEX -->"`` or ``"<!-- STRATEGY_INDEX -->"``).
        catalog_doc: Rendered catalog text to substitute for
            ``catalog_placeholder``.

    Returns:
        The template with ``catalog_placeholder`` and ``<!-- TOOL_INDEX -->``
        replaced by their rendered content.

    Raises:
        FileNotFoundError: If ``template_path`` is missing.
    """
    template = template_path.read_text(encoding="utf-8")
    tool_doc = format_tool_index_markdown(get_tool_index(names=tool_names))
    return (
        template
        .replace(catalog_placeholder, catalog_doc)
        .replace("<!-- TOOL_INDEX -->", tool_doc)
    )


__all__ = ["render_system_prompt", "resolve_tool_names"]
