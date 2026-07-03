"""File reading tool: read a UTF-8 file under the project root.

The agent can call :func:`read_file` to load any text file inside the
project tree (skill reference files, configs, source modules, …). Paths
are resolved relative to the project root; absolute paths and ``..``
segments that escape the project root are rejected.

This is the generic counterpart to ``load_skill``: ``load_skill`` is
for project-level ``SKILL.md`` files (and structured skill discovery),
while ``read_file`` is the escape hatch for any other file the LLM has
been told to consult.
"""
from __future__ import annotations

from pathlib import Path

from langchain.tools import tool

# Module-level constants — resolved once at import time.
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # src/<package>/
_PROJECT_ROOT = _PACKAGE_ROOT.parent.parent           # repo root (pyproject.toml)


def _read_file(path: str) -> str:
    """Read ``path`` as UTF-8 text, refusing to escape the project root.

    Args:
        path: Path to the file, relative to the project root (e.g.
            ``"src/stock_analysis_agent/skill/lark-doc/SKILL.md"``).
            Absolute paths that resolve inside the project root are
            also accepted.

    Returns:
        The file's content as a UTF-8 string.

    Raises:
        FileNotFoundError: If the resolved file does not exist.
        IsADirectoryError: If the resolved path is a directory.
        ValueError: If the resolved path escapes the project root
            (path-traversal guard).
    """
    if not path or not path.strip():
        raise ValueError("path cannot be empty")

    target = (_PROJECT_ROOT / path).resolve()
    if not target.is_relative_to(_PROJECT_ROOT):
        raise ValueError(
            f"path {path!r} resolves to {target}, "
            f"which is outside the project root {_PROJECT_ROOT}"
        )
    if target.is_dir():
        raise IsADirectoryError(f"{path!r} is a directory, not a file")
    if not target.is_file():
        raise FileNotFoundError(f"file not found: {path!r}")

    return target.read_text(encoding="utf-8")


@tool("read_file")
def read_file(path: str) -> str:
    """Read a UTF-8 file under the project root and return its content.

    Use this tool to load text files the agent has been told to
    consult — typically skill reference files (e.g.
    ``src/stock_analysis_agent/skill/lark-doc/references/lark-doc-xml.md``)
    or other project files. Paths are resolved relative to the project
    root; absolute paths and ``..`` segments that escape the project
    root are rejected with ``ValueError``.

    Args:
        path: Path to the file, relative to the project root. Example:
            ``"src/stock_analysis_agent/skill/lark-doc/SKILL.md"``.

    Returns:
        The file's content as a UTF-8 string.

    Raises:
        ValueError: ``path`` is empty or escapes the project root.
        IsADirectoryError: The resolved path is a directory.
        FileNotFoundError: The file does not exist.
    """
    return _read_file(path)


__all__ = ["read_file", "_read_file"]