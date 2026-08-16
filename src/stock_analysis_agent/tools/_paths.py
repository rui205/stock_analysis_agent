"""Shared filesystem path constants for the package and project root.

Centralizes the ``Path(__file__).resolve().parents[N]`` arithmetic that
several tools otherwise duplicate with a fragile magic index.
"""
from __future__ import annotations

from pathlib import Path

#: Importable package directory: ``src/stock_analysis_agent/``.
PACKAGE_ROOT: Path = Path(__file__).resolve().parents[1]

#: Repository root (the directory containing ``pyproject.toml``), two
#: levels above the package directory.
PROJECT_ROOT: Path = PACKAGE_ROOT.parent.parent

__all__ = ["PACKAGE_ROOT", "PROJECT_ROOT"]
