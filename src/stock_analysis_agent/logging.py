"""Structured logging via loguru.

Call `setup_logging()` exactly once, from the CLI entry point, before any
other code runs. Never configure logging at module import time — importing
this package from a notebook or test must not have side effects.
"""

from __future__ import annotations

import sys

from loguru import logger

from .config import LOG_LEVEL


def setup_logging() -> None:
    """Configure the loguru default sink to emit JSON to stdout.

    The JSON format keeps logs machine-parseable for downstream pipelines
    while remaining human-readable via `loguru` itself during development.
    """
    logger.remove()
    logger.add(
        sys.stdout,
        level=LOG_LEVEL,
        serialize=True,
        backtrace=False,
        diagnose=False,
        enqueue=False,
    )


__all__ = ["logger", "setup_logging"]
