"""Eval-test helpers: skip expensive end-to-end tests unless ``--run-slow``.

End-to-end strategy-match regression makes real LLM calls (~minutes + tokens
per case), so it must NOT run in the normal ``pytest`` path. Opt in with
``pytest --run-slow``.
"""
from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run expensive end-to-end eval tests (real LLM calls)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "slow: expensive end-to-end eval (real LLM calls)"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item],
) -> None:
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="slow eval — pass --run-slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
