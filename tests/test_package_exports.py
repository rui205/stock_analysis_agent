"""Smoke tests: top-level package re-exports the new public symbols."""
from __future__ import annotations


def test_agent_package_exports() -> None:
    from stock_analysis_agent.agent import StockAnalysis, StockAnalysisAgent

    assert StockAnalysis.__name__ == "StockAnalysis"
    assert StockAnalysisAgent.__name__ == "StockAnalysisAgent"


def test_tools_package_exports() -> None:
    from stock_analysis_agent.tools import FeishuCli, FeishuCliError, FeishuDocRef

    assert FeishuCli.__name__ == "FeishuCli"
    assert FeishuDocRef.__name__ == "FeishuDocRef"
    assert FeishuCliError.__name__ == "FeishuCliError"


def test_top_level_package_exports() -> None:
    import stock_analysis_agent

    expected = {
        "StockAnalysis",
        "StockAnalysisAgent",
        "FeishuCli",
        "FeishuDocRef",
        "FeishuCliError",
    }
    missing = expected - set(stock_analysis_agent.__all__)
    assert not missing, f"top-level __all__ missing: {missing}"
