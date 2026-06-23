"""Tests for the StockAnalysis pydantic schema."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from stock_analysis_agent.agent.analysis_schema import StockAnalysis


def _full_payload() -> dict[str, str]:
    return {
        "symbol": "02319.HK",
        "summary": "蒙牛乳业近期经营稳健,股价在 12 港元附近震荡。" * 2,
        "fundamentals": "乳制品行业龙头,PE 处于行业中位水平。",
        "technicals": "现价 12.34,日内小幅波动,成交量持平。",
        "peer_compare": "伊利股份 PE 略低,光明乳业规模较小。",
        "news": "近期发布半年报,营收同比+5%。",
        "risks": "原奶价格波动;消费需求疲软。",
        "recommendation": "关注,等待买点信号。",
    }


def test_stock_analysis_accepts_full_payload() -> None:
    a = StockAnalysis(**_full_payload())
    assert a.symbol == "02319.HK"
    assert "蒙牛" in a.summary


def test_stock_analysis_rejects_short_summary() -> None:
    payload = _full_payload()
    payload["summary"] = "太短"  # < 20 chars
    with pytest.raises(ValidationError):
        StockAnalysis(**payload)


def test_stock_analysis_rejects_empty_symbol() -> None:
    payload = _full_payload()
    payload["symbol"] = ""
    with pytest.raises(ValidationError):
        StockAnalysis(**payload)


def test_stock_analysis_rejects_missing_field() -> None:
    payload = _full_payload()
    payload.pop("risks")
    with pytest.raises(ValidationError):
        StockAnalysis(**payload)
