"""Tests for stock_analysis_agent.tools.market_data."""
from __future__ import annotations

import pytest

from stock_analysis_agent.tools.market_data import _translate


class TestTranslate:
    """_translate(symbol) -> dict[source_name, source_local_code]."""

    def test_translate_hk_symbol_to_all_sources(self) -> None:
        result = _translate("02319.HK")
        assert result == {
            "sina": "rt_hk02319",
            "tencent": "hk02319",
            "tushare": "02319.HK",
            "akshare": "02319",
            "mootdx": "23",
            "mootdx_symbol": "023190",
        }

    def test_translate_sh_symbol_to_all_sources(self) -> None:
        result = _translate("600519.SH")
        assert result == {
            "sina": "sh600519",
            "tencent": "sh600519",
            "tushare": "600519.SH",
            "akshare": "sh600519",
            "mootdx": "1",
            "mootdx_symbol": "600519",
        }

    def test_translate_sz_symbol_to_all_sources(self) -> None:
        result = _translate("000001.SZ")
        assert result == {
            "sina": "sz000001",
            "tencent": "sz000001",
            "tushare": "000001.SZ",
            "akshare": "sz000001",
            "mootdx": "0",
            "mootdx_symbol": "000001",
        }

    def test_translate_unknown_market_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unsupported market"):
            _translate("02319.XX")
