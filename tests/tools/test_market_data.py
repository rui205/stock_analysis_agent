"""Tests for stock_analysis_agent.tools.market_data."""
from __future__ import annotations

import httpx
import pytest

from stock_analysis_agent.tools import market_data as md
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


class TestFetchSina:
    """_fetch_sina(code) -> str using httpx against hq.sinajs.cn."""

    @pytest.mark.asyncio
    async def test_fetch_sina_parses_hk_quote(self) -> None:
        """Mock the httpx response, assert _fetch_sina returns a snippet
        that includes the parsed price/change fields."""
        sample_csv = (
            'var hq_str_rt_hk02319="MENGNIU DAIRY,蒙牛股份,15.940,15.570,'
            "15.940,15.340,15.890,0.320,2.055,15.880,15.890,"
            "278092437.740,17684472,36.161,0.000,17.411,13.374,"
            '2026/06/22,16:08:16,100|0,N|Y,Y,15.850|15.060|16.640,'
            "0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,..."
            '";'
        )

        def _h(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=sample_csv)

        result = await md._fetch_sina(
            "rt_hk02319",
            transport=httpx.MockTransport(_h),
        )
        # Result must contain the price and the change percent.
        assert "15.890" in result
        assert "0.320" in result
        assert "+2.06%" in result or "2.06%" in result
        # Header should mark this as the sina source.
        assert "[sina]" in result

    @pytest.mark.asyncio
    async def test_fetch_sina_parses_a_share_quote(self) -> None:
        """Mock the httpx response with a short A-share CSV (fewer than
        32 fields), assert the A-share branch renders name/open/prev_close/
        current/high/low."""
        sample_csv = (
            'var hq_str_sh600519="贵州茅台,1700.00,1690.00,1710.00,'
            '1705.00,1695.00,12345.67,12345678,1.234,5.6,...";'
        )

        def _h(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=sample_csv)

        result = await md._fetch_sina(
            "sh600519",
            transport=httpx.MockTransport(_h),
        )
        # A-share branch should render name + price + OHLC.
        assert "贵州茅台" in result
        assert "1710.000" in result
        assert "[sina]" in result

    @pytest.mark.asyncio
    async def test_fetch_sina_returns_error_segment_on_http_failure(self) -> None:
        """If httpx raises, _fetch_sina returns '[error: ...]' segment,
        not raising."""

        def _h(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom")

        result = await md._fetch_sina(
            "rt_hk02319",
            transport=httpx.MockTransport(_h),
        )
        assert "[error:" in result
        assert "[sina]" in result


class TestFetchTencent:
    """_fetch_tencent(code) -> str using httpx against qt.gtimg.cn."""

    @pytest.mark.asyncio
    async def test_fetch_tencent_parses_hk_quote(self) -> None:
        """Mock the httpx response, assert _fetch_tencent returns a
        snippet with the parsed price/PE fields."""
        sample = (
            'v_hk02319="100~蒙牛股份~02319~15.890~15.570~15.940~'
            "17684472.0~0~0~15.890~0~0~0~0~0~0~0~0~0~15.890~0~0~0~0~0~0~0~0~0~"
            "17684472.0~2026/06/22 16:08:17~0.320~2.06~15.940~15.340~15.890~"
            '17684472.0~278092437.740~0~36.00~~0~0~3.85~615.9721~615.9721~'
            'MENGNIU DAIRY~3.76~17.472~13.282~1.08~-96.57~0~0~0~0~0~36.00~'
            '1.38~0.46~1000~11.03~-4.33~GP~3.81~1.68~3.91~-5.71~1.78~'
            '3876476513.00~3876476513.00~36.00~0.598~15.725~10.03~HKD~1~50";'
        )

        def _h(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=sample)

        result = await md._fetch_tencent(
            "hk02319",
            transport=httpx.MockTransport(_h),
        )
        assert "[tencent]" in result
        assert "15.890" in result
        assert "蒙牛股份" in result or "MENGNIU" in result

    @pytest.mark.asyncio
    async def test_fetch_tencent_returns_error_segment_on_http_failure(self) -> None:
        def _h(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom")

        result = await md._fetch_tencent(
            "hk02319",
            transport=httpx.MockTransport(_h),
        )
        assert "[error:" in result
        assert "[tencent]" in result
