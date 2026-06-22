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
        # Strict assertions: change/change_pct/high/low/PE-TTM/PB must
        # render with their verified values, not the off-by-one
        # neighbors from fields[32..35,46,49].
        assert "涨跌: +0.320 (+2.06%)" in result
        assert "最高: 15.940" in result
        assert "最低: 15.340" in result
        assert "PE-TTM: 17.472" in result
        assert "PB: 13.282" in result
        # fields[1] is the CN name and must be present.
        assert "蒙牛股份" in result
        # fields[46] is the English name and must NOT leak into the
        # output (the buggy parser was rendering it as PB).
        assert "MENGNIU DAIRY" not in result

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


class TestFetchTushare:
    """_fetch_tushare(code, token) -> str using tushare.pro_api."""

    @pytest.mark.asyncio
    async def test_fetch_tushare_returns_error_when_token_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No TUSHARE_TOKEN env var -> [tushare]\\n[error: TUSHARE_TOKEN not set]\\n"""
        monkeypatch.delenv("TUSHARE_TOKEN", raising=False)

        result = await md._fetch_tushare("02319.HK", token=None)
        assert "[tushare]" in result
        assert "TUSHARE_TOKEN" in result
        assert "[error:" in result

    @pytest.mark.asyncio
    async def test_fetch_tushare_happy_path_with_mocked_pro_api(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When token is set and pro_api is monkeypatched to return fake
        DataFrames, _fetch_tushare should render a snapshot including
        price and PE fields."""
        import pandas as pd

        fake_daily = pd.DataFrame(
            [
                {
                    "ts_code": "02319.HK",
                    "trade_date": "20260622",
                    "open": 15.570,
                    "high": 15.940,
                    "low": 15.340,
                    "close": 15.890,
                    "pre_close": 15.570,
                    "change": 0.320,
                    "pct_chg": 2.060,
                    "vol": 17684472.0,
                    "amount": 278092437.74,
                }
            ]
        )
        fake_basic = pd.DataFrame(
            [
                {
                    "ts_code": "02319.HK",
                    "name": "蒙牛乳业",
                    "industry": "乳品",
                    "pe": 11.030,
                    "pb": 1.680,
                    "total_mv": 387647.0,
                }
            ]
        )

        class _FakePro:
            def daily(self, **kwargs):  # type: ignore[no-untyped-def]
                return fake_daily

            def stock_basic(self, **kwargs):  # type: ignore[no-untyped-def]
                return fake_basic

        import tushare as ts

        monkeypatch.setattr(ts, "pro_api", lambda token: _FakePro())

        result = await md._fetch_tushare("02319.HK", token="dummy")
        assert "[tushare]" in result
        assert "15.890" in result
        assert "涨跌: +0.320 (+2.06%)" in result
        assert "最高: 15.940" in result
        assert "最低: 15.340" in result
        assert "蒙牛乳业" in result or "乳品" in result


class TestFetchAkshare:
    """_fetch_akshare(code) -> str using akshare."""

    @pytest.mark.asyncio
    async def test_fetch_akshare_happy_path_with_mocked_ak(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Monkeypatch the specific akshare functions we use, assert
        the output contains price and PE fields."""
        import akshare as ak
        import pandas as pd

        fake_spot = pd.DataFrame(
            [
                {
                    "代码": "02319",
                    "名称": "蒙牛乳业",
                    "最新价": 15.89,
                    "涨跌额": 0.32,
                    "涨跌幅": 2.06,
                    "今开": 15.57,
                    "昨收": 15.57,
                    "最高": 15.94,
                    "最低": 15.34,
                    "成交量": 17684472,
                    "成交额": 278092437.74,
                    "市盈率": 11.03,
                    "市净率": 1.68,
                    "总市值": 387647651300,
                }
            ]
        )

        monkeypatch.setattr(
            ak, "stock_hk_spot_em", lambda: fake_spot
        )

        from stock_analysis_agent.tools import market_data as md

        result = await md._fetch_akshare("02319")
        assert "[akshare]" in result
        assert "15.890" in result
        assert "蒙牛乳业" in result
        assert "PE" in result

    @pytest.mark.asyncio
    async def test_fetch_akshare_returns_error_segment_on_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import akshare as ak

        def _boom() -> None:
            raise RuntimeError("akshare down")

        monkeypatch.setattr(ak, "stock_hk_spot_em", _boom)

        from stock_analysis_agent.tools import market_data as md

        result = await md._fetch_akshare("02319")
        assert "[akshare]" in result
        assert "[error:" in result


class TestFetchMootdx:
    """_fetch_mootdx(market, symbol) -> str using mootdx 0.11.7 API."""

    @pytest.mark.asyncio
    async def test_fetch_mootdx_returns_error_segment_on_connection_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If mootdx.quotes.StdQuotes raises on connect, _fetch_mootdx
        returns an [error: ...] segment instead of propagating."""
        from mootdx.quotes import StdQuotes

        def _boom(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise ConnectionError("tdx server unreachable")

        monkeypatch.setattr(StdQuotes, "__init__", _boom)

        from stock_analysis_agent.tools import market_data as md

        result = await md._fetch_mootdx("0", "000001")
        assert "[mootdx]" in result
        assert "[error:" in result

    @pytest.mark.asyncio
    async def test_fetch_mootdx_happy_path_with_mocked_dataframe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the client returns a non-empty DataFrame, _fetch_mootdx
        renders open/high/low/close/volume fields."""
        import pandas as pd
        from mootdx.quotes import StdQuotes

        fake_df = pd.DataFrame(
            [
                {
                    "open": 15.57,
                    "high": 15.94,
                    "low": 15.34,
                    "close": 15.89,
                    "volume": 17684472.0,
                }
            ]
        )

        monkeypatch.setattr(StdQuotes, "__init__", lambda *a, **kw: None)
        monkeypatch.setattr(StdQuotes, "bars", lambda self, **kw: fake_df)

        from stock_analysis_agent.tools import market_data as md

        result = await md._fetch_mootdx("0", "000001")
        assert "[mootdx]" in result
        assert "15.890" in result  # strict .3f format

    @pytest.mark.asyncio
    async def test_fetch_mootdx_empty_dataframe_returns_error_segment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """mootdx returns empty DataFrame for HK symbols (mootdx is
        A-share focused). Adapter must report this gracefully, not raise."""
        import pandas as pd
        from mootdx.quotes import StdQuotes

        empty_df = pd.DataFrame()

        monkeypatch.setattr(StdQuotes, "__init__", lambda *a, **kw: None)
        monkeypatch.setattr(StdQuotes, "bars", lambda self, **kw: empty_df)

        from stock_analysis_agent.tools import market_data as md

        result = await md._fetch_mootdx("23", "023190")
        assert "[mootdx]" in result
        assert "[error:" in result
