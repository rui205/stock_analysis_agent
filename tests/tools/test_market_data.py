"""Tests for stock_analysis_agent.tools.market_data."""
from __future__ import annotations

import json

import pytest

from stock_analysis_agent.tools import market_data as md
from stock_analysis_agent.tools.market_data import _translate


class TestTranslate:
    """_translate(symbol) -> dict[source_name, source_local_code]."""

    def test_translate_hk_symbol_to_all_sources(self) -> None:
        result = _translate("02319.HK")
        assert result == {
            "tushare": "02319.HK",
            "akshare": "02319",
            "mootdx": "23",
            "mootdx_symbol": "023190",
        }

    def test_translate_sh_symbol_to_all_sources(self) -> None:
        result = _translate("600519.SH")
        assert result == {
            "tushare": "600519.SH",
            "akshare": "sh600519",
            "mootdx": "1",
            "mootdx_symbol": "600519",
        }

    def test_translate_sz_symbol_to_all_sources(self) -> None:
        result = _translate("000001.SZ")
        assert result == {
            "tushare": "000001.SZ",
            "akshare": "sz000001",
            "mootdx": "0",
            "mootdx_symbol": "000001",
        }

    def test_translate_unknown_market_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unsupported market"):
            _translate("02319.XX")


class TestFetchTushare:
    """_fetch_tushare(code, token) -> dict: {"data", "row_index"} or {"error"}."""

    @pytest.mark.asyncio
    async def test_fetch_tushare_returns_error_when_token_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No TUSHARE_TOKEN env var -> {"error": {"type": "TushareTokenMissing", ...}}."""
        monkeypatch.delenv("TUSHARE_TOKEN", raising=False)

        result = await md._fetch_tushare("02319.HK", token=None)
        assert "error" in result
        assert "data" not in result
        assert result["error"]["type"] == "TushareTokenMissing"
        assert "TUSHARE_TOKEN" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_fetch_tushare_happy_path_returns_dict_with_merged_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When token is set and pro_api returns DataFrames, _fetch_tushare
        merges the daily + stock_basic rows into one flat dict and wraps it
        in {"data": ..., "row_index": 0}. No field filtering."""
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
        assert "data" in result
        assert result["row_index"] == 0
        d = result["data"]
        # daily fields preserved
        assert d["ts_code"] == "02319.HK"
        assert d["trade_date"] == "20260622"
        assert float(d["close"]) == 15.89
        assert float(d["vol"]) == 17684472.0
        # stock_basic fields merged in
        assert d["name"] == "蒙牛乳业"
        assert d["industry"] == "乳品"
        assert float(d["pe"]) == 11.03
        assert float(d["total_mv"]) == 387647.0

    @pytest.mark.asyncio
    async def test_fetch_tushare_returns_error_when_basic_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If stock_basic returns empty, _fetch_tushare returns {"error": ...}."""
        import pandas as pd

        class _FakePro:
            def daily(self, **kwargs):  # type: ignore[no-untyped-def]
                return pd.DataFrame()

            def stock_basic(self, **kwargs):  # type: ignore[no-untyped-def]
                return pd.DataFrame()

        import tushare as ts

        monkeypatch.setattr(ts, "pro_api", lambda token: _FakePro())

        result = await md._fetch_tushare("02319.HK", token="dummy")
        assert "error" in result
        assert result["error"]["type"] == "TushareEmpty"

    @pytest.mark.asyncio
    async def test_fetch_tushare_returns_error_on_unexpected_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Any exception other than the known paths becomes {"error": {"type": <ExcType>, ...}}."""
        import tushare as ts

        def _boom(token):  # type: ignore[no-untyped-def]
            raise RuntimeError("network down")

        monkeypatch.setattr(ts, "pro_api", _boom)

        result = await md._fetch_tushare("02319.HK", token="dummy")
        assert "error" in result
        assert result["error"]["type"] == "RuntimeError"
        assert "network down" in result["error"]["message"]


class TestFetchAkshare:
    """_fetch_akshare(code) -> dict: {"data", "row_index"} or {"error"}."""

    @pytest.mark.asyncio
    async def test_fetch_akshare_hk_happy_path_returns_full_sina_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """For an HK code, _fetch_akshare calls stock_hk_spot (sina) and
        returns the matching row's FULL dict (every sina column preserved,
        no field filtering). NaN values become None."""
        import akshare as ak
        import pandas as pd

        fake_spot = pd.DataFrame(
            [
                {
                    "代码": "02319",
                    "中文名称": "蒙牛乳业",
                    "英文名称": "MENGNIU DAIRY",
                    "最新价": 15.89,
                    "涨跌额": 0.32,
                    "涨跌幅": 2.06,
                    "昨收": 15.57,
                    "今开": 15.57,
                    "最高": 15.94,
                    "最低": 15.34,
                    "成交量": 17684472,
                    "成交额": 278092437.74,
                    "市盈率": float("nan"),  # NaN must become None in output
                }
            ]
        )

        monkeypatch.setattr(ak, "stock_hk_spot", lambda: fake_spot)

        result = await md._fetch_akshare("02319")
        assert "data" in result
        assert result["row_index"] == 0
        d = result["data"]
        # Every sina column must be preserved.
        assert d["代码"] == "02319"
        assert d["中文名称"] == "蒙牛乳业"
        assert d["英文名称"] == "MENGNIU DAIRY"  # no longer stripped
        assert d["最新价"] == 15.89
        assert d["涨跌幅"] == 2.06
        # NaN becomes None (JSON-safe).
        assert d["市盈率"] is None

    @pytest.mark.asyncio
    async def test_fetch_akshare_a_share_happy_path_returns_full_sina_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """For an A-share code, _fetch_akshare calls stock_zh_a_spot (sina)
        and returns the full row dict."""
        import akshare as ak
        import pandas as pd

        fake_spot = pd.DataFrame(
            [
                {
                    "代码": "sh600887",
                    "名称": "伊利股份",
                    "最新价": 24.53,
                    "涨跌额": -0.10,
                    "涨跌幅": -0.41,
                    "昨收": 24.63,
                    "今开": 24.56,
                    "最高": 25.11,
                    "最低": 24.51,
                    "成交量": 35784473,
                    "成交额": 886842735.0,
                }
            ]
        )

        monkeypatch.setattr(ak, "stock_zh_a_spot", lambda: fake_spot)

        result = await md._fetch_akshare("sh600887")
        assert "data" in result
        assert result["row_index"] == 0
        d = result["data"]
        assert d["代码"] == "sh600887"
        assert d["名称"] == "伊利股份"
        assert d["最新价"] == 24.53

    @pytest.mark.asyncio
    async def test_fetch_akshare_returns_error_when_code_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the spot DataFrame doesn't contain our code, return {"error": ...}."""
        import akshare as ak
        import pandas as pd

        fake_spot = pd.DataFrame(
            [{"代码": "00000", "名称": "不存在", "最新价": 0.0}]
        )
        monkeypatch.setattr(ak, "stock_hk_spot", lambda: fake_spot)

        result = await md._fetch_akshare("02319")
        assert "error" in result
        assert result["error"]["type"] == "AkshareCodeNotFound"
        assert "02319" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_fetch_akshare_returns_error_on_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When akshare raises, return {"error": {"type": <ExcType>, ...}}."""
        import akshare as ak

        def _boom() -> None:
            raise RuntimeError("akshare down")

        monkeypatch.setattr(ak, "stock_hk_spot", _boom)

        result = await md._fetch_akshare("02319")
        assert "error" in result
        assert result["error"]["type"] == "RuntimeError"
        assert "akshare down" in result["error"]["message"]


class TestFetchMootdx:
    """_fetch_mootdx(market, symbol) -> dict: {"data", "row_index"} or {"error"}."""

    @pytest.mark.asyncio
    async def test_fetch_mootdx_returns_error_on_connection_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If mootdx.quotes.StdQuotes raises on connect, _fetch_mootdx
        returns {"error": {"type": "ConnectionError", ...}}."""
        from mootdx.quotes import StdQuotes

        def _boom(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise ConnectionError("tdx server unreachable")

        monkeypatch.setattr(StdQuotes, "__init__", _boom)

        result = await md._fetch_mootdx("0", "000001")
        assert "error" in result
        assert result["error"]["type"] == "ConnectionError"
        assert "tdx server unreachable" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_fetch_mootdx_happy_path_returns_full_bar_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the client returns a non-empty DataFrame, _fetch_mootdx
        returns the full bar row as {"data": ..., "row_index": 0}."""
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
                    "amount": 278092437.74,
                }
            ]
        )

        monkeypatch.setattr(StdQuotes, "__init__", lambda *a, **kw: None)
        monkeypatch.setattr(StdQuotes, "bars", lambda self, **kw: fake_df)

        result = await md._fetch_mootdx("0", "000001")
        assert "data" in result
        assert result["row_index"] == 0
        d = result["data"]
        assert d["close"] == 15.89
        assert d["volume"] == 17684472.0
        # All mootdx columns preserved (no filtering).
        assert "amount" in d
        assert "open" in d

    @pytest.mark.asyncio
    async def test_fetch_mootdx_empty_dataframe_returns_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """mootdx returns empty DataFrame for HK symbols (A-share focused).
        Adapter must report {"error": {"type": "MootdxEmpty", ...}}."""
        import pandas as pd
        from mootdx.quotes import StdQuotes

        empty_df = pd.DataFrame()

        monkeypatch.setattr(StdQuotes, "__init__", lambda *a, **kw: None)
        monkeypatch.setattr(StdQuotes, "bars", lambda self, **kw: empty_df)

        result = await md._fetch_mootdx("23", "023190")
        assert "error" in result
        assert result["error"]["type"] == "MootdxEmpty"


class TestDetectPeers:
    """_detect_peers(symbol, peer_count) -> list[str] | None."""

    def test_detect_peers_happy_path_a_share(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """For an A-share symbol whose industry is found in akshare's
        industry table, return the top-N industry codes by market cap."""
        import akshare as ak
        import pandas as pd

        industries = pd.DataFrame(
            [{"板块名称": "白酒"}]
        )
        cons = pd.DataFrame(
            [
                {"代码": "600519", "名称": "贵州茅台", "总市值": 20000},
                {"代码": "000858", "名称": "五粮液", "总市值": 8000},
                {"代码": "000568", "名称": "泸州老窖", "总市值": 3000},
            ]
        )

        monkeypatch.setattr(ak, "stock_board_industry_name_em", lambda: industries)
        monkeypatch.setattr(
            ak,
            "stock_board_industry_cons_em",
            lambda symbol: cons,
        )

        fake_info = pd.DataFrame([{"行业": "白酒"}])
        monkeypatch.setattr(
            ak,
            "stock_individual_info_em",
            lambda symbol: fake_info,
        )

        from stock_analysis_agent.tools import market_data as md

        result = md._detect_peers("600519.SH", peer_count=2)
        assert result == ["600519.SH", "000858.SZ"]

    def test_detect_peers_returns_none_when_industry_cons_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If akshare's industry-cons endpoint raises, _detect_peers
        returns None and the aggregator should emit an [error: ...]
        peers segment."""
        import akshare as ak
        import pandas as pd

        industries = pd.DataFrame([{"板块名称": "乳品"}])

        monkeypatch.setattr(ak, "stock_board_industry_name_em", lambda: industries)

        def _boom_cons(symbol: str) -> None:
            raise RuntimeError("network down")

        monkeypatch.setattr(ak, "stock_board_industry_cons_em", _boom_cons)

        from stock_analysis_agent.tools import market_data as md

        result = md._detect_peers("02319.HK", peer_count=2)
        assert result is None

    def test_detect_peers_uses_hk_industry_hint_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """For an HK code, fall back to HK_INDUSTRY_HINTS to find the
        akshare industry name."""
        import akshare as ak
        import pandas as pd

        industries = pd.DataFrame([{"板块名称": "乳品"}])
        cons = pd.DataFrame(
            [
                {"代码": "600887", "名称": "伊利股份", "总市值": 15000},
                {"代码": "600597", "名称": "光明乳业", "总市值": 1500},
            ]
        )

        monkeypatch.setattr(ak, "stock_board_industry_name_em", lambda: industries)
        monkeypatch.setattr(ak, "stock_board_industry_cons_em", lambda symbol: cons)

        from stock_analysis_agent.tools import market_data as md

        result = md._detect_peers("02319.HK", peer_count=2)
        assert result == ["600887.SH", "600597.SH"]

    def test_detect_peers_invokes_em_request_hook(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_detect_peers must call _install_em_request_hook so that
        downstream akshare ``*_em`` requests carry EM_HEADERS. The
        hook is idempotent — calling _detect_peers multiple times
        should not re-patch requests.get."""
        import akshare as ak
        import pandas as pd

        industries = pd.DataFrame([{"板块名称": "白酒"}])
        cons = pd.DataFrame(
            [
                {"代码": "600519", "名称": "贵州茅台", "总市值": 20000},
            ]
        )

        monkeypatch.setattr(ak, "stock_board_industry_name_em", lambda: industries)
        monkeypatch.setattr(ak, "stock_board_industry_cons_em", lambda symbol: cons)
        fake_info = pd.DataFrame([{"行业": "白酒"}])
        monkeypatch.setattr(ak, "stock_individual_info_em", lambda symbol: fake_info)

        from stock_analysis_agent.tools import market_data as md

        # Reset hook state so the test exercises the install path.
        md._em_hook_installed = False
        original = md._install_em_request_hook
        calls: list[int] = []

        def _spy() -> None:
            calls.append(1)
            return original()

        monkeypatch.setattr(md, "_install_em_request_hook", _spy)

        md._detect_peers("600519.SH", peer_count=1)
        md._detect_peers("600519.SH", peer_count=1)
        # _install_em_request_hook is referenced from _detect_peers; the
        # spy records two invocations even though the underlying
        # install is idempotent.
        assert len(calls) == 2


class TestInstallEmRequestHook:
    """_install_em_request_hook: monkey-patch requests.get for eastmoney."""

    def test_hook_patches_requests_get_for_eastmoney_urls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the URL contains 'eastmoney.com', the wrapped get must
        inject EM_HEADERS into kwargs['headers'] before delegating."""
        import requests

        from stock_analysis_agent.tools import market_data as md

        captured: dict[str, object] = {}

        def _fake_get(url, **kwargs):  # type: ignore[no-untyped-def]
            captured["url"] = url
            captured["kwargs"] = kwargs
            return None

        monkeypatch.setattr(requests, "get", _fake_get)
        # Force re-install so the test sees the patched get.
        md._em_hook_installed = False
        md._install_em_request_hook()

        # Call through the patched requests.get.
        requests.get("https://push2.eastmoney.com/api/qt/clist/get")

        kwargs = captured["kwargs"]  # type: ignore[index]
        assert "headers" in kwargs
        headers = kwargs["headers"]  # type: ignore[index]
        for key, value in md.EM_HEADERS.items():
            assert headers[key] == value  # type: ignore[index]
        # Restore for downstream tests.
        md._em_hook_installed = False

    def test_hook_does_not_touch_non_eastmoney_urls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-eastmoney URLs should pass through to the original get
        without any header injection."""
        import requests

        from stock_analysis_agent.tools import market_data as md

        captured: dict[str, object] = {}

        def _fake_get(url, **kwargs):  # type: ignore[no-untyped-def]
            captured["url"] = url
            captured["kwargs"] = kwargs
            return None

        monkeypatch.setattr(requests, "get", _fake_get)
        md._em_hook_installed = False
        md._install_em_request_hook()

        requests.get("https://hq.sinajs.cn/list=sh600887", headers={"X-Custom": "y"})

        # Original headers must be preserved untouched.
        assert captured["kwargs"] == {"headers": {"X-Custom": "y"}}  # type: ignore[index]
        md._em_hook_installed = False

    def test_hook_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Calling _install_em_request_hook twice must not re-wrap
        requests.get (which would stack wrappers and re-inject headers)."""
        import requests

        from stock_analysis_agent.tools import market_data as md

        # Reset hook state for the test.
        md._em_hook_installed = False
        original_get = requests.get
        try:
            md._install_em_request_hook()
            patched_once = requests.get
            md._install_em_request_hook()
            patched_twice = requests.get
            assert patched_once is patched_twice
        finally:
            requests.get = original_get
            md._em_hook_installed = False


class TestFetchAndConcat:
    """_fetch_and_concat aggregator returns nested dict with top-level keys."""

    @pytest.mark.asyncio
    async def test_concat_returns_dict_with_symbol_and_fetched_at(self) -> None:
        """Result is a dict with <symbol> (containing per-source dicts)
        and fetched_at (ISO 8601 string)."""
        from stock_analysis_agent.tools import market_data as md

        async def _ok(*args, **kwargs):  # type: ignore[no-untyped-def]
            return {"data": {"ts_code": "02319.HK", "name": "蒙牛"}, "row_index": 0}

        original_tushare = md._fetch_tushare
        original_akshare = md._fetch_akshare
        original_mootdx = md._fetch_mootdx
        md._fetch_tushare = _ok  # type: ignore[assignment]
        md._fetch_akshare = _ok  # type: ignore[assignment]
        md._fetch_mootdx = _ok  # type: ignore[assignment]
        try:
            result = await md._fetch_and_concat(
                "02319.HK",
                sources=("tushare", "akshare", "mootdx"),
                include_peers=False,
                peer_count=0,
                cache=None,
            )
        finally:
            md._fetch_tushare = original_tushare  # type: ignore[assignment]
            md._fetch_akshare = original_akshare  # type: ignore[assignment]
            md._fetch_mootdx = original_mootdx  # type: ignore[assignment]

        assert isinstance(result, dict)
        assert "02319.HK" in result
        assert "fetched_at" in result
        assert "peers" not in result  # include_peers=False
        # Per-source dicts present
        assert "tushare" in result["02319.HK"]
        assert "akshare" in result["02319.HK"]
        assert "mootdx" in result["02319.HK"]

    @pytest.mark.asyncio
    async def test_concat_partial_failure_keeps_working_sources(self) -> None:
        """If one source errors, others still appear with their data blocks."""
        from stock_analysis_agent.tools import market_data as md

        async def _ok(*args, **kwargs):  # type: ignore[no-untyped-def]
            return {"data": {"ts_code": "02319.HK"}, "row_index": 0}

        async def _boom(*args, **kwargs):  # type: ignore[no-untyped-def]
            return {"error": {"type": "ConnectError", "message": "nope"}}

        original_tushare = md._fetch_tushare
        original_akshare = md._fetch_akshare
        md._fetch_tushare = _ok  # type: ignore[assignment]
        md._fetch_akshare = _boom  # type: ignore[assignment]
        try:
            result = await md._fetch_and_concat(
                "02319.HK",
                sources=("tushare", "akshare"),
                include_peers=False,
                peer_count=0,
                cache=None,
            )
        finally:
            md._fetch_tushare = original_tushare  # type: ignore[assignment]
            md._fetch_akshare = original_akshare  # type: ignore[assignment]

        assert "data" in result["02319.HK"]["tushare"]
        assert "error" in result["02319.HK"]["akshare"]
        assert result["02319.HK"]["akshare"]["error"]["type"] == "ConnectError"

    @pytest.mark.asyncio
    async def test_concat_all_failure_raises_tool_execution_error(self) -> None:
        """When every source errors, raise ToolExecutionError so retry middleware can act."""
        from stock_analysis_agent.agent.exceptions import ToolExecutionError
        from stock_analysis_agent.tools import market_data as md

        async def _boom1(*args, **kwargs):  # type: ignore[no-untyped-def]
            return {"error": {"type": "ConnectError", "message": "nope"}}

        async def _boom2(*args, **kwargs):  # type: ignore[no-untyped-def]
            return {"error": {"type": "ConnectError", "message": "nope"}}

        original_tushare = md._fetch_tushare
        original_akshare = md._fetch_akshare
        md._fetch_tushare = _boom1  # type: ignore[assignment]
        md._fetch_akshare = _boom2  # type: ignore[assignment]
        try:
            with pytest.raises(ToolExecutionError, match="all sources failed"):
                await md._fetch_and_concat(
                    "02319.HK",
                    sources=("tushare", "akshare"),
                    include_peers=False,
                    peer_count=0,
                    cache=None,
                )
        finally:
            md._fetch_tushare = original_tushare  # type: ignore[assignment]
            md._fetch_akshare = original_akshare  # type: ignore[assignment]

    @pytest.mark.asyncio
    async def test_concat_includes_peers_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When include_peers=True, peers dict appears at top level,
        keyed by peer symbol, with akshare as the only nested source."""
        from stock_analysis_agent.tools import market_data as md

        async def _ok(*args, **kwargs):  # type: ignore[no-untyped-def]
            return {"data": {"代码": "02319"}, "row_index": 0}

        def _fake_detect(symbol: str, peer_count: int) -> list[str]:
            return ["600887.SH", "600597.SH"]

        monkeypatch.setattr(md, "_fetch_tushare", _ok)
        monkeypatch.setattr(md, "_fetch_akshare", _ok)
        monkeypatch.setattr(md, "_fetch_mootdx", _ok)
        monkeypatch.setattr(md, "_detect_peers", _fake_detect)

        result = await md._fetch_and_concat(
            "02319.HK",
            sources=("tushare", "akshare"),
            include_peers=True,
            peer_count=2,
            cache=None,
        )
        assert "peers" in result
        assert "600887.SH" in result["peers"]
        assert "600597.SH" in result["peers"]
        # Only akshare populated per peer
        assert "akshare" in result["peers"]["600887.SH"]
        assert "tushare" not in result["peers"]["600887.SH"]

    @pytest.mark.asyncio
    async def test_concat_peers_error_when_detection_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When _detect_peers returns None, peers dict has _error placeholder."""
        from stock_analysis_agent.tools import market_data as md

        async def _ok(*args, **kwargs):  # type: ignore[no-untyped-def]
            return {"data": {"代码": "02319"}, "row_index": 0}

        monkeypatch.setattr(md, "_fetch_tushare", _ok)
        monkeypatch.setattr(md, "_fetch_akshare", _ok)
        monkeypatch.setattr(md, "_fetch_mootdx", _ok)
        monkeypatch.setattr(md, "_detect_peers", lambda s, n: None)

        result = await md._fetch_and_concat(
            "02319.HK",
            sources=("tushare", "akshare"),
            include_peers=True,
            peer_count=2,
            cache=None,
        )
        assert "peers" in result
        assert "_error" in result["peers"]
        assert result["peers"]["_error"]["type"] == "PeerDetectionError"

    @pytest.mark.asyncio
    async def test_concat_cache_roundtrip_returns_equal_dict(self, tmp_path) -> None:
        """Cache miss writes JSON; cache hit returns equal dict."""
        import json as _json
        from stock_analysis_agent.memory import _FileCache
        from stock_analysis_agent.tools import market_data as md

        async def _ok(*args, **kwargs):  # type: ignore[no-untyped-def]
            return {"data": {"ts_code": "02319.HK", "name": "蒙牛"}, "row_index": 0}

        original_tushare = md._fetch_tushare
        original_akshare = md._fetch_akshare
        original_mootdx = md._fetch_mootdx
        md._fetch_tushare = _ok  # type: ignore[assignment]
        md._fetch_akshare = _ok  # type: ignore[assignment]
        md._fetch_mootdx = _ok  # type: ignore[assignment]

        cache = _FileCache(tmp_path, ttl_seconds=60.0)
        try:
            # First call: writes cache.
            first = await md._fetch_and_concat(
                "02319.HK",
                sources=("tushare", "akshare", "mootdx"),
                include_peers=False,
                peer_count=0,
                cache=cache,
            )
            # Second call: hits cache.
            second = await md._fetch_and_concat(
                "02319.HK",
                sources=("tushare", "akshare", "mootdx"),
                include_peers=False,
                peer_count=0,
                cache=cache,
            )
        finally:
            md._fetch_tushare = original_tushare  # type: ignore[assignment]
            md._fetch_akshare = original_akshare  # type: ignore[assignment]
            md._fetch_mootdx = original_mootdx  # type: ignore[assignment]

        # Cache hit must return an equal dict.
        assert first == second
        # The cached text on disk must be valid JSON.
        # Cache key format: f"{symbol}|{','.join(sorted(sources))}|peers={peer_count if include_peers else 0}"
        cached_text = cache.get(
            site="market_data",
            query="02319.HK|akshare,mootdx,tushare|peers=0",
        )
        assert cached_text is not None
        parsed = _json.loads(cached_text)
        assert parsed["02319.HK"]["tushare"]["data"]["name"] == "蒙牛"

    @pytest.mark.asyncio
    async def test_concat_cache_handles_stale_text_entry_as_miss(
        self, tmp_path
    ) -> None:
        """If the on-disk cache holds a non-JSON string (legacy text format),
        the aggregator must treat it as a miss and re-fetch (no exception)."""
        from stock_analysis_agent.memory import _FileCache
        from stock_analysis_agent.tools import market_data as md

        cache = _FileCache(tmp_path, ttl_seconds=60.0)
        # Pre-seed a cache entry that holds legacy text, not JSON.
        legacy_key = "02319.HK|akshare,tushare|peers=0"
        cache.set(site="market_data", query=legacy_key, text="[tushare]\nlegacy text\n")

        async def _ok(*args, **kwargs):  # type: ignore[no-untyped-def]
            return {"data": {"ts_code": "02319.HK"}, "row_index": 0}

        original_tushare = md._fetch_tushare
        original_akshare = md._fetch_akshare
        md._fetch_tushare = _ok  # type: ignore[assignment]
        md._fetch_akshare = _ok  # type: ignore[assignment]
        try:
            # Must not raise even though legacy text is unparseable as JSON.
            result = await md._fetch_and_concat(
                "02319.HK",
                sources=("tushare", "akshare"),
                include_peers=False,
                peer_count=0,
                cache=cache,
            )
        finally:
            md._fetch_tushare = original_tushare  # type: ignore[assignment]
            md._fetch_akshare = original_akshare  # type: ignore[assignment]

        assert "02319.HK" in result

    @pytest.mark.asyncio
    async def test_concat_runs_in_parallel(self) -> None:
        """3 sources with ~100ms delay each → total < 250ms (parallel)."""
        import asyncio
        import time

        from stock_analysis_agent.tools import market_data as md

        async def _slow(*args, **kwargs):  # type: ignore[no-untyped-def]
            await asyncio.sleep(0.1)
            return {"data": {}, "row_index": 0}

        original_tushare = md._fetch_tushare
        original_akshare = md._fetch_akshare
        original_mootdx = md._fetch_mootdx
        md._fetch_tushare = _slow  # type: ignore[assignment]
        md._fetch_akshare = _slow  # type: ignore[assignment]
        md._fetch_mootdx = _slow  # type: ignore[assignment]
        try:
            start = time.monotonic()
            result = await md._fetch_and_concat(
                "02319.HK",
                sources=("tushare", "akshare", "mootdx"),
                include_peers=False,
                peer_count=0,
                cache=None,
            )
            elapsed = time.monotonic() - start
        finally:
            md._fetch_tushare = original_tushare  # type: ignore[assignment]
            md._fetch_akshare = original_akshare  # type: ignore[assignment]
            md._fetch_mootdx = original_mootdx  # type: ignore[assignment]

        assert elapsed < 0.25, f"expected parallel, took {elapsed:.3f}s"
        assert "02319.HK" in result


class TestGetStockSnapshotTool:
    """The @tool _get_stock_snapshot wrapper returns dict, not str."""

    def test_tool_name_is_get_stock_snapshot(self) -> None:
        assert md._get_stock_snapshot.name == "get_stock_snapshot"

    def test_tool_args_schema_has_symbol_and_sources(self) -> None:
        """The @tool exposes symbol and sources fields in its JSON schema."""
        schema = md._get_stock_snapshot.args
        if hasattr(schema, "model_json_schema"):
            schema = schema.model_json_schema()
        if isinstance(schema, dict) and "properties" in schema:
            properties = schema["properties"]
        else:
            properties = schema
        assert "symbol" in properties
        assert "sources" in properties

    def test_tool_return_annotation_is_dict(self) -> None:
        """The tool's annotated return type must be dict (not str)."""
        import typing
        coroutine = md._get_stock_snapshot.coroutine  # type: ignore[attr-defined]
        hints = typing.get_type_hints(coroutine)
        ret = hints.get("return")
        # Annotation is `dict[str, Any]` — check origin is dict, not bare match.
        assert typing.get_origin(ret) is dict, (
            f"expected dict origin, got {ret!r}"
        )

    @pytest.mark.asyncio
    async def test_tool_invokes_aggregator_and_returns_dict(self, tmp_path) -> None:
        """End-to-end: tool.ainvoke calls the aggregator and returns a dict."""
        from stock_analysis_agent.memory import _FileCache

        async def _fake_concat(symbol, **kwargs):  # type: ignore[no-untyped-def]
            return {
                symbol: {
                    "tushare": {"data": {"ts_code": symbol}, "row_index": 0},
                    "akshare": {"data": {"代码": "02319"}, "row_index": 0},
                },
                "fetched_at": "2026-06-23T15:30:00+08:00",
            }

        original = md._fetch_and_concat
        md._fetch_and_concat = _fake_concat  # type: ignore[assignment]
        cache = _FileCache(tmp_path, ttl_seconds=60.0)
        md._CACHE_PROVIDER.value = cache
        md._SOURCES_PROVIDER.value = md.ALL_SOURCES
        try:
            result = await md._get_stock_snapshot.ainvoke(
                {"symbol": "02319.HK", "sources": ["tushare", "akshare"]}
            )
        finally:
            md._fetch_and_concat = original  # type: ignore[assignment]
            md._CACHE_PROVIDER.value = None
            md._SOURCES_PROVIDER.value = None

        assert isinstance(result, dict)
        assert "02319.HK" in result
        assert result["02319.HK"]["tushare"]["data"]["ts_code"] == "02319.HK"
        assert "fetched_at" in result


class TestHelpers:
    """_now_iso() and _json_default() helpers used by the aggregator."""

    def test_now_iso_returns_iso8601_with_shanghai_offset(self) -> None:
        """_now_iso() returns an ISO 8601 string with +08:00 offset."""
        result = md._now_iso()
        # Format: YYYY-MM-DDTHH:MM:SS+08:00 (seconds precision).
        import re
        assert re.match(
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00$", result
        ), f"unexpected format: {result!r}"

    def test_json_default_serializes_date(self) -> None:
        """datetime.date objects become ISO 8601 strings."""
        import datetime as _dt
        result = md._json_default(_dt.date(2026, 6, 23))
        assert result == "2026-06-23"

    def test_json_default_serializes_datetime(self) -> None:
        """datetime.datetime objects become ISO 8601 strings."""
        import datetime as _dt
        dt = _dt.datetime(2026, 6, 23, 15, 30, 0)
        result = md._json_default(dt)
        assert result.startswith("2026-06-23T15:30:00")

    def test_json_default_raises_typeerror_for_unsupported_type(self) -> None:
        """Unsupported types raise TypeError with a helpful message."""
        with pytest.raises(TypeError, match="not JSON serializable"):
            md._json_default(object())


class TestJsonSerialization:
    """NaN/date/numpy serialization safety for cache + LLM transport."""

    def test_noneify_replaces_nan_with_none(self) -> None:
        """pandas NaN, float('nan'), and numpy.nan must become None."""
        import math

        import numpy as np

        assert md._noneify(float("nan")) is None
        assert md._noneify(math.nan) is None
        assert md._noneify(np.float64("nan")) is None
        # Non-NaN values pass through.
        assert md._noneify(0.0) == 0.0
        assert md._noneify(42) == 42
        assert md._noneify("hello") == "hello"
        assert md._noneify(None) is None

    def test_noneify_handles_pandas_nat(self) -> None:
        """pandas NaT (Not-a-Time) must become None."""
        import pandas as pd

        assert md._noneify(pd.NaT) is None

    def test_json_default_handles_date_and_datetime(self) -> None:
        """datetime.date / datetime.datetime become ISO strings via _json_default."""
        import datetime as _dt

        assert md._json_default(_dt.date(2026, 6, 23)) == "2026-06-23"
        assert md._json_default(_dt.datetime(2026, 6, 23, 15, 30, 0)) == "2026-06-23T15:30:00"

    def test_json_default_handles_numpy_scalars(self) -> None:
        """numpy.float64 / numpy.int64 become Python scalars."""
        import numpy as np

        assert md._json_default(np.float64(3.14)) == 3.14
        assert md._json_default(np.int64(42)) == 42

    def test_full_roundtrip_dict_to_json_to_dict(self) -> None:
        """A result dict with NaN + numpy + dates round-trips through json."""
        import datetime as _dt

        import numpy as np

        raw = {
            "02319.HK": {
                "tushare": {
                    "data": {
                        "ts_code": "02319.HK",
                        "close": np.float64(15.89),
                        "trade_date": _dt.date(2026, 6, 22),
                        "extra_nan": float("nan"),
                    },
                    "row_index": 0,
                }
            },
            "fetched_at": "2026-06-23T15:30:00+08:00",
        }
        # Production flow: walk leaves with _noneify (strips NaN -> None),
        # then encode with _json_default (handles dates / numpy scalars).
        # On decode, `parse_constant` collapses any leftover NaN/Infinity
        # literals to None — this is what `_noneify` already accomplishes
        # for the in-memory dict before it ever reaches `json.dumps`.
        sanitized = json.loads(
            json.dumps(raw, ensure_ascii=False, default=md._json_default),
            parse_constant=lambda _c: None,
        )

        assert sanitized["02319.HK"]["tushare"]["data"]["close"] == 15.89
        assert sanitized["02319.HK"]["tushare"]["data"]["trade_date"] == "2026-06-22"
        assert sanitized["02319.HK"]["tushare"]["data"]["extra_nan"] is None

    def test_cache_roundtrip_dict_via_json(self, tmp_path) -> None:
        """A result dict stored as JSON in the cache round-trips back equal."""
        from stock_analysis_agent.memory import _FileCache

        cache = _FileCache(tmp_path, ttl_seconds=60.0)
        original = {
            "02319.HK": {
                "tushare": {"data": {"ts_code": "02319.HK", "name": "蒙牛"}, "row_index": 0},
                "akshare": {"error": {"type": "ConnectError", "message": "down"}},
            },
            "fetched_at": "2026-06-23T15:30:00+08:00",
        }
        cache.set(
            site="market_data",
            query="02319.HK|akshare,tushare|peers=0",
            text=json.dumps(original, ensure_ascii=False, default=md._json_default),
        )
        hit = cache.get(site="market_data", query="02319.HK|akshare,tushare|peers=0")
        assert hit is not None
        parsed = json.loads(hit)
        assert parsed == original
