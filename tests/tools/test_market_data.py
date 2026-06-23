"""Tests for stock_analysis_agent.tools.market_data."""
from __future__ import annotations

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
    """_fetch_akshare(code) -> str using sina-backed akshare endpoints."""

    @pytest.mark.asyncio
    async def test_fetch_akshare_hk_happy_path_with_mocked_ak(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """For an HK code, _fetch_akshare calls stock_hk_spot (sina
        backend) and renders 中文名称 (not 名称 — that's the eastmoney
        column). Sina HK payload has no PE/PB/总市值 — those must NOT
        appear in the output."""
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
                }
            ]
        )

        monkeypatch.setattr(ak, "stock_hk_spot", lambda: fake_spot)

        from stock_analysis_agent.tools import market_data as md

        result = await md._fetch_akshare("02319")
        assert "[akshare]" in result
        assert "蒙牛乳业" in result
        # 15.89 is the raw value; the helper formats to .3f.
        assert "15.890" in result
        # Sina HK payload does not contain PE/PB/总市值.
        assert "PE" not in result
        assert "PB" not in result
        assert "总市值" not in result
        # English name (英文名称) must not leak into the rendered text.
        assert "MENGNIU DAIRY" not in result
        # Currency unit must be HKD for HK.
        assert "HKD" in result

    @pytest.mark.asyncio
    async def test_fetch_akshare_a_share_happy_path_with_mocked_ak(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """For an A-share code, _fetch_akshare calls stock_zh_a_spot
        (sina backend) and renders 名称 (not 中文名称). The A-share sina
        payload has no PE/PB/总市值 either."""
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

        from stock_analysis_agent.tools import market_data as md

        result = await md._fetch_akshare("sh600887")
        assert "[akshare]" in result
        assert "伊利股份" in result
        assert "24.530" in result
        # A-share sina payload has no PE/PB/总市值.
        assert "PE" not in result
        assert "PB" not in result
        assert "总市值" not in result
        # Currency unit must be CNY for A-share.
        assert "CNY" in result

    @pytest.mark.asyncio
    async def test_fetch_akshare_returns_error_segment_on_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import akshare as ak

        def _boom() -> None:
            raise RuntimeError("akshare down")

        monkeypatch.setattr(ak, "stock_hk_spot", _boom)

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
    """_fetch_and_concat aggregator + cache behavior."""

    @pytest.mark.asyncio
    async def test_concat_runs_in_parallel(self) -> None:
        """3 sources with ~100ms delay each → total < 250ms (parallel)."""
        import asyncio
        import time

        from stock_analysis_agent.tools import market_data as md

        async def _slow_tushare(*args, **kwargs):  # type: ignore[no-untyped-def]
            await asyncio.sleep(0.1)
            return "[tushare]\nok\n"

        async def _slow_akshare(*args, **kwargs):  # type: ignore[no-untyped-def]
            await asyncio.sleep(0.1)
            return "[akshare]\nok\n"

        async def _slow_mootdx(*args, **kwargs):  # type: ignore[no-untyped-def]
            await asyncio.sleep(0.1)
            return "[mootdx]\nok\n"

        original_tushare = md._fetch_tushare
        original_akshare = md._fetch_akshare
        original_mootdx = md._fetch_mootdx
        md._fetch_tushare = _slow_tushare  # type: ignore[assignment]
        md._fetch_akshare = _slow_akshare  # type: ignore[assignment]
        md._fetch_mootdx = _slow_mootdx  # type: ignore[assignment]
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
        assert "[tushare]" in result
        assert "[akshare]" in result
        assert "[mootdx]" in result

    @pytest.mark.asyncio
    async def test_concat_partial_failure_does_not_raise(self) -> None:
        """If one source fails, others still appear, no exception."""
        from stock_analysis_agent.tools import market_data as md

        async def _ok(*args, **kwargs):  # type: ignore[no-untyped-def]
            return "[tushare]\nok\n"

        async def _boom(*args, **kwargs):  # type: ignore[no-untyped-def]
            return "[akshare]\n[error: ConnectError: nope]\n"

        async def _ok2(*args, **kwargs):  # type: ignore[no-untyped-def]
            return "[mootdx]\nok\n"

        original_tushare = md._fetch_tushare
        original_akshare = md._fetch_akshare
        original_mootdx = md._fetch_mootdx
        md._fetch_tushare = _ok  # type: ignore[assignment]
        md._fetch_akshare = _boom  # type: ignore[assignment]
        md._fetch_mootdx = _ok2  # type: ignore[assignment]
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

        assert "[tushare]" in result
        assert "[mootdx]" in result
        assert "[akshare]" in result
        assert "[error:" in result

    @pytest.mark.asyncio
    async def test_concat_all_failure_raises_tool_execution_error(self) -> None:
        """When every source fails, raise ToolExecutionError so the
        retry middleware can act."""
        from stock_analysis_agent.agent.exceptions import ToolExecutionError
        from stock_analysis_agent.tools import market_data as md

        async def _boom1(*args, **kwargs):  # type: ignore[no-untyped-def]
            return "[tushare]\n[error: ConnectError: nope]\n"

        async def _boom2(*args, **kwargs):  # type: ignore[no-untyped-def]
            return "[akshare]\n[error: ConnectError: nope]\n"

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


class TestGetStockSnapshotTool:
    """The @tool _get_stock_snapshot wrapper."""

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

    @pytest.mark.asyncio
    async def test_tool_invokes_aggregator_and_returns_aggregated_text(
        self, tmp_path
    ) -> None:
        """End-to-end: tool.ainvoke calls the aggregator and returns text."""
        from stock_analysis_agent.memory import _FileCache

        async def _fake_concat(symbol, **kwargs):  # type: ignore[no-untyped-def]
            return (
                f"[tushare]\n{symbol}-tushare\n"
                f"[akshare]\n{symbol}-akshare\n"
            )

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

        assert "[tushare]" in result
        assert "[akshare]" in result
        assert "02319.HK-tushare" in result


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
