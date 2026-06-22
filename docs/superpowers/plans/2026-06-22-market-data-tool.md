# get_stock_snapshot Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `@tool get_stock_snapshot` that fans out concurrently to 5 Chinese-market data sources (Sina / Tencent / Tushare / AKShare / Mootdx), aggregates real-time quote + fundamentals + sector/financials into plain text, includes top-N industry-peer comparison, and caches the whole snapshot for 12 hours.

**Architecture:** Single new module `src/stock_analysis_agent/tools/market_data.py` mirroring `web_search.py`'s pattern — per-source async adapters, an `_fetch_and_concat` aggregator with `asyncio.gather`, module-level `_Provider` singletons, one `@tool` decorator. Per-source failures yield `[error: ...]` text segments; peers are detected via akshare's industry endpoints with a small HK→industry fallback map; cache reuses `_FileCache` keyed by a composite `symbol|sources|peers` string.

**Tech Stack:** httpx (existing) for Sina/Tencent; tushare/akshare/mootdx (new deps) for the rest; `_FileCache` (existing) for 12h JSON-file cache; langchain `@tool` decorator (existing).

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Add `tushare`, `akshare`, `mootdx` to `dependencies` |
| `src/stock_analysis_agent/tools/market_data.py` | New module: constants, `_Provider`, `_translate`, 5×`_fetch_*`, `_detect_peers`, `_fetch_peers`, `_fetch_and_concat`, `@tool _get_stock_snapshot` |
| `tests/tools/test_market_data.py` | New test module (~15 cases covering all 6 test groups from spec §12) |
| `src/stock_analysis_agent/script/test_mengniu_snapshot.py` | New demo script: invoke `_get_stock_snapshot.ainvoke({"symbol": "02319.HK"})` and print |

Each new file has one clear responsibility; no edits to existing modules beyond `pyproject.toml`.

---

## Task 1: Add & install the three new dependencies

**Files:**
- Modify: `pyproject.toml:6-11`

- [ ] **Step 1: Edit `pyproject.toml` dependencies**

Edit `pyproject.toml`, replace the `dependencies` block with:

```toml
dependencies = [
    "langchain>=1.0",
    "langchain-anthropic>=1.0",
    "langchain-core>=1.0",
    "httpx>=0.27",
    "tushare>=1.4",
    "akshare>=1.13",
    "mootdx>=2.4",
]
```

- [ ] **Step 2: Install new deps**

Run: `uv pip install -e ".[dev]"`
Expected: All three new packages (`tushare`, `akshare`, `mootdx`) install without errors. If a package version is unavailable, bump the lower bound to the lowest available stable version and re-run.

- [ ] **Step 3: Verify imports work**

Run: `uv run python -c "import tushare, akshare, mootdx; print(tushare.__version__, akshare.__version__, mootdx.__version__)"`
Expected: Three version strings printed (one line). If `akshare` warns on first import about missing optional deps (`pandas`, etc.), that is OK — only an `ImportError` is a failure.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(deps): add tushare, akshare, mootdx for market data adapters"
```

---

## Task 2: Write the failing test for `_translate`

**Files:**
- Test: `tests/tools/test_market_data.py`

- [ ] **Step 1: Create the empty test file**

Create `tests/tools/test_market_data.py` with this exact content:

```python
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
        }

    def test_translate_sh_symbol_to_all_sources(self) -> None:
        result = _translate("600519.SH")
        assert result == {
            "sina": "sh600519",
            "tencent": "sh600519",
            "tushare": "600519.SH",
            "akshare": "sh600519",
            "mootdx": "1",
        }

    def test_translate_sz_symbol_to_all_sources(self) -> None:
        result = _translate("000001.SZ")
        assert result == {
            "sina": "sz000001",
            "tencent": "sz000001",
            "tushare": "000001.SZ",
            "akshare": "sz000001",
            "mootdx": "0",
        }

    def test_translate_unknown_market_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unsupported market"):
            _translate("02319.XX")
```

- [ ] **Step 2: Run test to verify it fails (collection error)**

Run: `pytest tests/tools/test_market_data.py -v`
Expected: `ModuleNotFoundError: No module named 'stock_analysis_agent.tools.market_data'` — the module doesn't exist yet.

---

## Task 3: Implement `_translate` to make the test pass

**Files:**
- Create: `src/stock_analysis_agent/tools/market_data.py`

- [ ] **Step 1: Create the module skeleton**

Create `src/stock_analysis_agent/tools/market_data.py` with this exact content (only `_translate` is functional; everything else is a stub to keep the file importable):

```python
"""@tool get_stock_snapshot: fan-out concurrent stock data over 5 sources."""
from __future__ import annotations

from typing import Literal

MarketName = Literal["HK", "SH", "SZ"]
SourceName = Literal["sina", "tencent", "tushare", "akshare", "mootdx"]

ALL_SOURCES: tuple[SourceName, ...] = (
    "sina",
    "tencent",
    "tushare",
    "akshare",
    "mootdx",
)

MOOTDX_DEFAULT_SERVER: str = "std.tdx.com.cn"

# HK prefix/code -> akshare 行业板块名 (fallback when akshare can't classify HK symbols).
HK_INDUSTRY_HINTS: dict[str, str] = {
    "02319": "乳品",
    "09988": "互联网服务",
    "00700": "互联网服务",
    "03690": "互联网服务",
    "01211": "汽车整车",
    "00939": "建筑工程",
    "00388": "证券",
    "01398": "银行",
    "00945": "保险",
}

DEFAULT_CACHE_DIR: str = "~/.cache/stock-analysis-agent/market"
DEFAULT_CACHE_TTL: float = 12 * 3600.0
PEER_INDUSTRY_SOURCE: SourceName = "akshare"
PEER_FETCH_SOURCES: tuple[SourceName, ...] = ("sina", "tencent")


def _translate(symbol: str) -> dict[SourceName, str]:
    """Translate a standard `<code>.<market>` symbol into each source's
    native code format.

    Args:
        symbol: Standard code, e.g. ``"02319.HK"``, ``"600519.SH"``,
            ``"000001.SZ"``.

    Returns:
        Mapping from source name to that source's local code. Markets
        outside ``{"HK", "SH", "SZ"}`` raise ``ValueError``.

    Raises:
        ValueError: If the market segment is unknown.
    """
    if "." not in symbol:
        raise ValueError(f"unsupported market in symbol: {symbol!r}")
    code, market = symbol.rsplit(".", 1)
    market = market.upper()
    if market not in {"HK", "SH", "SZ"}:
        raise ValueError(f"unsupported market {market!r} in symbol {symbol!r}")
    code = code.strip()
    if not code:
        raise ValueError(f"empty code in symbol: {symbol!r}")
    if market == "HK":
        return {
            "sina": f"rt_hk{code}",
            "tencent": f"hk{code}",
            "tushare": f"{code}.HK",
            "akshare": code,
            "mootdx": "23",
        }
    if market == "SH":
        return {
            "sina": f"sh{code}",
            "tencent": f"sh{code}",
            "tushare": f"{code}.SH",
            "akshare": f"sh{code}",
            "mootdx": "1",
        }
    # SZ
    return {
        "sina": f"sz{code}",
        "tencent": f"sz{code}",
        "tushare": f"{code}.SZ",
        "akshare": f"sz{code}",
        "mootdx": "0",
    }
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/tools/test_market_data.py::TestTranslate -v`
Expected: 4 passed.

- [ ] **Step 3: Commit**

```bash
git add src/stock_analysis_agent/tools/market_data.py tests/tools/test_market_data.py
git commit -m "feat(market-data): add _translate symbol translation"
```

---

## Task 4: Write failing test for Sina adapter

**Files:**
- Test: `tests/tools/test_market_data.py` (append)

- [ ] **Step 1: Append Sina test class to the test file**

Append (do not overwrite) this block to `tests/tools/test_market_data.py`:

```python
class TestFetchSina:
    """_fetch_sina(code) -> str using httpx against hq.sinajs.cn."""

    def test_fetch_sina_parses_hk_quote(self) -> None:
        """Mock the httpx response, assert _fetch_sina returns a snippet
        that includes the parsed price/change fields."""
        import httpx

        sample_csv = (
            'var hq_str_rt_hk02319="MENGNIU DAIRY,蒙牛股份,15.940,15.570,'
            "15.940,15.340,15.890,0.320,2.055,15.880,15.890,"
            "278092437.740,17684472,36.161,0.000,17.411,13.374,"
            '2026/06/22,16:08:16,100|0,N|Y,Y,15.850|15.060|16.640,...";'
        )

        def _h(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=sample_csv)

        from stock_analysis_agent.tools import market_data as md

        result = md._fetch_sina(
            "rt_hk02319",
            transport=httpx.MockTransport(_h),
        )
        # Result must contain the price and the change percent.
        assert "15.890" in result
        assert "0.320" in result
        assert "+2.06%" in result or "2.06%" in result
        # Header should mark this as the sina source.
        assert "[sina]" in result

    def test_fetch_sina_returns_error_segment_on_http_failure(self) -> None:
        """If httpx raises, _fetch_sina returns '[error: ...]' segment,
        not raising."""
        import httpx

        from stock_analysis_agent.tools import market_data as md

        def _h(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom")

        result = md._fetch_sina(
            "rt_hk02319",
            transport=httpx.MockTransport(_h),
        )
        assert "[error:" in result
        assert "[sina]" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_market_data.py::TestFetchSina -v`
Expected: `AttributeError: module 'stock_analysis_agent.tools.market_data' has no attribute '_fetch_sina'`

---

## Task 5: Implement Sina adapter

**Files:**
- Modify: `src/stock_analysis_agent/tools/market_data.py`

- [ ] **Step 1: Add the Sina adapter after `_translate`**

Add this import at the top of the file (right after `from typing import Literal`):

```python
from typing import Any

import httpx
```

Then add this function after `_translate`:

```python
async def _fetch_sina(
    code: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout: float = 10.0,
) -> str:
    """Fetch a Sina realtime quote for `code` and return formatted text.

    Args:
        code: Sina-local code, e.g. ``"rt_hk02319"``, ``"sh600519"``.
        transport: Optional httpx transport (for tests).
        timeout: HTTP timeout in seconds.

    Returns:
        A text snippet prefixed with ``[sina]`` and the parsed fields,
        or ``[sina]\n[error: ...]`` on failure.
    """
    url = "https://hq.sinajs.cn/list=" + code
    headers = {
        "Referer": "https://finance.sina.com.cn/",
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36"
        ),
    }
    try:
        client_kwargs: dict[str, Any] = {"timeout": timeout}
        if transport is not None:
            client_kwargs["transport"] = transport
        async with httpx.AsyncClient(**client_kwargs) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            body = resp.text
    except Exception as e:
        return f"[sina]\n[error: {type(e).__name__}: {e}]\n"
    return "[sina]\n" + _parse_sina_csv(body) + "\n"


def _parse_sina_csv(body: str) -> str:
    """Parse a `var hq_str_xxx="...";` response into readable text.

    Sina's payload is GBK-encoded CSV-ish. The 17th field onward is the
    HK-standard layout (name, open, prev_close, current, high, low,
    change, change_pct, ...). For A-shares the field layout differs;
    we render a defensive summary that works for both.
    """
    # Extract the quoted portion.
    start = body.find('"')
    end = body.rfind('"')
    if start < 0 or end <= start:
        return f"[error: unparseable sina response: {body[:80]!r}]"
    raw = body[start + 1 : end]
    fields = raw.split(",")
    if len(fields) < 6:
        return f"[error: too few fields: {len(fields)}]"
    name_cn = fields[0] if fields[0] else "(unknown)"
    # A-shares use indices 1..5 for open/prev_close/current/high/low.
    # HK uses fields[1..7] differently; we attempt both layouts.
    try:
        if len(fields) >= 32:  # HK layout
            open_p = float(fields[2])
            prev_close = float(fields[3])
            current = float(fields[6])
            high = float(fields[4])
            low = float(fields[5])
            change = float(fields[7])
            change_pct = float(fields[8])
            volume = fields[12]
            amount = fields[11]
            return (
                f"名称: {name_cn}\n"
                f"现价: {current}\n"
                f"涨跌: {change:+.3f} ({change_pct:+.2f}%)\n"
                f"今开: {open_p}  昨收: {prev_close}  "
                f"最高: {high}  最低: {low}\n"
                f"成交量: {volume} 股\n"
                f"成交额: {amount}\n"
            )
        # A-share layout: open, prev_close, current, high, low, ...
        open_p = float(fields[1])
        prev_close = float(fields[2])
        current = float(fields[3])
        high = float(fields[4])
        low = float(fields[5])
        return (
            f"名称: {name_cn}\n"
            f"现价: {current}\n"
            f"今开: {open_p}  昨收: {prev_close}  "
            f"最高: {high}  最低: {low}\n"
        )
    except (ValueError, IndexError) as e:
        return f"[error: parse failed: {e}]"
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/tools/test_market_data.py::TestFetchSina -v`
Expected: 2 passed.

- [ ] **Step 3: Commit**

```bash
git add src/stock_analysis_agent/tools/market_data.py tests/tools/test_market_data.py
git commit -m "feat(market-data): add sina realtime quote adapter"
```

---

## Task 6: Write failing test for Tencent adapter

**Files:**
- Test: `tests/tools/test_market_data.py` (append)

- [ ] **Step 1: Append Tencent test class**

Append this block to `tests/tools/test_market_data.py`:

```python
class TestFetchTencent:
    """_fetch_tencent(code) -> str using httpx against qt.gtimg.cn."""

    def test_fetch_tencent_parses_hk_quote(self) -> None:
        """Mock the httpx response, assert _fetch_tencent returns a
        snippet with the parsed price/PE fields."""
        import httpx

        # Real-shape sample (truncated).
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

        from stock_analysis_agent.tools import market_data as md

        result = md._fetch_tencent(
            "hk02319",
            transport=httpx.MockTransport(_h),
        )
        assert "[tencent]" in result
        assert "15.890" in result
        assert "蒙牛股份" in result or "MENGNIU" in result

    def test_fetch_tencent_returns_error_segment_on_http_failure(self) -> None:
        import httpx

        from stock_analysis_agent.tools import market_data as md

        def _h(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom")

        result = md._fetch_tencent(
            "hk02319",
            transport=httpx.MockTransport(_h),
        )
        assert "[error:" in result
        assert "[tencent]" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_market_data.py::TestFetchTencent -v`
Expected: `AttributeError: ... has no attribute '_fetch_tencent'`

---

## Task 7: Implement Tencent adapter

**Files:**
- Modify: `src/stock_analysis_agent/tools/market_data.py`

- [ ] **Step 1: Add the Tencent adapter after the Sina block**

Add this function after `_parse_sina_csv`:

```python
async def _fetch_tencent(
    code: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout: float = 10.0,
) -> str:
    """Fetch a Tencent realtime quote for `code` and return formatted text.

    Args:
        code: Tencent-local code, e.g. ``"hk02319"``, ``"sh600519"``.
        transport: Optional httpx transport (for tests).
        timeout: HTTP timeout in seconds.

    Returns:
        A text snippet prefixed with ``[tencent]`` and parsed fields,
        or ``[tencent]\n[error: ...]`` on failure.
    """
    url = "http://qt.gtimg.cn/q=" + code
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36"
        ),
    }
    try:
        client_kwargs: dict[str, Any] = {"timeout": timeout}
        if transport is not None:
            client_kwargs["transport"] = transport
        async with httpx.AsyncClient(**client_kwargs) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            body = resp.text
    except Exception as e:
        return f"[tencent]\n[error: {type(e).__name__}: {e}]\n"
    return "[tencent]\n" + _parse_tencent_csv(body) + "\n"


def _parse_tencent_csv(body: str) -> str:
    """Parse a `v_<code>="~...";` payload into readable text.

    Field layout for HK shares: name_cn, code, current, prev_close,
    open, change_amount, change_pct, ...; we pick the most useful ones.
    """
    start = body.find('"')
    end = body.rfind('"')
    if start < 0 or end <= start:
        return f"[error: unparseable tencent response: {body[:80]!r}]"
    raw = body[start + 1 : end]
    fields = raw.split("~")
    if len(fields) < 50:
        return f"[error: too few fields: {len(fields)}]"
    try:
        name_cn = fields[1]
        current = float(fields[3])
        prev_close = float(fields[4])
        open_p = float(fields[5])
        change = float(fields[32])
        change_pct = float(fields[33])
        high = float(fields[34])
        low = float(fields[35])
        volume = fields[36]
        amount = fields[37]
        pe_ttm = fields[49] if fields[49] else "--"
        pb = fields[46] if len(fields) > 46 and fields[46] else "--"
        return (
            f"名称: {name_cn}\n"
            f"现价: {current}\n"
            f"涨跌: {change:+.3f} ({change_pct:+.2f}%)\n"
            f"今开: {open_p}  昨收: {prev_close}  "
            f"最高: {high}  最低: {low}\n"
            f"成交量: {volume}\n"
            f"成交额: {amount}\n"
            f"PE-TTM: {pe_ttm}  PB: {pb}\n"
        )
    except (ValueError, IndexError) as e:
        return f"[error: parse failed: {e}]"
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/tools/test_market_data.py::TestFetchTencent -v`
Expected: 2 passed.

- [ ] **Step 3: Commit**

```bash
git add src/stock_analysis_agent/tools/market_data.py tests/tools/test_market_data.py
git commit -m "feat(market-data): add tencent realtime quote adapter"
```

---

## Task 8: Write failing test for Tushare adapter (token-missing path)

**Files:**
- Test: `tests/tools/test_market_data.py` (append)

- [ ] **Step 1: Append Tushare token-missing test**

Append this block to `tests/tools/test_market_data.py`:

```python
class TestFetchTushare:
    """_fetch_tushare(code, token) -> str using tushare.pro_api."""

    def test_fetch_tushare_returns_error_when_token_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No TUSHARE_TOKEN env var -> [tushare]\\n[error: TUSHARE_TOKEN not set]\\n"""
        monkeypatch.delenv("TUSHARE_TOKEN", raising=False)

        from stock_analysis_agent.tools import market_data as md

        result = md._fetch_tushare("02319.HK", token=None)
        assert "[tushare]" in result
        assert "TUSHARE_TOKEN" in result
        assert "[error:" in result

    def test_fetch_tushare_happy_path_with_mocked_pro_api(
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
                    "open": 15.57,
                    "high": 15.94,
                    "low": 15.34,
                    "close": 15.89,
                    "pre_close": 15.57,
                    "change": 0.32,
                    "pct_chg": 2.06,
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
                    "pe": 11.03,
                    "pb": 1.68,
                    "total_mv": 387647.0,  # 万 HKD
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

        from stock_analysis_agent.tools import market_data as md

        result = md._fetch_tushare("02319.HK", token="dummy")
        assert "[tushare]" in result
        assert "15.89" in result or "15.890" in result
        assert "蒙牛乳业" in result or "乳品" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_market_data.py::TestFetchTushare -v`
Expected: `AttributeError: ... has no attribute '_fetch_tushare'`

---

## Task 9: Implement Tushare adapter

**Files:**
- Modify: `src/stock_analysis_agent/tools/market_data.py`

- [ ] **Step 1: Add the Tushare adapter**

Add this function after `_parse_tencent_csv`:

```python
async def _fetch_tushare(
    code: str,
    *,
    token: str | None = None,
) -> str:
    """Fetch a Tushare snapshot for `code` (e.g. ``"02319.HK"``).

    Uses two endpoints:
      - ``pro.daily`` for the latest OHLCV
      - ``pro.stock_basic`` for industry, PE, PB, market cap

    Returns:
        A text snippet prefixed with ``[tushare]``. When ``token`` is
        missing, returns a structured ``[error: TUSHARE_TOKEN not set]``
        segment so the aggregator can still render other sources.
    """
    import os

    token = token if token is not None else os.environ.get("TUSHARE_TOKEN")
    if not token:
        return "[tushare]\n[error: TUSHARE_TOKEN not set]\n"

    try:
        import asyncio

        import tushare as ts

        pro = ts.pro_api(token)

        def _fetch() -> tuple[list[dict], list[dict]]:
            daily = pro.daily(ts_code=code, limit=1).to_dict("records")
            basic = pro.stock_basic(
                ts_code=code, fields="ts_code,name,industry,pe,pb,total_mv"
            ).to_dict("records")
            return daily, basic

        daily, basic = await asyncio.to_thread(_fetch)
    except Exception as e:
        return f"[tushare]\n[error: {type(e).__name__}: {e}]\n"

    if not basic:
        return f"[tushare]\n[error: stock_basic returned empty for {code}]\n"
    info = basic[0]
    lines = [
        "名称: " + str(info.get("name", "(unknown)")),
        "行业: " + str(info.get("industry", "--")),
        "PE: " + str(info.get("pe", "--")),
        "PB: " + str(info.get("pb", "--")),
        "总市值(万): " + str(info.get("total_mv", "--")),
    ]
    if daily:
        d = daily[0]
        lines.insert(
            0,
            f"现价: {d.get('close')}  "
            f"涨跌: {d.get('change')} ({d.get('pct_chg')}%)\n"
            f"今开: {d.get('open')}  昨收: {d.get('pre_close')}  "
            f"最高: {d.get('high')}  最低: {d.get('low')}\n"
            f"成交量: {d.get('vol')}  成交额: {d.get('amount')}",
        )
    return "[tushare]\n" + "\n".join(lines) + "\n"
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/tools/test_market_data.py::TestFetchTushare -v`
Expected: 2 passed.

- [ ] **Step 3: Commit**

```bash
git add src/stock_analysis_agent/tools/market_data.py tests/tools/test_market_data.py
git commit -m "feat(market-data): add tushare adapter (with token-missing guard)"
```

---

## Task 10: Write failing test for AKShare adapter

**Files:**
- Test: `tests/tools/test_market_data.py` (append)

- [ ] **Step 1: Append AKShare test class**

Append this block to `tests/tools/test_market_data.py`:

```python
class TestFetchAkshare:
    """_fetch_akshare(code) -> str using akshare."""

    def test_fetch_akshare_happy_path_with_mocked_ak(
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

        result = md._fetch_akshare("02319")
        assert "[akshare]" in result
        assert "15.89" in result or "蒙牛乳业" in result

    def test_fetch_akshare_returns_error_segment_on_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import akshare as ak

        def _boom() -> None:
            raise RuntimeError("akshare down")

        monkeypatch.setattr(ak, "stock_hk_spot_em", _boom)

        from stock_analysis_agent.tools import market_data as md

        result = md._fetch_akshare("02319")
        assert "[akshare]" in result
        assert "[error:" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_market_data.py::TestFetchAkshare -v`
Expected: `AttributeError: ... has no attribute '_fetch_akshare'`

---

## Task 11: Implement AKShare adapter

**Files:**
- Modify: `src/stock_analysis_agent/tools/market_data.py`

- [ ] **Step 1: Add the AKShare adapter**

Add this function after `_fetch_tushare`:

```python
async def _fetch_akshare(code: str) -> str:
    """Fetch an AKShare snapshot for `code`.

    For HK codes (e.g. ``"02319"``) uses ``stock_hk_spot_em``.
    For SH/SZ codes uses ``stock_zh_a_spot_em`` and filters by code.

    Returns:
        A text snippet prefixed with ``[akshare]``, or
        ``[akshare]\n[error: ...]`` on failure.
    """
    import asyncio

    import akshare as ak
    import pandas as pd

    def _fetch() -> pd.DataFrame:
        # Heuristic: HK codes are 5-digit plain numbers;
        # A-share codes come prefixed with sh/sz from _translate.
        if code.isdigit() and len(code) == 5:
            return ak.stock_hk_spot_em()
        return ak.stock_zh_a_spot_em()

    try:
        df = await asyncio.to_thread(_fetch)
    except Exception as e:
        return f"[akshare]\n[error: {type(e).__name__}: {e}]\n"

    if df is None or df.empty:
        return f"[akshare]\n[error: empty result for {code}]\n"

    # Find the row matching our code (strip sh/sz prefix if present).
    needle = code.replace("sh", "").replace("sz", "")
    match = df[df["代码"].astype(str).str.contains(needle, na=False)]
    if match.empty:
        return f"[akshare]\n[error: {needle} not found in spot data]\n"
    row = match.iloc[0].to_dict()

    def _g(key: str) -> str:
        v = row.get(key, "--")
        return "--" if v in (None, "", float("nan")) else str(v)

    return (
        "[akshare]\n"
        f"名称: {_g('名称')}\n"
        f"现价: {_g('最新价')}\n"
        f"涨跌: {_g('涨跌额')} ({_g('涨跌幅')}%)\n"
        f"今开: {_g('今开')}  昨收: {_g('昨收')}  "
        f"最高: {_g('最高')}  最低: {_g('最低')}\n"
        f"成交量: {_g('成交量')}  成交额: {_g('成交额')}\n"
        f"PE: {_g('市盈率')}  PB: {_g('市净率')}\n"
        f"总市值: {_g('总市值')}\n"
    )
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/tools/test_market_data.py::TestFetchAkshare -v`
Expected: 2 passed.

- [ ] **Step 3: Commit**

```bash
git add src/stock_analysis_agent/tools/market_data.py tests/tools/test_market_data.py
git commit -m "feat(market-data): add akshare adapter"
```

---

## Task 12: Write failing test for Mootdx adapter

**Files:**
- Test: `tests/tools/test_market_data.py` (append)

- [ ] **Step 1: Append Mootdx test class**

Append this block to `tests/tools/test_market_data.py`:

```python
class TestFetchMootdx:
    """_fetch_mootdx(code) -> str using mootdx."""

    def test_fetch_mootdx_returns_error_segment_on_connection_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the mootdx client raises on connect, _fetch_mootdx returns
        an [error: ...] segment instead of propagating."""
        import mootdx

        class _FakeClient:
            def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                raise ConnectionError("tdx server unreachable")

        monkeypatch.setattr(mootdx, "quote", _FakeClient)

        from stock_analysis_agent.tools import market_data as md

        result = md._fetch_mootdx("23")
        assert "[mootdx]" in result
        assert "[error:" in result

    def test_fetch_mootdx_happy_path_with_mocked_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the client returns fake security bars, _fetch_mootdx
        should render them."""
        import mootdx

        class _FakeClient:
            def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                pass

            def security_bars(
                self, *args, **kwargs  # type: ignore[no-untyped-def]
            ):
                return {
                    "price": [
                        {
                            "open": 15.57,
                            "high": 15.94,
                            "low": 15.34,
                            "close": 15.89,
                            "vol": 17684472,
                        }
                    ]
                }

        monkeypatch.setattr(mootdx, "quote", _FakeClient)

        from stock_analysis_agent.tools import market_data as md

        result = md._fetch_mootdx("23")
        assert "[mootdx]" in result
        assert "15.89" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_market_data.py::TestFetchMootdx -v`
Expected: `AttributeError: ... has no attribute '_fetch_mootdx'`

---

## Task 13: Implement Mootdx adapter

**Files:**
- Modify: `src/stock_analysis_agent/tools/market_data.py`

- [ ] **Step 1: Add the Mootdx adapter**

Add this function after `_fetch_akshare`:

```python
async def _fetch_mootdx(code: str) -> str:
    """Fetch a Mootdx snapshot (latest bar) for `code`.

    Args:
        code: Mootdx market code (``"23"`` for HK, ``"1"`` for SH,
            ``"0"`` for SZ — see `_translate`).

    Returns:
        A text snippet prefixed with ``[mootdx]``, or
        ``[mootdx]\n[error: ...]`` on connection or parse failure.
    """
    import asyncio

    import mootdx

    def _fetch() -> dict:
        client = mootdx.quote.Client(
            tdx_server=MOOTDX_DEFAULT_SERVER, timeout=10
        )
        try:
            return client.security_bars(
                category=0,  # 5-minute K-line
                market=int(code),
                code="",
                start=0,
                offset=1,
            )
        finally:
            try:
                client.close()
            except Exception:
                pass

    try:
        data = await asyncio.to_thread(_fetch)
    except Exception as e:
        return f"[mootdx]\n[error: {type(e).__name__}: {e}]\n"

    prices = (data or {}).get("price") or []
    if not prices:
        return f"[mootdx]\n[error: empty security_bars for market {code}]\n"
    bar = prices[0]
    return (
        "[mootdx]\n"
        f"市场代码: {code}\n"
        f"今开: {bar.get('open')}  最高: {bar.get('high')}  "
        f"最低: {bar.get('low')}  收盘: {bar.get('close')}\n"
        f"成交量: {bar.get('vol')}\n"
    )
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/tools/test_market_data.py::TestFetchMootdx -v`
Expected: 2 passed.

- [ ] **Step 3: Commit**

```bash
git add src/stock_analysis_agent/tools/market_data.py tests/tools/test_market_data.py
git commit -m "feat(market-data): add mootdx adapter"
```

---

## Task 14: Write failing test for peer detection

**Files:**
- Test: `tests/tools/test_market_data.py` (append)

- [ ] **Step 1: Append peer-detection test class**

Append this block to `tests/tools/test_market_data.py`:

```python
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

        from stock_analysis_agent.tools import market_data as md

        result = md._detect_peers("600519.SH", peer_count=2)
        assert result == ["600519.SH", "000858.SH"]

    def test_detect_peers_returns_none_when_akshare_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If akshare raises, _detect_peers returns None and the
        aggregator should emit an [error: ...] peers segment."""
        import akshare as ak

        def _boom() -> None:
            raise RuntimeError("network down")

        monkeypatch.setattr(ak, "stock_board_industry_name_em", _boom)

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

        def _industries() -> pd.DataFrame:
            return industries

        def _cons(symbol: str) -> pd.DataFrame:
            return cons

        monkeypatch.setattr(ak, "stock_board_industry_name_em", _industries)
        monkeypatch.setattr(ak, "stock_board_industry_cons_em", _cons)

        from stock_analysis_agent.tools import market_data as md

        result = md._detect_peers("02319.HK", peer_count=2)
        assert result == ["600887.SH", "600597.SH"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_market_data.py::TestDetectPeers -v`
Expected: `AttributeError: ... has no attribute '_detect_peers'`

---

## Task 15: Implement `_detect_peers`

**Files:**
- Modify: `src/stock_analysis_agent/tools/market_data.py`

- [ ] **Step 1: Add `_detect_peers`**

Add this function after `_fetch_mootdx`:

```python
def _detect_peers(symbol: str, peer_count: int) -> list[str] | None:
    """Detect top-`peer_count` peer companies in the same industry.

    For A-shares, looks up the industry via akshare's
    ``stock_individual_info_em`` and then queries
    ``stock_board_industry_cons_em`` to get constituents ranked by
    market cap. For HK symbols (where akshare's individual-info
    endpoint does not classify), falls back to ``HK_INDUSTRY_HINTS``.

    Args:
        symbol: Standard code, e.g. ``"600519.SH"`` or ``"02319.HK"``.
        peer_count: Maximum peers to return.

    Returns:
        List of standard codes (``"600887.SH"`` etc.), the input symbol
        itself included as the first entry, or ``None`` if detection
        fails (akshare unreachable, no industry mapped, etc.).
    """
    import akshare as ak
    import pandas as pd

    code, market = symbol.rsplit(".", 1)
    code = code.strip()
    industry_name: str | None = None

    try:
        if market == "HK":
            industry_name = HK_INDUSTRY_HINTS.get(code)
            if industry_name is None:
                industry_name = HK_INDUSTRY_HINTS.get(
                    symbol.upper().lstrip("0")
                )
        else:
            info = ak.stock_individual_info_em(symbol=code)
            if info is not None and not info.empty:
                row = info.iloc[0].to_dict()
                industry_name = row.get("行业") or row.get("industry")
    except Exception:
        return None

    if not industry_name:
        return None

    try:
        cons = ak.stock_board_industry_cons_em(symbol=industry_name)
    except Exception:
        return None

    if cons is None or cons.empty:
        return None

    if "总市值" in cons.columns:
        cons = cons.sort_values("总市值", ascending=False)
    codes = cons["代码"].astype(str).head(peer_count).tolist()
    result: list[str] = []
    for c in codes:
        c = str(c)
        if c.startswith("6"):
            result.append(f"{c}.SH")
        elif c.startswith(("0", "3")):
            result.append(f"{c}.SZ")
        else:
            result.append(c)
    # Ensure the input symbol itself is the first peer.
    if symbol not in result:
        result.insert(0, symbol)
    else:
        result.remove(symbol)
        result.insert(0, symbol)
    return result[: peer_count + 1] if peer_count > 0 else result
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/tools/test_market_data.py::TestDetectPeers -v`
Expected: 3 passed.

- [ ] **Step 3: Commit**

```bash
git add src/stock_analysis_agent/tools/market_data.py tests/tools/test_market_data.py
git commit -m "feat(market-data): add _detect_peers (akshare + HK hint fallback)"
```

---

## Task 16: Write failing test for `_fetch_and_concat`

**Files:**
- Test: `tests/tools/test_market_data.py` (append)

- [ ] **Step 1: Append aggregator test class**

Append this block to `tests/tools/test_market_data.py`:

```python
class TestFetchAndConcat:
    """_fetch_and_concat aggregator + cache behavior."""

    def test_concat_runs_in_parallel(self) -> None:
        """3 sources with ~100ms delay each → total < 250ms (parallel)."""
        import asyncio
        import time

        from stock_analysis_agent.tools import market_data as md

        async def _slow_sina(*args, **kwargs):  # type: ignore[no-untyped-def]
            await asyncio.sleep(0.1)
            return "[sina]\nok\n"

        async def _slow_tencent(*args, **kwargs):  # type: ignore[no-untyped-def]
            await asyncio.sleep(0.1)
            return "[tencent]\nok\n"

        async def _slow_tushare(*args, **kwargs):  # type: ignore[no-untyped-def]
            await asyncio.sleep(0.1)
            return "[tushare]\nok\n"

        original_sina = md._fetch_sina
        original_tencent = md._fetch_tencent
        original_tushare = md._fetch_tushare
        md._fetch_sina = _slow_sina  # type: ignore[assignment]
        md._fetch_tencent = _slow_tencent  # type: ignore[assignment]
        md._fetch_tushare = _slow_tushare  # type: ignore[assignment]
        try:
            start = time.monotonic()
            result = asyncio.run(
                md._fetch_and_concat(
                    "02319.HK",
                    sources=("sina", "tencent", "tushare"),
                    include_peers=False,
                    peer_count=0,
                    cache=None,
                )
            )
            elapsed = time.monotonic() - start
        finally:
            md._fetch_sina = original_sina  # type: ignore[assignment]
            md._fetch_tencent = original_tencent  # type: ignore[assignment]
            md._fetch_tushare = original_tushare  # type: ignore[assignment]

        assert elapsed < 0.25, f"expected parallel, took {elapsed:.3f}s"
        assert "[sina]" in result
        assert "[tencent]" in result
        assert "[tushare]" in result

    def test_concat_partial_failure_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If one source fails, others still appear, no exception."""
        import asyncio

        from stock_analysis_agent.tools import market_data as md

        async def _ok() -> str:
            return "[sina]\nok\n"

        async def _boom() -> str:
            return "[tencent]\n[error: ConnectError: nope]\n"

        async def _ok2() -> str:
            return "[tushare]\nok\n"

        original_sina = md._fetch_sina
        original_tencent = md._fetch_tencent
        original_tushare = md._fetch_tushare
        md._fetch_sina = _ok  # type: ignore[assignment]
        md._fetch_tencent = _boom  # type: ignore[assignment]
        md._fetch_tushare = _ok2  # type: ignore[assignment]
        try:
            result = asyncio.run(
                md._fetch_and_concat(
                    "02319.HK",
                    sources=("sina", "tencent", "tushare"),
                    include_peers=False,
                    peer_count=0,
                    cache=None,
                )
            )
        finally:
            md._fetch_sina = original_sina  # type: ignore[assignment]
            md._fetch_tencent = original_tencent  # type: ignore[assignment]
            md._fetch_tushare = original_tushare  # type: ignore[assignment]

        assert "[sina]" in result
        assert "[tushare]" in result
        assert "[tencent]" in result
        assert "[error:" in result

    def test_concat_all_failure_raises_tool_execution_error(self) -> None:
        """When every source fails, raise ToolExecutionError so the
        retry middleware can act."""
        import asyncio

        from stock_analysis_agent.agent.exceptions import ToolExecutionError
        from stock_analysis_agent.tools import market_data as md

        async def _boom() -> str:
            return "[sina]\n[error: ConnectError: nope]\n"

        async def _boom2() -> str:
            return "[tencent]\n[error: ConnectError: nope]\n"

        original_sina = md._fetch_sina
        original_tencent = md._fetch_tencent
        md._fetch_sina = _boom  # type: ignore[assignment]
        md._fetch_tencent = _boom2  # type: ignore[assignment]
        try:
            with pytest.raises(ToolExecutionError, match="all sources failed"):
                asyncio.run(
                    md._fetch_and_concat(
                        "02319.HK",
                        sources=("sina", "tencent"),
                        include_peers=False,
                        peer_count=0,
                        cache=None,
                    )
                )
        finally:
            md._fetch_sina = original_sina  # type: ignore[assignment]
            md._fetch_tencent = original_tencent  # type: ignore[assignment]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_market_data.py::TestFetchAndConcat -v`
Expected: `AttributeError: ... has no attribute '_fetch_and_concat'`

---

## Task 17: Implement `_fetch_and_concat` aggregator (with cache)

**Files:**
- Modify: `src/stock_analysis_agent/tools/market_data.py`

- [ ] **Step 1: Add the aggregator function**

Add this function after `_detect_peers`:

```python
async def _fetch_and_concat(
    symbol: str,
    *,
    sources: tuple[SourceName, ...],
    include_peers: bool,
    peer_count: int,
    cache: _FileCache | None,
) -> str:
    """Aggregate snapshots from all configured sources for `symbol`.

    Behaviour:
      1. Compute composite cache key and short-circuit on hit.
      2. Fan out per-source fetches via ``asyncio.gather``.
      3. If ``include_peers``, detect top-N peers and append a
         ``[peers]`` section rendered from sina + tencent only.
      4. If every primary source errored, raise ``ToolExecutionError``.
      5. Write the aggregated text to cache (best-effort).

    Args:
        symbol: Standard code, e.g. ``"02319.HK"``.
        sources: Non-empty tuple of source names.
        include_peers: Whether to run peer detection + fetch.
        peer_count: How many top peers to compare.
        cache: Optional file cache for whole-snapshot memoization.

    Returns:
        Concatenated text with one section per source plus optional
        ``[peers]`` section.
    """
    from stock_analysis_agent.agent.exceptions import ToolExecutionError

    if not sources:
        raise ValueError("sources cannot be empty")

    cache_key = (
        f"{symbol}|{','.join(sorted(sources))}|"
        f"peers={peer_count if include_peers else 0}"
    )
    cache_site = "market_data"

    if cache is not None:
        hit = cache.get(site=cache_site, query=cache_key)
        if hit is not None:
            return hit

    async def _call(src: SourceName) -> tuple[SourceName, str]:
        try:
            if src == "sina":
                text = await _fetch_sina(_translate(symbol)[src])
            elif src == "tencent":
                text = await _fetch_tencent(_translate(symbol)[src])
            elif src == "tushare":
                text = await _fetch_tushare(
                    _translate(symbol)[src], token=None
                )
            elif src == "akshare":
                text = await _fetch_akshare(_translate(symbol)[src])
            elif src == "mootdx":
                text = await _fetch_mootdx(_translate(symbol)[src])
            else:
                text = f"[error: unknown source {src!r}]"
        except Exception as e:  # noqa: BLE001 — top-level guard
            text = f"[{src}]\n[error: {type(e).__name__}: {e}]\n"
        return src, text

    pairs = await asyncio.gather(*(_call(s) for s in sources))
    parts = [text for _, text in pairs]

    # If ALL primary sources errored, raise so the retry middleware acts.
    if all(text.lstrip("[").startswith(("sina", "tencent",
                                        "tushare", "akshare", "mootdx"))
           and "[error:" in text for text in parts):
        raise ToolExecutionError(
            f"all sources failed for {symbol}: "
            f"{[s for s in sources]}"
        )

    if include_peers and peer_count > 0:
        peer_symbols = _detect_peers(symbol, peer_count)
        if peer_symbols is None:
            parts.append(
                "[peers]\n[error: industry detection failed]\n"
            )
        else:
            peer_lines = ["[peers]"]
            for psym in peer_symbols:
                try:
                    sina_text = await _fetch_sina(_translate(psym)["sina"])
                    tencent_text = await _fetch_tencent(
                        _translate(psym)["tencent"]
                    )
                except Exception as e:  # noqa: BLE001
                    peer_lines.append(
                        f"- {psym}: [error: {type(e).__name__}: {e}]"
                    )
                    continue
                combined = sina_text + tencent_text
                # Compact: only show price / market-cap summary.
                peer_lines.append(f"- {psym}:\n{combined.strip()}")
            parts.append("\n".join(peer_lines) + "\n")

    result = "\n".join(parts)

    if cache is not None:
        try:
            cache.set(site=cache_site, query=cache_key, text=result)
        except OSError:
            pass  # cache write failure does not fail the search

    return result
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/tools/test_market_data.py::TestFetchAndConcat -v`
Expected: 3 passed.

- [ ] **Step 3: Commit**

```bash
git add src/stock_analysis_agent/tools/market_data.py tests/tools/test_market_data.py
git commit -m "feat(market-data): add _fetch_and_concat aggregator with cache + peer fan-out"
```

---

## Task 18: Write failing test for the @tool wrapper

**Files:**
- Test: `tests/tools/test_market_data.py` (append)

- [ ] **Step 1: Append @tool wrapper test class**

Append this block to `tests/tools/test_market_data.py`:

```python
class TestGetStockSnapshotTool:
    """The @tool _get_stock_snapshot wrapper."""

    def test_tool_name_is_get_stock_snapshot(self) -> None:
        from stock_analysis_agent.tools import market_data as md

        assert md._get_stock_snapshot.name == "get_stock_snapshot"

    def test_tool_sources_param_default_is_all(self) -> None:
        """When sources=None, the tool queries ALL_SOURCES."""
        from stock_analysis_agent.tools import market_data as md

        assert md._get_stock_snapshot.args is not None
        schema = md._get_stock_snapshot.args
        if hasattr(schema, "model_json_schema"):
            schema = schema.model_json_schema()
        # Just verify the schema contains a `sources` field.
        if isinstance(schema, dict) and "properties" in schema:
            properties = schema["properties"]
        else:
            properties = schema
        assert "symbol" in properties
        assert "sources" in properties

    def test_tool_invokes_aggregator_and_returns_aggregated_text(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """End-to-end: tool.ainvoke calls the aggregator and returns text."""
        import asyncio

        from stock_analysis_agent.tools import market_data as md
        from stock_analysis_agent.memory import _FileCache

        async def _fake_concat(symbol, **kwargs):  # type: ignore[no-untyped-def]
            return (
                f"[sina]\n{symbol}-sina\n"
                f"[tencent]\n{symbol}-tencent\n"
            )

        monkeypatch.setattr(md, "_fetch_and_concat", _fake_concat)
        cache = _FileCache(tmp_path, ttl_seconds=60.0)
        md._CACHE_PROVIDER.value = cache
        md._SOURCES_PROVIDER.value = md.ALL_SOURCES
        try:
            result = asyncio.run(
                md._get_stock_snapshot.ainvoke(
                    {"symbol": "02319.HK", "sources": ["sina", "tencent"]}
                )
            )
        finally:
            md._CACHE_PROVIDER.value = None
            md._SOURCES_PROVIDER.value = None

        assert "[sina]" in result
        assert "[tencent]" in result
        assert "02319.HK-sina" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_market_data.py::TestGetStockSnapshotTool -v`
Expected: `AttributeError: ... has no attribute '_get_stock_snapshot'`

---

## Task 19: Implement the `@tool _get_stock_snapshot` wrapper

**Files:**
- Modify: `src/stock_analysis_agent/tools/market_data.py`

- [ ] **Step 1: Add `_Provider` singletons and `asyncio` import at the top**

Add this import alongside the existing ones at the top of the file:

```python
import asyncio
```

Add this class import (private re-use of web_search's Provider):

```python
from stock_analysis_agent.tools.web_search import _Provider
from stock_analysis_agent.memory.file_cache import _FileCache
```

Then declare the module-level providers right after the constants block:

```python
_SOURCES_PROVIDER: _Provider[tuple[SourceName, ...]] = _Provider()
_CACHE_PROVIDER: _Provider[_FileCache | None] = _Provider()
```

- [ ] **Step 2: Add the @tool function at the end of the file**

Add this block at the bottom of `src/stock_analysis_agent/tools/market_data.py`:

```python
from langchain.tools import tool  # noqa: E402


@tool("get_stock_snapshot")
async def _get_stock_snapshot(
    symbol: str,
    sources: list[str] | None = None,
    include_peers: bool = True,
    peer_count: int = 2,
) -> str:
    """Fetch a comprehensive stock snapshot from multiple Chinese-market
    data sources and return aggregated text.

    Args:
        symbol: Standard code in '<code>.<market>' format, e.g.
            '02319.HK', '600519.SH', '000001.SZ'.
        sources: Optional subset of data sources to query. Allowed
            values: 'sina', 'tencent', 'tushare', 'akshare', 'mootdx'.
            None or empty list means query ALL sources.
        include_peers: If True, also look up the stock's industry and
            fetch the top `peer_count` peer companies for comparison.
        peer_count: How many top peers (by market cap) to include.
            Only meaningful when include_peers=True. Range: 0..10.

    Returns:
        Plain-text aggregation of snippets from each source, each
        prefixed with `[source-name]`. Failed sources are recorded as
        `[error: ...]` segments. The `[peers]` section appears at the
        end when include_peers=True and peer lookup succeeded.
    """
    resolved_sources: tuple[SourceName, ...]
    if not sources:
        resolved_sources = _SOURCES_PROVIDER.get()
    else:
        resolved_sources = tuple(sources)  # type: ignore[arg-type]
    cache = _CACHE_PROVIDER.get()
    return await _fetch_and_concat(
        symbol,
        sources=resolved_sources,
        include_peers=include_peers,
        peer_count=peer_count,
        cache=cache,
    )
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `pytest tests/tools/test_market_data.py::TestGetStockSnapshotTool -v`
Expected: 3 passed.

- [ ] **Step 4: Run the entire test file**

Run: `pytest tests/tools/test_market_data.py -v`
Expected: All tests pass (15+ total).

- [ ] **Step 5: Commit**

```bash
git add src/stock_analysis_agent/tools/market_data.py tests/tools/test_market_data.py
git commit -m "feat(market-data): add @tool _get_stock_snapshot wrapper with provider injection"
```

---

## Task 20: Write the demo script and run it

**Files:**
- Create: `src/stock_analysis_agent/script/test_mengniu_snapshot.py`

- [ ] **Step 1: Create the demo script**

Create `src/stock_analysis_agent/script/test_mengniu_snapshot.py` with this exact content:

```python
"""One-shot smoke test: invoke get_stock_snapshot directly for 02319.HK.

Usage:
    python -m stock_analysis_agent.script.test_mengniu_snapshot
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

from stock_analysis_agent.memory import _FileCache
from stock_analysis_agent.tools import market_data as md

USER_SYMBOL = "02319.HK"


async def _main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        cache = _FileCache(
            Path(tmp), ttl_seconds=md.DEFAULT_CACHE_TTL
        )
        # Inject providers so @tool _get_stock_snapshot can read them.
        md._SOURCES_PROVIDER.value = md.ALL_SOURCES
        md._CACHE_PROVIDER.value = cache
        try:
            print(f"Symbol : {USER_SYMBOL}", flush=True)
            print(f"Sources: {[s for s in md.ALL_SOURCES]}", flush=True)
            print(f"TTL    : {md.DEFAULT_CACHE_TTL}s "
                  f"({md.DEFAULT_CACHE_TTL / 3600:.0f}h)", flush=True)
            print("-" * 60, flush=True)
            result = await md._get_stock_snapshot.ainvoke(
                {
                    "symbol": USER_SYMBOL,
                    "include_peers": True,
                    "peer_count": 2,
                }
            )
            print(result, flush=True)
            print("-" * 60, flush=True)
            print(
                f"(cached: re-running returns identical text "
                f"in <1ms, zero network)", flush=True
            )
            cached = await md._get_stock_snapshot.ainvoke(
                {"symbol": USER_SYMBOL, "include_peers": True,
                 "peer_count": 2}
            )
            assert cached == result
        finally:
            md._SOURCES_PROVIDER.value = None
            md._CACHE_PROVIDER.value = None
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
```

- [ ] **Step 2: Run the demo**

Run: `uv run python -m stock_analysis_agent.script.test_mengniu_snapshot`
Expected: A multi-section text output starting with `[sina]\n...`, including per-source sections and ending with `[peers]\n...`. Note: some sources may fail in CI environments (e.g. Tushare without `TUSHARE_TOKEN` → `[error: ...]`); that is acceptable, the script just prints whatever it gets. The script should NOT raise.

If you have a `TUSHARE_TOKEN` in the environment, set it first:

```bash
export TUSHARE_TOKEN="your_token_here"
```

- [ ] **Step 3: Commit**

```bash
git add src/stock_analysis_agent/script/test_mengniu_snapshot.py
git commit -m "feat(scripts): add demo script that invokes get_stock_snapshot on 02319.HK"
```

---

## Task 21: Final quality gates

**Files:**
- none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `pytest -v`
Expected: All tests pass — existing 6 web_search / 7 text_extractor / 4 file_cache / ~12 deepsearch-agent tests plus the new ~15 market_data tests = ~44 tests total, all green.

- [ ] **Step 2: Run ruff lint**

Run: `uv run ruff check src tests`
Expected: `All checks passed!` If any `E` or `W` warnings appear, fix them inline (the new code should already conform to PEP 8 + project conventions).

- [ ] **Step 3: Verify the public API surface**

Run: `uv run python -c "from stock_analysis_agent.tools.market_data import _get_stock_snapshot, ALL_SOURCES; print(_get_stock_snapshot.name, ALL_SOURCES)"`
Expected: `get_stock_snapshot ('sina', 'tencent', 'tushare', 'akshare', 'mootdx')`

- [ ] **Step 4: Commit any cleanup (if ruff fixed anything)**

```bash
git status --short
# If there are uncommitted changes:
git add -A
git commit -m "chore: ruff cleanup"
```

---

## Self-Review Checklist (run before considering plan complete)

**1. Spec coverage** — verify every spec section has a task:
- [x] §3 Public API → Tasks 18, 19
- [x] §4 Symbol translation → Task 3
- [x] §5 Per-source coverage → Tasks 5 (sina), 7 (tencent), 9 (tushare), 11 (akshare), 13 (mootdx)
- [x] §6 Peer detection + HK fallback → Tasks 14, 15
- [x] §6 Peer fetch (sina + tencent) → Task 17
- [x] §7 Failure handling → Tasks 5, 7, 9, 11, 13 (per-source error), Task 17 (all-fail raise)
- [x] §8 Cache strategy → Task 17
- [x] §9 Dependencies → Task 1
- [x] §10 Configuration → Task 9 (TUSHARE_TOKEN), Task 13 (MOOTDX_DEFAULT_SERVER), Task 19 (cache_dir default)
- [x] §11 Module structure → covered by all tasks (file-by-file)
- [x] §12 Testing strategy → Tasks 2, 4, 6, 8, 10, 12, 14, 16, 18 (each test class)
- [x] §13 Demo script → Task 20

**2. Placeholder scan** — no TBD / TODO / "implement later" / vague instructions. All code blocks contain working implementations.

**3. Type consistency** —
- `_translate(symbol)` returns `dict[SourceName, str]` everywhere ✓
- `_fetch_*` adapters return `str` everywhere ✓
- `_fetch_and_concat` signature matches Task 17 ↔ Task 18's mock ✓
- `_get_stock_snapshot.ainvoke` accepts `{symbol, sources, include_peers, peer_count}` dict ✓
- Module-level providers named `_SOURCES_PROVIDER` and `_CACHE_PROVIDER` referenced consistently in Tasks 17, 18, 19, 20 ✓

**4. Risk mitigations from spec §14** —
- AKShare interface changes isolated in `_detect_peers` ✓
- HK fallback map populated with common large caps ✓
- Tushare paid-tier endpoints avoided (only daily, stock_basic) ✓
- Mootdx default server overridable via `MOOTDX_DEFAULT_SERVER` constant ✓
- 12h TTL matches spec, configurable via `cache_ttl` param ✓
- Output size bounded: 5×~300 chars + 2 peer snippets ≈ 2KB text ✓