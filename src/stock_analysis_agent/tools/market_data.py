"""@tool get_stock_snapshot: fan-out concurrent stock data over 3 sources."""
from __future__ import annotations

import asyncio
import datetime as _dt
from typing import Any, Literal
from zoneinfo import ZoneInfo

from stock_analysis_agent.memory.file_cache import _FileCache
from stock_analysis_agent.tools.web_search import _Provider

MarketName = Literal["HK", "SH", "SZ"]
SourceName = Literal["tushare", "akshare", "mootdx"]

ALL_SOURCES: tuple[SourceName, ...] = (
    "tushare",
    "akshare",
    "mootdx",
)

# HTTP headers sent on every eastmoney request. Eastmoney's
# `push2.eastmoney.com` endpoints reject bare ``requests.get`` calls
# with no User-Agent / Referer, returning proxy errors. These headers
# match what quote.eastmoney.com sends in the browser and unblock
# the ``*_em`` family of akshare functions used by ``_detect_peers``.
EM_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://quote.eastmoney.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
}
EM_HOST_MARKER: str = "eastmoney.com"

# Tracks whether ``_install_em_request_hook`` has already been applied
# so we only patch ``requests.get`` once per process.
_em_hook_installed: bool = False


def _install_em_request_hook() -> None:
    """Wrap ``requests.get`` so eastmoney URLs get EM_HEADERS injected.

    Akshare's ``*_em`` functions call ``requests.get(url, params=...)``
    with no headers; eastmoney's CDN rejects those requests with proxy
    errors. The hook here is a one-time, process-wide monkey-patch that
    inspects the URL and, when it targets eastmoney, injects
    ``EM_HEADERS`` via ``kwargs['headers'].update(...)`` before
    delegating to the original ``requests.get``.
    """
    global _em_hook_installed
    if _em_hook_installed:
        return
    import requests

    _original_get = requests.get

    def _patched_get(  # type: ignore[no-untyped-def]
        url: Any, **kwargs: Any
    ):
        if isinstance(url, str) and EM_HOST_MARKER in url:
            kwargs.setdefault("headers", {}).update(EM_HEADERS)
        return _original_get(url, **kwargs)

    requests.get = _patched_get  # type: ignore[assignment]
    _em_hook_installed = True

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
PEER_FETCH_SOURCES: tuple[SourceName, ...] = ("akshare",)

_SOURCES_PROVIDER: _Provider[tuple[SourceName, ...]] = _Provider()
_CACHE_PROVIDER: _Provider[_FileCache | None] = _Provider()


def _now_iso() -> str:
    """Return the current wall-clock time in Asia/Shanghai as ISO 8601.

    Format: ``YYYY-MM-DDTHH:MM:SS+08:00`` (seconds precision, fixed
    +08:00 offset, no microseconds). Used to stamp ``fetched_at`` in
    the structured snapshot output.
    """
    return _dt.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def _json_default(obj: object) -> object:
    """Default hook for ``json.dumps`` to serialize non-JSON-native types.

    Handles:
      - ``datetime.date`` / ``datetime.datetime`` → ISO 8601 string
      - ``numpy.generic`` (numpy scalar types) → Python scalar via ``.item()``

    Raises:
        TypeError: with a message identifying the offending type, so
            cache writes surface serialization bugs immediately.
    """
    if isinstance(obj, _dt.date):
        return obj.isoformat()
    if isinstance(obj, _dt.datetime):
        return obj.isoformat()
    # numpy scalar types (numpy.float64, numpy.int64, etc.)
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except (ValueError, TypeError):
            pass
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


def _noneify(v: object) -> object:
    """Replace pandas NaN / NaT with ``None`` so the value is JSON-safe.

    Non-pandas scalars are returned unchanged. ``float('nan')`` and
    ``pandas.NaT`` both compare equal to themselves but not to None,
    so we use ``pd.isna`` on the pandas side and a float check otherwise.
    """
    if v is None:
        return None
    # Catch both numpy.nan and float('nan') without importing pandas at module top.
    try:
        import math
        if isinstance(v, float) and math.isnan(v):
            return None
    except Exception:
        pass
    try:
        import pandas as pd  # local import keeps the helper lightweight
        if pd.isna(v):
            return None
    except Exception:
        pass
    return v


def _translate(symbol: str) -> dict[SourceName, str]:
    """Translate a standard `<code>.<market>` symbol into each source's
    native code format.

    Args:
        symbol: Standard code, e.g. ``"02319.HK"``, ``"600519.SH"``,
            ``"000001.SZ"``.

    Returns:
        Mapping from source name to that source's local code, plus a
        ``"mootdx_symbol"`` entry holding the 6-digit mootdx symbol
        (left-zero-padded for HK, zero-filled for SH/SZ). Markets
        outside ``{"HK", "SH", "SZ"}`` raise ``ValueError``.

    Raises:
        ValueError: If the symbol has no ``.`` separator, the market
            segment is unknown, or the code is empty.
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
            "tushare": f"{code}.HK",
            "akshare": code,
            "mootdx": "23",
            "mootdx_symbol": code.ljust(6, "0")[:6],
        }
    if market == "SH":
        return {
            "tushare": f"{code}.SH",
            "akshare": f"sh{code}",
            "mootdx": "1",
            "mootdx_symbol": code.zfill(6),
        }
    # SZ
    return {
        "tushare": f"{code}.SZ",
        "akshare": f"sz{code}",
        "mootdx": "0",
        "mootdx_symbol": code.zfill(6),
    }


async def _fetch_tushare(
    code: str,
    *,
    token: str | None = None,
) -> dict[str, Any]:
    """Fetch a Tushare snapshot for `code` (e.g. ``"02319.HK"``).

    Calls two endpoints in a single thread hop:
      - ``pro.daily`` for the latest OHLCV
      - ``pro.stock_basic`` for industry / PE / PB / market cap

    Returns the merged dict from both endpoints with no field filtering
    (raw row data preserved) so downstream consumers can pick which
    fields to surface.

    Args:
        code: Tushare-format ts_code, e.g. ``"02319.HK"``.
        token: Optional explicit token; if ``None``, falls back to the
            ``TUSHARE_TOKEN`` environment variable.

    Returns:
        One of:
          - ``{"data": <merged row dict>, "row_index": 0}`` on success
          - ``{"error": {"type": str, "message": str}}`` on failure

        Error types:
          - ``"TushareTokenMissing"`` when ``TUSHARE_TOKEN`` is unset
          - ``"TushareEmpty"`` when ``stock_basic`` returns no rows
          - ``<ExceptionClassName>`` for any other raised exception
    """
    import os

    token = token if token is not None else os.environ.get("TUSHARE_TOKEN")
    if not token:
        return {
            "error": {
                "type": "TushareTokenMissing",
                "message": "TUSHARE_TOKEN not set",
            }
        }

    try:
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
        return {
            "error": {
                "type": type(e).__name__,
                "message": str(e),
            }
        }

    if not basic:
        return {
            "error": {
                "type": "TushareEmpty",
                "message": f"stock_basic returned empty for {code}",
            }
        }

    # Merge: union of keys from both rows. When keys overlap (only ts_code
    # in practice), stock_basic wins because it's the more authoritative
    # source for the symbol identity. NaN -> None for JSON-safe output.
    merged: dict[str, Any] = {}
    if daily:
        for k, v in daily[0].items():
            merged[k] = _noneify(v)
    for k, v in basic[0].items():
        merged[k] = _noneify(v)

    return {"data": merged, "row_index": 0}


async def _fetch_akshare(code: str) -> str:
    """Fetch an AKShare snapshot for `code` using the Sina backend.

    For HK codes (e.g. ``"02319"``) uses ``stock_hk_spot``.
    For SH/SZ codes uses ``stock_zh_a_spot`` and filters by code.

    The Sina-backed endpoints are reachable when the eastmoney proxy
    used by ``*_em`` functions is blocked. They expose price / OHLCV
    only; PE / PB / market-cap are not in the sina payload and are
    omitted from the rendered text. Use tushare for fundamentals.

    The blocking SDK calls are wrapped in ``asyncio.to_thread`` so the
    event loop is not stalled.

    Args:
        code: AKShare-local code, e.g. ``"02319"`` (HK) or
            ``"sh600519"`` / ``"sz000001"`` (A-share).

    Returns:
        A text snippet prefixed with ``[akshare]``, or
        ``[akshare]\n[error: ...]`` on failure or empty result.
    """
    import asyncio

    import akshare as ak
    import pandas as pd

    def _fetch() -> tuple[pd.DataFrame, str, str]:
        # Heuristic: HK codes are 5-digit plain numbers;
        # A-share codes come prefixed with sh/sz from _translate.
        if code.isdigit() and len(code) == 5:
            return ak.stock_hk_spot(), "中文名称", "HKD"
        return ak.stock_zh_a_spot(), "名称", "CNY"

    try:
        df, name_col, ccy = await asyncio.to_thread(_fetch)
    except Exception as e:
        return f"[akshare]\n[error: {type(e).__name__}: {e}]\n"

    if df is None or df.empty:
        return f"[akshare]\n[error: empty result for {code}]\n"

    # Find the row matching our code. Sina A-share spot data keeps the
    # `sh`/`sz`/`bj` prefix in the `代码` column (e.g. `sh600887`), and
    # HK spot data uses bare 5-digit codes (e.g. `02319`). The input
    # `code` from `_translate` already follows the right format for
    # each market, so a direct equality match works.
    match = df[df["代码"].astype(str) == code]
    if match.empty:
        return f"[akshare]\n[error: {code} not found in spot data]\n"
    row = match.iloc[0].to_dict()

    def _g(key: str) -> str:
        v = row.get(key, "--")
        return "--" if v in (None, "", float("nan")) else str(v)

    def _price(key: str) -> str:
        v = _g(key)
        if v == "--":
            return "--"
        try:
            return f"{float(v):.3f}"
        except ValueError:
            return v

    return (
        "[akshare]\n"
        f"名称: {_g(name_col)}\n"
        f"现价: {_price('最新价')}\n"
        f"涨跌: {_g('涨跌额')} ({_g('涨跌幅')}%)\n"
        f"今开: {_g('今开')}  昨收: {_g('昨收')}  "
        f"最高: {_g('最高')}  最低: {_g('最低')}\n"
        f"成交量: {_g('成交量')}  成交额: {_g('成交额')} {ccy}\n"
    )


async def _fetch_mootdx(market: str, symbol: str) -> str:
    """Fetch a Mootdx snapshot (latest daily bar) for `symbol`.

    Uses ``mootdx.quotes.StdQuotes`` (mootdx 0.11.7 API) with the
    daily-frequency ``bars()`` method. mootdx is primarily an A-share
    data source; HK symbols return an empty DataFrame which is
    surfaced as an ``[error: ...]`` segment rather than an exception.

    The blocking SDK calls are wrapped in ``asyncio.to_thread`` so the
    event loop is not stalled.

    Args:
        market: Mootdx market code (``"23"`` for HK, ``"1"`` for SH,
            ``"0"`` for SZ — see ``_translate``).
        symbol: 6-digit mootdx symbol, e.g. ``"000001"``, ``"600519"``,
            ``"023190"``.

    Returns:
        A text snippet prefixed with ``[mootdx]``, or
        ``[mootdx]\n[error: ...]`` on connection failure, empty
        DataFrame, or any other exception.
    """
    import asyncio

    import pandas as pd
    from mootdx.quotes import StdQuotes

    def _fetch() -> pd.DataFrame:
        client = StdQuotes(
            server=MOOTDX_DEFAULT_SERVER, timeout=10
        )
        try:
            return client.bars(
                symbol=symbol, frequency=9, start=0, offset=1
            )
        finally:
            try:
                client.close()
            except Exception:
                pass

    try:
        df = await asyncio.to_thread(_fetch)
    except Exception as e:
        return f"[mootdx]\n[error: {type(e).__name__}: {e}]\n"

    if df is None or df.empty:
        return (
            f"[mootdx]\n[error: empty bars for market={market} "
            f"symbol={symbol} (mootdx 0.11.x is A-share focused; "
            f"HK symbols may return empty)]\n"
        )

    row = df.iloc[0].to_dict()
    return (
        "[mootdx]\n"
        f"市场代码: {market}\n"
        f"今开: {float(row.get('open', 0)):.3f}  "
        f"最高: {float(row.get('high', 0)):.3f}  "
        f"最低: {float(row.get('low', 0)):.3f}  "
        f"收盘: {float(row.get('close', 0)):.3f}\n"
        f"成交量: {row.get('volume')}\n"
    )


def _detect_peers(symbol: str, peer_count: int) -> list[str] | None:
    """Detect top-`peer_count` peer companies in the same industry.

    For A-shares, looks up the industry via akshare's
    ``stock_individual_info_em`` and then queries
    ``stock_board_industry_cons_em`` to get constituents ranked by
    market cap. For HK symbols (where akshare's individual-info
    endpoint does not classify), falls back to ``HK_INDUSTRY_HINTS``.

    Note: industry / board endpoints are only available via the
    eastmoney backend (``*_em``) in current akshare; there is no
    sina-based equivalent. Peer detection will fail when eastmoney
    is unreachable even though the main quote source is sina-based.

    Args:
        symbol: Standard code, e.g. ``"600519.SH"`` or ``"02319.HK"``.
        peer_count: Maximum peers to return.

    Returns:
        List of standard codes (e.g. ``"600887.SH"``). For A-share inputs
        the symbol itself is promoted to the first entry when it appears
        in the cons list. Returns ``None`` if detection fails (akshare
        unreachable, no industry mapped, empty cons, etc.).
    """
    import akshare as ak

    # Ensure the eastmoney request hook is active so that all
    # ``ak.*_em`` calls below carry the User-Agent / Referer that
    # quote.eastmoney.com expects. Idempotent: subsequent calls are
    # no-ops thanks to ``_em_hook_installed``.
    _install_em_request_hook()

    if "." not in symbol:
        return None
    code, market = symbol.rsplit(".", 1)
    code = code.strip()
    industry_name: str | None = None

    try:
        if market == "HK":
            # HK is not classified by akshare; fall back to hint map.
            industry_name = HK_INDUSTRY_HINTS.get(code)
            if industry_name is None:
                # Try without leading zeros.
                industry_name = HK_INDUSTRY_HINTS.get(
                    code.lstrip("0") or "0"
                )
        else:
            try:
                info = ak.stock_individual_info_em(symbol=code)
            except Exception:
                info = None
            if info is not None and not info.empty:
                row = info.iloc[0].to_dict()
                industry_name = row.get("行业") or row.get("industry")
            if not industry_name:
                # Fallback: if akshare couldn't classify the symbol
                # directly, scan the industry table and pick the first
                # industry whose cons list contains our code. If the
                # table is degenerate (single industry) this is just
                # the one industry — best-effort detection.
                try:
                    industries_df = ak.stock_board_industry_name_em()
                except Exception:
                    industries_df = None
                if industries_df is not None and not industries_df.empty:
                    name_col = (
                        "板块名称"
                        if "板块名称" in industries_df.columns
                        else "name"
                    )
                    for candidate in industries_df[name_col].tolist():
                        try:
                            candidate_cons = ak.stock_board_industry_cons_em(
                                symbol=candidate
                            )
                        except Exception:
                            continue
                        if (
                            candidate_cons is not None
                            and not candidate_cons.empty
                            and "代码" in candidate_cons.columns
                            and (candidate_cons["代码"].astype(str) == code).any()
                        ):
                            industry_name = str(candidate)
                            break
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
    # If the input symbol itself is already among the peers (e.g. an
    # A-share whose code is in the cons list), promote it to the front.
    # Otherwise leave the list untouched — peer detection is best-effort
    # and the input symbol doesn't need to be force-prepended when it's
    # not actually a member of the returned peer set (e.g. an HK code
    # whose industry peers are all A-share).
    if symbol in result:
        result.remove(symbol)
        result.insert(0, symbol)
    if peer_count > 0:
        return result[: peer_count + 1]
    return result


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
         ``[peers]`` section rendered from akshare only.
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

    Raises:
        ValueError: If ``sources`` is empty.
        ToolExecutionError: If every configured primary source returned
            an error segment.
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

    translated = _translate(symbol)

    async def _call(src: SourceName) -> str:
        try:
            if src == "tushare":
                return await _fetch_tushare(
                    translated["tushare"], token=None
                )
            if src == "akshare":
                return await _fetch_akshare(translated["akshare"])
            if src == "mootdx":
                return await _fetch_mootdx(
                    translated["mootdx"], translated["mootdx_symbol"]
                )
            return f"[error: unknown source {src!r}]"
        except Exception as e:  # noqa: BLE001 — top-level guard
            return f"[{src}]\n[error: {type(e).__name__}: {e}]\n"

    parts = await asyncio.gather(*(_call(s) for s in sources))

    # If ALL primary sources errored, raise so the retry middleware acts.
    if parts and all("[error:" in p for p in parts):
        raise ToolExecutionError(
            f"all sources failed for {symbol}: {list(sources)}"
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
                    ptrans = _translate(psym)
                    peer_text = await _fetch_akshare(ptrans["akshare"])
                except Exception as e:  # noqa: BLE001
                    peer_lines.append(
                        f"- {psym}: [error: "
                        f"{type(e).__name__}: {e}]"
                    )
                    continue
                peer_lines.append(f"- {psym}:\n{peer_text.strip()}")
            parts.append("\n".join(peer_lines) + "\n")

    result = "\n".join(parts)

    if cache is not None:
        try:
            cache.set(site=cache_site, query=cache_key, text=result)
        except OSError:
            pass  # cache write failure does not fail the search

    return result


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
            values: 'tushare', 'akshare', 'mootdx'. None or empty list
            means query ALL sources configured via the module-level
            _SOURCES_PROVIDER (typically all three).
        include_peers: If True, also look up the stock's industry and
            fetch the top `peer_count` peer companies for comparison.
            Peer rendering is done through the akshare source.
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
