"""@tool get_stock_snapshot: fan-out concurrent stock data over 5 sources."""
from __future__ import annotations

from typing import Any, Literal

import httpx

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
            "sina": f"rt_hk{code}",
            "tencent": f"hk{code}",
            "tushare": f"{code}.HK",
            "akshare": code,
            "mootdx": "23",
            "mootdx_symbol": code.ljust(6, "0")[:6],
        }
    if market == "SH":
        return {
            "sina": f"sh{code}",
            "tencent": f"sh{code}",
            "tushare": f"{code}.SH",
            "akshare": f"sh{code}",
            "mootdx": "1",
            "mootdx_symbol": code.zfill(6),
        }
    # SZ
    return {
        "sina": f"sz{code}",
        "tencent": f"sz{code}",
        "tushare": f"{code}.SZ",
        "akshare": f"sz{code}",
        "mootdx": "0",
        "mootdx_symbol": code.zfill(6),
    }


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

    Sina's payload is GBK-encoded CSV-ish. The HK-standard layout
    (name, open, prev_close, current, high, low, change, change_pct,
    ...) is used when ``len(fields) >= 32``; A-share shares a shorter
    layout. We render a defensive summary for both.
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
                f"现价: {current:.3f}\n"
                f"涨跌: {change:+.3f} ({change_pct:+.2f}%)\n"
                f"今开: {open_p:.3f}  昨收: {prev_close:.3f}  "
                f"最高: {high:.3f}  最低: {low:.3f}\n"
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
            f"现价: {current:.3f}\n"
            f"今开: {open_p:.3f}  昨收: {prev_close:.3f}  "
            f"最高: {high:.3f}  最低: {low:.3f}\n"
        )
    except (ValueError, IndexError) as e:
        return f"[error: parse failed: {e}]"


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

    Field layout (0-indexed after splitting on `~`) for HK shares:
    1=name_cn, 3=current, 4=prev_close, 5=open, 31=change,
    32=change_pct, 33=high, 34=low, 36=volume, 37=amount,
    48=PE-TTM, 49=PB.
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
        change = float(fields[31])
        change_pct = float(fields[32])
        high = float(fields[33])
        low = float(fields[34])
        volume = fields[36]
        amount = fields[37]
        pe_ttm = fields[48] or "--"
        pb = fields[49] or "--"
        return (
            f"名称: {name_cn}\n"
            f"现价: {current:.3f}\n"
            f"涨跌: {change:+.3f} ({change_pct:+.2f}%)\n"
            f"今开: {open_p:.3f}  昨收: {prev_close:.3f}  "
            f"最高: {high:.3f}  最低: {low:.3f}\n"
            f"成交量: {volume}\n"
            f"成交额: {amount}\n"
            f"PE-TTM: {pe_ttm}  PB: {pb}\n"
        )
    except (ValueError, IndexError) as e:
        return f"[error: parse failed: {e}]"
