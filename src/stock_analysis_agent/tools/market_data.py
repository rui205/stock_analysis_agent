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
