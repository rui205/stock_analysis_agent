"""One-shot smoke test: invoke get_stock_snapshot directly for 02319.HK.

Usage:
    python -m stock_analysis_agent.script.test_mengniu_snapshot
"""
from __future__ import annotations

import asyncio
import json
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
            print(
                f"TTL    : {md.DEFAULT_CACHE_TTL}s "
                f"({md.DEFAULT_CACHE_TTL / 3600:.0f}h)",
                flush=True,
            )
            print("-" * 60, flush=True)
            result = await md._get_stock_snapshot.ainvoke(
                {
                    "symbol": USER_SYMBOL,
                    "include_peers": True,
                    "peer_count": 2,
                }
            )
            print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
            print("-" * 60, flush=True)
            print(
                "(cached: re-running returns identical dict "
                "in <1ms, zero network)",
                flush=True,
            )
            cached = await md._get_stock_snapshot.ainvoke(
                {
                    "symbol": USER_SYMBOL,
                    "include_peers": True,
                    "peer_count": 2,
                }
            )
            assert cached == result
        finally:
            md._SOURCES_PROVIDER.value = None
            md._CACHE_PROVIDER.value = None
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
