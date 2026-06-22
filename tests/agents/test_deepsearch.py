"""Tests for stock_analysis_agent.agents.deepsearch.DeepSearchAgent."""
from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from stock_analysis_agent.agents.deepsearch import _FileCache, _extract_text
from stock_analysis_agent.agents.exceptions import ToolExecutionError


def test_extract_text_strips_script_and_style() -> None:
    """<script> and <style> blocks must be removed entirely."""
    html = "<script>alert(1)</script><p>hello</p><style>p{}</style>"
    assert _extract_text(html) == "hello"


def test_extract_text_folds_whitespace() -> None:
    """Runs of whitespace (newlines, tabs, multiple spaces) collapse to single space."""
    html = "<p>hello   world</p>\n<p>foo\tbar</p>"
    assert _extract_text(html) == "hello world foo bar"


def test_extract_text_empty_input() -> None:
    """Empty HTML returns empty string."""
    assert _extract_text("") == ""


def test_extract_text_preserves_text_outside_tags() -> None:
    """Plain text between tags is preserved."""
    html = "before <b>middle</b> after"
    assert _extract_text(html) == "before middle after"


def test_cache_miss_when_file_absent(tmp_path: Path) -> None:
    """A cache directory with no files returns None for any get()."""
    cache = _FileCache(tmp_path, ttl_seconds=60.0)
    assert cache.get(site="https://a.test", query="hello") is None


def test_cache_hit_returns_stored_text(tmp_path: Path) -> None:
    """After set(), get() returns the same text."""
    cache = _FileCache(tmp_path, ttl_seconds=60.0)
    cache.set(site="https://a.test", query="hello", text="world")
    assert cache.get(site="https://a.test", query="hello") == "world"


def test_cache_expired_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entry older than ttl_seconds is treated as a miss."""
    cache = _FileCache(tmp_path, ttl_seconds=10.0)
    cache.set(site="https://a.test", query="hello", text="world")

    # Advance "now" by 11 seconds so the entry is expired.
    import time

    base = time.time()
    monkeypatch.setattr("time.time", lambda: base + 11.0)

    assert cache.get(site="https://a.test", query="hello") is None


def test_cache_ttl_none_means_never_expire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ttl_seconds=None disables expiration entirely."""
    cache = _FileCache(tmp_path, ttl_seconds=None)
    cache.set(site="https://a.test", query="hello", text="world")

    import time

    base = time.time()
    monkeypatch.setattr("time.time", lambda: base + 1_000_000.0)

    assert cache.get(site="https://a.test", query="hello") == "world"


def test_cache_corrupt_json_returns_none(tmp_path: Path) -> None:
    """A cache file with invalid JSON is treated as a miss, not an error."""
    cache = _FileCache(tmp_path, ttl_seconds=60.0)
    key = _FileCache._key("https://a.test", "hello")
    (tmp_path / f"{key}.json").write_text("not valid json {{{", encoding="utf-8")

    assert cache.get(site="https://a.test", query="hello") is None


def test_cache_creates_dir_on_init(tmp_path: Path) -> None:
    """A non-existent cache_dir is created on construction."""
    nested = tmp_path / "a" / "b" / "c"
    assert not nested.exists()
    _FileCache(nested, ttl_seconds=60.0)
    assert nested.is_dir()


def test_cache_set_is_atomic(tmp_path: Path) -> None:
    """After set() returns, no .tmp file is left behind."""
    cache = _FileCache(tmp_path, ttl_seconds=60.0)
    cache.set(site="https://a.test", query="hello", text="world")
    remaining = list(tmp_path.iterdir())
    assert all(p.suffix != ".tmp" for p in remaining), f"tmp residue: {remaining!r}"
    assert any(p.suffix == ".json" for p in remaining), f"no json file: {remaining!r}"


def test_cache_key_is_query_site_specific(tmp_path: Path) -> None:
    """Different (query, site) pairs map to different cache files."""
    cache = _FileCache(tmp_path, ttl_seconds=60.0)
    cache.set(site="https://a.test", query="hello", text="A")
    cache.set(site="https://a.test", query="world", text="B")
    cache.set(site="https://b.test", query="hello", text="C")

    assert cache.get(site="https://a.test", query="hello") == "A"
    assert cache.get(site="https://a.test", query="world") == "B"
    assert cache.get(site="https://b.test", query="hello") == "C"


def _make_mock_transport(
    handler,
) -> httpx.MockTransport:
    """Build an httpx.MockTransport from a synchronous handler."""
    return httpx.MockTransport(handler)


def _ok_handler(html: str = "<p>hello</p>"):
    def _h(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html)
    return _h


def _fail_handler(
    exc: type[Exception] = httpx.ConnectError, msg: str = "boom"
):
    def _h(request: httpx.Request) -> httpx.Response:
        raise exc(msg)
    return _h


@pytest.mark.asyncio
async def test_fetch_empty_site_list_raises_value_error(
    tmp_path: Path,
) -> None:
    """An empty site_list is a programmer error, not a runtime error."""
    from stock_analysis_agent.agents.deepsearch import _fetch_and_concat

    cache = _FileCache(tmp_path, ttl_seconds=60.0)
    with pytest.raises(ValueError, match="site_list cannot be empty"):
        await _fetch_and_concat("q", [], cache=cache, transport=httpx.MockTransport(_ok_handler()))


@pytest.mark.asyncio
async def test_fetch_all_sites_fail_raises_tool_execution_error() -> None:
    """If every site fails, raise ToolExecutionError after per-site attempts."""
    from stock_analysis_agent.agents.deepsearch import _fetch_and_concat

    transport = httpx.MockTransport(_fail_handler(httpx.ConnectError, "nope"))
    with pytest.raises(ToolExecutionError, match="all sites failed"):
        await _fetch_and_concat(
            "q",
            ["https://a.test", "https://b.test"],
            cache=None,
            transport=transport,
        )


@pytest.mark.asyncio
async def test_fetch_partial_failure_returns_text_with_error_segment() -> None:
    """One site succeeds, one fails → text contains both, no exception."""
    from stock_analysis_agent.agents.deepsearch import _fetch_and_concat

    def _h(request: httpx.Request) -> httpx.Response:
        if "a.test" in str(request.url):
            return httpx.Response(200, text="<p>good</p>")
        raise httpx.ConnectError("down")

    transport = httpx.MockTransport(_h)
    result = await _fetch_and_concat(
        "q",
        ["https://a.test", "https://b.test"],
        cache=None,
        transport=transport,
    )

    assert "https://a.test" in result
    assert "good" in result
    assert "https://b.test" in result
    assert "[error:" in result


@pytest.mark.asyncio
async def test_fetch_runs_in_parallel() -> None:
    """All sites are fetched concurrently (total time ≈ slowest, not sum)."""
    import time as time_mod

    from stock_analysis_agent.agents.deepsearch import _fetch_and_concat

    delay = 0.1

    def _h(request: httpx.Request) -> httpx.Response:
        # Use asyncio.sleep (cooperative) so parallel gather actually yields.
        # httpx.MockTransport supports a coroutine return value.
        import asyncio as _asyncio

        async def _respond() -> httpx.Response:
            await _asyncio.sleep(delay)
            return httpx.Response(200, text="<p>ok</p>")

        return _respond()

    transport = httpx.MockTransport(_h)

    start = time_mod.monotonic()
    await _fetch_and_concat(
        "q",
        ["https://a.test", "https://b.test", "https://c.test"],
        cache=None,
        transport=transport,
    )
    elapsed = time_mod.monotonic() - start

    # Parallel should be ~delay; sequential would be ~3*delay. Allow some headroom.
    assert elapsed < delay * 2.5, f"expected parallel, took {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_fetch_uses_cache_when_present(tmp_path: Path) -> None:
    """A cache hit means no HTTP call is made."""
    from stock_analysis_agent.agents.deepsearch import _fetch_and_concat

    cache = _FileCache(tmp_path, ttl_seconds=60.0)
    cache.set(site="https://a.test", query="q", text="cached-A")

    calls: list[str] = []

    def _h(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, text="<p>fresh</p>")

    transport = httpx.MockTransport(_h)
    result = await _fetch_and_concat(
        "q",
        ["https://a.test"],
        cache=cache,
        transport=transport,
    )

    assert calls == [], f"expected no HTTP, got {calls!r}"
    assert "cached-A" in result


@pytest.mark.asyncio
async def test_fetch_writes_through_cache_on_miss(tmp_path: Path) -> None:
    """On a miss, the fetched text is written to cache."""
    from stock_analysis_agent.agents.deepsearch import _fetch_and_concat

    cache = _FileCache(tmp_path, ttl_seconds=60.0)
    transport = httpx.MockTransport(_ok_handler("<p>fetched</p>"))

    await _fetch_and_concat(
        "q",
        ["https://a.test"],
        cache=cache,
        transport=transport,
    )

    assert cache.get(site="https://a.test", query="q") == "fetched"


@pytest.mark.asyncio
async def test_fetch_does_not_write_cache_on_failure(tmp_path: Path) -> None:
    """HTTP failure must not pollute the cache with an error string."""
    from stock_analysis_agent.agents.deepsearch import _fetch_and_concat

    cache = _FileCache(tmp_path, ttl_seconds=60.0)

    def _h(request: httpx.Request) -> httpx.Response:
        if "a.test" in str(request.url):
            return httpx.Response(200, text="<p>good</p>")
        raise httpx.ConnectError("down")

    transport = httpx.MockTransport(_h)
    await _fetch_and_concat(
        "q",
        ["https://a.test", "https://b.test"],
        cache=cache,
        transport=transport,
    )

    assert cache.get(site="https://a.test", query="q") == "good"
    assert cache.get(site="https://b.test", query="q") is None


@pytest.mark.asyncio
async def test_fetch_cache_write_failure_does_not_abort_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache.set() failure (e.g. disk full) must not fail the search."""
    from stock_analysis_agent.agents.deepsearch import _fetch_and_concat

    cache = _FileCache(tmp_path, ttl_seconds=60.0)
    transport = httpx.MockTransport(_ok_handler("<p>ok</p>"))

    def _boom_set(**kwargs) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(cache, "set", _boom_set)

    result = await _fetch_and_concat(
        "q",
        ["https://a.test"],
        cache=cache,
        transport=transport,
    )

    assert "ok" in result


@pytest.mark.asyncio
async def test_fetch_query_param_is_passed(tmp_path: Path) -> None:
    """The query must be sent as the `q` query parameter."""
    from stock_analysis_agent.agents.deepsearch import _fetch_and_concat

    seen_urls: list[str] = []

    def _h(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(200, text="<p>x</p>")

    transport = httpx.MockTransport(_h)
    await _fetch_and_concat(
        "search-term",
        ["https://a.test"],
        cache=None,
        transport=transport,
    )

    assert any("q=search-term" in u for u in seen_urls), f"expected q= param, got {seen_urls!r}"


def test_web_search_tool_metadata() -> None:
    """The @tool _web_search exposes the expected name and args schema."""
    from stock_analysis_agent.agents.deepsearch import _web_search

    assert _web_search.name == "web_search"
    # The args schema must include a `query` string field. langchain versions
    # differ in whether `_web_search.args` is a Pydantic model, a full JSON
    # schema dict (with a `properties` key), or the properties dict itself.
    schema = _web_search.args
    if hasattr(schema, "model_json_schema"):
        schema = schema.model_json_schema()
    if isinstance(schema, dict) and "properties" in schema and isinstance(schema["properties"], dict):
        properties = schema["properties"]
    else:
        properties = schema
    assert "query" in (properties or {}), f"missing query in {schema!r}"