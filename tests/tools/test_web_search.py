"""Tests for stock_analysis_agent.tools.web_search.

Covers _fetch_and_concat (9 tests) and the _web_search @tool metadata (1 test).
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from stock_analysis_agent.agent.exceptions import ToolExecutionError
from stock_analysis_agent.memory import _FileCache
from stock_analysis_agent.tools.web_search import (
    _DEFAULT_USER_AGENT,
    _fetch_and_concat,
    _is_captcha_response,
    _web_search,
)


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
    cache = _FileCache(tmp_path, ttl_seconds=60.0)
    with pytest.raises(ValueError, match="site_list cannot be empty"):
        await _fetch_and_concat("q", [], cache=cache, transport=httpx.MockTransport(_ok_handler()))


@pytest.mark.asyncio
async def test_fetch_all_sites_fail_raises_tool_execution_error() -> None:
    """If every site fails, raise ToolExecutionError after per-site attempts."""
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


@pytest.mark.asyncio
async def test_fetch_sends_default_chrome_user_agent_header() -> None:
    """By default, requests must carry a Chrome-like UA (not python-httpx/...).

    Bing and DuckDuckGo HTML reject the default httpx user-agent with 302
    / CAPTCHA, so the tool injects a current Chrome UA. This guards
    against accidental regression.
    """
    seen_uas: list[str] = []

    def _h(request: httpx.Request) -> httpx.Response:
        seen_uas.append(request.headers.get("User-Agent", ""))
        return httpx.Response(200, text="<p>ok</p>")

    transport = httpx.MockTransport(_h)
    await _fetch_and_concat(
        "q", ["https://a.test"], cache=None, transport=transport
    )

    assert seen_uas == [_DEFAULT_USER_AGENT], (
        f"expected Chrome UA, got {seen_uas!r}"
    )
    assert "python-httpx" not in _DEFAULT_USER_AGENT


@pytest.mark.asyncio
async def test_fetch_accepts_custom_user_agent_override() -> None:
    """Callers can override the UA via the ``user_agent`` parameter."""
    seen_uas: list[str] = []
    custom_ua = "Mozilla/5.0 (test-bot/1.0)"

    def _h(request: httpx.Request) -> httpx.Response:
        seen_uas.append(request.headers.get("User-Agent", ""))
        return httpx.Response(200, text="<p>ok</p>")

    transport = httpx.MockTransport(_h)
    await _fetch_and_concat(
        "q",
        ["https://a.test"],
        cache=None,
        transport=transport,
        user_agent=custom_ua,
    )

    assert seen_uas == [custom_ua]


@pytest.mark.asyncio
async def test_fetch_follows_redirects_on_302() -> None:
    """httpx's default is ``follow_redirects=False`` for AsyncClient.
    That breaks Bing and Baidu (both 302 to a real endpoint) because
    ``raise_for_status()`` then fires on the redirect itself. This test
    guards the fix: a 302 must be followed transparently, not raised.
    """
    seen_urls: list[str] = []

    def _h(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        # First request: 302 redirect. Second: 200 OK with real content.
        if "redirect.test" in str(request.url):
            return httpx.Response(
                302,
                headers={"Location": "https://final.test/result"},
                text="redirecting",
            )
        return httpx.Response(200, text="<p>real-results</p>")

    transport = httpx.MockTransport(_h)
    result = await _fetch_and_concat(
        "q",
        ["https://redirect.test/"],
        cache=None,
        transport=transport,
    )

    assert "real-results" in result, (
        f"302 not followed; result was {result!r}. seen_urls={seen_urls!r}"
    )


@pytest.mark.asyncio
async def test_fetch_constructs_httpx_client_with_follow_redirects_true() -> None:
    """The httpx.AsyncClient must be created with ``follow_redirects=True``.
    Direct constructor-kwarg assertion, independent of MockTransport behavior."""
    import stock_analysis_agent.tools.web_search as ws_module

    captured: dict = {}
    real_async_client = httpx.AsyncClient

    class _RecordingClient:
        def __init__(self, *args, **kwargs) -> None:
            captured.update(kwargs)
            self._real = real_async_client(*args, **kwargs)

        async def __aenter__(self) -> httpx.AsyncClient:
            return await self._real.__aenter__()

        async def __aexit__(self, *args: object) -> None:
            await self._real.__aexit__(*args)

    ws_module.httpx.AsyncClient = _RecordingClient  # type: ignore[misc]
    try:
        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, text="<p>ok</p>")
        )
        await _fetch_and_concat(
            "q", ["https://a.test"], cache=None, transport=transport
        )
    finally:
        ws_module.httpx.AsyncClient = real_async_client  # type: ignore[misc]

    assert captured.get("follow_redirects") is True, (
        f"expected follow_redirects=True on AsyncClient, got {captured!r}"
    )


# ---------------------------------------------------------------------------
# CAPTCHA detection — surface captcha pages as [error: ...] segments so the
# LLM does not loop trying to "use" the captcha UI text.
# ---------------------------------------------------------------------------


def test_captcha_detected_by_url() -> None:
    """Final URL on a known captcha domain → captcha flagged."""
    assert _is_captcha_response(
        url="https://wappass.baidu.com/static/captcha/tuxing_v2.html",
        text="<html>some text</html>",
    )
    assert _is_captcha_response(
        url="https://qcaptcha.so.com/?ret=...",
        text="ok",
    )


def test_captcha_detected_by_text() -> None:
    """Verification-page text → captcha flagged even on a benign URL."""
    assert _is_captcha_response(
        url="https://www.example.com/s",
        text="请输入验证码 to continue",
    )
    assert _is_captcha_response(
        url="https://www.example.com/s",
        text="<p>人机识别</p>",
    )
    assert _is_captcha_response(
        url="https://www.example.com/s",
        text="<title>captcha</title>",
    )


def test_normal_search_response_not_flagged() -> None:
    """A real search-results page must NOT be flagged as captcha."""
    assert not _is_captcha_response(
        url="https://cn.bing.com/search?q=foo",
        text="<h2>伊利股份 2025 年报</h2><p>净利润同比增长 ...</p>",
    )
    assert not _is_captcha_response(
        url="https://www.example.com/article/123",
        text="Some normal page content without verification markers.",
    )


@pytest.mark.asyncio
async def test_fetch_treats_captcha_response_as_error_segment() -> None:
    """A 200 OK that lands on a captcha page must be surfaced as
    ``[error: captcha ...]`` rather than returned as if it were results.
    Otherwise the LLM will loop, trying to reason about captcha UI text.
    """
    def _h(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "b.test" in url:
            return httpx.Response(200, text="<p>real-results</p>")
        if "captcha-final.test" in url:
            # The redirected-to URL: 200 with captcha-like body.
            return httpx.Response(200, text="<p>请输入验证码</p>")
        # First request: 302 → captcha-final.test
        return httpx.Response(
            302,
            headers={"Location": "https://captcha-final.test/captcha"},
        )

    result = await _fetch_and_concat(
        "q",
        ["https://a.test", "https://b.test"],
        cache=None,
        transport=httpx.MockTransport(_h),
    )

    # Normal site returns its real content.
    assert "real-results" in result
    # Captcha site is flagged as error, not surfaced as content.
    assert "请输入验证码" not in result, (
        f"captcha text leaked to output: {result!r}"
    )
    assert "[error: captcha page returned]" in result, (
        f"captcha not flagged as error: {result!r}"
    )