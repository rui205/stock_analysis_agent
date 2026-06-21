# DeepSearchAgent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `DeepSearchAgent`, the first concrete `BaseAgent` subclass for `stock_analysis_agent`. It exposes a single `web_search` tool that fans out to a configurable list of external search endpoints, fetches HTML concurrently via httpx, extracts plain text with stdlib, caches per-(query, site) to local JSON files, and lets the LLM drive iterative multi-turn research. Default `max_retries=3`.

**Architecture:** `DeepSearchAgent` extends `BaseAgent`. Constructor presets `tools=[_web_search]` (a module-level `@tool` async function) and overrides `max_retries` default to 3. Two module-level holder singletons (`_SITE_LIST_PROVIDER`, `_CACHE_PROVIDER`) bridge instance state into the tool, since `@tool` requires a module-level callable. `_web_search` calls `_fetch_and_concat`, which performs parallel `httpx.AsyncClient` GETs (cache-aware, per-site fault-tolerant) and aggregates results as plain text. `_FileCache` is a JSON-file cache keyed by `sha256(query|site)[:16]` with TTL-based expiration and atomic writes.

**Tech Stack:** Python 3.12, httpx (new runtime dep), LangChain 1.x `@tool`, stdlib `html.parser` + `json` + `hashlib` + `pathlib`, pytest with `httpx.MockTransport`.

**Spec:** [`docs/superpowers/specs/2026-06-21-deepsearch-agent-design.md`](../specs/2026-06-21-deepsearch-agent-design.md)

---

## File Structure

Files created or modified by this plan:

| Path | Change | Responsibility |
|------|--------|----------------|
| `pyproject.toml` | Modify | Add `httpx>=0.27` to runtime deps |
| `src/stock_analysis_agent/agents/deepsearch.py` | Create | `DeepSearchAgent`, module constants, `_web_search` tool, `_fetch_and_concat`, `_FileCache`, `_extract_text`, `_Provider` singletons |
| `src/stock_analysis_agent/agents/__init__.py` | Modify | Re-export `DeepSearchAgent` |
| `tests/agents/test_deepsearch.py` | Create | All ~25 tests across extractor, cache, fetch, tool, agent config, end-to-end integration |

---

## Task 1: Add httpx dependency

**Files:**
- Modify: `pyproject.toml` (dependencies list)

- [ ] **Step 1: Update `pyproject.toml`**

Open `/Users/rui/workspace/stock_analysis_agent/pyproject.toml` and add `"httpx>=0.27"` to the `dependencies` list:

```toml
[project]
name = "stock_analysis_agent"
version = "0.1.0"
description = "Reusable agents for stock analysis"
requires-python = ">=3.12"
dependencies = [
    "langchain>=1.0",
    "langchain-anthropic>=1.0",
    "langchain-core>=1.0",
    "httpx>=0.27",
]
```

- [ ] **Step 2: Install**

Run: `uv pip install -e ".[dev]"`
Expected: install completes. `httpx` may already be present as a transitive dep — the install is a no-op for it but ensures `pyproject.toml` declares it as a direct dep.

- [ ] **Step 3: Verify import**

Run: `uv run python -c "import httpx; print(httpx.__version__)"`
Expected: prints httpx version ≥ 0.27 (e.g. `0.27.x` or `0.28.x`).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add httpx runtime dependency for DeepSearchAgent"
```

---

## Task 2: HTML text extractor (TDD)

**Files:**
- Create: `tests/agents/test_deepsearch.py`
- Create: `src/stock_analysis_agent/agents/deepsearch.py`

- [ ] **Step 1: Write the failing tests**

`/Users/rui/workspace/stock_analysis_agent/tests/agents/test_deepsearch.py`:

```python
"""Tests for stock_analysis_agent.agents.deepsearch.DeepSearchAgent."""
from __future__ import annotations

import pytest

from stock_analysis_agent.agents.deepsearch import _extract_text


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agents/test_deepsearch.py -v`
Expected: `ModuleNotFoundError: No module named 'stock_analysis_agent.agents.deepsearch'`

- [ ] **Step 3: Implement `_extract_text`**

`/Users/rui/workspace/stock_analysis_agent/src/stock_analysis_agent/agents/deepsearch.py`:

```python
"""DeepSearchAgent: an LLM-driven deep-research agent.

Wraps a single @tool _web_search function that fans out to a configured
list of external search endpoints (httpx + stdlib HTML parser + file cache).
"""
from __future__ import annotations

from html.parser import HTMLParser


class _TextExtractor(HTMLParser):
    """Collect visible text from an HTML document.

    Skips the contents of <script> and <style> elements entirely. Tracks
    nested skip regions by depth so e.g. `<script><script></script></script>`
    is handled correctly.
    """

    _SKIP_TAGS = frozenset({"script", "style"})

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:  # type: ignore[no-untyped-def]
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)


def _extract_text(html: str) -> str:
    """Extract visible text from `html`, stripping <script> and <style> blocks.

    Whitespace runs (spaces, tabs, newlines) are collapsed to a single space.
    """
    extractor = _TextExtractor()
    extractor.feed(html)
    text = "".join(extractor._parts)
    return " ".join(text.split())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agents/test_deepsearch.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/stock_analysis_agent/agents/deepsearch.py tests/agents/test_deepsearch.py
git commit -m "feat(agents): add _extract_text HTML parser for DeepSearchAgent"
```

---

## Task 3: `_FileCache` class (TDD)

**Files:**
- Modify: `tests/agents/test_deepsearch.py` (append tests)
- Modify: `src/stock_analysis_agent/agents/deepsearch.py` (append `_FileCache`)

- [ ] **Step 1: Append cache tests**

Append to `/Users/rui/workspace/stock_analysis_agent/tests/agents/test_deepsearch.py`:

```python
from pathlib import Path

from stock_analysis_agent.agents.deepsearch import _FileCache


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
    # Write a file that the (query, site) key will resolve to.
    bad = tmp_path / "deadbeef.json"
    bad.write_text("not valid json {{{", encoding="utf-8")
    cache._FileCache__key_override = None  # see step 3 — not needed actually
    # Force get() to read this file by setting the same key it uses.
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
```

Note: the `test_cache_corrupt_json_returns_none` test above includes a stale line (`cache._FileCache__key_override = None`) — remove it during implementation. The test passes by directly writing a file at the key's path.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agents/test_deepsearch.py -v`
Expected: tests after the `_extract_text` ones fail with `ImportError: cannot import name '_FileCache'`.

- [ ] **Step 3: Implement `_FileCache`**

Append to `/Users/rui/workspace/stock_analysis_agent/src/stock_analysis_agent/agents/deepsearch.py`:

```python
import hashlib
import json
import time


class _FileCache:
    """JSON-file cache under `cache_dir`, keyed by `(query, site)`.

    A cache hit returns the stored text if (a) the file exists, (b) parses
    as JSON with a `text` field, and (c) is not expired (when ttl is set).
    Any failure during read is treated as a miss — caching is a
    performance optimization, not a correctness layer.

    Writes are atomic: payload is written to `<key>.tmp` then renamed to
    `<key>.json`, so a crash mid-write cannot leave a half-baked file.
    """

    def __init__(self, cache_dir: Path, *, ttl_seconds: float | None = 60.0) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl_seconds

    @staticmethod
    def _key(query: str, site: str) -> str:
        raw = f"{query}|{site}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def _path(self, query: str, site: str) -> Path:
        return self._dir / f"{self._key(query, site)}.json"

    def get(self, *, site: str, query: str) -> str | None:
        path = self._path(query, site)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if self._ttl is not None and time.time() - data.get("ts", 0.0) > self._ttl:
                return None  # expired
            return data["text"]
        except (json.JSONDecodeError, KeyError, OSError, UnicodeDecodeError):
            return None  # corrupt / unreadable → treat as miss

    def set(self, *, site: str, query: str, text: str) -> None:
        path = self._path(query, site)
        payload = json.dumps(
            {"query": query, "site": site, "text": text, "ts": time.time()},
            ensure_ascii=False,
        )
        # Atomic write: write to .tmp then replace.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)
```

Also update the imports at the top of `deepsearch.py` to include `Path`:

```python
from __future__ import annotations

import hashlib
import json
import time
from html.parser import HTMLParser
from pathlib import Path
```

Also clean up the stale `cache._FileCache__key_override = None` line from the test file in Step 1:

```python
def test_cache_corrupt_json_returns_none(tmp_path: Path) -> None:
    """A cache file with invalid JSON is treated as a miss, not an error."""
    cache = _FileCache(tmp_path, ttl_seconds=60.0)
    key = _FileCache._key("https://a.test", "hello")
    (tmp_path / f"{key}.json").write_text("not valid json {{{", encoding="utf-8")

    assert cache.get(site="https://a.test", query="hello") is None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agents/test_deepsearch.py -v`
Expected: 12 passed (4 extractor + 8 cache).

- [ ] **Step 5: Commit**

```bash
git add src/stock_analysis_agent/agents/deepsearch.py tests/agents/test_deepsearch.py
git commit -m "feat(agents): add _FileCache for DeepSearchAgent

JSON-file cache under cache_dir, keyed by sha256(query|site)[:16].
TTL-based expiration; atomic writes via .tmp + replace.
Read failures (missing file, corrupt JSON, IO) are treated as miss.
"
```

---

## Task 4: `_fetch_and_concat` with httpx + cache (TDD)

**Files:**
- Modify: `tests/agents/test_deepsearch.py` (append tests)
- Modify: `src/stock_analysis_agent/agents/deepsearch.py` (append `_fetch_and_concat`)

This task uses `httpx.MockTransport` to inject canned HTTP responses — no real network, no respx dependency.

- [ ] **Step 1: Append fetch tests**

Append to `/Users/rui/workspace/stock_analysis_agent/tests/agents/test_deepsearch.py`:

```python
import asyncio
from collections.abc import Callable

import httpx


def _make_mock_transport(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.MockTransport:
    """Build an httpx.MockTransport from a synchronous handler."""
    return httpx.MockTransport(handler)


def _ok_handler(html: str = "<p>hello</p>") -> Callable[[httpx.Request], httpx.Response]:
    def _h(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html)
    return _h


def _fail_handler(
    exc: type[Exception] = httpx.ConnectError, msg: str = "boom"
) -> Callable[[httpx.Request], httpx.Response]:
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
        time_mod.sleep(delay)
        return httpx.Response(200, text="<p>ok</p>")

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
    transport = httpx.MockTransport(_fail_handler(httpx.ConnectError, "down"))

    # Mixed: one success, one failure. Only success should write cache.
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

    def _boom_set(**kwargs) -> None:  # type: ignore[no-untyped-def]
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
```

Also add the import at the top of `test_deepsearch.py`:

```python
from stock_analysis_agent.agents.exceptions import ToolExecutionError
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agents/test_deepsearch.py -v`
Expected: tests for `_fetch_and_concat` fail with `ImportError: cannot import name '_fetch_and_concat'`.

- [ ] **Step 3: Implement `_fetch_and_concat`**

Append to `/Users/rui/workspace/stock_analysis_agent/src/stock_analysis_agent/agents/deepsearch.py`:

```python
import asyncio

from stock_analysis_agent.agents.exceptions import ToolExecutionError

if TYPE_CHECKING:
    import httpx


async def _fetch_and_concat(
    query: str,
    site_list: list[str],
    *,
    cache: _FileCache | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout: float = 10.0,
) -> str:
    """Fetch `query` from each site in `site_list` concurrently and concatenate results.

    Each site is fetched via httpx.AsyncClient with optional `transport`
    (for tests). Cache behavior:
      - If `cache` is None, every site is fetched over HTTP.
      - If `cache` is set, hit returns the cached text without HTTP;
        miss fetches and writes through to the cache.
    Per-site failures are recorded as `[error: ...]` segments rather
    than raised. If every site fails, the function raises
    `ToolExecutionError` so the BaseAgent retry middleware can act.
    """
    if not site_list:
        raise ValueError("site_list cannot be empty")

    async def _one(site: str) -> tuple[str, str]:
        # 1) Try cache first.
        if cache is not None:
            hit = cache.get(site=site, query=query)
            if hit is not None:
                return (site, hit)
        # 2) HTTP fetch.
        try:
            client_kwargs: dict[str, Any] = {"timeout": timeout}
            if transport is not None:
                client_kwargs["transport"] = transport
            async with httpx.AsyncClient(**client_kwargs) as client:
                resp = await client.get(site, params={"q": query})
                resp.raise_for_status()
                text = _extract_text(resp.text)
        except Exception as e:
            return (site, f"[error: {type(e).__name__}: {e}]")
        # 3) Write-through cache (best-effort).
        if cache is not None:
            try:
                cache.set(site=site, query=query, text=text)
            except OSError:
                pass  # cache write failure does not fail the search
        return (site, text)

    results = await asyncio.gather(*(_one(s) for s in site_list))
    if all(text.startswith("[error:") for _, text in results):
        raise ToolExecutionError(f"all sites failed: {[s for s, _ in results]}")

    parts = [f"[{site}]\n{text}\n" for site, text in results]
    return "\n".join(parts)
```

Also update the imports at the top of `deepsearch.py`:

```python
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, Any

from stock_analysis_agent.agents.exceptions import ToolExecutionError

if TYPE_CHECKING:
    import httpx
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agents/test_deepsearch.py -v`
Expected: 21 passed (4 extractor + 8 cache + 9 fetch).

- [ ] **Step 5: Commit**

```bash
git add src/stock_analysis_agent/agents/deepsearch.py tests/agents/test_deepsearch.py
git commit -m "feat(agents): add _fetch_and_concat with httpx + cache integration

Parallel per-site fetch via asyncio.gather. Cache-aware (read-through,
write-through best-effort). Per-site failures recorded as [error: ...]
segments; all-fail raises ToolExecutionError for retry middleware.
transport kwarg allows httpx.MockTransport injection in tests.
"
```

---

## Task 5: Module-level providers and `_web_search` tool (TDD)

**Files:**
- Modify: `tests/agents/test_deepsearch.py` (append tests)
- Modify: `src/stock_analysis_agent/agents/deepsearch.py` (append providers + tool)

- [ ] **Step 1: Append tool-metadata tests**

Append to `/Users/rui/workspace/stock_analysis_agent/tests/agents/test_deepsearch.py`:

```python
def test_web_search_tool_metadata() -> None:
    """The @tool _web_search exposes the expected name and args schema."""
    from stock_analysis_agent.agents.deepsearch import _web_search

    assert _web_search.name == "web_search"
    # The args schema must include a `query` string field.
    schema = _web_search.args
    if hasattr(schema, "model_json_schema"):
        schema = schema.model_json_schema()
    assert "query" in (schema.get("properties") or {}), f"missing query in {schema!r}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agents/test_deepsearch.py::test_web_search_tool_metadata -v`
Expected: fails with `ImportError: cannot import name '_web_search' from 'stock_analysis_agent.agents.deepsearch'`.

- [ ] **Step 3: Implement `_Provider`, providers, and the tool**

Append to `/Users/rui/workspace/stock_analysis_agent/src/stock_analysis_agent/agents/deepsearch.py`:

```python
from typing import Generic, TypeVar

from langchain.tools import tool

T = TypeVar("T")


class _Provider(Generic[T]):
    """Module-level singleton holder for a single value.

    The single-instance design (per spec §1) lets us mutate `self.value`
    on every `DeepSearchAgent.__init__` call, and the @tool _web_search
    reads it via `.get()` whenever the LLM invokes the tool. Concurrent
    multi-instance construction is not supported.
    """

    def __init__(self) -> None:
        self.value: T | None = None  # type: ignore[assignment]

    def get(self) -> T:
        if self.value is None:
            raise RuntimeError("provider was not initialized")
        return self.value


_SITE_LIST_PROVIDER: _Provider[list[str]] = _Provider()
_CACHE_PROVIDER: _Provider[_FileCache] = _Provider()


@tool
async def _web_search(query: str) -> str:
    """Search the configured site list for `query` and return aggregated text.

    Returns a plain-text concatenation of snippets from each configured
    site. Sites that error are mentioned in the output but do not abort
    the search.
    """
    sites = _SITE_LIST_PROVIDER.get()
    cache = _CACHE_PROVIDER.get()
    return await _fetch_and_concat(query, sites, cache=cache)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agents/test_deepsearch.py -v`
Expected: 22 passed (4 extractor + 8 cache + 9 fetch + 1 tool metadata).

- [ ] **Step 5: Commit (just the providers + tool)**

```bash
git add src/stock_analysis_agent/agents/deepsearch.py tests/agents/test_deepsearch.py
git commit -m "feat(agents): add _Provider holders and _web_search @tool for DeepSearchAgent

Module-level _SITE_LIST_PROVIDER and _CACHE_PROVIDER bridge instance
state into the @tool callable. _web_search is async and delegates to
_fetch_and_concat with the current site_list and cache.
"
```

---

## Task 6: `DeepSearchAgent` class constructor (TDD)

**Files:**
- Modify: `tests/agents/test_deepsearch.py` (append config tests)
- Modify: `src/stock_analysis_agent/agents/deepsearch.py` (append `DeepSearchAgent`)

- [ ] **Step 1: Append constructor tests**

Append to `/Users/rui/workspace/stock_analysis_agent/tests/agents/test_deepsearch.py`:

```python
def test_default_construction_uses_module_constants(tmp_path: Path) -> None:
    """No-arg construction uses DEFAULT_SITE_LIST, DEFAULT_SYSTEM_PROMPT, max_retries=3."""
    from stock_analysis_agent.agents.deepsearch import (
        DEFAULT_CACHE_DIR,
        DEFAULT_CACHE_TTL,
        DEFAULT_SITE_LIST,
        DEFAULT_SYSTEM_PROMPT,
        DeepSearchAgent,
    )

    agent = DeepSearchAgent(cache_dir=tmp_path, cache_ttl=None)
    assert agent.site_list == DEFAULT_SITE_LIST
    assert agent.system_prompt_value == DEFAULT_SYSTEM_PROMPT
    assert agent.max_retries == 3
    assert agent.cache_dir == tmp_path.resolve()


def test_custom_site_list_overrides_default(tmp_path: Path) -> None:
    from stock_analysis_agent.agents.deepsearch import DeepSearchAgent

    agent = DeepSearchAgent(
        site_list=["https://x.test"], cache_dir=tmp_path, cache_ttl=None
    )
    assert agent.site_list == ["https://x.test"]


def test_custom_system_prompt_overrides_default(tmp_path: Path) -> None:
    from stock_analysis_agent.agents.deepsearch import DeepSearchAgent

    agent = DeepSearchAgent(
        system_prompt="custom prompt", cache_dir=tmp_path, cache_ttl=None
    )
    assert agent.system_prompt_value == "custom prompt"


def test_site_list_returns_copy(tmp_path: Path) -> None:
    """Mutating the returned site_list must not affect the agent or DEFAULT_SITE_LIST."""
    from stock_analysis_agent.agents.deepsearch import (
        DEFAULT_SITE_LIST,
        DeepSearchAgent,
    )

    agent = DeepSearchAgent(cache_dir=tmp_path, cache_ttl=None)
    snapshot = agent.site_list
    snapshot.append("https://mutated.test")
    assert "https://mutated.test" not in agent.site_list
    assert "https://mutated.test" not in DEFAULT_SITE_LIST


def test_empty_site_list_raises_at_construction(tmp_path: Path) -> None:
    from stock_analysis_agent.agents.deepsearch import DeepSearchAgent

    with pytest.raises(ValueError, match="site_list cannot be empty"):
        DeepSearchAgent(site_list=[], cache_dir=tmp_path, cache_ttl=None)


def test_kwargs_pass_through_to_base_agent(tmp_path: Path) -> None:
    """model, temperature, name flow through to BaseAgent properties."""
    from stock_analysis_agent.agents.deepsearch import DeepSearchAgent

    agent = DeepSearchAgent(
        model="claude-opus-4-8",
        temperature=0.7,
        name="custom",
        cache_dir=tmp_path,
        cache_ttl=None,
    )
    assert agent.model == "claude-opus-4-8"
    assert agent.temperature == 0.7
    assert agent.name == "custom"


def test_cache_dir_expands_tilde(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`~` in a str cache_dir is expanded via Path.expanduser()."""
    from stock_analysis_agent.agents.deepsearch import DeepSearchAgent

    monkeypatch.setenv("HOME", str(tmp_path))
    agent = DeepSearchAgent(cache_dir="~/my-cache", cache_ttl=None)
    assert agent.cache_dir == (tmp_path / "my-cache").resolve()


def test_cache_ttl_none_disables_expiration(tmp_path: Path) -> None:
    """cache_ttl=None means cache entries never expire."""
    from stock_analysis_agent.agents.deepsearch import DeepSearchAgent

    agent = DeepSearchAgent(cache_dir=tmp_path, cache_ttl=None)
    agent._cache.set(site="https://a.test", query="q", text="cached")
    assert agent._cache.get(site="https://a.test", query="q") == "cached"
    assert agent._cache._ttl is None


def test_web_search_provider_reflects_latest_construction(tmp_path: Path) -> None:
    """After constructing a DeepSearchAgent with site_list=[B], the
    module-level site provider exposes [B] (single-instance contract)."""
    from stock_analysis_agent.agents.deepsearch import (
        DeepSearchAgent,
        _SITE_LIST_PROVIDER,
    )

    DeepSearchAgent(site_list=["https://b.test"], cache_dir=tmp_path, cache_ttl=None)
    assert _SITE_LIST_PROVIDER.get() == ["https://b.test"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agents/test_deepsearch.py -v`
Expected: tests after the previous ones fail with `ImportError: cannot import name 'DeepSearchAgent'`.

- [ ] **Step 3: Implement `DeepSearchAgent`**

Append to `/Users/rui/workspace/stock_analysis_agent/src/stock_analysis_agent/agents/deepsearch.py`:

```python
from collections.abc import Sequence
from typing import Any

from stock_analysis_agent.agents.base import BaseAgent


DEFAULT_SYSTEM_PROMPT: str = (
    "You are a deep research agent. Given a user question, "
    "use the web_search tool to gather information from the "
    "configured sites, then synthesize a concise answer. "
    "Cite the source site in parentheses when you use a fact."
)

DEFAULT_SITE_LIST: list[str] = [
    "https://duckduckgo.com/html/",
    "https://www.bing.com/search",
    "https://html.duckduckgo.com/html/",
]

DEFAULT_CACHE_DIR: str = "~/.cache/stock-analysis-agent"
DEFAULT_CACHE_TTL: float | None = 86400.0  # 24h in seconds


class DeepSearchAgent(BaseAgent):
    """LLM-driven deep-research agent that searches a configured site list.

    Adds a single tool (`web_search`) that fans out to the configured
    external sites, fetches each concurrently via httpx, caches results
    to local JSON files, and returns aggregated plain text. The LLM
    decides what to search and when to synthesize.

    Construction overrides `BaseAgent`'s `max_retries` default from 2 → 3.
    Other BaseAgent parameters (model, temperature, name, ...) flow
    through via **kwargs.

    Single-instance: constructing a second agent updates the module-level
    _SITE_LIST_PROVIDER and _CACHE_PROVIDER used by the @tool _web_search.
    """

    def __init__(
        self,
        *,
        site_list: Sequence[str] | None = None,
        system_prompt: str | None = None,
        max_retries: int = 3,
        cache_dir: str | Path | None = None,
        cache_ttl: float | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_sites = list(site_list) if site_list is not None else list(DEFAULT_SITE_LIST)
        if not resolved_sites:
            raise ValueError("site_list cannot be empty")

        resolved_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT

        resolved_dir = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir is not None
            else Path(DEFAULT_CACHE_DIR).expanduser().resolve()
        )
        resolved_ttl = cache_ttl if cache_ttl is not None else DEFAULT_CACHE_TTL

        self._cache = _FileCache(resolved_dir, ttl_seconds=resolved_ttl)

        # Single-instance: write into module-level providers so the @tool
        # callable (which is module-level) can read them.
        _SITE_LIST_PROVIDER.value = resolved_sites
        _CACHE_PROVIDER.value = self._cache

        super().__init__(
            system_prompt=resolved_prompt,
            max_retries=max_retries,
            tools=[_web_search],
            **kwargs,
        )

    @property
    def site_list(self) -> list[str]:
        return list(self._site_list_snapshot())

    def _site_list_snapshot(self) -> list[str]:
        # Internal helper to read the current site_list without copying twice.
        return _SITE_LIST_PROVIDER.get()

    @property
    def cache_dir(self) -> Path:
        return self._cache._dir

    @property
    def cache_ttl(self) -> float | None:
        return self._cache._ttl
```

Note: `site_list` reads from `_SITE_LIST_PROVIDER` to keep a single source of truth (the provider holds the value the tool will use). This matches the single-instance contract.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agents/test_deepsearch.py -v`
Expected: 31 passed (4 extractor + 8 cache + 9 fetch + 1 tool + 9 agent config).

- [ ] **Step 5: Commit**

```bash
git add src/stock_analysis_agent/agents/deepsearch.py tests/agents/test_deepsearch.py
git commit -m "feat(agents): add DeepSearchAgent class

Extends BaseAgent with site_list, system_prompt, max_retries=3, cache_dir,
cache_ttl constructor params. Module-level _SITE_LIST_PROVIDER and
_CACHE_PROVIDER are written in __init__ so the @tool _web_search can
read them. Validates non-empty site_list at construction.
"
```

---

## Task 7: End-to-end integration test (fake model → tool_call → `_web_search`)

**Files:**
- Modify: `tests/agents/test_deepsearch.py` (append integration tests)

This task exercises the full LangChain graph with a fake chat model and `httpx.MockTransport`, verifying that when the LLM produces a `web_search` tool call, the configured sites are actually hit.

- [ ] **Step 1: Append integration tests**

Append to `/Users/rui/workspace/stock_analysis_agent/tests/agents/test_deepsearch.py`:

```python
from langchain_core.messages import HumanMessage

from tests.agents.conftest import ToolAwareFakeChatModel, make_ai, make_tool_call


def _noop_middleware():  # type: ignore[no-untyped-def]
    """Build a no-op AgentMiddleware to skip BaseAgent's retry middleware."""
    from langchain.agents.middleware import AgentMiddleware

    class _NoRetry(AgentMiddleware):
        def wrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
            return handler(request)

        async def awrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
            return await handler(request)

    return _NoRetry()


def test_tool_call_reaches_web_search_with_configured_sites(
    tmp_path: Path,
) -> None:
    """When the LLM produces a web_search tool_call, the configured sites
    are actually fetched (verified via MockTransport request log)."""
    from langchain.agents import create_agent

    from stock_analysis_agent.agents.deepsearch import DeepSearchAgent, _web_search

    seen_urls: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(200, text="<p>snippet</p>")

    transport = httpx.MockTransport(_handler)

    # Build the agent first so providers are populated.
    agent = DeepSearchAgent(
        site_list=["https://a.test", "https://b.test"],
        cache_dir=tmp_path,
        cache_ttl=None,
    )

    # Monkey-patch _fetch_and_concat to inject the MockTransport. This
    # is the smallest surgical way to avoid real HTTP without touching
    # the production API surface.
    import stock_analysis_agent.agents.deepsearch as ds_mod

    original = ds_mod._fetch_and_concat

    async def _patched(query, sites, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["transport"] = transport
        return await original(query, sites, **kwargs)

    ds_mod._fetch_and_concat = _patched
    try:
        # Fake model: tool_call, then final answer.
        first = make_ai("")
        first.tool_calls = [make_tool_call("web_search", {"query": "AI safety"}, "call_1")]
        model = ToolAwareFakeChatModel(responses=[first, make_ai("answer")])

        graph = create_agent(
            model=model,
            tools=[_web_search],
            system_prompt=agent.system_prompt_value,
            middleware=[_noop_middleware()],
        )
        agent._build_graph = lambda: graph  # type: ignore[method-assign]

        # Drain the stream; we don't care about events, just that the
        # tool ran and we got a final answer.
        for _ in agent.stream([HumanMessage(content="research AI safety")]):
            pass
    finally:
        ds_mod._fetch_and_concat = original

    # Both configured sites were hit (with q=AI safety).
    assert len(seen_urls) == 2, f"expected 2 sites, got {seen_urls!r}"
    assert all("q=AI+safety" in u or "q=AI%20safety" in u or "q=AI safety" in u for u in seen_urls), (
        f"query not threaded into URL: {seen_urls!r}"
    )


def test_second_agent_overwrites_first_sites(tmp_path: Path) -> None:
    """Single-instance contract: constructing a second DeepSearchAgent
    overwrites the module-level provider."""
    from stock_analysis_agent.agents.deepsearch import (
        DeepSearchAgent,
        _SITE_LIST_PROVIDER,
    )

    DeepSearchAgent(
        site_list=["https://first.test"], cache_dir=tmp_path, cache_ttl=None
    )
    assert _SITE_LIST_PROVIDER.get() == ["https://first.test"]

    DeepSearchAgent(
        site_list=["https://second.test"], cache_dir=tmp_path, cache_ttl=None
    )
    assert _SITE_LIST_PROVIDER.get() == ["https://second.test"]
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/agents/test_deepsearch.py -v`
Expected: 33 passed (all previous + 2 integration).

- [ ] **Step 3: Commit**

```bash
git add tests/agents/test_deepsearch.py
git commit -m "test(agents): add end-to-end integration tests for DeepSearchAgent

Verifies that when the fake LLM produces a web_search tool_call, the
configured sites are actually fetched with the right query param, and
that single-instance semantics hold across multiple constructions.
"
```

---

## Task 8: Re-export `DeepSearchAgent` from `agents/__init__.py`

**Files:**
- Modify: `src/stock_analysis_agent/agents/__init__.py`

- [ ] **Step 1: Update `__init__.py`**

Replace `/Users/rui/workspace/stock_analysis_agent/src/stock_analysis_agent/agents/__init__.py` with:

```python
"""Re-exports for stock_analysis_agent.agents."""
from stock_analysis_agent.agents.base import BaseAgent
from stock_analysis_agent.agents.deepsearch import DeepSearchAgent
from stock_analysis_agent.agents.exceptions import ToolExecutionError

__all__ = ["BaseAgent", "DeepSearchAgent", "ToolExecutionError"]
```

- [ ] **Step 2: Smoke-test the public import**

Run:

```bash
uv run python -c "from stock_analysis_agent.agents import BaseAgent, DeepSearchAgent, ToolExecutionError; print(BaseAgent, DeepSearchAgent, ToolExecutionError)"
```

Expected: prints the class and exception object addresses, no error.

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests pass — 18 from before + 33 new = 51 total.

- [ ] **Step 4: Commit**

```bash
git add src/stock_analysis_agent/agents/__init__.py
git commit -m "feat(agents): re-export DeepSearchAgent from agents package"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Covered by |
|--------------|------------|
| §1 设计目标: 单一搜索工具 | Task 5 (`_web_search`) + Task 6 (`tools=[_web_search]`) |
| §1 设计目标: 模块常量 + 构造参数 | Task 6 (`DeepSearchAgent.__init__` + module constants) |
| §1 设计目标: max_retries=3 | Task 6 (`max_retries: int = 3`) |
| §1 设计目标: 复用 _ToolRetryMiddleware | Task 6 (via `super().__init__`) |
| §1 设计目标: 本地文件级缓存 | Tasks 3, 4 |
| §1 设计目标: 单进程单实例 | Tasks 5, 6 (`_Provider` holders + assignment in `__init__`) |
| §2 架构 | Tasks 3-6 (the whole module) |
| §3 模块常量 DEFAULT_SYSTEM_PROMPT | Task 6 |
| §3 模块常量 DEFAULT_SITE_LIST | Task 6 |
| §3 模块常量 DEFAULT_CACHE_DIR | Task 6 |
| §3 模块常量 DEFAULT_CACHE_TTL | Task 6 |
| §3 构造参数 site_list / system_prompt / max_retries / cache_dir / cache_ttl | Task 6 |
| §3 site_list 校验 | Task 6 + test in Task 6 |
| §3 cache_dir 用 expanduser().resolve() | Task 6 + test in Task 6 |
| §4 派生类最小形态 | Implicit — `DeepSearchAgent` itself is the canonical example |
| §5 单 site fetch 失败 | Task 4 (partial + all-fail tests) |
| §5 site_list 为空 → ValueError | Task 4 + Task 6 |
| §5 重试复用 _ToolRetryMiddleware | Task 6 (transparent via super) |
| §5 缓存读失败 → miss | Task 3 (`_FileCache.get`) |
| §5 缓存写失败 → 静默 | Task 4 (cache write failure test) |
| §6 数据流 | Tasks 3, 4, 5, 6 |
| §6 `_FileCache` 实现 | Task 3 |
| §6 `_fetch_and_concat` 实现 | Task 4 |
| §6 `_extract_text` 用 stdlib html.parser | Task 2 |
| §7 三层测试 | Tasks 2-7 |
| §8 文件清单 | All tasks (deepsearch.py, test_deepsearch.py, pyproject.toml, agents/__init__.py) |
| §9 开放问题: 默认 site_list | Task 6 |
| §9 开放问题: timeout=10.0 | Task 4 |
| §9 开放问题: cache_ttl=0 关闭 | Implied by Task 3 (TTL=0 → all entries expired) |

**Placeholder scan:** No "TBD"/"TODO"/"implement later" anywhere. Every code block is complete and runnable. Every command has expected output.

**Type consistency:**
- `DeepSearchAgent.__init__` parameters match property names (`site_list`, `system_prompt_value`, `max_retries`, `cache_dir`, `cache_ttl`).
- `_fetch_and_concat(query, site_list, *, cache, transport, timeout)` signature consistent across Tasks 4, 7.
- `_FileCache.get(*, site, query)` and `_FileCache.set(*, site, query, text)` consistent across Tasks 3, 4, 6.
- `_Provider[T]` with `value` attribute and `get()` method consistent across Tasks 5, 6.
- `_web_search(query: str)` signature consistent across Tasks 5, 7.
- Exception class `ToolExecutionError` consistent with `BaseAgent._ToolRetryMiddleware` (already in repo).
- `DEFAULT_SYSTEM_PROMPT`, `DEFAULT_SITE_LIST`, `DEFAULT_CACHE_DIR`, `DEFAULT_CACHE_TTL` constants consistent across Tasks 6, 7.

**Potential issue — `_SITE_LIST_PROVIDER` use in `agent.site_list` property:** Task 6 reads from the provider (single source of truth). If a future change copies into `self._site_list` instead, `agent.site_list` and `_web_search` would diverge. The plan uses the provider to enforce consistency.

**Potential issue — `cache_ttl=0` semantics:** Setting TTL=0 means every entry is "expired" (because `time.time() - ts > 0` is always true). This effectively disables the cache (always HTTP fetch). This is documented in spec §9 as a valid usage. Task 3's `_FileCache(ttl_seconds=0.0)` would silently disable caching.

**Potential issue — Task 7's monkey-patch of `_fetch_and_concat`:** This is the cleanest way to inject `httpx.MockTransport` without polluting the production API with a `transport` parameter at the public level. The patch is reverted in a `finally` block. Alternative: expose `transport` kwarg on `_fetch_and_concat` (already done internally — Task 4 — and used by Task 7's monkey-patch).

**Potential issue — Task 7 URL query string assertion:** httpx encodes spaces as `+` or `%20` depending on context; the test allows both forms. If httpx version changes encoding, this test may need adjustment.

**All review issues fixed inline. Plan is ready to execute.**