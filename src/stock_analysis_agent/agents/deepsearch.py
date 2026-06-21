"""DeepSearchAgent: an LLM-driven deep-research agent.

Wraps a single @tool _web_search function that fans out to a configured
list of external search endpoints (httpx + stdlib HTML parser + file cache).
"""
from __future__ import annotations

import hashlib
import json
import time
from html.parser import HTMLParser
from pathlib import Path


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