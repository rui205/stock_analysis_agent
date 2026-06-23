"""Tests for stock_analysis_agent.memory._FileCache."""
from __future__ import annotations

from pathlib import Path

import pytest

from stock_analysis_agent.memory import _FileCache


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