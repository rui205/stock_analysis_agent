"""HTML → plain-text extraction (stdlib only, strips <script>/<style>)."""
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