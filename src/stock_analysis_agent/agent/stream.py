"""Helpers for collecting text from LangChain agent stream events."""
from __future__ import annotations

from collections.abc import Iterator

from langchain_core.runnables.schema import StreamEvent


def chunk_text(content: object) -> str:
    """Return the text carried by a streamed chat chunk's ``content``.

    A chunk ``content`` is either a plain string or a list of content
    blocks (each ``{"type": "text", "text": ...}``). Any other shape
    yields the empty string so callers can accumulate unconditionally.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def collect_final_text(events: Iterator[StreamEvent]) -> str:
    """Accumulate the text of every ``on_chat_model_stream`` event.

    Args:
        events: Iterator of LangChain stream-event dicts (as yielded by
            :meth:`BaseAgent.stream` / ``astream``).

    Returns:
        Concatenated text of all streamed chat chunks, in arrival order.
    """
    last_text = ""
    for event in events:
        if event.get("event") != "on_chat_model_stream":
            continue
        chunk = event.get("data", {}).get("chunk", {})
        last_text += chunk_text(getattr(chunk, "content", ""))
    return last_text


__all__ = ["chunk_text", "collect_final_text"]
