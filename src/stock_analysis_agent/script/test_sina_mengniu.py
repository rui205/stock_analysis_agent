"""One-shot smoke test: drive DeepSearchAgent against Sina search API.

Usage:
    python -m stock_analysis_agent.script.test_sina_mengniu
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from langchain_core.messages import HumanMessage

from stock_analysis_agent.agent.deepsearch import DeepSearchAgent

# Sina's /api/search endpoint accepts GET ?q=... and returns JSON. The
# web_search tool appends ?q=<query> automatically, so the site URL must
# not already contain a `q` placeholder. Extra params (`range`, `c`)
# are appended after `q` by httpx, yielding:
#   .../api/search?q=<query>&range=all&c=news
SINA_SEARCH_URL = "https://search.sina.com.cn/api/search?range=all&c=news"

USER_QUESTION = "请查询新浪财经关于蒙牛股份(02319.HK)的最新分析情况,包括近期业绩、市场表现和分析师观点。"


def _print_event(event: dict) -> None:
    kind = event.get("event", "")
    name = event.get("name", "")
    if kind == "on_chain_start" and name in ("agent", "model", "tools"):
        print(f"[chain-start] {name}", flush=True)
    elif kind == "on_chain_end" and name in ("agent", "model", "tools"):
        print(f"[chain-end]   {name}", flush=True)
    elif kind == "on_tool_start":
        print(f"[tool-start]  {name}  args={event['data'].get('input')}", flush=True)
    elif kind == "on_tool_end":
        output = event["data"].get("output")
        snippet = repr(output)[:300] if output is not None else "None"
        print(f"[tool-end]    {name}  output={snippet}", flush=True)
    elif kind == "on_chat_model_start":
        print(f"[llm-start]   {name}", flush=True)
    elif kind == "on_chat_model_end":
        msg = event["data"].get("output", {})
        content = getattr(msg, "content", "")
        tool_calls = getattr(msg, "tool_calls", [])
        if content:
            print(f"[llm-text]    {content[:200]}", flush=True)
        for tc in tool_calls or []:
            print(f"[llm-tool]    -> {tc.get('name')}({tc.get('args')})", flush=True)
        if not content and not tool_calls:
            print(f"[llm-end]     {name}  (no content/tool_calls)", flush=True)


def main() -> int:
    site_list = [SINA_SEARCH_URL]
    with tempfile.TemporaryDirectory() as tmp:
        agent = DeepSearchAgent(
            site_list=site_list,
            cache_dir=Path(tmp),
            cache_ttl=None,  # disable TTL so we always get fresh fetches
            max_retries=2,
            system_prompt=(
                "你是一个财经研究助手。调用 web_search 工具检索用户提到的财经信息,"
                "工具会返回新浪财经的原始搜索结果(已抽取可见文本)。请综合这些片段,"
                "用中文给出简洁、结构化的分析回答,并在引用具体事实时用括号标注来源。"
            ),
        )
        print(f"site_list = {agent.site_list}", flush=True)
        print(f"cache_dir = {agent.cache_dir}", flush=True)
        print(f"question  = {USER_QUESTION}", flush=True)
        print("-" * 60, flush=True)

        messages = [HumanMessage(content=USER_QUESTION)]
        final_text: str | None = None
        for event in agent.stream(messages):
            _print_event(event)
            # Capture the last AIMessage content from the chat-model end event.
            if event.get("event") == "on_chat_model_end":
                msg = event["data"].get("output", {})
                content = getattr(msg, "content", None)
                if isinstance(content, str) and content.strip():
                    final_text = content
                elif isinstance(content, list):
                    parts = [
                        blk.get("text", "")
                        for blk in content
                        if isinstance(blk, dict) and blk.get("type") == "text"
                    ]
                    joined = "\n".join(p for p in parts if p)
                    if joined.strip():
                        final_text = joined

    print("-" * 60, flush=True)
    print("FINAL ANSWER:", flush=True)
    print(final_text or "(no final answer captured)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())