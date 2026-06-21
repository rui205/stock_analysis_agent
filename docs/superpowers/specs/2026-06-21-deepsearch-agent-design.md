# DeepSearchAgent 设计规范

**日期:** 2026-06-21
**状态:** Draft (待用户审阅)
**范围:** `stock_analysis_agent` 项目首个具体 Agent,基于 `BaseAgent` 的 deep research 能力

---

## 1. 背景与目标

`BaseAgent` 已于 2026-06-16 完成,提供流式接口、工具调用、瞬时错误重试等横切关注点。本规范定义**第一个派生 Agent** —— `DeepSearchAgent`,它通过调用一组外部网站搜索入口,让 LLM 自主决定何时搜、搜什么,综合多源信息给出答案。

**设计目标:**
- 暴露**单一搜索工具**(`web_search(query)`),工具内部并行访问 `site_list` 里的多个 URL
- `site_list` 与 `system_prompt` 都以**模块常量 + 构造参数覆盖**的形式提供
- 默认 `max_retries = 3`(覆盖 `BaseAgent` 的默认值 2)
- 复用 `BaseAgent._ToolRetryMiddleware` 做 HTTP 瞬时错误的指数退避重试
- 单进程单实例假设:同一时刻只构造一个 `DeepSearchAgent`

**非目标(YAGNI):**
- 不做缓存层(同一 query 多次调用会重复 HTTP 请求)
- 不引入 `beautifulsoup4` 等 HTML 抽取库,只用 stdlib `html.parser`
- 不实现 `Fetcher` 抽象层,工具直接调用 httpx
- 不做 rate limiting(由调用方控制调用频次)
- 不支持站点级认证 / Cookie / 自定义请求头
- 不暴露 `_web_search` 工具本身给派生类替换;如需换工具,整体覆盖 `_build_graph`

---

## 2. 架构

```
调用方 (cli.py / scripts)
    │
    │  stream(messages) -> Iterator[StreamEvent]
    ▼
DeepSearchAgent (src/stock_analysis_agent/agents/deepsearch.py)
    │
    │  继承 BaseAgent;预设 tools=[_web_search]
    │  _build_graph() 调用 super()._build_graph()
    ▼
BaseAgent._build_graph()
    │
    │  create_agent(model, tools, system_prompt,
    │                middleware=[_ToolRetryMiddleware(max_retries=3)])
    ▼
CompiledStateGraph (LangChain)
    │
    │  LLM 节点  ──决定调 _web_search(query="...")──►
    ▼
@tool _web_search(query)  ──► _fetch_and_concat(query, _SITE_LIST_PROVIDER.get())
                                            │
                                            ├─ httpx.AsyncClient 并行 GET (asyncio.gather)
                                            ├─ per-site 容错
                                            └─ 拼接纯文本返回
```

**`_SITE_LIST_PROVIDER` 间接拿值的设计:**

LangChain `@tool` 装饰器只能装饰模块级 callable,不能装饰实例方法。`DeepSearchAgent.__init__` 在构造时把 `self._site_list` 写入一个模块级 holder(`_SiteListProvider` 单例),`_web_search` 通过 holder 拿值。

单进程单实例假设下,这个 holder 不会出现竞争;后续如要多实例,需改用 `RunnableConfig` 注入或闭包工厂。

---

## 3. 类 API

**文件:** `src/stock_analysis_agent/agents/deepsearch.py`

```python
from collections.abc import Sequence
from typing import Any

from stock_analysis_agent.agents.base import BaseAgent
from stock_analysis_agent.agents.exceptions import ToolExecutionError

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

class DeepSearchAgent(BaseAgent):
    def __init__(
        self,
        *,
        site_list: Sequence[str] | None = None,
        system_prompt: str | None = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        ...

    @property
    def site_list(self) -> list[str]:
        ...
```

**模块常量:**

| 常量 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `DEFAULT_SYSTEM_PROMPT` | `str` | 通用 deep research 提示词 | `DeepSearchAgent` 不传 `system_prompt` 时使用 |
| `DEFAULT_SITE_LIST` | `list[str]` | 三个支持 `?q=` query 参数的搜索引擎入口 | `DeepSearchAgent` 不传 `site_list` 时使用 |

**构造参数语义:**

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `site_list` | `Sequence[str] \| None` | `None` → 用 `DEFAULT_SITE_LIST` | 搜索引擎入口 URL 列表;构造时拷贝,后续修改不影响常量 |
| `system_prompt` | `str \| None` | `None` → 用 `DEFAULT_SYSTEM_PROMPT` | 透传给 `BaseAgent.__init__` |
| `max_retries` | `int` | `3` | 覆盖 `BaseAgent` 的默认值 2,透传给 `BaseAgent.__init__` |
| `**kwargs` | `Any` | - | 透传给 `BaseAgent.__init__`(`model`、`temperature`、`max_tokens`、`name`) |

**`site_list` 校验:**
- 构造时 `len(site_list) > 0`,否则 `ValueError("site_list cannot be empty")`

---

## 4. 派生类最小形态

```python
# src/stock_analysis_agent/agents/finance_search.py
from stock_analysis_agent.agents.deepsearch import (
    DeepSearchAgent,
    DEFAULT_SYSTEM_PROMPT,
)

FINANCE_SITE_LIST = [
    "https://www.reuters.com/search/news",
    "https://finance.yahoo.com/lookup",
]

FINANCE_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT + (
    " Focus on publicly traded companies and SEC filings."
)

class FinanceSearchAgent(DeepSearchAgent):
    """A deepsearch agent specialized for financial news."""

    def __init__(self) -> None:
        super().__init__(
            site_list=FINANCE_SITE_LIST,
            system_prompt=FINANCE_SYSTEM_PROMPT,
            name="finance-search",
        )
```

派生类只预设 `site_list` / `system_prompt` / `name`,不关心 HTTP、并发、重试等横切关注点。`max_retries=3` 不需要重复声明。

---

## 5. 错误处理

**异常类型:** 复用 `stock_analysis_agent.agents.exceptions.ToolExecutionError`,不新增异常类。

**重试机制:** 完全复用 `BaseAgent._ToolRetryMiddleware`,不新增 middleware。

**错误分类:**

| 层 | 错误 | 处理 |
|---|---|---|
| 单 site fetch | `httpx.HTTPError` / `TimeoutException` / 5xx | catch 后写到结果段(不抛),search 继续 |
| 单 site fetch | 4xx(404 等) | 同上,写到 `[error: 404]` 段,search 继续 |
| 整个 search 工具 | 全部 site 失败(0 个成功) | 抛 `ToolExecutionError("all sites failed: ...")` |
| 整个 search 工具 | `site_list` 为空 | 抛 `ValueError("site_list cannot be empty")`(构造期校验,非运行期) |
| 工具重试 | `ToolExecutionError` 被 `_ToolRetryMiddleware` 捕获 | 按 `max_retries=3` 重试,带指数退避 |
| LLM 节点 | 模型错误 | 由 `BaseAgent.stream` 透传 |

**HTML 抽取失败:** 若单 site 返回非 HTML(`resp.text` 为空或不含 `<` 字符),仍按 `[site_url]\n<text>\n` 段写入,不抛错。

---

## 6. 数据流与可观测性

**正常流程:**

```
用户消息 → BaseAgent.stream → graph.astream_events
  ↓
LLM 节点:看到 tool 描述 → 决定调 web_search(query="...")
  ↓
@tool web_search(query) → _fetch_and_concat(query, _SITE_LIST_PROVIDER.get())
  ├─ httpx.AsyncClient 并行 GET 每个 site (asyncio.gather, return_exceptions=False)
  ├─ 剥 <script>/<style>,再用 html.parser 拿 <body> 文本
  └─ 拼接 "[site_url]\n<text>\n\n" 段;失败的 site 写成 "[site_url]\n[error: <msg>]\n\n"
  ↓
返回纯文本 (作为 ToolMessage.content)
  ↓
LLM 节点:看到结果 → 综合成最终 AIMessage
  ↓
on_chain_end 事件含 messages → stream 消费者拿到
```

**`_fetch_and_concat` 关键实现:**

```python
async def _fetch_and_concat(query: str, site_list: list[str], *, timeout: float = 10.0) -> str:
    if not site_list:
        raise ValueError("site_list cannot be empty")

    async def _one(site: str) -> tuple[str, str]:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(site, params={"q": query})
                resp.raise_for_status()
                return (site, _extract_text(resp.text))
        except Exception as e:
            return (site, f"[error: {type(e).__name__}: {e}]")

    results = await asyncio.gather(*(_one(s) for s in site_list))
    if all(text.startswith("[error:") for _, text in results):
        raise ToolExecutionError(f"all sites failed: {[s for s, _ in results]}")

    parts = [f"[{site}]\n{text}\n" for site, text in results]
    return "\n".join(parts)
```

**HTML 抽取(`_extract_text`):** 用 `html.parser.HTMLParser` 子类,跳过 `<script>` / `<style>` 节点,其余节点 `data` 拼起来,折叠连续空白为单空格。**不引入 BS4**。

**可观测性:** 完全沿用 `BaseAgent` 的 LangChain 标准事件流(`on_tool_start` / `on_tool_end` / `on_chain_end`),不新增项目级 logging。

---

## 7. 测试策略

**文件:** `tests/agents/test_deepsearch.py`

**三层测试,各自职责:**

### 7.1 工具函数单元测试(`_fetch_and_concat`)

直接测,不走 LangChain:

- `test_fetch_empty_site_list_raises_value_error` —— 空 list 抛 `ValueError`
- `test_fetch_all_sites_fail_raises_tool_execution_error` —— 所有 site 抛 → `ToolExecutionError`
- `test_fetch_partial_failure_returns_text_with_error_segment` —— 部分失败 → 拼接文本含 `[error: ...]` 段,**不抛**
- `test_fetch_runs_in_parallel` —— monkey-patch `asyncio.gather` 验证并行;或计时验证总耗时 ≪ 串行耗时
- `test_extract_text_strips_script_and_style` —— `<script>/<style>` 被剥掉
- `test_extract_text_folds_whitespace` —— 连续空白折叠为单空格
- `test_fetch_timeout_does_not_abort_others` —— 单 site 超时不阻断其它 site

**测试方式:** httpx 自带 `MockTransport`,在测试里手写 transport 返回假响应。**不引 respx**。

### 7.2 `@tool _web_search` 接口契约测试

- `test_web_search_tool_metadata` —— 工具 `name` / `description` / args schema 正确
- `test_web_search_invoke_returns_string` —— `tool.invoke({"query": "x"})` 返回 `str`

### 7.3 `DeepSearchAgent` 集成测试(走完整 graph)

沿用 `test_base.py` 的 pattern: `ToolAwareFakeChatModel` + monkey-patch `agent._build_graph`:

- `test_default_construction_uses_module_constants` —— 不传参 → `site_list == DEFAULT_SITE_LIST`、`system_prompt_value == DEFAULT_SYSTEM_PROMPT`、`max_retries == 3`
- `test_custom_site_list_overrides_default` —— 传 `site_list=[...]` 生效
- `test_custom_system_prompt_overrides_default` —— 传 `system_prompt="..."` 生效
- `test_site_list_returns_copy` —— 改 `agent.site_list` 不影响 `DEFAULT_SITE_LIST`
- `test_empty_site_list_raises_at_construction` —— 构造期 `ValueError`
- `test_tool_call_invokes_web_search_with_query` —— fake model 产出 tool_call → agent stream 触发 `_web_search` → 验证 holder 拿到正确 site_list
- `test_single_instance_holder_reflects_latest_construction` —— 第二个 `DeepSearchAgent(site_list=[B])` 构造后,holder 里是 `[B]`
- `test_kwargs_pass_through_to_base_agent` —— `DeepSearchAgent(model="x", temperature=0.7)` → `agent.model == "x"`、`agent.temperature == 0.7`

### 7.4 不测的(明确 YAGNI)

- ❌ 真实 HTTP 调用(无网络依赖)
- ❌ LLM 实际决策(用 fake model)
- ❌ HTML 抽取对各种奇葩 HTML 的鲁棒性(只测标准 `<script>/<style>/<p>` 模式)

---

## 8. 文件清单

**新增文件:**
- `src/stock_analysis_agent/agents/deepsearch.py` —— `DeepSearchAgent` + 模块常量 + 内部 `_web_search` + `_fetch_and_concat` + `_extract_text`
- `tests/agents/test_deepsearch.py` —— 三层测试
- `docs/superpowers/specs/2026-06-21-deepsearch-agent-design.md` —— 本文档

**修改文件:**
- `pyproject.toml` —— `dependencies` 增加 `httpx>=0.27`
- `src/stock_analysis_agent/agents/__init__.py` —— 重导出 `DeepSearchAgent`

**不在本次范围:**
- `cli.py`(只读取流,不创建)
- `FinanceSearchAgent` 等更具体的派生类(后续单独规范)
- `tools/` 目录(本次单文件 MVP,不动)

---

## 9. 开放问题(留待实现阶段)

- 默认 `site_list` 的具体 URL 是否要换成更稳定的搜索引擎?—— **决定**:保持草案中的 3 个 URL(DuckDuckGo HTML、Bing、HTML.duckduckgo);实现阶段如有失败再调整
- `timeout=10.0` 是否过短?—— **决定**:保留 10s;调用方如需调整,通过覆盖 `_fetch_and_concat` 实现(本规范不暴露参数)
- 是否需要在 `DeepSearchAgent` 上加 `query_param_name` 参数(有些搜索引擎用 `query` 而不是 `q`)?—— **决定**:不加,固定 `q`;后续如有需求,在派生类覆盖 `_build_graph`
- `_fetch_and_concat` 是否要加 `max_response_size` 限制(防止返回体过大撑爆 LLM 上下文)?—— **决定**:暂不加,实现阶段验证返回体大小后再决定