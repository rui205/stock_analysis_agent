# BaseAgent 设计规范

**日期:** 2026-06-16
**状态:** Draft (待用户审阅)
**范围:** `stock_analysis_agent` 项目首个可复用 Agent 基类

---

## 1. 背景与目标

`stock_analysis_agent` 仓库刚初始化,目录里只有空的 `agents/`、`data_sources/`、`tools/` 三个子目录,业务目标(基本面/技术面/情绪面分析)将由多个具体 Agent 协作完成。本规范定义**所有具体 Agent 共享的基类**,作为后续派生类(FundamentalAgent、TechnicalAgent、SentimentAgent 等)的基础设施。

**设计目标:**
- 派生一个具体 Agent 只需提供 `system_prompt` + `tools`
- 暴露**流式**事件接口,支持交互式 CLI 和 Web UI
- 与 LangChain 1.x 生态(LangSmith、LangGraph Studio)兼容
- 无状态:对话历史由调用方管理,符合"消息即数据"原则

**非目标(YAGNI):**
- 不实现同步 `invoke()`(调用方可用 `next(stream(...))` 拿最终结果)
- 不内置 session / checkpointer(派生类如需可覆盖)
- 不做内置的日志/打印输出
- 不提供 LangGraph 内部对象直接访问

---

## 2. 架构

```
调用方 (cli.py / scripts)
    │
    │  stream(messages) -> Iterator[StreamEvent]
    ▼
BaseAgent  (抽象基类, src/stock_analysis_agent/agents/base.py)
    │
    │  内部: langchain.agents.create_agent(...)
    │        + 自定义 ToolRetryMiddleware
    │        + ChatAnthropic(model=..., temperature=..., max_tokens=...)
    ▼
CompiledStateGraph (LangChain 内置)
```

调用方通过 `stream(messages)` 拿到 LangChain 标准事件流(`on_chat_model_start` / `on_chat_model_stream` / `on_tool_start` / `on_tool_end` / `on_chain_end` 等),不直接接触底层 `CompiledStateGraph` 对象。

---

## 3. 类 API

**文件:** `src/stock_analysis_agent/agents/base.py`

```python
import abc
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from langchain.tools import BaseTool
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

class BaseAgent(abc.ABC):
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_retries: int = 2,
        tools: Sequence[BaseTool | Callable] | None = None,
    ) -> None: ...

    @property
    @abstractmethod
    def system_prompt(self) -> str: ...

    @property
    @abstractmethod
    def tools(self) -> list[BaseTool]: ...

    def stream(
        self,
        messages: list[BaseMessage],
        *,
        config: RunnableConfig | None = None,
    ) -> Iterator: ...

    async def astream(
        self,
        messages: list[BaseMessage],
        *,
        config: RunnableConfig | None = None,
    ) -> AsyncIterator: ...
```

**构造参数语义:**

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `model` | `str` | `"claude-sonnet-4-6"` | 透传给 `init_chat_model` |
| `temperature` | `float` | `0.0` | 适合分析类任务 |
| `max_tokens` | `int` | `4096` | 单次模型输出上限 |
| `max_retries` | `int` | `2` | 工具执行失败时的重试次数(不含首次执行) |
| `tools` | `Sequence[BaseTool \| Callable]` | `None` | 运行时追加工具;派生类 `tools` 属性返回的列表优先 |

派生类**必须**实现 `system_prompt` 与 `tools` 两个抽象属性,否则 `BaseAgent()` 实例化时抛 `TypeError`(Python ABC 内置行为)。

---

## 4. 派生类最小形态

```python
# src/stock_analysis_agent/agents/fundamental.py
from stock_analysis_agent.agents.base import BaseAgent
from stock_analysis_agent.tools.financial import get_financial_statements
from stock_analysis_agent.tools.company import get_company_profile

class FundamentalAgent(BaseAgent):
    @property
    def system_prompt(self) -> str:
        return "You are a fundamental analysis expert..."

    @property
    def tools(self) -> list[BaseTool]:
        return [get_financial_statements, get_company_profile]
```

派生类只关注"提示词 + 工具",不关心模型调用、流式、重试这些横切关注点。

---

## 5. 错误处理

**文件:** `src/stock_analysis_agent/agents/exceptions.py`

```python
class ToolExecutionError(RuntimeError):
    """工具执行在重试耗尽后仍然失败。"""
```

**Middleware:** `ToolRetryMiddleware` 通过 `langchain.agents.middleware.AgentMiddleware.wrap_tool_call` 钩子实现。

**重试规则:**
- **瞬时错误**(`TimeoutError`、`RateLimitError`、网络层 `httpx` 异常)→ 指数退避后重试,最多 `max_retries` 次
- **业务错误**(工具内部抛 `ToolException`、参数校验失败)→ 不重试,直接抛 `ToolExecutionError`
- **重试耗尽** → 抛 `ToolExecutionError`,`__cause__` 指向最后一次的原始异常

退避策略:第 N 次重试前等待 `min(2 ** N, 30)` 秒(`2, 4, 8, 16, 30, 30, ...`)。

---

## 6. 可观测性

不内置日志输出,而是通过 LangChain 标准事件流暴露中间状态:

| 事件 | 触发时机 | `data` 字段含义 |
|------|---------|----------------|
| `on_chat_model_start` | 模型调用开始 | `input` = 完整 messages |
| `on_chat_model_stream` | 模型增量输出 | `chunk` = 增量 token/AIMessageChunk |
| `on_tool_start` | 工具开始执行 | `input` = 工具入参 |
| `on_tool_end` | 工具执行完成 | `output` = 工具返回 |
| `on_chain_error` | 任一节点异常 | `error` = 异常对象 |
| `on_chain_end` | 整轮对话结束 | `output` = 最终 messages |

调用方按需订阅事件,自行决定 `print` / `logging` / WebSocket 推送。

---

## 7. 测试策略

**文件:** `tests/agents/test_base.py`

TDD 顺序(先红后绿):

1. `test_subclass_must_implement_abstract_props`
   - 不实现 `system_prompt` 或 `tools` 的子类不能实例化 → `TypeError`
2. `test_stream_returns_final_ai_message`
   - 喂入简单问题(`"hi"`),遍历事件流到 `on_chain_end`,断言最终 messages 含一条 `AIMessage`
3. `test_stream_emits_tool_events`
   - 喂入需要工具调用的问题(用 `FakeChatModel` 注入预设的工具调用序列)
   - 断言事件流中出现 `on_tool_start` 与 `on_tool_end`
4. `test_tool_error_retries_then_raises`
   - mock 工具函数,前两次抛 `TimeoutError`,第三次抛
   - 断言:工具被调用 3 次(1 次首次 + 2 次重试),最终抛 `ToolExecutionError`
5. `test_messages_are_stateless`
   - 实例化一个 agent,先后两次 `stream()` 用相同 `messages` 输入
   - 断言:两次执行互不影响(无消息污染),且最终结果一致

**测试依赖:**
- `pytest`
- `pytest-asyncio`(用于 `astream` 异步测试,本次不要求实现异步测试)
- `langchain_core.language_models.fake_chat_models.FakeListChatModel`(注入预设模型响应)
- `unittest.mock`(mock 工具函数)

---

## 8. 文件清单

新增文件:
- `src/stock_analysis_agent/agents/__init__.py`
- `src/stock_analysis_agent/agents/base.py` — `BaseAgent` + `ToolRetryMiddleware`
- `src/stock_analysis_agent/agents/exceptions.py` — `ToolExecutionError`
- `tests/agents/__init__.py`
- `tests/agents/test_base.py`

不在本次范围:
- `cli.py`(只读取流,不创建)
- 具体派生 Agent(后续单独规范)
- `data_sources/`、`tools/` 下的具体实现

---

## 9. 开放问题(留待实现阶段)

- 是否需要在 `BaseAgent` 上加 `name: str` 字段(便于 LangGraph Studio 显示)?—— **决定**:加,作为可选构造参数,默认 `None`
- `max_tokens=4096` 是否偏小?—— **决定**:保留 4096,调用方可在派生类 `__init__` 覆盖
- `temperature=0.0` 是否对所有分析类任务都合适?—— **决定**:保留 0.0 作为基类默认,具体 Agent 可在 `__init__` 调高
