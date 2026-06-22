# get_stock_snapshot Tool 设计规范

**日期:** 2026-06-22
**状态:** Draft (待用户审阅)
**范围:** `stock_analysis_agent` 第二个 `@tool`,多源行情/基本面/同行业对比快照

---

## 1. 背景与目标

`web_search` 已于 2026-06-21 完成,提供多站点文本检索能力。本规范定义**第二个具体 Tool** —— `get_stock_snapshot`,它接收一个标准化股票代码(如 `02319.HK`),并发 fan-out 到 5 个数据源(tushare / 新浪 / 腾讯 / mootdx / akshare),聚合返回该股的**全景快照**(实时报价 + 基本面 + 财务 + 同行业前 N 龙头对比),并自带 12 小时本地文件缓存。

**设计目标:**
- 暴露**单一工具** `get_stock_snapshot(symbol, sources=None, include_peers=True, peer_count=2)`,工具内部并发查询
- 接受标准化代码 `代码.市场` 格式(`02319.HK` / `600519.SH` / `000001.SZ`),内部翻译为各源本地格式
- `sources` 可选子集,`None` 或空列表表示**全查**
- 当实际查询的数据源超过 1 个时,**并发**发起请求(`asyncio.gather`)
- 自动补充**同行业前 N 龙头**对比,数据来源统一
- **12 小时本地文件缓存**,复用现有 `_FileCache`
- 单进程单实例,沿用 `_Provider[T]` 模式注入 site list 和 cache

**非目标(YAGNI):**
- 不做分布式缓存
- 不做实时推送 / WebSocket,只快照
- 不做自定义请求头 / Cookie / 站点认证
- 不引入 beautifulsoup4 / lxml,文本提取沿用 stdlib
- 不做 scheduler / 定时刷新
- 不实现回测、模拟交易、订单管理
- 不暴露 `Fetcher` 抽象层,各源适配器是私有 `_fetch_*` 函数

---

## 2. 架构

```
调用方 (cli.py / scripts / Agent)
    │
    │  _get_stock_snapshot.ainvoke({"symbol": "02319.HK", ...})
    ▼
@tool _get_stock_snapshot(symbol, sources, include_peers, peer_count)
    │
    │  1) cache key  = "{symbol}|{sorted(sources)}|peers={peer_count}"
    │     cache hit  -> 直接返回缓存文本
    │     cache miss -> 继续
    │
    │  2) _translate(symbol) -> 各源本地代码
    │
    │  3) [可选] peer 识别:
    │     akshare.stock_board_industry_name_em / _cons_em
    │     -> 取主 symbol 所属行业 + 龙头 N 只
    │
    │  4) asyncio.gather(
    │       _fetch_sina(code),
    │       _fetch_tencent(code),
    │       _fetch_tushare(code, token),
    │       _fetch_akshare(code),
    │       _fetch_mootdx(code),
    │       *_fetch_peers(peer_symbols, sources),  # 同行业龙头
    │     )
    │
    │  5) 拼接 "[source]\ntext\n" 写入 cache,返回拼接结果
    ▼
返回: 多段文本 (每个数据源一段,失败为 [error: ...])
```

---

## 3. 公共 API

**文件:** `src/stock_analysis_agent/tools/market_data.py`

```python
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

MarketName = Literal["HK", "SH", "SZ"]
SourceName = Literal["sina", "tencent", "tushare", "akshare", "mootdx"]

ALL_SOURCES: tuple[SourceName, ...] = ("sina", "tencent", "tushare", "akshare", "mootdx")
PEER_INDUSTRY_SOURCE: SourceName = "akshare"   # 行业/龙头识别权威源
DEFAULT_CACHE_DIR: str = "~/.cache/stock-analysis-agent/market"
DEFAULT_CACHE_TTL: float = 12 * 3600.0          # 12 小时


@tool("get_stock_snapshot")
async def _get_stock_snapshot(
    symbol: str,
    sources: Sequence[str] | None = None,
    include_peers: bool = True,
    peer_count: int = 2,
) -> str:
    """Fetch a comprehensive stock snapshot from multiple Chinese-market
    data sources and return aggregated text.

    Args:
        symbol: Standard code in '<code>.<market>' format, e.g.
            '02319.HK', '600519.SH', '000001.SZ'.
        sources: Optional subset of data sources to query. Allowed values:
            'sina', 'tencent', 'tushare', 'akshare', 'mootdx'.
            None or empty list means query ALL sources.
        include_peers: If True, also look up the stock's industry and
            fetch the top `peer_count` peer companies for comparison.
        peer_count: How many top peers (by market cap) to include.
            Only meaningful when include_peers=True. Range: 0..10.

    Returns:
        Plain-text aggregation of snippets from each source, each
        prefixed with `[source-name]`. Failed sources are recorded as
        `[error: ...]` segments. The `[peers]` section appears at the
        end when include_peers=True and peer lookup succeeded.
    """
    ...
```

**模块级 Provider**(沿用 web_search 的 `_Provider[T]` 单例):
```python
_SOURCES_PROVIDER: _Provider[tuple[SourceName, ...]] = _Provider()
_CACHE_PROVIDER: _Provider[_FileCache | None] = _Provider()
```

调用方(`MarketDataAgent` 或脚本)在构造时写入 provider 值;`_get_stock_snapshot` 读取。

---

## 4. 符号翻译表

| 标准输入 | sina | tencent | tushare | akshare | mootdx |
|---|---|---|---|---|---|
| `02319.HK` | `rt_hk02319` | `hk02319` | `02319.HK` | `02319` | `23`(市场号) |
| `600519.SH` | `sh600519` | `sh600519` | `600519.SH` | `sh600519` | `1`(市场号) |
| `000001.SZ` | `sz000001` | `sz000001` | `000001.SZ` | `sz000001` | `0`(市场号) |

实现:`_translate(symbol: str) -> dict[SourceName, str]`,market 不在 `{HK, SH, SZ}` 时抛 `ValueError`。

---

## 5. 各源覆盖范围(尽力原则)

每个适配器**只输出它能拿到的字段**,字段缺失不假装有,直接省略该行。

| | 实时报价 | 基本面 | 财务 | 新闻/公告 | K线 | 资金流 |
|---|---|---|---|---|---|---|
| sina    | ✅ 完整 | 部分(简介/PE) | ❌ | ❌ | ❌ | 部分(成交额) |
| tencent | ✅ 完整 | 部分(简介/PE/PB) | ❌ | ❌ | ❌ | 部分(成交额) |
| tushare | ✅ 完整 | ✅ | ✅ 4-8 季报 | ❌ | ✅ | 部分 |
| akshare | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| mootdx  | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |

**输出格式参考(单源):**
```
[sina]
代码: 02319.HK 蒙牛股份
现价: 15.890 HKD
涨跌: +0.320 (+2.06%)
今开: 15.570  最高: 15.940  最低: 15.340  昨收: 15.570
成交量: 17,684,472 股
成交额: 278,092,437.74 HKD
换手率: --
市盈率(动): 11.03  市盈率(静): 13.37  市净率: 1.68
市值: 3,876 亿 HKD
```

文本可读格式,人类和 LLM 都能直接综合。

---

## 6. 同行业前 N 龙头对比

**实现路径(单一权威源 = akshare):**

1. `include_peers=True` 时,从 akshare 拉 A 股行业列表:
   ```python
   industries_df = ak.stock_board_industry_name_em()
   ```
2. 通过主 symbol 在 akshare 个股信息接口查到所属行业名:
   ```python
   info = ak.stock_individual_info_em(symbol=translated_ak_code)
   ```
   注:对于 HK 标的(如 02319),akshare 个股接口不直接覆盖,需要通过公司名/代码查**映射表**(见下)。
3. 查行业成分股:
   ```python
   cons_df = ak.stock_board_industry_cons_em(symbol=industry_name)
   ```
4. 按市值降序取前 `peer_count`,作为对比 peer 列表(每个 peer 是标准代码格式 `代码.SH`/`代码.SZ`)。
5. 对每个 peer 用 **sina + tencent**(免费、覆盖现价/PE/市值已够)发起实时报价快照,不递归查 peer。
6. 拼到主输出末尾,新增段:
   ```
   [peers]
   - 伊利股份 600887.SH: 现价 X, PE Y, 市值 Z
   - 光明乳业 600597.SH: 现价 X, PE Y, 市值 Z
   ```

**HK → A 股行业映射表(降级方案):**
- 当主 symbol 是 HK 标的且 akshare 个股接口查不到行业时,fallback 到内置的小型映射表 `HK_INDUSTRY_HINTS: dict[str, str]`,键是 HK 代码前缀或公司名,值是 akshare 行业名。
- 例: `"02319": "乳品"`、`"09988": "互联网服务"`、`"00700": "互联网服务"`
- 映射表未命中 → `[peers]\n[error: industry not mapped for HK symbol]\n`

**peer 递归限制:** peers 不再查 peers(避免循环)。

---

## 7. 失败处理(沿用 web_search)

- **单源失败** → 该源段输出 `[error: ...]`,其他源继续,整体正常返回。
- **所有源失败** → 抛 `ToolExecutionError`(由 retry middleware 处理)。
- **Tushare token 缺失** → 该源段 `[tushare]\n[error: TUSHARE_TOKEN not set]\n`,跳过(不抛)。
- **Mootdx 连接失败** → 该源段 `[mootdx]\n[error: <ExceptionType>: <msg>]\n`,跳过。
- **akshare 行业检测失败** → 跳过 `[peers]` 段,不影响主查询。

**错误信息格式:** `[error: {ExceptionType}: {msg}]`(与 web_search 一致)。

---

## 8. 缓存策略

**复用 `stock_analysis_agent.memory.file_cache._FileCache`**,不重新实现。

| 项 | 值 |
|---|---|
| 缓存目录 | `~/.cache/stock-analysis-agent/market/`(可通过构造参数覆盖) |
| 缓存 key | `f"{symbol}|{','.join(sorted(sources))}|peers={peer_count}"` |
| 缓存 value | 拼接好的纯文本字符串(与未命中时返回一致) |
| TTL | `12 * 3600 = 43200` 秒(默认;`None` 表示永不过期) |
| 命中行为 | **零 HTTP / 零 SDK 调用**,直接返回缓存文本 |
| 写入时机 | 完整 fan-out 完成后,**整体**写入(不允许部分写入) |
| Provider | `_CACHE_PROVIDER: _Provider[_FileCache \| None] = _Provider()`,`_get_stock_snapshot` 内部 `.get()` 读取 |

**为什么整段缓存而不是分源缓存:**
- 不同 sources 子集返回的字段差异大,分源缓存反而难组合。
- 整段缓存一次写入一次读出,实现简单。
- 缓存粒度 `(symbol, sources, peer_count)` 与用户调用一致,可预测。

---

## 9. 依赖

新增到 `pyproject.toml dependencies`:
```toml
dependencies = [
    "langchain>=1.0",
    "langchain-anthropic>=1.0",
    "langchain-core>=1.0",
    "httpx>=0.27",          # 已有,新浪/腾讯用
    "tushare>=1.4",         # ← 新增
    "akshare>=1.13",        # ← 新增(包含 akshare 作为 peer 识别源 + 独立数据源)
    "mootdx>=2.4",          # ← 新增
]
```

**AKShare 同时承担两个角色:**
1. 独立数据源(直接提供行情/财务/新闻/K线)
2. 同行业龙头识别的**权威查找源**

避免引入第二个行业分类依赖。

**锁版本区间:** 用 `>=` + 合理上界(后续测试时根据实测稳定版本调整),不写 `*`。

---

## 10. 配置

| 项 | 来源 | 默认值 | 缺失行为 |
|---|---|---|---|
| `TUSHARE_TOKEN` | 环境变量 | 无 | 该源段返回 `[error: TUSHARE_TOKEN not set]`,其他源继续 |
| Mootdx 服务器 | 模块常量 `MOOTDX_DEFAULT_SERVER` | `std.tdx.com.cn` | 连接失败 → 该源段 `[error: ...]` |
| 缓存目录 | `_get_stock_snapshot` 构造参数 `cache_dir` | `~/.cache/stock-analysis-agent/market` | 自动创建(失败 → 不缓存,继续查询) |
| 缓存 TTL | 构造参数 `cache_ttl` | `12 * 3600.0` | `None` = 永不过期 |

---

## 11. 模块结构

```
src/stock_analysis_agent/tools/market_data.py
├── 常量: ALL_SOURCES, DEFAULT_CACHE_DIR, DEFAULT_CACHE_TTL,
│        PEER_INDUSTRY_SOURCE, MOOTDX_DEFAULT_SERVER, HK_INDUSTRY_HINTS
├── 类型: MarketName, SourceName
├── _Provider 单例: _SOURCES_PROVIDER, _CACHE_PROVIDER
├── _translate(symbol: str) -> dict[SourceName, str]
├── _fetch_sina(code: str) -> str                  # async,httpx GET hq.sinajs.cn
├── _fetch_tencent(code: str) -> str               # async,httpx GET qt.gtimg.cn
├── _fetch_tushare(code: str, token: str | None) -> str  # async,asyncio.to_thread(pro)
├── _fetch_akshare(code: str) -> str               # async,asyncio.to_thread
├── _fetch_mootdx(code: str) -> str                # async,asyncio.to_thread
├── _detect_peers(symbol: str, peer_count: int) -> list[str] | None
│                                                    # akshare 行业识别 + 映射表降级
├── _fetch_peers(peer_symbols: list[str],
│                sources: tuple[SourceName, ...]) -> dict[str, str]
│                                                    # 对每个 peer 并发查(sina + tencent 实时报价足够)
├── _fetch_and_concat(symbol: str,
│                     sources: tuple[SourceName, ...],
│                     include_peers: bool,
│                     peer_count: int,
│                     cache: _FileCache | None) -> str
│                                                    # cache check → fan-out → cache write
└── @tool("get_stock_snapshot")
    async def _get_stock_snapshot(symbol, sources, include_peers, peer_count) -> str
```

**为什么 `_fetch_peers` 只用 sina/tencent:** 龙头对比只需"现价 + 市值 + PE",新浪和腾讯这两个免费实时源已覆盖;递归用 tushare/akshare 一是慢、二是 peers 本身可能没完整数据。

---

## 12. 测试策略

`tests/tools/test_market_data.py`(~15 用例,分 5 组):

**A. `_translate`(3 用例)**
- HK / SH / SZ 三种输入返回正确 5 个源代码
- 未知 market 抛 `ValueError`
- 数字代码长度错误不抛(允许透传)

**B. 单源适配器(每个源 ≥1 用例,共 5+ 用例)**
- sina: Mock httpx,验证返回文本含"现价""涨跌"
- tencent: Mock httpx,同上
- tushare: monkeypatch `pro_api` 接口,验证 TUSHARE_TOKEN 缺失路径
- akshare: monkeypatch `ak.stock_*`,验证 happy path
- mootdx: monkeypatch `mootdx` 客户端,验证连接失败路径

**C. `_detect_peers`(2 用例)**
- akshare 返回正常行业 → 龙头列表正确
- akshare 抛异常或 HK 未命中映射 → 返回 None

**D. `_fetch_and_concat` 聚合与并发(3 用例)**
- 并发测时:4 源各 100ms 延迟,总耗时 < 250ms(确认并行)
- 单源失败:其它正常返回,该源段含 `[error:`
- 全源失败:抛 `ToolExecutionError`

**E. 缓存(3 用例)**
- 写入:首次调用后,`_CACHE_PROVIDER.get().get(...)` 能取到相同文本
- 命中:第二次调用,所有 HTTP/SDK 调用计数为 0
- TTL:cache_ttl=0.001 + sleep 0.01 后,再次调用 → HTTP/SDK 计数 > 0

**F. peer_count 与 include_peers(2 用例)**
- `include_peers=False` → 绝不调用 peer 检测
- `peer_count=0` → 调用检测但不发起 peer 查询

---

## 13. 演示脚本

`src/stock_analysis_agent/script/test_mengniu_snapshot.py`:
- 构造一个最小化的 `MarketDataAgent`(或直接用 provider 注入),启用 12h 缓存
- 调用 `await _get_stock_snapshot.ainvoke({"symbol": "02319.HK"})`
- 打印完整快照输出

不通过 LLM,直接走 tool,验证**零 LLM 成本**也能拿到完整快照(对比上一版 web_search 必须经 LLM 综合)。

---

## 14. 风险与缓解

| 风险 | 缓解 |
|---|---|
| `akshare.stock_board_industry_*` 接口变动 | 在 `_detect_peers` 内单点封装,接口变动只改一处 |
| HK 行业映射表不全 | 至少覆盖常见 HK 大盘股(腾讯/阿里/蒙牛/美团/比亚迪 等),未命中 fallback 报错 |
| Tushare 高级接口要积分 | 适配器只用 `daily` / `stock_basic` / `income` 等基础接口,免费额度够 |
| Mootdx 默认服务器网络抖动 | 单源失败容错已覆盖;可后续通过构造参数 `mootdx_server` 切换 |
| 12h 缓存导致数据过期 | TTL 与 web_search 默认 24h 一致,在合理范围;用户可传 `cache_ttl=None` |
| 输出文本极长(token 爆) | 单源 ~200-500 字符,5 源 + 2 peer = ~3500 字符,在 LLM 上下文安全范围内 |

---

## 15. 实施步骤概要(供 writing-plans 展开)

1. 加 `tushare` / `akshare` / `mootdx` 到 `pyproject.toml` 依赖,`uv pip install` 验证
2. 写 `_translate` + 单源适配器(sina / tencent 先,后 3 个 SDK)
3. 写 `_detect_peers` + HK_INDUSTRY_HINTS 映射表
4. 写 `_fetch_and_concat` + 缓存集成
5. 写 `@tool _get_stock_snapshot` + provider 单例
6. 写 `tests/tools/test_market_data.py`(~15 用例)
7. 写演示脚本 `test_mengniu_snapshot.py`,跑通 `02319.HK` 完整快照
8. `ruff check .` + `pytest` 全绿

---

## 16. 版本

- v1.0 — 初版

## 版本变更

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-06-22 | v1.0 | 初版设计完成 |