---
name: stock-snapshot-format
description: |
  Format the nested multi-source JSON output of `get_stock_snapshot` (Tushare / AkShare / mooTDX / 同类股票数据工具)
  into a standardized company profile + optional peer-comparison section. Load this skill whenever the user mentions
  `get_stock_snapshot`、股票快照、公司画像、上市公司画像、财报快照、stock snapshot / company profile,
  or pastes raw stock-tool output and asks for a structured summary.
  Data source priority (同字段冲突时): Tushare > AkShare > mooTDX > 其他.
  mooTDX 主要补全价格维度字段(high/low/open/volume),不参与 PE/PB 等基本面字段合并(它本来就没有)。
  Output sections: 公司简介 / 主营业务 / 当前股价与估值 / 财务概览 / 近期公告与新闻 / 治理变动 / 数据声明 / (可选) 同业对比.
  Do NOT use for: 交易建议、投资组合分析、DCF/估值建模、行业研究 (→ industry-research-report)、单点财务问答.
---

# Stock Snapshot Format

## Inputs to collect

- `get_stock_snapshot` 的结构化返回(嵌套 dict):
  - 顶层键: `<symbol>`(必选)、`peers`(可选)、`fetched_at`(可选)
  - `<symbol>` 下面的二级键: `tushare` / `akshare` / `mootdx`(按调用顺序)
  - 每个源要么是 `{"data": <row dict>, "row_index": int}` 要么是 `{"error": {"type", "message"}}`
  - `peers`(若存在)以同样嵌套结构组织,每个 peer 只有 `akshare` 一项
- 公司代码 `ts_code` — 与 `<symbol>` 顶层键一致,若缺失或格式异常,在 §7 提示
- 抓取时间 — 取 `fetched_at`,若缺失则标注"快照时间未知"

## Procedure

1. **校验顶层结构**。检查入参是否含 `<symbol>` 顶层键;若缺失,输出 "工具返回结构异常:缺少顶层 symbol 键",不进入七段渲染。校验 `ts_code` 格式(`^\d{6}\.(SH|SZ|BJ)$` 或 `^\d{5}\.HK$`),异常时 §7 标注。
2. **数据源优先级**。读取任一字段时,按 `tushare > akshare > mootdx` 顺序遍历,取第一个含该字段 `data` 块的源。tushare 基本面最完整,akshare 提供换手率,mootdx 仅做 OHLCV 兜底。
3. **逐段映射**。每段按 §Output contract 检查对应键是否存在;缺失字段写 `[数据源未提供]`,**禁止用先验知识补全**(不能因为"知道是茅台"就补写主营业务)。
4. **估值字段双口径**。`pe` 视为静态 PE、`pe_ttm` 视为 TTM PE,二者至少给一个;`pb` / `ps` / `ps_ttm` 同理。口径不明确时输出"PE:静态 / TTM"两值,日期附后。
5. **财务概览给方向,不预测**。同比/环比只展示工具返回的数字 + 方向(`↑` / `↓` / `≈`),不写"未来走势"。
6. **缺失字段汇总到 §7**。把 §3-§6 中所有 `[数据源未提供]` 汇成清单,便于用户决定是否补查(单独查 `anns` / `board_change` 等接口)。
7. **peer 数据使用**。若 `peers` 含有效条目,在 §3 末尾附上"同行 PE/PB 中位数"或"市值最高 2 家";若 `peers._error` 存在,§3 末尾追加"同行对比:数据源未提供"。
8. **ts_code 格式校验**。A 股形如 `XXXXXX.SH` / `XXXXXX.SZ` / `XXXXXX.BJ`,港股形如 `XXXXX.HK`,否则 §7 提示"ts_code 格式异常"。

## Output contract

输出按以下固定顺序,每段标题用三级 markdown(`###`),字段用 bullet;§8 用 markdown 表格。

### 1. 公司简介
- 公司名称、股票代码、所属行业、上市市场 / 交易所、上市日期

### 2. 主营业务
- 主营业务概述 / 主要产品 / 收入构成(top 3-5 项);若仅给行业分类无业务细节,原样输出

### 3. 当前股价与估值
- 最新收盘价 + 交易日(若有)
- PE(静态)、PE(TTM)、PB、PS(TTM)— 至少给 PE 一项
- 总市值、流通市值(统一"亿元",§7 注明原始单位与数据源)
- 换手率(若有,标 `%`)
- 可选:`open` / `high` / `low`(mootdx 提供时)

### 4. 财务概览
- 最新报告期
- 营业收入 + 同比 / 环比(若有)
- 归母净利润 + 同比 / 环比(若有)
- ROE、毛利率、净利率(若有)
- 资产负债率、经营性现金流(可选)

### 5. 近期公告与新闻
- 近 N 条公告 / 新闻(标题 + 日期);数据源缺失时整段写 `[数据源未提供]`

### 6. 治理变动
- 董事 / 监事 / 高管变动记录(姓名 + 职务 + 变动类型 + 日期);数据源缺失时整段写 `[数据源未提供]`

### 7. 数据声明(主体)
- 数据源可用性:`tushare ✅ / akshare ✅ / mootdx ❌ / ...`(用 ✅ / ❌)
- 数据源错误摘要:对每个返回 `{"error": ...}` 的源,记录一句话摘要(例如 `tushare: 限频 / akshare: 接口超时`),不要贴完整堆栈
- 抓取 / 快照时间
- 缺失字段清单
- **字段来源子表**:字段名 → 取自哪个数据源;发生覆盖时附 `(覆盖 <源>: <旧值>)`
- ts_code 格式校验结果

### 8. 同业对比(**仅当 `peers` 存在时输出**)
markdown 表格,固定列:

| Peer Code | Name | Close | PE | PB | ROE | 总市值(亿) | 数据源 |

- 行序按 `peers` 字典迭代顺序(原样保留用户传进来的顺序)
- 任一列缺数据写 `[未提供]`,peer 维度不强行跨源补齐
- `数据源` 列填该 peer 实际生效的数据源(如有覆盖也附注)
- 若所有 peer 都完全没数据(全部 `❌`),整段写 `[所有 peer 数据均不可用]`,不渲染空表

## Failure handling

- **空 dict / 工具报错** → 输出 `get_stock_snapshot 返回空数据,请检查 ts_code 是否正确(沪深:XXXXXX.SH / SZ,北交所:XXXXXX.BJ,港股:XXXXX.HK)`,不进入七段渲染。
- **结构异常**(缺 `<symbol>` 顶层键) → 输出 `工具返回结构异常:缺少顶层 symbol 键`,不进入七段渲染。
- **字段部分缺失** → 对应位置写 `[数据源未提供]`,§7 汇总列出,不静默跳过。
- **估值口径混用** → 同时输出静态 + TTM,标注日期;不让用户猜口径。
- **ts_code 格式异常** → §7 标注,但仍按现有字段渲染(防止工具部分成功)。
- **市值单位不一** → Tushare `total_mv` 单位是"万元",统一除以 `1e4` 转亿元再输出;其他数据源按其原始单位处理并在 §7 注明。
- **同行对比不可用**(`peers._error` 存在) → §3 末尾追加"同行对比:数据源未提供"。
- **数据源错误**(`<source>` 含 `error` 键) → 该源在 §7 列出,字段由下一优先源补足。

## Field mapping (nested dict → output)

工具返回的嵌套 dict 路径: `result[<symbol>][<source>]["data"]`。

每个输出字段按 `首选源` 列出的数据源顺序查找,首个含该键的源胜出。
未列出的键先尝试语义匹配。源失败(`{"error": ...}`)或键缺失 → `[数据源未提供]`。

| 输出字段 | 常见键名 | 首选源 |
|---------|---------|-------|
| 公司名称 | `name` / `company_name` | tushare |
| 股票代码 | `ts_code` / `symbol` | tushare |
| 所属行业 | `industry` | tushare |
| 上市市场 | `market` / `exchange` | tushare |
| 上市日期 | `list_date` / `ipo_date` | tushare |
| 主营业务 | `main_business` / `business_scope` / `intro` | tushare |
| 收盘价 | `close` | tushare |
| 交易日 | `trade_date` | tushare |
| PE 静态 | `pe` | tushare |
| PE TTM | `pe_ttm` | tushare |
| PB | `pb` | tushare |
| PS TTM | `ps_ttm` | tushare |
| 总市值 | `total_mv` | tushare |
| 流通市值 | `circ_mv` | tushare |
| 换手率 | `turnover_rate` | akshare |
| 营收 | `revenue` | tushare |
| 营收同比 | `revenue_yoy` | tushare |
| 归母净利润 | `n_income` / `net_profit` | tushare |
| 净利润同比 | `n_income_yoy` / `net_profit_yoy` | tushare |
| ROE | `roe` / `roe_yearly` | tushare |
| 毛利率 | `gross_profit_margin` | tushare |
| 净利率 | `net_profit_margin` | tushare |
| 资产负债率 | `debt_to_assets` | tushare |
| 经营现金流 | `n_cashflow_act` / `ocf` | tushare |
| 公告 | `announcements` / `news` | (当前三个数据源都不提供,标记 §5 为 `[数据源未提供]`) |
| 治理变动 | `board_changes` / `management_changes` | (当前三个数据源都不提供,标记 §6 为 `[数据源未提供]`) |
| 抓取时间 | `fetched_at` / `update_time` | top-level `fetched_at` |

**peers 字段**(若存在):
- 同行 PE/PB/市值中位数 → 遍历 `result["peers"][*]["akshare"]["data"]` 取值后聚合
- 行业分类 → 取被分析股票的 `industry`,作为 peer 集合的归类依据

## Examples

**Input** (from `get_stock_snapshot`):
```jsonc
{
  "02319.HK": {
    "tushare": {
      "data": {
        "ts_code": "02319.HK",
        "name": "蒙牛乳业",
        "industry": "乳品",
        "pe": 11.03,
        "pe_ttm": 12.5,
        "pb": 1.68,
        "total_mv": 387647.0,
        "trade_date": "20260620",
        "close": 15.89,
        "revenue_yoy": 8.5,
        "n_income_yoy": 12.3,
        "roe": 14.2,
        "gross_profit_margin": 38.5,
        "net_profit_margin": 9.1,
        "list_date": "20040610"
      },
      "row_index": 0
    },
    "akshare": {
      "data": {
        "代码": "02319",
        "中文名称": "蒙牛乳业",
        "最新价": 15.89,
        "涨跌幅": 2.06,
        "换手率": 0.32,
        "市盈率": 11.03
      },
      "row_index": 0
    },
    "mootdx": {
      "error": {"type": "MootdxEmpty", "message": "..."}
    }
  },
  "peers": {
    "600887.SH": {"akshare": {"data": {"代码": "sh600887", "最新价": 24.53}, "row_index": 0}},
    "600597.SH": {"akshare": {"data": {"代码": "sh600597", "最新价": 9.85}, "row_index": 0}}
  },
  "fetched_at": "2026-06-23T15:30:00+08:00"
}
```

**合并逻辑**:
- 主体 tushare ❌(限频),仅 akshare + mootdx 可用
- `close` ← akshare 1681.5(mootdx 1683.0 仅作参考,不覆盖,§7 标注 mootdx 提供了 `open/high/low` 补全)
- `pe` / `pb` / 财务字段 ← akshare(mooTDX 无基本面字段,不参与)
- 总市值 ← akshare 2112000000000 元 → 21120 亿元
- §7 错误摘要:`tushare: 限频(RATE_LIMIT)`

**Output**:
```
### 公司画像:贵州茅台 (600519.SH)

#### 1. 公司简介
- 公司名称:贵州茅台
- 股票代码:600519.SH
- 所属行业:白酒
- 上市市场:[数据源未提供]
- 上市日期:[数据源未提供]

#### 2. 主营业务
- [数据源未提供]

#### 3. 当前股价与估值
- 收盘价:1681.50 元(2026-06-20)
- 今开:1680.00 元(mootdx 补全)
- 最高:1690.00 元(mootdx 补全)
- 最低:1675.00 元(mootdx 补全)
- PE(静态):26.0
- PE(TTM):28.5
- PB:9.2
- PS(TTM):[数据源未提供]
- 总市值:21120.00 亿元(akshare,原值 2112000000000 元)
- 流通市值:[数据源未提供]
- 换手率:[数据源未提供]

#### 4. 财务概览
- 营业收入同比:+15.2% ↑
- 归母净利润同比:+18.3% ↑
- ROE:33.1%
- 毛利率:91.5%
- 净利率:[数据源未提供]

#### 5. 近期公告与新闻
- [数据源未提供]

#### 6. 治理变动
- [数据源未提供]

#### 7. 数据声明(主体)
- 数据源可用性:tushare ❌ / akshare ✅ / mootdx ✅
- 数据源错误摘要:tushare: 限频(RATE_LIMIT)
- 抓取时间:快照时间未知
- 缺失字段:上市市场、上市日期、主营业务、流通市值、换手率、PS-TTM、净利率、资产负债率、经营现金流、近期公告、治理变动
- 字段来源:
  - `name` / `industry` / `trade_date` / `pe` / `pe_ttm` / `pb` / `revenue_yoy` / `n_income_yoy` / `roe` / `gross_profit_margin` ← akshare(tushare 不可用)
  - `close` ← akshare 1681.5(mootdx 提供了 1683.0,未覆盖)
  - `total_mv` ← akshare(原值 2112000000000 元,转亿元)
  - `open` / `high` / `low` / `volume` ← mootdx 补全
- ts_code 格式:正常

#### 8. 同业对比

| Peer Code | Name | Close | PE | PB | ROE | 总市值(亿) | 数据源 |
|-----------|------|-------|----|----|-----|-----------|--------|
| 000858.SZ | 五粮液 | 158.00 | 22.5 | 4.8 | 25.0 | 6135.00 | akshare |
| 600809.SH | [未提供] | [未提供] | [未提供] | [未提供] | [未提供] | [未提供] | ❌(akshare 超时) |
```
