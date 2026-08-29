---
name: technical-capital
description: Use when the user asks for technical analysis (技术面) and/or capital-flow analysis (资金面) on a single stock — trend / momentum / pattern / volume + main-force / northbound / large-order / turnover — to get key support/resistance levels and timing signals (何时进出 / 加仓 / 止损). Complementary to `stock-analysis` (估值视角"值不值"): this skill answers "何时进出". Do NOT use for fundamental valuation (→ stock-analysis), industry screening (→ mx-stocks-screener), or macro research (→ mx-macro-data).
---

# Technical + Capital Flow Analysis Workflow

Single-ticker **trading-perspective** deep dive: technical indicators → capital-flow → key levels → timing signals. Output is a Markdown report returned to the caller (the strategy-match orchestrator folds it into per-criterion matching); it does **not** publish to 飞书.

## 定位

- `stock-analysis` 回答"**值不值**"（估值 + 基本面 + 事件 + 机构观点，结论偏事实）。
- 本 skill 回答"**何时进出**"（趋势 / 动量 / 形态 / 量能 + 资金博弈，数据客观、解读与择时结论偏交易）。
- 两者分离：技术面超买 / 资金流出的结论不污染基本面的估值判断，反之亦然。

## Inputs to collect

- **股票代码 + 市场** (required): A 股 / 港股 / 美股 / ETF
- **分析重点** (optional): 趋势 / 动量 / 形态 / 量能 / 资金博弈（默认全做）
- **时间窗口** (optional): 默认日线近 60–120 个交易日

**缺失追问**:代码或市场缺失时追问一句；技术指标口径在 A/港/美差异不大，但资金面口径差异大（A 股看主力/北向/龙虎榜，港股看港股通/大行席位，美股看机构持仓/大宗交易）。

## Quick reference

| 场景 | 关键调整 |
|------|---------|
| 港股 / 美股 | 资金面改用港股通持仓变化 / 机构持仓与大宗交易，北向资金不适用 |
| ETF / 指数基金 | 技术面照常，资金面改"份额变化 / 折溢价 / 成交额"，不套用个股主力/北向 |
| 无资金流数据 | 只出技术面结论，资金面章节标"数据源未提供"，不臆造 |
| `mx-finance-data` / `mx-stocks-screener` 大面积无结果 | 降级：省略对应章节，报告顶部注明缺失项 |

## Procedure

### Step 1. 技术指标 (mx-finance-data)
MA / MACD / KDJ / RSI / BOLL 等常用指标，近 60–120 日。

**调用**:`load_skill(name="mx-finance-data")` 拿规范后:

```bash
python3 {baseDir}/scripts/get_data.py --query "<股票名> <代码> 近120日技术指标" --indicators "技术指标"
```

> 指标必须落到具体数值（金叉/死叉、超买/超卖、均线多头/空头排列），禁止只写"走势向好"。

### Step 2. 资金流向 (mx-finance-data)
主力资金净流入、大单/超大单、换手率、量比等。

**调用**:

```bash
python3 {baseDir}/scripts/get_data.py --query "<股票名> <代码> 近60日资金流向" --indicators "资金流向"
```

### Step 3. 技术信号与资金面筛选 (mx-stocks-screener)
突破均线、连续上涨/下跌、主力流入等复合条件；港股/美股用对应 `--select-type`。

**调用**:`load_skill(name="mx-stocks-screener")` 后:

```bash
python3 {baseDir}/scripts/get_data.py --query "<信号条件>" --select-type "<A股|港股|美股>"
```

例:`--query "股价突破20日均线且主力资金净流入" --select-type A股`

### Step 4. 综合判断 (不调 skill)

1. **趋势**:多头 / 空头 / 震荡（均线排列 + 价格位置）
2. **动量**:MACD/RSI/KDJ 的方向与背离
3. **量能**:放量 / 缩量 / 量价配合（量价背离要单列）
4. **资金**:净流入 / 净流出 / 平衡，标出时间窗口
5. **关键价位**:支撑位（前低 / 均线 / 密集成交区）与压力位（前高 / 均线 / 缺口），给**区间**不给点预测
6. **择时信号**:每个信号带"触发条件 → 动作"（如"有效放量站上 20 日线 → 试仓"，"跌破 60 日线且资金持续流出 → 离场"），不保证收益

## Output contract

产出 Markdown 报告（**返回给调用方，不发布飞书**），固定结构：

```
### 0. 摘要
一句话：趋势 + 资金 + 最关键的一个择时信号

### 1. 技术面
- 趋势：…（数据来源）
- 动量：MACD/RSI/KDJ 具体数值与方向
- 形态 / 量能：…

### 2. 资金面
- 主力净流入 / 大单 / 换手 / 量比（标时间窗口）
- 数据缺失时写 `[数据源未提供]`

### 3. 关键价位
- 支撑位区间 / 压力位区间

### 4. 择时信号
- 每个信号：触发条件 → 动作（可试仓/加仓/持有/减仓/离场）

### 5. 数据声明 + 免责声明
- 数据源列表 + "本报告由 AI 生成，不构成投资建议"
```

## When NOT to use this skill

- **基本面估值 / 财报 / 机构观点** → `stock-analysis`
- **行业扫描 / 板块筛选** → `mx-stocks-screener`（本 skill 只用它做单票信号，不跑行业）
- **宏观研究 / 大类资产** → `mx-macro-data`

## Common mistakes

- **只写"走势向好 / 资金流入"不给数值** → 指标必须落到具体数字与方向
- **给"目标价 XX 元"点预测** → 只给支撑/压力**区间**
- **择时信号不带触发条件** → 每个信号必须"触发条件 → 动作"
- **港股/美股套用北向资金** → 市场口径混用
- **数据缺失时静默跳过** → 在对应章节写 `[数据源未提供]`，报告顶部汇总
- **把本报告发布到飞书** → 本 skill 返回 Markdown 给调用方，不走 `lark-doc`
