---
name: strategy-match
description: Use when the user wants to evaluate whether a single stock fits a personal selection strategy (策略匹配 / 这支票符合我的策略吗 / 帮我看看 X 是否值得买). Wraps the existing `StockAnalysisAgent` as a subagent, loads a user-defined Markdown strategy from `conf/strategies/*.md`, and emits a standalone `StrategyMatchReport`. Do NOT use for raw fundamentals analysis (→ stock-analysis) or industry screening (→ mx-stocks-screener).
---

# Strategy Match Workflow

## Inputs

- **股票代码** (required):与 `analyze_stock` 接受的格式一致(A/港/美/ETF)
- **策略名称** (required):必须对应 `conf/strategies/<name>.md` 的文件名(无后缀)
- **输出形态** (optional):默认 both(本地 md + 飞书);也可 local-only / feishu-only

**缺失追问**:策略名不在 conf/strategies/ 时,CLI 启动时报错并列出可用策略;运行中如 agent 误用错名,`load_strategy` tool 会抛 FileNotFoundError 并列出可用项。

## Quick reference

| 场景 | 关键调整 |
|------|----------|
| 策略含硬阈值(如 "PE<15") | agent 把硬阈值当作 "fit/mismatch" 的清晰二值判据,模糊时给 partial |
| 策略是纯软偏好(如 "回避烧钱赛道") | evidence 必须引用基本面数据(行业、ROE、FCF 等),无数据时给 partial |
| 港股 / 美股 | subagent 已在 step 2 处理估值口径差异,strategy-match 这层不重复 |
| ETF / 指数基金 | subagent 跳过个股估值,strategy-match 按策略原则给 partial + "ETF 不适用个股估值" |
| 已持仓场景 | strategy-match 不单独处理,用户先用 stock-analysis 拿到仓位建议,再来评估策略匹配 |
| 飞书发布失败 | 降级到本地 md,顶部加 `⚠️ 飞书文档创建失败(<err>),以下为会话内输出` |

## Procedure

### Step 1. 加载策略

```
load_strategy(name="<strategy_name>")
```

返回策略全文(YAML frontmatter + 自然语言原则)。frontmatter 解析在脚本层完成,
`StrategyMatchReport.strategy_name` 取自 frontmatter 的 `name` 字段,`strategy_version` 取自 `version` 字段(缺省 `"unversioned"`)。

### Step 2. 跑基本面 subagent

```
run_analyze_stock(symbol="<symbol>")
```

返回 subagent 产出的完整 Markdown 分析报告(verbatim,不做任何格式化或抽字段)。
agent 自己从报告里捞需要的章节(verdict / score / 主要风险等)喂给策略匹配逻辑;
不需要重新跑所有 skill。

失败处理:
- `[ERROR] analyze_stock tool failed: ...` → agent 自决:可基于已有信息给 "avoid" +
  confidence="low",或直接中止并在报告中说明数据缺失

### Step 2.5 数据不足 → deepresearch 补充(≤3 次)

当 `run_analyze_stock` 返回 `[ERROR]`、或报告中某条策略原则所需字段缺失/无法验证时,
**不要直接给 fit / mismatch**:

1. 挑出「证据不足」的策略原则,提炼为具体维度(如 `["盈利质量-ROE"]`)。
2. 调 `run_deepresearch(symbol=..., dimensions=[...])`,等返回 Markdown 报告。
3. 把深研结论回填到对应 criterion 的 evidence / reasoning。
4. **最多 3 次**;仍不足则基于现有信息给结论,`confidence=low`,并在
   `judgment_rationale` 标注缺失维度。
5. 绝不编造深研数据。

### Step 3. 逐条匹配

针对策略中**所有可独立验证**的原则(忽略纯定性描述如"长期持有"),逐条生成一条
`StrategyCriterionMatch`:

- `criterion` — 原文引用
- `match_level` — `fit` / `partial` / `mismatch`
- `evidence` — 引用 subagent 摘要里的具体数字/字段
- `reasoning` — 为什么是这个评级

### Step 4. 综合判断

- `overall_fit`:
  - `buy` — 大部分原则 fit,且无 mismatch,基本面 buy/强烈买入
  - `hold` — 多数 partial,基本面 hold,或基本面 buy 但策略部分 mismatch
  - `avoid` — 任何硬阈值 mismatch,或基本面 sell + 策略多项 mismatch
- `fit_score` (0-10):参考 criterion_matches 中 fit 的占比,结合 subagent 报告的 verdict/score 综合打分
- `summary` — 一句话,30-80 字
- `action_recommendation` — 具体仓位/等待/放弃,30-200 字

### Step 5. 输出

**严格按 StrategyMatchReport schema 输出 JSON**,不附加解释。schema 现含
`data_sources`(stock_analysis / deepresearch 两个来源摘要)与
`judgment_rationale`(判断理论)两个新字段。LLM 输出后由
`script.evaluate_strategy.run` 校验 + 渲染。

## Output delivery (--delivery 决定)

| `--delivery` | 行为 |
|--------------|------|
| `local` (默认) | 写 `output/strategy-match-<sym>-<ts>.md`(见 §"本地 md 模板") |
| `feishu` | 调 `lark-cli docs +create --api-version v2` 发到飞书(见 §"飞书 XML 模板") |
| `both` | 两者都做 |

### 本地 md 模板

```markdown
# [{symbol}] 策略匹配报告 · {YYYY-MM-DD}

> 策略: **{strategy_name}** v{strategy_version}
> 适合度: **{overall_fit}** (score: {fit_score}/10, confidence: {confidence})

## 摘要
{summary}

## 策略原则逐条匹配
| # | 原则 | 评级 | 证据 | 推理 |
|---|------|------|------|------|
{for each criterion_match:}
| {i} | {criterion} | {match_level} | {evidence} | {reasoning} |

## 数据来源
### 来自 stock_analysis
{data_sources.stock_analysis}

### 来自 deepresearch
{data_sources.deepresearch}(未调用则写"未调用 deepresearch")

## 判断理论
{judgment_rationale}

## 行动建议
{action_recommendation}

---
*本报告由 AI 生成,不构成投资建议*
```

### 飞书 XML 模板(9 节)

1. **执行摘要** — `<callout type="info">` + 表格:overall_fit + fit_score + confidence + summary
2. **策略信息** — strategy_name + strategy_version + 适用市场
3. **逐条匹配** — 表格:原则 / 评级 / 证据 / 推理
4. **数据来源 · stock_analysis** — 来自 `data_sources.stock_analysis`
5. **数据来源 · deepresearch** — 来自 `data_sources.deepresearch`(未调用标注"未调用")
6. **判断理论** — `judgment_rationale` 段落
7. **行动建议** — action_recommendation 段落
8. **数据声明** — 数据源列表 + 免责声明
9. (可选)**完整报告链接** — 如果 subagent 已发布飞书,这里附链接

lark-cli 命令细节、`lark-cli docs +create` / `+update` 选择、`<callout>` /
`<h1>` 等 XML 标签规范,均在 `lark-doc` skill 里 — **先** `load_skill("lark-doc")`。

### 错误处理(降级)

| 场景 | 处理 |
|------|------|
| `lark-cli` 未安装 / 认证失败 | 走本地 md 路径,顶部加 `⚠️ 飞书文档创建失败(<err>),以下为会话内输出` |
| 文档创建成功但内容截断/部分 block 报错 | 重试 1 次;仍失败降级同上 |
| 用户本轮说"不要建文档" | 走 local-only,exit 0 |
| 网络/限流 | 最多重试 2 次(指数退避 1s/3s);失败后降级 |

## When NOT to use this skill

- **行业扫描 / 板块筛选** → `mx-stocks-screener`
- **个股基本面** → `stock-analysis` skill(本 skill 内部 subagent 间接调用)
- **组合配置 / 多标的仓位** → 不在 MVP 范围
- **宏观研究 / 大类资产** → `mx-macro-data`

## Common mistakes

- **匹配原则时编造数据** → evidence 必须引用 subagent 摘要里的真实字段;无数据时给 mismatch + reasoning="基本面数据缺失,无法验证"
- **orchestrator 绕过 subagent 自己跑数** → 基本面数据一律以 `run_analyze_stock` 返回的报告为准。即使本 agent 带 `run_command`(用于 lark-cli 发布),也不要自己调 mx-* skill 补数——自己跑出来的结果进不了匹配上下文,只会浪费查询;数据不全时在报告里如实标注"数据缺失",由用户决定是否开 `--include-shell-tool` 重跑
- **`criterion_matches` 漏掉策略中所有可验证原则** → 必须穷举,原则描述不清时给 partial + reasoning="原则表述模糊"
- **`fit_score` 超 0-10 范围** → schema 校验会失败,务必保证
- **`overall_fit` 选错** → 严格按 §4 触发条件,不要因为 "分数高" 就直接 buy
- **策略中只有定性描述** → 仍要逐条给出 criterion_match,evidence 写 "定性原则,无量化判据"
- **数据不足却强行给 fit/mismatch** → 先按 Step 2.5 调 `run_deepresearch` 补充,别硬凑
- **deepresearch 调用超过 3 次** → 上限 3 次,超了就用已有信息给 `confidence=low`
- **编造深研数据** → deepresearch 结果必须来自 `run_deepresearch` 返回的报告,不得臆造