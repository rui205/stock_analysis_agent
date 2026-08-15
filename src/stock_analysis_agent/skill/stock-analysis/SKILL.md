---
name: stock-analysis
description: Use when the user asks to analyze a single stock (个股 / 单只股票 / 这支票怎么样 / 帮我看看 / 给个买卖建议), wants a comprehensive valuation + fundamentals + events + analyst views report on one ticker across A-shares / HK / US / ETF, or needs a fair-value range and a buy/hold/sell verdict for one symbol. Do NOT use for industry / sector screening (→ mx-stocks-screener), index ETF basket construction, macro research, or pure technical chart pattern analysis.
---

# Stock Analysis Workflow

Single-ticker deep dive: snapshot → valuation positioning → recent events → analyst views → verdict + risks. Output goes to 飞书云文档 (see `lark-doc`), not the chat.

## Inputs to collect

- **股票代码 + 市场** (required): A 股 / 港股 / 美股 / ETF
- **分析重点** (optional): 用户特别想看的角度 (估值 / 业绩拐点 / 政策受益)
- **持仓状态** (optional): 是否已持仓、持仓成本、持有时长、目标仓位

**缺失追问**:代码或市场缺失时追问一句——估值口径在 A/港/美差异很大(A 股 PE-TTM、港股常用 PB+股息率、美股 GAAP vs Non-GAAP);持仓已 deep-ITM/OTM 时,焦点会从"估值"变成"止损/加仓"。

## Quick reference

| 场景 | 关键调整 |
|------|---------|
| 美股 / 中概股 | DCF 用 USD;Non-GAAP EPS 与 GAAP EPS 至少各给一行 |
| 港股 | PB + 股息率为主,PE 需注意 HKD 计价 |
| 未盈利 / 亏损公司 | 跳过 PE,改 PS / EV/EBITDA / PB |
| ETF / 指数基金 | 跳过 Step 1 公司画像 + Step 5 个股估值,改"跟踪误差 + 折溢价 + 成分股 top10" |
| 已持仓 (含成本) | Step 5 必须含**止损位 / 加仓位**,不只是"目标价" |
| 宏观冲击期 (政策/加息) | 额外跑一次 `mx-macro-data` 拿宏观背景,并入风险章节 |
| `mx-finance-search` 大面积无结果 | 走降级路径:省略事件/机构观点章节,列出缺失项,报告顶部注明 |

## Procedure

### Step 1. 基本面快照 (mx-finance-data)
公司信息、实时行情、近 3 年核心财务 (营收 / 归母净利 / ROE / 毛利率 / 净利率 / 负债率 / FCF)、分红与回购历史。

> 估值倍数必须用盈利质量解读——低 PE 可能是真低估,也可能是利润即将崩塌。

**调用**:`load_skill(name="mx-finance-data")` 拿规范后,`python3 {baseDir}/scripts/get_data.py --query "<问句>" --indicators "<指标>"`。该 skill 在 system prompt 的 catalog 里。失败时 Step 2 估值章节标"基本面数据缺失,不可计算"。

### Step 2. 估值定位 (mx-finance-data + mx-stocks-screener)
- **历史分位**:当前 PE-TTM / PB / PS 相对近 3-5 年的分位数 (<30% 偏低 / >70% 偏高)
- **同行对比**:`mx-stocks-screener` 拉同行业 3-5 家可比公司,PE/PB/PEG/股息率横向比
- **市场差异**:A 股直接 PE-TTM / PB;港股 PB + 股息率为主;美股 GAAP 与 Non-GAAP 估值并列;亏损公司改 PS / EV/EBITDA / PB 并标注"暂不适用 PE"

**调用**:`mx-stocks-screener` 同样在 catalog 里,`load_skill` 后跑 `python3 {baseDir}/scripts/get_data.py --query "<问句>" --select-type "<A 股|港股|美股>"`。

### Step 3. 公司动态 (mx-finance-search)
拉最近 90 天关键事件,按"利好 / 利空 / 中性"分类,每条带信源:财报、分红、回购、增减持、解禁、重组、关联交易、监管问询、业务进展、政策影响、行业事件、高管变动。

**调用**:`load_skill(name="mx-finance-search")` 后用自然语言一句话拉:
```bash
python3 {baseDir}/scripts/get_data.py "<股票名> <代码> 最近90天公告与重要事件"
```
`mx-finance-search` 是 catalog 里**唯一**做"公告+新闻+研报"统一检索的 skill——之前用的 `announcement-search` / `news-search` 不可用,统一用它替代。

### Step 4. 机构观点 (mx-finance-search)
拉最近 60 天主流机构研报:评级分布、目标价区间、共识 EPS、深度报告逻辑。

**调用**:同样用 `mx-finance-search`,只是 query 改成:
```bash
python3 {baseDir}/scripts/get_data.py "<股票名> <代码> 最近60天券商研报与目标价"
```
之前用的 `report-search` 不可用,统一用 `mx-finance-search` 替代。

### Step 5. 综合判断 (不调 skill)

1. **合理估值区间**:
   - **相对估值法**:历史分位中位数 ± 同行均值给 PE/PB 区间 × EPS/BVPS = 价格区间
   - **DCF 简版** (仅适用盈利稳定的成熟公司):近 3 年平均 FCF、8-10% 永续增速、10% 折现率做敏感性
   - 取两种方法的**重叠区间**为最终合理估值
2. **投资建议 (5 档)**:

   | 档位 | 触发条件 |
   |---|---|
   | 强烈买入 | 当前价 < 合理估值下限 + 基本面 & 消息面均向上 |
   | 买入 | 当前价 ≤ 合理估值中位 + 无重大风险 |
   | 持有 | 当前价在合理估值区间内 ±10% |
   | 卖出 | 当前价 ≥ 合理估值上限 或 基本面明显恶化 |
   | 强烈卖出 | 估值高估 + 基本面恶化 + 重大利空叠加 |

3. **关键风险**:3-5 条最可能打破判断的风险点,每条带"触发条件 → 对结论的影响"。
4. **(若已持仓) 持仓行动建议**:持仓成本 × 当前价 → 浮盈/浮亏百分比决定"持有/加仓/减仓/止损";给**止损位** (成本 -1×ATR 或关键支撑) 与**加仓位** (合理估值下限 ±10%)。与第 2 项矛盾时,以此项优先——用户真实问题是"我该怎么办"。

## Output contract

完成 7 节分析后,**不**在会话内输出报告正文,改用 `lark-doc` 把全量报告发布到飞书云文档,会话内**只**返回链接 + 一句话摘要。

### 调用方式

```bash
lark-cli docs +create --api-version v2 \
  --content '<title>...</title><h1>...</h1>...'
```

> 首次使用前需 `lark-cli auth login`;XML 语法细节随 `lark-doc` skill 一起提供,直接 `load_skill("lark-doc")` 拿到完整 XML 规范。

### 文档标题

```
[{symbol}] 股票分析报告 · {YYYY-MM-DD}
```

例:`[600519] 股票分析报告 · 2026-06-29`

### 文档正文(7 节,XML 格式)

1. **执行摘要** — `<callout type="info">` + 表格:投资建议(decision_label + confidence) + 估值区间 + 目标价 + 当前价
2. **公司画像** — `<h2>` + 段落:主营业务、行业地位、近期重要事件(每条带来源标注);"七段式渲染"按 `stock-snapshot-format` skill
3. **多维评分** — 表格:4 维度(fundamental / technical / news_catalyst / peer_positioning)+ 加权总分
4. **价位计划** — 表格 + 列表:current_price / entry_zone / add_zone / target_price / stop_loss / risk_reward_ratio / time_horizon
5. **基本面 + 技术面分析** — `<h3>` + bullets:highlights、concerns 分点列,带数据来源
6. **风险与行动方案** — 风险表格(type 6 选 1 + severity)+ 仓位建议 + review_triggers
7. **数据声明与免责声明** — `<callout type="warning">`:数据源列表 + "本报告由 AI 生成,不构成投资建议"

### 会话内输出(只回链接)

成功路径下,会话内**只**输出:

```
📄 [{symbol}] 股票分析报告已生成

🔗 <飞书文档 URL>

摘要:<一句话,30-80 字,包含 verdict + 关键价位>
```

**禁止**在对话内重复 7 节正文、把整段 XML 粘到对话里、或输出 markdown 形式的报告(那是文档的事,不是对话的事)。

### 判断分流

- 本轮用户消息含 `feishu.cn/docx/` URL 或 docx token → 改用 `lark-cli docs +update --command append/overwrite --api-version v2 --doc <URL_or_token> --content <...>` 写入该文档
- 否则 → 走 `+create` 新建路径

### 错误处理(降级)

| 场景 | 处理 |
|------|------|
| `lark-cli` 未安装 (`FileNotFoundError`) | **降级**:会话内输出 7 节报告 markdown 正文,顶部加 `⚠️ 飞书文档创建失败(<err>),以下为会话内输出` |
| `lark-cli` 认证失败 (`[LARK_AUTH_REQUIRED]` 信号) | **不要降级**。先 `load_skill("lark-shared")` 拿到授权 split-flow,然后:`lark-cli auth login --scope "docs:document:create" --no-wait --json` → 生成二维码 → 把 `verification_url` + 二维码发给用户 → **本轮结束**。等用户回复"已授权"后再跑 `--device-code` 完成登录,回到 `lark-cli docs +create` 重试原步骤。 |
| `lark-cli` 高风险门禁 (`[LARK_CONFIRMATION_REQUIRED]`, exit 10) | 把 `action` / `hint` 贴给用户,**明确告知是高风险操作**,等用户显式同意;同意后在原 argv 末尾加 `--yes` 重试。**禁止**默认加 `--yes` 静默重试。 |
| `lark-cli` 其他结构化错误 (`[LARK_ERROR]` 信号) | 把 `type` / `message` / `hint` 贴给用户,按 `error.hint` 指示修复(常见: `permission_violations` → 提示用户去飞书开发者后台开 scope;`rate_limited` → 1s/3s 指数退避后再试 2 次)。仍失败才降级。 |
| 文档创建成功但内容截断/部分 block 报错 | 重试 1 次(同一个 `--content` 整体重发);仍失败则降级同上 |
| 用户本轮明确说"不要建文档,直接说结论" | 跳过 `lark-doc` 步骤,会话内只输出结论摘要(不输出 7 节正文) |
| 网络/限流错误 | 最多重试 2 次(指数退避 1s/3s);失败后降级 |

> **关键规则**:`run_command` 在 `lark-cli` 失败时会把 `[LARK_AUTH_REQUIRED]` / `[LARK_CONFIRMATION_REQUIRED]` / `[LARK_ERROR]` 作为结果第一行返回。看到这些前缀,按上面表格分支处理 —— **认证失败不允许直接降级 markdown**,必须走 `lark-shared` 授权 split-flow。

降级路径在每轮都要准备好,不要让 lark-cli 报错时无所适从。

## When NOT to use this skill

- **行业扫描 / 板块筛选** → `mx-stocks-screener`
- **纯技术面 K 线 / 形态分析** → 专门的图表 skill;本 skill 不覆盖技术指标
- **组合配置 / 多标的仓位** → 本 skill 只看单只
- **宏观研究 / 大类资产** → `mx-macro-data`
- **基金定投 / 推荐式荐股** → 不在 system_prompt 的"我做"清单内

## Common mistakes

- **直接给"目标价 XX 元"点预测** → 违反原则,只给区间
- **5 档建议缺触发条件** → 每档必须可验证
- **风险章节只写"市场风险"空话** → 每条必须有"触发条件 → 对结论的影响"链路
- **Step 3-4 数据缺失时静默跳过** → 在"数据声明"节列出缺失项,不能省略
- **DCF 用在未盈利公司** → 仅限"盈利稳定的成熟公司"
- **忽略市场差异** → 美股用 HKD EPS / 港股用 A 股 PE 分位 / ETF 当个股估都会算错
- **已持仓用户只给"目标价",不给止损 / 加仓位** → 用户真实问题是仓位管理
- **走 markdown 而不是 lark-doc** → 默认必须走飞书云文档,markdown 仅降级