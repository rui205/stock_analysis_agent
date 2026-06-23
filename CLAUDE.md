# CLAUDE.md — Python 工程 Agent 约束

> 适用于 Claude Code / Cursor / 其他 AI Coding Agent。放在项目根目录,Agent 启动时会读取。

---

## 一、核心原则

1. **先读后改**:改动前必须先读相关文件,理解上下文,不要瞎猜。
2. **最小变更**:只改需要改的地方,不重构无关代码,不"顺手优化"。
3. **保持运行**:每次改动后确保 `pytest` 能跑过(如果项目有测试)。
4. **不要沉默地破坏**:发现现有代码与约束冲突时,**先告知用户**,不要自作主张重写。

---

## 二、项目结构约定

项目采用 **src 布局**:`src/<package>/` 是主包,内部按职责分子目录:

```
project/
├── src/<package>/      # 主包(src 布局)
│   ├── mcp/            # MCP server 实现(Model Context Protocol)
│   ├── web/            # HTTP 接口(FastAPI / Starlette 路由、请求/响应 schema)
│   ├── tools/          # 工具函数(LangChain @tool / 通用工具)
│   ├── agent/          # Agent 类 + middleware(LLM 驱动的 agent)
│   ├── script/         # 脚本入口(CLI / 一次性脚本)
│   ├── conf/           # 配置(pydantic-settings / yaml / toml)
│   ├── memory/         # 长期记忆 / 状态持久化(cache、向量存储等)
│   ├── prompts/        # prompt 模板(系统 prompt、few-shot 等)
│   ├── skill/          # 项目级 skill 定义(供 AI Agent 加载的 SKILL.md / 子目录)
│   └── __init__.py     # 显式 __all__
├── tests/              # 测试,镜像 <package>/ 结构
├── docs/               # 设计文档 / spec / plan
├── pyproject.toml      # 唯一依赖/打包配置
├── README.md
├── .env.example        # 环境变量样例(不要提交真实 .env)
└── CLAUDE.md           # 本文件
```

### 子目录职责

- `mcp/` — MCP server 入口,暴露给 Claude / 其他 LLM 客户端的资源与工具。
- `web/` — HTTP 接口(FastAPI / Starlette router、request/response pydantic schema、依赖注入、依赖的服务)。与 `mcp/` 平行但面向通用 HTTP 客户端(浏览器、其他后端服务),不是 LLM 协议。
- `tools/` — 无状态工具(纯函数 / LangChain `@tool`),可被 agent 调用。
- `agent/` — Agent 类与配套 middleware,每个子类一个文件(如 `agent/deepsearch.py`)。
- `script/` — CLI 入口或一次性脚本(用 `python -m <package>.script.xxx` 运行)。
- `conf/` — 配置加载(`pydantic-settings` / `BaseSettings`),**不允许 hardcode**。
- `memory/` — 长期记忆 / 状态持久化(cache、conversation history、向量存储等)。
- `prompts/` — prompt 模板文件(`.md` / `.txt`),按用途命名,代码里读文件加载。
- `skill/` — 项目级 skill 定义(供 AI Agent 加载的约定/参考文档)。约定:
  - 每个 skill 一个子目录,如 `src/<package>/skill/<skill-name>/SKILL.md`。
  - `SKILL.md` 是入口文档(描述 skill 的用途、触发条件、关键约定)。
  - skill 子目录内可放 `references/`、`examples/`、`assets/` 等辅助文件。
  - skill 是**文档而非 Python 代码**,子目录里**不需要** `__init__.py`,内部也不应被业务代码 `import`。
  - 顶层 `src/<package>/skill/` 不放 `__init__.py`(保持为数据目录)。
  - 若不希望随 wheel 打包,在 `pyproject.toml` 的 `[tool.hatch.build.targets.wheel]` 用 `exclude` 排除。
  - **不要**把 skill 文件散落在根目录、`docs/`、`tests/` 下,统一进 `src/<package>/skill/`。

### 强制项

- 包代码必须在 `src/<package>/` 下,根目录不直接放 Python 文件(脚本入口除外)。
- 所有包目录必须有 `__init__.py`,且显式 `__all__`。
- 测试文件命名 `test_*.py`,函数命名 `test_*`,类命名 `Test*`。
- 不要把 `__pycache__`、`*.pyc`、`.venv`、`.env`、`dist/`、`build/` 提交进 git。
- 任何子目录**不强制存在**——根据实际需要创建,不要为了"对齐"硬塞空目录。
- 新建文件前先确认它属于哪个子目录;跨职责(如 agent 里硬塞工具)需要明确理由。

---

## 三、代码风格

### 强制项

- **Python ≥ 3.10**,使用新语法(`match`、`|` 类型、`X | Y`、`dict[str, int]`)。
- **全部函数必须有类型注解**,包括返回值。内部 helper 也不例外。
- **公共 API 必须有 docstring**(Google 风格),包含 Args / Returns / Raises。
- **遵循 PEP 8**:`ruff` 或 `black` 自动格式化,行长 100。
- **import 顺序**:标准库 → 第三方 → 本地,三组用空行分隔(用 `ruff` 自动排)。
- **命名**:
  - 函数/变量:`snake_case`
  - 类:`PascalCase`
  - 常量:`UPPER_SNAKE_CASE`
  - 私有:`_` 前缀

### 推荐项

- 用 `pathlib.Path` 而不是 `os.path`。
- 用 `f-string` 不用 `%` 或 `.format()`。
- 用 `dataclass` 或 `pydantic.BaseModel` 不用裸 dict 传复杂数据。
- 列表推导式优先,但别嵌套超过 2 层(可读性差就改 for 循环)。

---

## 四、依赖与环境

- **统一用 `pyproject.toml`**,禁用 `setup.py` / `requirements.txt`(除非对接旧工具链)。
- **锁版本**:`pydantic>=2.5,<3` 这种区间写法,**不要写 `*`**。
- **依赖分层**:
  - `dependencies`:运行时必须
  - `[project.optional-dependencies]`:dev / test / docs 分组
- **不要引入没必要的依赖**:能用标准库解决的不要装新包。
- **虚拟环境**:用 `uv` 或 `poetry`,不要污染全局 Python。
- **环境变量**:用 `pydantic-settings` 或类似的加载,**不要在代码里 hardcode**。

---

## 五、测试要求

- **核心逻辑必须有测试**(纯函数、业务规则、边界条件)。
- 改一个 bug:**先写一个能复现的测试**,让它失败,再修。
- 测试要独立、可重复、不依赖真实网络/数据库,需要时用 mock。
- 一个测试只测一件事,名字要说明意图:`test_login_with_invalid_password_returns_401`。
- **不要为了覆盖率堆测试**,没意义的 assertion 不要写。

运行:
```bash
pytest                          # 跑全部
pytest tests/test_xxx.py -v     # 跑某个文件
pytest -k "关键字" -v           # 跑某类
```

---

## 六、错误处理

- **不要裸 `except:` 或 `except Exception:`**,必须指定异常类型。
- **不要吞异常**:捕获了就要处理或重新 raise,不要 `pass`。
- **业务错误用自定义异常**,继承自领域基类,例如 `class UserNotFoundError(DomainError)`。
- **错误信息要带上下文**:`raise ValueError(f"user_id={user_id} 不存在")`,不要 `"无效输入"`。
- **对外 API**(CLI/HTTP)用统一的错误包装,内部细节别直接暴露。

---

## 七、日志与输出

- 用 `logging`,**不要用 `print()`** 调试业务逻辑(脚本入口例外)。
- logger 命名:`logger = logging.getLogger(__name__)`。
- 级别:DEBUG 详细信息 / INFO 关键节点 / WARNING 可恢复异常 / ERROR 失败 / CRITICAL 致命。
- 敏感信息(密码、token、个人信息)**不要记日志**。

---

## 八、Git 提交

- 提交前自检:
  - [ ] `ruff check .` 通过
  - [ ] `pytest` 通过
  - [ ] 没有遗留 `print`、`breakpoint()`、`import pdb`
  - [ ] 没有遗留 `.pyc`、调试文件
- Commit message 格式:
  ```
  <type>(<scope>): <subject>
  
  <body>(可选)
  ```
  type: `feat` / `fix` / `refactor` / `docs` / `test` / `chore` / `perf`
- 一个 commit 做一件事,不要把不相关的改动混在一起。
- **不要 commit 到 main / master**,所有改动走分支。

---

## 九、Agent 行为红线(禁止事项)

- ❌ **不要删文件 / 删代码**除非用户明确要求,删之前先确认。
- ❌ **不要 push 到远程**除非用户明确要求。
- ❌ **不要修改 git 配置**(user.name / user.email)。
- ❌ **不要在代码里写密钥、token、密码**(包括示例)。用环境变量或占位符 `<YOUR_API_KEY>`。
- ❌ **不要绕过测试**(`pytest.skip`、`@unittest.skip`、`--no-verify`)。
- ❌ **不要批量 rename / 重构无关代码**,除非用户明确说"重构 X"。
- ❌ **不要使用 `--force`、`reset --hard`、`clean -fd`** 这类破坏性命令。
- ❌ **不要在 commit 信息里写 "Generated with Claude Code"** 或类似自动签名,除非用户要求。
- ❌ **不要臆造依赖/库/API**,不确定就先说"我不确定 X 是否存在"。
- ❌ **不要为不存在的文件写代码**(避免幻觉),先 `ls` 或 `glob` 确认。

---

## 十、完成任务前自检清单

每次交付代码前,逐条检查:

- [ ] 代码能跑(`python -c "import mypkg"` 或对应入口)
- [ ] 类型注解完整
- [ ] 公共函数有 docstring
- [ ] 没有 `print`、`TODO`、`FIXME`、`pass  # TODO`(除非刻意留)
- [ ] 没有 hardcode 的密钥或环境相关的绝对路径
- [ ] 测试已加 / 已更新,且本地通过
- [ ] `git diff` 看过,改动符合预期,没有夹带杂物
- [ ] 改动在合理范围内(没有顺手改无关文件)

---

## 十一、与用户沟通

- **不确定就问**,不要瞎猜。但只问真正影响结果的问题。
- **给方案时给推荐**,不要只列 pros/cons 然后说"看你"。
- **报错信息贴全**,不要只说"出错了"。
- **改完说明改了什么 + 为什么**,不要沉默交付。
- **完成任务后主动总结**,但简短,别写小作文。

---

## 十二、项目特定信息(按需填写)

<!--
下面这些是占位符,改成本项目的实际情况:
-->

- **Python 版本**:3.11
- **包管理器**:uv / poetry / pip
- **测试框架**:pytest
- **代码风格工具**:ruff + black(或 ruff format)
- **CI**:GitHub Actions / GitLab CI
- **主要依赖**:列在这里
- **部署方式**:Docker / 脚本 / 平台托管
- **关键目录**:列出来并简述用途

---

## 版本

- v1.2 — 新增 `src/<package>/web/`(HTTP 接口)与 `src/<package>/skill/`(项目级 skill 定义)约定
- v1.1 — 保留 src 布局,在包内新增 mcp/tools/agent/script/conf/memory/prompts 分层约定
- v1.0 — 初版,通用 Python 工程约束
