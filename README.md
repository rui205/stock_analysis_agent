# stock_analysis_agent

A multi-market (A股 / 美股 / 港股) stock analysis agent built on **LangChain** + **Claude (Anthropic)** with **akshare** and **yfinance** as data sources.

## Requirements

- [uv](https://docs.astral.sh/uv/) (install via `brew install uv` or the official installer)
- Python 3.12 (provisioned automatically by uv via `.python-version`)
- An Anthropic API key

## Install

```bash
uv sync
```

This downloads the pinned Python 3.12 interpreter (if not already present) and installs all dependencies into a project-local `.venv/`.

## Configure

```bash
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY
```

The agent reads the key from the `ANTHROPIC_API_KEY` environment variable.

## Run

One-shot query:

```bash
uv run stock-analysis-agent "What is the latest price of AAPL?"
uv run stock-analysis-agent "查询一下贵州茅台(600519)最近的收盘价"
```

Interactive REPL:

```bash
uv run stock-analysis-agent
```

Or as a module:

```bash
uv run python -m stock_analysis_agent "How did Tencent (0700.HK) move today?"
```

## Supported markets

| Market   | Source    | Symbol format             | Example       |
|----------|-----------|---------------------------|---------------|
| A股      | akshare   | 6-digit numeric           | `600519`      |
| 美股     | yfinance  | Ticker                    | `AAPL`        |
| 港股     | yfinance  | 4-digit zero-padded + `.HK` | `0700.HK`   |

## Architecture (v1)

```
src/stock_analysis_agent/
├── cli.py            # argparse + REPL entry point
├── config.py         # env reads (ANTHROPIC_API_KEY, MODEL, LOG_LEVEL)
├── logging.py        # loguru JSON sink
├── errors.py         # DataSourceError, SymbolNotFoundError, RateLimitError
├── models.py         # Pydantic models + Market enum
├── llm.py            # ChatAnthropic factory (claude-opus-4-8 + adaptive thinking)
├── data_sources/     # Protocol-based per-market adapters
├── tools/            # @tool wrappers
└── agents/           # create_agent (langchain.agents) with the system prompt
```

## Data-source attribution

- **akshare** — open-source A-share data aggregator. https://github.com/akfamily/akshare
- **yfinance** — unofficial Yahoo Finance API wrapper. For personal/research use. https://github.com/ranaroussi/yfinance

**Future upgrade path:** [Tushare Pro](https://tushare.pro/) for higher-quality A-share data (requires a free token). Not implemented at v1.

## Limitations

- No persistence — every query is fresh.
- No testing framework, linter, or type checker is configured at v1. The user's explicit choice.
- yfinance can rate-limit or return partial data silently; the data source layer validates responses.
- A-share symbol lists from akshare are re-fetched on each session; consider caching for production use.

## License

Personal use. See upstream data-source licenses for redistribution terms.
