"""LLM settings loader for stock_analysis_agent.

Reads configuration from environment variables so the project never
hardcodes secrets (model API keys, etc.).

The model source is selected via ``SELECT_SOURCE`` (default ``"qwen"``):

* ``qwen`` — model ``qwen3.8-max`` reached through the Anthropic-protocol
  gateway (``ANTHROPIC_BASE_URL``) with ``ANTHROPIC_API_KEY``.
* ``deepseek`` — model ``deepseek-v4-pro`` reached through
  ``DEEPSEEK_BASE_URL`` with ``DEEPSEEK_API_KEY``.

Both routes reuse the ``anthropic`` provider protocol
(:data:`DEFAULT_MODEL_PROVIDER`); only the model id, endpoint, and API key
change when the source switches.

The module exposes:

* :class:`LLMSettings` — frozen dataclass holding the resolved config.
* :func:`load_llm_settings` — builder; reads env, returns a fresh
  :class:`LLMSettings`. Cached so callers can simply import the
  module-level ``settings`` instance.
* :func:`get_settings` — process-wide singleton accessor; on first
  call it logs the resolved config (model, provider, base URL, masked
  API key) at INFO level so an operator can see at a glance whether
  the subprocess inherited the right env vars.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

# Env-var name for the LLM API key. Kept as a module constant so tests
# and downstream callers can reference one source of truth.
API_KEY_ENV_VAR: str = "ANTHROPIC_API_KEY"

# Env-var name for the LLM endpoint. Read by the Anthropic SDK at
# request time — we surface it in the startup log so a misconfigured
# subprocess (one that didn't inherit this var) is obvious.
BASE_URL_ENV_VAR: str = "ANTHROPIC_BASE_URL"

# Model source selection. ``select_source`` decides which model/endpoint/
# credential triplet is used; valid values are ``qwen`` (default) and
# ``deepseek``.
SELECT_SOURCE_ENV_VAR: str = "SELECT_SOURCE"
SOURCE_QWEN: str = "qwen"
SOURCE_DEEPSEEK: str = "deepseek"
DEFAULT_SELECT_SOURCE: str = SOURCE_QWEN

# DeepSeek-specific env vars. DeepSeek exposes an Anthropic-compatible
# endpoint, so it reuses the ``anthropic`` provider with a different model
# id, base URL, and API key.
DEEPSEEK_API_KEY_ENV_VAR: str = "DEEPSEEK_API_KEY"
DEEPSEEK_BASE_URL_ENV_VAR: str = "DEEPSEEK_BASE_URL"
DEEPSEEK_MODEL: str = "deepseek-v4-pro"

# Default model identifier (qwen source), routed via the Anthropic-protocol
# gateway (see :data:`DEFAULT_MODEL_PROVIDER`).
DEFAULT_MODEL: str = "qwen3.8-max"

# The Anthropic SDK is used to call MiniMax because MiniMax exposes an
# Anthropic-compatible endpoint (``$ANTHROPIC_BASE_URL``). LangChain's
# ``init_chat_model`` cannot infer a provider from a bare ``MiniMax-M3``
# name, so we declare the provider here.
DEFAULT_MODEL_PROVIDER: str = "anthropic"

# Sensible defaults for sampling parameters; BaseAgent reads these too.
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_MAX_TOKENS: int = 32768


class MissingAPIKeyError(RuntimeError):
    """Raised when the LLM API key env var is not set.

    The project deliberately refuses to fall back to a hardcoded key —
    a missing :data:`API_KEY_ENV_VAR` must be surfaced early so the
    operator notices before a runtime call fails deep inside the
    LangChain stack.
    """


@dataclass(frozen=True)
class LLMSettings:
    """Resolved LLM configuration.

    Attributes:
        model: The model identifier passed to ``init_chat_model``. Depends
            on ``select_source``: :data:`DEFAULT_MODEL` for ``qwen``,
            :data:`DEEPSEEK_MODEL` for ``deepseek``.
        api_key: The API key for the resolved source. ``qwen`` reads
            :data:`API_KEY_ENV_VAR`; ``deepseek`` reads
            :data:`DEEPSEEK_API_KEY_ENV_VAR`.
        provider: LangChain provider string (e.g. ``"anthropic"``) that
            ``init_chat_model`` should route to. Cannot be inferred
            from the bare model name, so it must be set explicitly.
            Defaults to :data:`DEFAULT_MODEL_PROVIDER`.
        base_url: Custom endpoint URL forwarded to ``init_chat_model``
            (``None`` lets the SDK use its default). ``qwen`` reads
            :data:`BASE_URL_ENV_VAR`; ``deepseek`` reads
            :data:`DEEPSEEK_BASE_URL_ENV_VAR`.
        select_source: The resolved model source, ``"qwen"`` or
            ``"deepseek"``. Defaults to :data:`DEFAULT_SELECT_SOURCE`.
        temperature: Sampling temperature forwarded to LangChain.
        max_tokens: Output token cap forwarded to LangChain.
    """

    model: str
    api_key: str
    provider: str = DEFAULT_MODEL_PROVIDER
    base_url: str | None = None
    select_source: str = DEFAULT_SELECT_SOURCE
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS


def _read_env(name: str) -> str:
    """Read a required env var or raise :class:`MissingAPIKeyError`.

    Args:
        name: Environment variable name to look up.

    Returns:
        The non-empty value of the variable.

    Raises:
        MissingAPIKeyError: If the variable is unset or empty.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        raise MissingAPIKeyError(
            f"environment variable {name!r} is not set; "
            "export it before running stock_analysis_agent"
        )
    return value


def _resolve_select_source(select_source: str | None) -> str:
    """Resolve the model source from an explicit arg or the env var.

    Args:
        select_source: Explicit source override, or ``None`` to read
            :data:`SELECT_SOURCE_ENV_VAR` and fall back to
            :data:`DEFAULT_SELECT_SOURCE`.

    Returns:
        The normalized (lowercase) source name.

    Raises:
        ValueError: If the resolved source is neither :data:`SOURCE_QWEN`
            nor :data:`SOURCE_DEEPSEEK`.
    """
    source = (
        select_source or os.environ.get(SELECT_SOURCE_ENV_VAR) or DEFAULT_SELECT_SOURCE
    )
    raw = source.strip().lower()
    if raw not in (SOURCE_QWEN, SOURCE_DEEPSEEK):
        raise ValueError(
            f"invalid selectSource={raw!r}; expected {SOURCE_QWEN!r} or "
            f"{SOURCE_DEEPSEEK!r}"
        )
    return raw


def load_llm_settings(
    *,
    select_source: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> LLMSettings:
    """Build a :class:`LLMSettings` from env, with explicit overrides.

    The model source is chosen by ``select_source`` (or the
    :data:`SELECT_SOURCE_ENV_VAR` env var). ``"qwen"`` (default) resolves
    the model, API key, and endpoint from the Anthropic-protocol env vars;
    ``"deepseek"`` resolves them from ``DEEPSEEK_API_KEY`` /
    ``DEEPSEEK_BASE_URL`` with model ``deepseek-v4-pro``.

    Args:
        select_source: Override the model source (``"qwen"`` or
            ``"deepseek"``). Defaults to the env var or
            :data:`DEFAULT_SELECT_SOURCE`.
        model: Override the model identifier.
        api_key: Override the API key (skip env lookup).
        provider: Override the LangChain provider string.
        base_url: Override the endpoint URL.
        temperature: Override sampling temperature.
        max_tokens: Override output token cap.

    Returns:
        A fresh :class:`LLMSettings` instance.

    Raises:
        MissingAPIKeyError: If the source's API key (or DeepSeek base URL)
            env var is unset and no override is supplied.
        ValueError: If ``select_source`` is not a known source.
    """
    resolved_source = _resolve_select_source(select_source)
    if resolved_source == SOURCE_DEEPSEEK:
        resolved_model = model if model is not None else DEEPSEEK_MODEL
        resolved_api_key = api_key if api_key else _read_env(DEEPSEEK_API_KEY_ENV_VAR)
        resolved_base_url = (
            base_url if base_url is not None else _read_env(DEEPSEEK_BASE_URL_ENV_VAR)
        )
    else:
        resolved_model = model if model is not None else DEFAULT_MODEL
        resolved_api_key = api_key if api_key else _read_env(API_KEY_ENV_VAR)
        env_base_url = os.environ.get(BASE_URL_ENV_VAR) or None
        resolved_base_url = base_url if base_url is not None else env_base_url

    return LLMSettings(
        model=resolved_model,
        api_key=resolved_api_key,
        provider=provider if provider is not None else DEFAULT_MODEL_PROVIDER,
        base_url=resolved_base_url,
        select_source=resolved_source,
        temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS,
    )


def _mask_key(key: str) -> str:
    """Return a redacted form of an API key for safe logging.

    Keeps the first 4 and last 4 characters so an operator can confirm
    "is this the key I expected?" without the full secret landing in
    log files.

    Args:
        key: The full API key.

    Returns:
        A masked representation suitable for logging.
    """
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def _log_resolved_settings(settings: LLMSettings) -> None:
    """Log the resolved LLM config once at first build.

    Intended for early diagnostics: if the source's base URL env var is
    unset in the running subprocess, the log will show ``<unset>`` and the
    request will silently fall back to ``https://api.anthropic.com`` —
    which is almost never what an operator wants in this project.
    """
    base_url = settings.base_url or "<unset>"
    logger.info(
        "LLM config: model=%s provider=%s source=%s base_url=%s api_key=%s",
        settings.model,
        settings.provider,
        settings.select_source,
        base_url,
        _mask_key(settings.api_key),
    )


@lru_cache(maxsize=1)
def _cached_settings() -> LLMSettings:
    """Process-wide singleton, lazily initialized on first call."""
    s = load_llm_settings()
    _log_resolved_settings(s)
    return s


def get_settings() -> LLMSettings:
    """Return the module-level :class:`LLMSettings` singleton.

    Lazy and cached: the env is only read once per process. Tests that
    need a fresh instance should call :func:`load_llm_settings`
    directly.
    """
    return _cached_settings()
