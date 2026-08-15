"""LLM settings loader for stock_analysis_agent.

Reads configuration from environment variables so the project never
hardcodes secrets (model API keys, etc.). The configured model is
``MiniMax-M3`` (routed via an Anthropic-protocol gateway) and the API
key is sourced from ``ANTHROPIC_API_KEY``.

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

# Default model identifier. ``MiniMax-M3`` is the model id routed via the
# Anthropic-protocol gateway (see :data:`DEFAULT_MODEL_PROVIDER`).
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
        model: The model identifier passed to ``init_chat_model``.
            Defaults to :data:`DEFAULT_MODEL`.
        api_key: The API key read from :data:`API_KEY_ENV_VAR`.
        provider: LangChain provider string (e.g. ``"anthropic"``) that
            ``init_chat_model`` should route to. Cannot be inferred
            from the model name ``MiniMax-M3``, so it must be set
            explicitly. Defaults to :data:`DEFAULT_MODEL_PROVIDER`.
        temperature: Sampling temperature forwarded to LangChain.
        max_tokens: Output token cap forwarded to LangChain.
    """

    model: str
    api_key: str
    provider: str = DEFAULT_MODEL_PROVIDER
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


def load_llm_settings(
    *,
    model: str | None = None,
    api_key: str | None = None,
    provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> LLMSettings:
    """Build a :class:`LLMSettings` from env, with explicit overrides.

    Reads :data:`API_KEY_ENV_VAR` from the environment unless ``api_key``
    is passed explicitly (useful for tests). Other fields fall back to
    their defaults if not provided.

    Args:
        model: Override the model identifier.
        api_key: Override the API key (skip env lookup).
        provider: Override the LangChain provider string.
        temperature: Override sampling temperature.
        max_tokens: Override output token cap.

    Returns:
        A fresh :class:`LLMSettings` instance.
    """
    resolved_api_key = api_key if api_key else _read_env(API_KEY_ENV_VAR)
    return LLMSettings(
        model=model if model is not None else DEFAULT_MODEL,
        api_key=resolved_api_key,
        provider=provider if provider is not None else DEFAULT_MODEL_PROVIDER,
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

    Intended for early diagnostics: if ``$ANTHROPIC_BASE_URL`` is unset
    in the running subprocess, the log will show ``<unset>`` and the
    request will silently fall back to ``https://api.anthropic.com`` —
    which is almost never what an operator wants in this project.
    """
    base_url = os.environ.get(BASE_URL_ENV_VAR) or "<unset>"
    logger.info(
        "LLM config: model=%s provider=%s base_url=%s api_key=%s",
        settings.model,
        settings.provider,
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
