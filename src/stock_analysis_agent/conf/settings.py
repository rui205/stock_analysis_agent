"""LLM settings loader for stock_analysis_agent.

Reads model credentials (model id, endpoint, API key) from a local JSON
registry instead of environment variables. The registry is keyed by model
id; the active model is selected via the ``MODEL`` env var (default
``qwen3.8-max``), and each entry carries the ``base_url`` / ``api_key`` /
``model`` triplet the gateway needs.

The module exposes:

* :class:`ModelEntry` — frozen dataclass for a single registry entry.
* :class:`LLMSettings` — frozen dataclass holding the resolved config.
* :func:`load_llm_settings` — builder; reads the JSON, returns a fresh
  :class:`LLMSettings`.
* :func:`resolve_subagent_model` — returns the light "flash" model id that
  the mechanical sub-agents should use for the selected entry.
* :func:`get_settings` — process-wide singleton accessor; on first call it
  logs the resolved config (model, provider, base URL, masked API key) at
  INFO level so an operator can see at a glance which entry was resolved.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Absolute path to the model registry JSON. User-specified local path — the
# operator keeps every model's base_url/api_key/model here rather than in
# the process environment.
MODEL_CONFIG_PATH: str = "/Users/rui/Desktop/model.json"

# Env-var name selecting which entry of :data:`MODEL_CONFIG_PATH` to use.
MODEL_ENV_VAR: str = "MODEL"

# Default model id — also the default entry key when ``MODEL`` is unset.
DEFAULT_MODEL: str = "qwen3.8-max"

# The Anthropic SDK is used to call these gateways because qwen and deepseek
# both expose an Anthropic-compatible endpoint. LangChain's ``init_chat_model``
# cannot infer a provider from a bare model name, so we declare it here; each
# entry may override it via an optional ``provider`` field.
DEFAULT_MODEL_PROVIDER: str = "anthropic"

# Sensible defaults for sampling parameters; BaseAgent reads these too.
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_MAX_TOKENS: int = 32768


class ModelConfigError(RuntimeError):
    """Raised when the model registry JSON is missing or malformed."""


class MissingAPIKeyError(RuntimeError):
    """Raised when the selected registry entry has no API key.

    The project deliberately refuses to fall back to a hardcoded key — a
    missing ``api_key`` on the selected entry must be surfaced early so the
    operator notices before a runtime call fails deep inside the LangChain
    stack.
    """


@dataclass(frozen=True)
class ModelEntry:
    """A single entry from the model registry JSON.

    Attributes:
        model: The model identifier passed to ``init_chat_model``.
        api_key: The API key for this model's gateway.
        base_url: Custom endpoint URL (``None`` lets the SDK use its default).
        provider: LangChain provider string (e.g. ``"anthropic"``) that
            ``init_chat_model`` should route to.
        flash_model: Optional key of a cheaper sibling entry that the
            mechanical sub-agents should run on. ``None`` means the
            sub-agents reuse this entry's model.
    """

    model: str
    api_key: str
    base_url: str | None = None
    provider: str = DEFAULT_MODEL_PROVIDER
    flash_model: str | None = None


@dataclass(frozen=True)
class LLMSettings:
    """Resolved LLM configuration.

    Attributes:
        model: The model identifier passed to ``init_chat_model``.
        api_key: The API key for the resolved model.
        provider: LangChain provider string (e.g. ``"anthropic"``).
        base_url: Custom endpoint URL forwarded to ``init_chat_model``
            (``None`` lets the SDK use its default).
        temperature: Sampling temperature forwarded to LangChain.
        max_tokens: Output token cap forwarded to LangChain.
    """

    model: str
    api_key: str
    provider: str = DEFAULT_MODEL_PROVIDER
    base_url: str | None = None
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS


def _select_model_key(model: str | None) -> str:
    """Resolve the registry key from an explicit arg or the env var.

    Args:
        model: Explicit model id override, or ``None`` to read
            :data:`MODEL_ENV_VAR` and fall back to :data:`DEFAULT_MODEL`.

    Returns:
        The stripped registry key to look up.
    """
    raw = model or os.environ.get(MODEL_ENV_VAR) or DEFAULT_MODEL
    return raw.strip()


def _entry_from_dict(key: str, value: dict) -> ModelEntry:
    """Build a :class:`ModelEntry` from a raw JSON object.

    ``model`` defaults to the key when absent (so a bare ``{base_url, api_key}``
    entry is still self-describing); ``api_key`` is required non-empty and
    validated by the caller.
    """
    return ModelEntry(
        model=value.get("model") or key,
        api_key=(value.get("api_key") or "").strip(),
        base_url=value.get("base_url") or None,
        provider=value.get("provider") or DEFAULT_MODEL_PROVIDER,
        flash_model=value.get("flash_model") or None,
    )


def _read_model_config(path: str | None = None) -> dict[str, ModelEntry]:
    """Read and parse the model registry JSON into entries.

    Args:
        path: Registry path; defaults to :data:`MODEL_CONFIG_PATH`.

    Returns:
        A mapping of model id → :class:`ModelEntry`.

    Raises:
        ModelConfigError: If the file is missing, not a JSON object, or a
            top-level value is not a JSON object.
    """
    config_path = Path(path or MODEL_CONFIG_PATH)
    if not config_path.is_file():
        raise ModelConfigError(f"model config file not found: {config_path}")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelConfigError(f"invalid JSON in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ModelConfigError(
            f"model config must be a JSON object, got {type(raw).__name__}"
        )
    entries: dict[str, ModelEntry] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            raise ModelConfigError(f"model entry {key!r} must be a JSON object")
        entries[key] = _entry_from_dict(key, value)
    return entries


def load_llm_settings(
    *,
    model: str | None = None,
    api_key: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    config_path: str | None = None,
) -> LLMSettings:
    """Build a :class:`LLMSettings` from the model registry, with overrides.

    The active entry is chosen by ``model`` (or the :data:`MODEL_ENV_VAR` env
    var, or :data:`DEFAULT_MODEL`). Its ``model`` / ``api_key`` / ``base_url`` /
    ``provider`` fields are resolved from the JSON; explicit keyword arguments
    win over the file.

    Args:
        model: Override the registry key (model id) to select.
        api_key: Override the API key (skip the registry value).
        provider: Override the LangChain provider string.
        base_url: Override the endpoint URL.
        temperature: Override sampling temperature.
        max_tokens: Override output token cap.
        config_path: Override the registry path (mainly for tests). Defaults
            to :data:`MODEL_CONFIG_PATH`.

    Returns:
        A fresh :class:`LLMSettings` instance.

    Raises:
        ModelConfigError: If the registry file is missing or malformed.
        ValueError: If the selected key is not present in the registry.
        MissingAPIKeyError: If the selected entry has no ``api_key`` and no
            ``api_key`` override is supplied.
    """
    entries = _read_model_config(config_path)
    key = _select_model_key(model)
    entry = entries.get(key)
    if entry is None:
        available = ", ".join(sorted(entries)) or "<none>"
        raise ValueError(
            f"model {key!r} not found in {config_path or MODEL_CONFIG_PATH}; "
            f"available: {available}"
        )
    resolved_api_key = api_key if api_key else entry.api_key
    if not resolved_api_key:
        raise MissingAPIKeyError(
            f"model {key!r} has no api_key in {config_path or MODEL_CONFIG_PATH}"
        )
    return LLMSettings(
        model=entry.model,
        api_key=resolved_api_key,
        provider=provider if provider is not None else entry.provider,
        base_url=base_url if base_url is not None else entry.base_url,
        temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS,
    )


def resolve_subagent_model(
    model: str | None = None,
    config_path: str | None = None,
) -> str:
    """Return the model id for the mechanical sub-agents, entry-aware.

    ``run_analyze_stock`` / ``run_technical_capital`` do mostly structured data
    fetching plus moderate synthesis, so they run on the selected entry's
    ``flash_model`` when one is declared (e.g. ``qwen3.8-max`` → ``qwen3.8-flash``).
    Entries without a lighter sibling (``deepseek-v4-pro``) fall back to their
    own model.

    Args:
        model: Override the registry key (model id) to select.
        config_path: Override the registry path (mainly for tests). Defaults
            to :data:`MODEL_CONFIG_PATH`.

    Returns:
        The ``flash_model`` of the selected entry, or the entry's own model
        when no ``flash_model`` is declared.

    Raises:
        ModelConfigError: If the registry file is missing or malformed.
        ValueError: If the selected key is not present in the registry.
    """
    entries = _read_model_config(config_path)
    key = _select_model_key(model)
    entry = entries.get(key)
    if entry is None:
        available = ", ".join(sorted(entries)) or "<none>"
        raise ValueError(
            f"model {key!r} not found in {config_path or MODEL_CONFIG_PATH}; "
            f"available: {available}"
        )
    return entry.flash_model or entry.model


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

    Intended for early diagnostics: if the selected entry has no ``base_url``,
    the log shows ``<unset>`` and the request silently falls back to
    ``https://api.anthropic.com`` — almost never what an operator wants.
    """
    base_url = settings.base_url or "<unset>"
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

    Lazy and cached: the registry is only read once per process. Tests that
    need a fresh instance should call :func:`load_llm_settings` directly.
    """
    return _cached_settings()
