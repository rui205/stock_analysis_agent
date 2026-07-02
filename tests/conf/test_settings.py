"""Tests for stock_analysis_agent.conf.settings."""
from __future__ import annotations

import logging

import pytest

from stock_analysis_agent.conf import LLMSettings, load_llm_settings
from stock_analysis_agent.conf.settings import (
    API_KEY_ENV_VAR,
    BASE_URL_ENV_VAR,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_MODEL_PROVIDER,
    DEFAULT_TEMPERATURE,
    MissingAPIKeyError,
    _mask_key,
    get_settings,
)


def test_default_model_uses_settings_constant() -> None:
    """Spec: model identifier mirrors the ``DEFAULT_MODEL`` constant
    declared in :mod:`stock_analysis_agent.conf.settings`."""
    settings = load_llm_settings(api_key="dummy")
    assert settings.model == DEFAULT_MODEL


def test_default_provider_is_anthropic() -> None:
    """Spec: provider defaults to ``anthropic`` because MiniMax is reached
    via an Anthropic-protocol endpoint (see DEFAULT_MODEL_PROVIDER)."""
    settings = load_llm_settings(api_key="dummy")
    assert settings.provider == DEFAULT_MODEL_PROVIDER == "anthropic"


def test_explicit_provider_override() -> None:
    """The provider can be overridden per-call for non-anthropic routes."""
    settings = load_llm_settings(api_key="dummy", provider="openai")
    assert settings.provider == "openai"


def test_api_key_is_read_from_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    """ANTHROPIC_API_KEY in env must be picked up automatically."""
    monkeypatch.setenv(API_KEY_ENV_VAR, "secret-from-env")
    settings = load_llm_settings()
    assert settings.api_key == "secret-from-env"


def test_explicit_api_key_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit api_key argument wins over the env var."""
    monkeypatch.setenv(API_KEY_ENV_VAR, "secret-from-env")
    settings = load_llm_settings(api_key="explicit-key")
    assert settings.api_key == "explicit-key"


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no env var and no override, a clear error is raised."""
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
    with pytest.raises(MissingAPIKeyError):
        load_llm_settings()


def test_blank_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A whitespace-only env value is treated as missing."""
    monkeypatch.setenv(API_KEY_ENV_VAR, "   ")
    with pytest.raises(MissingAPIKeyError):
        load_llm_settings()


def test_default_temperature_and_max_tokens() -> None:
    """Sampling defaults match BaseAgent's baseline."""
    settings = load_llm_settings(api_key="dummy")
    assert settings.temperature == DEFAULT_TEMPERATURE == 0.0
    assert settings.max_tokens == DEFAULT_MAX_TOKENS == 32768


def test_overrides_apply_per_field() -> None:
    """Each scalar field can be overridden independently."""
    settings = load_llm_settings(
        api_key="dummy",
        model="custom-model",
        temperature=0.7,
        max_tokens=4096,
    )
    assert settings.model == "custom-model"
    assert settings.temperature == 0.7
    assert settings.max_tokens == 4096


def test_settings_is_frozen() -> None:
    """LLMSettings is immutable — protects against accidental mutation."""
    settings = load_llm_settings(api_key="dummy")
    with pytest.raises((AttributeError, Exception)):  # FrozenInstanceError
        settings.model = "something-else"  # type: ignore[misc]


def test_get_settings_singleton_uses_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The module-level accessor reads the env via the same code path."""
    monkeypatch.setenv(API_KEY_ENV_VAR, "singleton-key")
    # The cache must be cleared for the test environment to be honored.
    from stock_analysis_agent.conf import settings as settings_module

    settings_module._cached_settings.cache_clear()
    try:
        singleton = get_settings()
        assert isinstance(singleton, LLMSettings)
        assert singleton.api_key == "singleton-key"
    finally:
        settings_module._cached_settings.cache_clear()


def test_export_surface_is_minimal() -> None:
    """Only the intended names are exposed from stock_analysis_agent.conf."""
    from stock_analysis_agent import conf

    assert sorted(conf.__all__) == ["LLMSettings", "load_llm_settings"]


# ---------------------------------------------------------------------------
# _mask_key — used by the startup self-check log
# ---------------------------------------------------------------------------


def test_mask_key_long_enough_to_show_both_ends() -> None:
    """Long keys show first 4 + last 4 with '...' in the middle."""
    assert _mask_key("sk-cp-abcdefghijklmnop") == "sk-c...mnop"


def test_mask_key_short_key_redacted_entirely() -> None:
    """Keys <= 8 chars are redacted to a placeholder."""
    assert _mask_key("short") == "***"
    assert _mask_key("12345678") == "***"


def test_mask_key_empty_redacted() -> None:
    """Empty input is redacted too (defense-in-depth)."""
    assert _mask_key("") == "***"


# ---------------------------------------------------------------------------
# Startup self-check log
# ---------------------------------------------------------------------------


def test_get_settings_logs_resolved_config(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``get_settings`` must emit a single INFO line on first build
    showing model/provider/base_url/masked api_key — so the operator
    can confirm env vars were inherited by the subprocess."""
    from stock_analysis_agent.conf import settings as settings_module

    monkeypatch.setenv(API_KEY_ENV_VAR, "sk-cp-supersecret-key-12345")
    monkeypatch.setenv(BASE_URL_ENV_VAR, "https://api.minimaxi.com/anthropic")
    settings_module._cached_settings.cache_clear()
    caplog.set_level(logging.INFO, logger="stock_analysis_agent.conf.settings")
    try:
        s = get_settings()
        assert s.model == DEFAULT_MODEL
        assert s.provider == "anthropic"
        assert s.api_key == "sk-cp-supersecret-key-12345"

        msgs = [r.message for r in caplog.records]
        # Exactly one config log on first build.
        config_logs = [m for m in msgs if "LLM config:" in m]
        assert len(config_logs) == 1, f"expected 1 LLM config log, got: {msgs!r}"
        line = config_logs[0]
        assert f"model={DEFAULT_MODEL}" in line
        assert "provider=anthropic" in line
        assert "base_url=https://api.minimaxi.com/anthropic" in line
        # Key must be masked, not leaked.
        assert "supersecret-key-12345" not in line
        assert "sk-c" in line  # first 4 chars preserved
        assert "2345" in line  # last 4 chars preserved
    finally:
        settings_module._cached_settings.cache_clear()


def test_get_settings_logs_unset_base_url(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A missing ``$ANTHROPIC_BASE_URL`` must be surfaced as ``<unset>``
    in the log so the operator notices before the request goes to the
    default (real Anthropic) endpoint."""
    from stock_analysis_agent.conf import settings as settings_module

    monkeypatch.setenv(API_KEY_ENV_VAR, "sk-cp-somekey")
    monkeypatch.delenv(BASE_URL_ENV_VAR, raising=False)
    settings_module._cached_settings.cache_clear()
    caplog.set_level(logging.INFO, logger="stock_analysis_agent.conf.settings")
    try:
        get_settings()
        line = next(
            r.message
            for r in caplog.records
            if "LLM config:" in r.message
        )
        assert "base_url=<unset>" in line
    finally:
        settings_module._cached_settings.cache_clear()


def test_get_settings_does_not_re_log_on_subsequent_calls(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The lru_cache guarantees one log per process — repeat calls
    should not spam the operator."""
    from stock_analysis_agent.conf import settings as settings_module

    monkeypatch.setenv(API_KEY_ENV_VAR, "sk-cp-once")
    settings_module._cached_settings.cache_clear()
    caplog.set_level(logging.INFO, logger="stock_analysis_agent.conf.settings")
    try:
        get_settings()
        get_settings()
        get_settings()
        config_logs = [r for r in caplog.records if "LLM config:" in r.message]
        assert len(config_logs) == 1
    finally:
        settings_module._cached_settings.cache_clear()
