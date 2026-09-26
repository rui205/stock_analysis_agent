"""Tests for stock_analysis_agent.conf.settings."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from stock_analysis_agent.conf import LLMSettings, load_llm_settings
from stock_analysis_agent.conf.settings import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_MODEL_PROVIDER,
    DEFAULT_TEMPERATURE,
    MODEL_ENV_VAR,
    MissingAPIKeyError,
    ModelConfigError,
    _mask_key,
    get_settings,
    resolve_subagent_model,
)


def _write_config(tmp_path: Path, data: dict) -> str:
    """Write ``data`` to a temp ``model.json`` and return its path."""
    path = tmp_path / "model.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def _qwen_entry(**overrides) -> dict:
    """A valid ``qwen3.8-max`` entry, overridable per-field."""
    entry: dict = {
        "model": "qwen3.8-max",
        "api_key": "dummy-key",
        "base_url": "https://token-plan.cn-beijing.maas.aliyuncs.com/apps/anthropic",
    }
    entry.update(overrides)
    return entry


# ---------------------------------------------------------------------------
# model id + provider resolution
# ---------------------------------------------------------------------------


def test_default_model_uses_settings_constant(tmp_path: Path) -> None:
    """The default registry key mirrors the ``DEFAULT_MODEL`` constant."""
    path = _write_config(tmp_path, {DEFAULT_MODEL: _qwen_entry()})
    settings = load_llm_settings(config_path=path)
    assert settings.model == DEFAULT_MODEL


def test_default_provider_is_anthropic(tmp_path: Path) -> None:
    """provider defaults to ``anthropic`` when the entry omits it."""
    path = _write_config(tmp_path, {DEFAULT_MODEL: _qwen_entry()})
    settings = load_llm_settings(config_path=path)
    assert settings.provider == DEFAULT_MODEL_PROVIDER == "anthropic"


def test_explicit_provider_override(tmp_path: Path) -> None:
    """An entry-level ``provider`` field is honored."""
    path = _write_config(tmp_path, {DEFAULT_MODEL: _qwen_entry(provider="openai")})
    settings = load_llm_settings(config_path=path)
    assert settings.provider == "openai"


def test_provider_kwarg_overrides_entry(tmp_path: Path) -> None:
    """A caller-supplied ``provider`` wins over the registry value."""
    path = _write_config(tmp_path, {DEFAULT_MODEL: _qwen_entry(provider="openai")})
    settings = load_llm_settings(config_path=path, provider="anthropic")
    assert settings.provider == "anthropic"


# ---------------------------------------------------------------------------
# model selection — explicit arg vs MODEL env var vs default
# ---------------------------------------------------------------------------


def test_model_is_read_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The active entry is selected via ``MODEL`` without an explicit arg."""
    data = {
        DEFAULT_MODEL: _qwen_entry(),
        "deepseek-v4-pro": {
            "model": "deepseek-v4-pro",
            "api_key": "deepseek-key",
            "base_url": "https://api.deepseek.com/anthropic",
        },
    }
    path = _write_config(tmp_path, data)
    monkeypatch.setenv(MODEL_ENV_VAR, "deepseek-v4-pro")
    settings = load_llm_settings(config_path=path)
    assert settings.model == "deepseek-v4-pro"
    assert settings.api_key == "deepseek-key"
    assert settings.base_url == "https://api.deepseek.com/anthropic"


def test_explicit_model_arg_wins_over_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit ``model=`` key wins over the ``MODEL`` env var."""
    data = {
        DEFAULT_MODEL: _qwen_entry(),
        "deepseek-v4-pro": {
            "model": "deepseek-v4-pro",
            "api_key": "deepseek-key",
        },
    }
    path = _write_config(tmp_path, data)
    monkeypatch.setenv(MODEL_ENV_VAR, "deepseek-v4-pro")
    settings = load_llm_settings(config_path=path, model=DEFAULT_MODEL)
    assert settings.model == DEFAULT_MODEL
    assert settings.api_key == "dummy-key"


def test_unknown_model_raises(tmp_path: Path) -> None:
    """A key absent from the registry raises a clear ``ValueError``."""
    path = _write_config(tmp_path, {DEFAULT_MODEL: _qwen_entry()})
    with pytest.raises(ValueError, match="gpt-4"):
        load_llm_settings(config_path=path, model="gpt-4")


# ---------------------------------------------------------------------------
# api_key / base_url binding + error cases
# ---------------------------------------------------------------------------


def test_api_key_and_base_url_come_from_entry(tmp_path: Path) -> None:
    """The selected entry's ``api_key`` and ``base_url`` are resolved."""
    path = _write_config(
        tmp_path,
        {DEFAULT_MODEL: _qwen_entry(api_key="real-key", base_url="https://gw.example")},
    )
    settings = load_llm_settings(config_path=path)
    assert settings.api_key == "real-key"
    assert settings.base_url == "https://gw.example"


def test_explicit_api_key_overrides_entry(tmp_path: Path) -> None:
    """An explicit ``api_key`` argument wins over the registry value."""
    path = _write_config(tmp_path, {DEFAULT_MODEL: _qwen_entry(api_key="from-file")})
    settings = load_llm_settings(config_path=path, api_key="explicit-key")
    assert settings.api_key == "explicit-key"


def test_missing_api_key_raises(tmp_path: Path) -> None:
    """An entry with no ``api_key`` fails fast with :class:`MissingAPIKeyError`."""
    path = _write_config(tmp_path, {DEFAULT_MODEL: _qwen_entry(api_key="")})
    with pytest.raises(MissingAPIKeyError):
        load_llm_settings(config_path=path)


def test_blank_api_key_raises(tmp_path: Path) -> None:
    """A whitespace-only ``api_key`` is treated as missing."""
    path = _write_config(tmp_path, {DEFAULT_MODEL: _qwen_entry(api_key="   ")})
    with pytest.raises(MissingAPIKeyError):
        load_llm_settings(config_path=path)


def test_missing_config_file_raises(tmp_path: Path) -> None:
    """A non-existent registry path raises :class:`ModelConfigError`."""
    with pytest.raises(ModelConfigError, match="not found"):
        load_llm_settings(config_path=str(tmp_path / "nope.json"))


def test_invalid_json_raises(tmp_path: Path) -> None:
    """Malformed JSON raises :class:`ModelConfigError`."""
    path = tmp_path / "model.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ModelConfigError, match="invalid JSON"):
        load_llm_settings(config_path=str(path))


def test_non_object_config_raises(tmp_path: Path) -> None:
    """A non-object top-level document raises :class:`ModelConfigError`."""
    path = tmp_path / "model.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ModelConfigError, match="JSON object"):
        load_llm_settings(config_path=str(path))


# ---------------------------------------------------------------------------
# sampling defaults + overrides
# ---------------------------------------------------------------------------


def test_default_temperature_and_max_tokens(tmp_path: Path) -> None:
    """Sampling defaults match BaseAgent's baseline."""
    path = _write_config(tmp_path, {DEFAULT_MODEL: _qwen_entry()})
    settings = load_llm_settings(config_path=path)
    assert settings.temperature == DEFAULT_TEMPERATURE == 0.0
    assert settings.max_tokens == DEFAULT_MAX_TOKENS == 32768


def test_overrides_apply_per_field(tmp_path: Path) -> None:
    """Each scalar field can be overridden independently."""
    path = _write_config(tmp_path, {DEFAULT_MODEL: _qwen_entry()})
    settings = load_llm_settings(
        config_path=path,
        temperature=0.7,
        max_tokens=4096,
    )
    assert settings.temperature == 0.7
    assert settings.max_tokens == 4096


def test_settings_is_frozen(tmp_path: Path) -> None:
    """LLMSettings is immutable — protects against accidental mutation."""
    path = _write_config(tmp_path, {DEFAULT_MODEL: _qwen_entry()})
    settings = load_llm_settings(config_path=path)
    with pytest.raises((AttributeError, Exception)):  # FrozenInstanceError
        settings.model = "something-else"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# resolve_subagent_model — flash-tier selection
# ---------------------------------------------------------------------------


def test_subagent_model_returns_flash_model(tmp_path: Path) -> None:
    """An entry with ``flash_model`` routes sub-agents to the lighter tier."""
    path = _write_config(
        tmp_path,
        {
            DEFAULT_MODEL: _qwen_entry(flash_model="qwen3.8-flash"),
            "qwen3.8-flash": _qwen_entry(model="qwen3.8-flash"),
        },
    )
    assert resolve_subagent_model(config_path=path) == "qwen3.8-flash"


def test_subagent_model_falls_back_to_model(tmp_path: Path) -> None:
    """An entry without ``flash_model`` reuses its own model."""
    path = _write_config(tmp_path, {DEFAULT_MODEL: _qwen_entry()})
    assert resolve_subagent_model(config_path=path) == DEFAULT_MODEL


def test_subagent_model_selected_via_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``resolve_subagent_model`` honors the ``MODEL`` env var."""
    data = {
        DEFAULT_MODEL: _qwen_entry(flash_model="qwen3.8-flash"),
        "deepseek-v4-pro": {
            "model": "deepseek-v4-pro",
            "api_key": "deepseek-key",
        },
    }
    path = _write_config(tmp_path, data)
    monkeypatch.setenv(MODEL_ENV_VAR, "deepseek-v4-pro")
    # deepseek has no flash tier → falls back to its own model.
    assert resolve_subagent_model(config_path=path) == "deepseek-v4-pro"


def test_subagent_model_unknown_raises(tmp_path: Path) -> None:
    """A missing key raises ``ValueError`` from ``resolve_subagent_model``."""
    path = _write_config(tmp_path, {DEFAULT_MODEL: _qwen_entry()})
    with pytest.raises(ValueError):
        resolve_subagent_model(config_path=path, model="gpt-4")


# ---------------------------------------------------------------------------
# singleton accessor + startup log
# ---------------------------------------------------------------------------


def test_get_settings_singleton_uses_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The module-level accessor reads the registry via the same code path."""
    from stock_analysis_agent.conf import settings as settings_module

    path = _write_config(tmp_path, {DEFAULT_MODEL: _qwen_entry(api_key="singleton-key")})
    monkeypatch.setattr(settings_module, "MODEL_CONFIG_PATH", path)
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``get_settings`` must emit a single INFO line on first build showing
    model/provider/base_url/masked api_key — so the operator can confirm the
    right registry entry was resolved."""
    from stock_analysis_agent.conf import settings as settings_module

    path = _write_config(
        tmp_path,
        {
            DEFAULT_MODEL: _qwen_entry(
                api_key="sk-cp-supersecret-key-12345",
                base_url="https://api.minimaxi.com/anthropic",
            )
        },
    )
    monkeypatch.setattr(settings_module, "MODEL_CONFIG_PATH", path)
    settings_module._cached_settings.cache_clear()
    caplog.set_level(logging.INFO, logger="stock_analysis_agent.conf.settings")
    try:
        s = get_settings()
        assert s.model == DEFAULT_MODEL
        assert s.provider == "anthropic"
        assert s.api_key == "sk-cp-supersecret-key-12345"

        msgs = [r.message for r in caplog.records]
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An entry without ``base_url`` must be surfaced as ``<unset>`` in the
    log so the operator notices before the request goes to the default
    (real Anthropic) endpoint."""
    from stock_analysis_agent.conf import settings as settings_module

    path = _write_config(
        tmp_path, {DEFAULT_MODEL: _qwen_entry(base_url=None, api_key="sk-cp-somekey")}
    )
    monkeypatch.setattr(settings_module, "MODEL_CONFIG_PATH", path)
    settings_module._cached_settings.cache_clear()
    caplog.set_level(logging.INFO, logger="stock_analysis_agent.conf.settings")
    try:
        get_settings()
        line = next(
            r.message for r in caplog.records if "LLM config:" in r.message
        )
        assert "base_url=<unset>" in line
    finally:
        settings_module._cached_settings.cache_clear()


def test_get_settings_does_not_re_log_on_subsequent_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The lru_cache guarantees one log per process — repeat calls
    should not spam the operator."""
    from stock_analysis_agent.conf import settings as settings_module

    path = _write_config(tmp_path, {DEFAULT_MODEL: _qwen_entry(api_key="sk-cp-once")})
    monkeypatch.setattr(settings_module, "MODEL_CONFIG_PATH", path)
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
