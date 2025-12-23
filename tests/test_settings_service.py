from __future__ import annotations

from modelcypher.core.use_cases.settings_service import SettingsService


def test_settings_snapshot_defaults(monkeypatch) -> None:
    monkeypatch.delenv(SettingsService.ENV_IDLE_TRAINING_ENABLED, raising=False)
    monkeypatch.delenv(SettingsService.ENV_IDLE_TRAINING_MIN_IDLE_SECONDS, raising=False)
    monkeypatch.delenv(SettingsService.ENV_IDLE_TRAINING_MAX_THERMAL_STATE, raising=False)
    monkeypatch.delenv(SettingsService.ENV_MAX_MEMORY_USAGE_PERCENT, raising=False)
    monkeypatch.delenv(SettingsService.ENV_AUTO_SAVE_CHECKPOINTS, raising=False)
    monkeypatch.delenv(SettingsService.ENV_PLATFORM_LOGGING_OPT_IN, raising=False)

    snapshot = SettingsService().snapshot()
    assert snapshot.idle_training_enabled is False
    assert snapshot.idle_training_min_idle_seconds is None
    assert snapshot.idle_training_max_thermal_state is None
    assert snapshot.max_memory_usage_percent is None
    assert snapshot.auto_save_checkpoints is False
    assert snapshot.platform_logging_opt_in is False


def test_settings_snapshot_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv(SettingsService.ENV_IDLE_TRAINING_ENABLED, "true")
    monkeypatch.setenv(SettingsService.ENV_IDLE_TRAINING_MIN_IDLE_SECONDS, "600")
    monkeypatch.setenv(SettingsService.ENV_IDLE_TRAINING_MAX_THERMAL_STATE, "2")
    monkeypatch.setenv(SettingsService.ENV_MAX_MEMORY_USAGE_PERCENT, "85")
    monkeypatch.setenv(SettingsService.ENV_AUTO_SAVE_CHECKPOINTS, "1")
    monkeypatch.setenv(SettingsService.ENV_PLATFORM_LOGGING_OPT_IN, "yes")

    snapshot = SettingsService().snapshot()
    assert snapshot.idle_training_enabled is True
    assert snapshot.idle_training_min_idle_seconds == 600
    assert snapshot.idle_training_max_thermal_state == 2
    assert snapshot.max_memory_usage_percent == 85
    assert snapshot.auto_save_checkpoints is True
    assert snapshot.platform_logging_opt_in is True


def test_settings_individual_getters(monkeypatch) -> None:
    service = SettingsService()
    
    monkeypatch.setenv(SettingsService.ENV_IDLE_TRAINING_ENABLED, "true")
    assert service.idle_training_enabled is True
    
    monkeypatch.setenv(SettingsService.ENV_MAX_MEMORY_USAGE_PERCENT, "99")
    assert service.max_memory_usage_percent == 99


def test_settings_boolean_parsing(monkeypatch) -> None:
    service = SettingsService()
    
    for val in ["true", "1", "yes", "on"]:
        monkeypatch.setenv(SettingsService.ENV_PLATFORM_LOGGING_OPT_IN, val)
        assert service.platform_logging_opt_in is True
        
    for val in ["false", "0", "no", "off", ""]:
        monkeypatch.setenv(SettingsService.ENV_PLATFORM_LOGGING_OPT_IN, val)
        assert service.platform_logging_opt_in is False
