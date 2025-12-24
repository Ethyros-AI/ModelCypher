from __future__ import annotations

import os


from modelcypher.core.domain.settings import SettingsSnapshot


class SettingsService:
    ENV_IDLE_TRAINING_ENABLED = "MC_IDLE_TRAINING_ENABLED"
    ENV_IDLE_TRAINING_MIN_IDLE_SECONDS = "MC_IDLE_TRAINING_MIN_IDLE_SECONDS"
    ENV_IDLE_TRAINING_MAX_THERMAL_STATE = "MC_IDLE_TRAINING_MAX_THERMAL_STATE"
    ENV_MAX_MEMORY_USAGE_PERCENT = "MC_MAX_MEMORY_USAGE_PERCENT"
    ENV_AUTO_SAVE_CHECKPOINTS = "MC_AUTO_SAVE_CHECKPOINTS"
    ENV_PLATFORM_LOGGING_OPT_IN = "MC_PLATFORM_LOGGING_OPT_IN"

    @property
    def idle_training_enabled(self) -> bool:
        return _parse_bool(os.environ.get(self.ENV_IDLE_TRAINING_ENABLED))

    @property
    def idle_training_min_idle_seconds(self) -> int | None:
        return _parse_optional_int(os.environ.get(self.ENV_IDLE_TRAINING_MIN_IDLE_SECONDS))

    @property
    def idle_training_max_thermal_state(self) -> int | None:
        return _parse_optional_int(os.environ.get(self.ENV_IDLE_TRAINING_MAX_THERMAL_STATE))

    @property
    def max_memory_usage_percent(self) -> int | None:
        return _parse_optional_int(os.environ.get(self.ENV_MAX_MEMORY_USAGE_PERCENT))

    @property
    def auto_save_checkpoints(self) -> bool:
        return _parse_bool(os.environ.get(self.ENV_AUTO_SAVE_CHECKPOINTS))

    @property
    def platform_logging_opt_in(self) -> bool:
        return _parse_bool(os.environ.get(self.ENV_PLATFORM_LOGGING_OPT_IN))

    def snapshot(self) -> SettingsSnapshot:
        return SettingsSnapshot(
            idle_training_enabled=_parse_bool(os.environ.get(self.ENV_IDLE_TRAINING_ENABLED)),
            idle_training_min_idle_seconds=_parse_optional_int(
                os.environ.get(self.ENV_IDLE_TRAINING_MIN_IDLE_SECONDS)
            ),
            idle_training_max_thermal_state=_parse_optional_int(
                os.environ.get(self.ENV_IDLE_TRAINING_MAX_THERMAL_STATE)
            ),
            max_memory_usage_percent=_parse_optional_int(os.environ.get(self.ENV_MAX_MEMORY_USAGE_PERCENT)),
            auto_save_checkpoints=_parse_bool(os.environ.get(self.ENV_AUTO_SAVE_CHECKPOINTS)),
            platform_logging_opt_in=_parse_bool(os.environ.get(self.ENV_PLATFORM_LOGGING_OPT_IN)),
        )


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return False


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    try:
        return int(normalized)
    except ValueError:
        return None
