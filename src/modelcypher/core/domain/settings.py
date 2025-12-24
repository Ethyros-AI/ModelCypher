from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SettingsSnapshot:
    idle_training_enabled: bool
    idle_training_min_idle_seconds: int | None
    idle_training_max_thermal_state: int | None
    max_memory_usage_percent: int | None
    auto_save_checkpoints: bool
    platform_logging_opt_in: bool

    def as_dict(self) -> dict:
        return {
            "idleTrainingEnabled": self.idle_training_enabled,
            "idleTrainingMinIdleSeconds": self.idle_training_min_idle_seconds,
            "idleTrainingMaxThermalState": self.idle_training_max_thermal_state,
            "maxMemoryUsagePercent": self.max_memory_usage_percent,
            "autoSaveCheckpoints": self.auto_save_checkpoints,
            "platformLoggingOptIn": self.platform_logging_opt_in,
        }
