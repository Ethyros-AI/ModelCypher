from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SettingsSnapshot:
    idle_training_enabled: bool
    idle_training_min_idle_seconds: Optional[int]
    idle_training_max_thermal_state: Optional[int]
    max_memory_usage_percent: Optional[int]
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
