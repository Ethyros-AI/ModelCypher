# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

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
