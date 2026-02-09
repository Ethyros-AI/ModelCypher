# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from modelcypher.core.domain.settings import SettingsSnapshot
from modelcypher.core.domain.storage_usage import DiskStats, StorageSnapshot, StorageUsage


def test_settings_snapshot_as_dict_uses_expected_api_keys() -> None:
    snapshot = SettingsSnapshot(
        idle_training_enabled=True,
        idle_training_min_idle_seconds=120,
        idle_training_max_thermal_state=2,
        max_memory_usage_percent=80,
        auto_save_checkpoints=False,
        platform_logging_opt_in=True,
    )

    assert snapshot.as_dict() == {
        "idleTrainingEnabled": True,
        "idleTrainingMinIdleSeconds": 120,
        "idleTrainingMaxThermalState": 2,
        "maxMemoryUsagePercent": 80,
        "autoSaveCheckpoints": False,
        "platformLoggingOptIn": True,
    }


def test_settings_snapshot_is_frozen() -> None:
    snapshot = SettingsSnapshot(
        idle_training_enabled=False,
        idle_training_min_idle_seconds=None,
        idle_training_max_thermal_state=None,
        max_memory_usage_percent=None,
        auto_save_checkpoints=True,
        platform_logging_opt_in=False,
    )

    with pytest.raises(FrozenInstanceError):
        snapshot.auto_save_checkpoints = False  # type: ignore[misc]


def test_storage_snapshot_round_trip_fields() -> None:
    usage = StorageUsage(
        total_gb=120.0,
        models_gb=60.0,
        checkpoints_gb=20.0,
        other_gb=40.0,
    )
    disk = DiskStats(total_bytes=1_000_000_000, free_bytes=250_000_000)
    snapshot = StorageSnapshot(usage=usage, disk=disk)

    assert snapshot.usage.total_gb == 120.0
    assert snapshot.usage.models_gb + snapshot.usage.checkpoints_gb + snapshot.usage.other_gb == 120.0
    assert snapshot.disk.total_bytes - snapshot.disk.free_bytes == 750_000_000


def test_storage_dataclasses_are_frozen() -> None:
    usage = StorageUsage(total_gb=1.0, models_gb=0.5, checkpoints_gb=0.25, other_gb=0.25)
    disk = DiskStats(total_bytes=10, free_bytes=2)

    with pytest.raises(FrozenInstanceError):
        usage.total_gb = 2.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        disk.free_bytes = 1  # type: ignore[misc]

