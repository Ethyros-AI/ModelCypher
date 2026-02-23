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

import pytest

from modelcypher.experimental.idle_training_scheduler import (
    ExperimentalIdleTrainingScheduler,
    SchedulerPolicy,
)


def test_scheduler_policy_requires_all_fields() -> None:
    with pytest.raises(TypeError):
        SchedulerPolicy()  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        SchedulerPolicy(  # type: ignore[call-arg]
            enabled=True,
            min_idle_seconds=1.0,
            max_thermal_state_raw=1,
            evaluation_interval=5.0,
            cooldown_duration=30.0,
        )


def test_scheduler_constructor_requires_policy() -> None:
    with pytest.raises(TypeError):
        ExperimentalIdleTrainingScheduler()  # type: ignore[call-arg]


def test_scheduler_accepts_explicit_policy(tmp_path) -> None:
    policy = SchedulerPolicy(
        enabled=True,
        min_idle_seconds=60.0,
        max_thermal_state_raw=1,
        evaluation_interval=5.0,
        cooldown_duration=30.0,
        memory_cache_valid_duration=2.0,
    )
    scheduler = ExperimentalIdleTrainingScheduler(
        policy=policy,
        state_file_path=str(tmp_path / "idle_scheduler_state.json"),
    )

    assert scheduler.policy == policy
    assert scheduler.policy.memory_cache_valid_duration == 2.0
