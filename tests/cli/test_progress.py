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

from modelcypher.cli.progress import ProgressReporter


def test_training_loop_started_reports_precision_cap_context():
    events = []
    reporter = ProgressReporter(callback=events.append)

    reporter.training_loop_started(
        max_iters=141953,
        iters_per_epoch=49,
        precision_floor_epochs=2897,
    )

    assert len(events) == 1
    payload = events[0].to_dict()
    assert payload["_type"] == "training_progress"
    assert payload["stage"] == "train"
    assert payload["substage"] == "started"
    assert payload["progress"]["max_iters"] == 141953
    assert payload["progress"]["iters_per_epoch"] == 49
    assert payload["progress"]["precision_floor_epochs"] == 2897
    assert "geometric LoRA" in payload["what"]
    assert "geometry-derived LoRA" in payload["why"]
    assert "safety cap" in payload["geometry"]["explanation"]
