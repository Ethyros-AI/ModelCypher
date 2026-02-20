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

"""Tests for adapter routing benchmark scoring and rescoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from modelcypher.core.use_cases.adapter_routing_benchmark_service import AdapterRoutingBenchmarkService


class _StubRoutingService:
    def __init__(self, backend: Any) -> None:
        self._backend = backend


def _make_service(any_backend: Any) -> AdapterRoutingBenchmarkService:
    return AdapterRoutingBenchmarkService(_StubRoutingService(any_backend))


def test_text_contradiction_guard_blocks_false_positive(any_backend: Any) -> None:
    service = _make_service(any_backend)
    expected = "The student took Chemistry."
    response = "We cannot conclude that the student took Chemistry."

    lenient = service.score_lenient(expected, response)
    strict = service.score_strict(expected, response)

    assert not lenient.is_correct
    assert lenient.contradiction
    assert not strict.is_correct
    assert strict.contradiction


def test_numeric_strict_uses_final_answer(any_backend: Any) -> None:
    service = _make_service(any_backend)
    expected = "8 minutes (eggs boil simultaneously)"
    response = (
        "One egg takes 8 minutes. For four eggs, 8*4 = 32. "
        "Therefore 32 minutes. #### 32"
    )

    lenient = service.score_lenient(expected, response)
    strict = service.score_strict(expected, response)

    assert not lenient.is_correct
    assert not strict.is_correct
    assert strict.rule == "numeric_strict_final"


def test_yes_no_contradiction_guard(any_backend: Any) -> None:
    service = _make_service(any_backend)
    expected = "yes"
    response = "Yes, but actually no."

    lenient = service.score_lenient(expected, response)
    strict = service.score_strict(expected, response)

    assert not lenient.is_correct
    assert lenient.contradiction
    assert not strict.is_correct
    assert strict.contradiction


def test_text_strict_requires_expected_in_final_answer(any_backend: Any) -> None:
    service = _make_service(any_backend)
    expected = "Students can study at home."
    response = (
        "Given the chain, students can study at home. "
        "Final answer: students can borrow books."
    )

    lenient = service.score_lenient(expected, response)
    strict = service.score_strict(expected, response)

    assert lenient.is_correct
    assert not strict.is_correct


def test_rescore_existing_report_writes_strict_and_lenient(
    any_backend: Any,
    tmp_path: Path,
) -> None:
    service = _make_service(any_backend)
    report_path = tmp_path / "report.json"
    report_payload = {
        "systems": ["base", "routed_min_kl"],
        "cases": [
            {
                "id": "case_1",
                "category": "logic",
                "expected": "The student took Chemistry.",
                "responses": {
                    "base": "The student took Chemistry.",
                    "routed_min_kl": "We cannot conclude that the student took Chemistry.",
                },
            },
            {
                "id": "case_2",
                "category": "numeric",
                "expected": "0.10",
                "responses": {
                    "base": "#### 0.10",
                    "routed_min_kl": "Work... #### 0.20",
                },
            },
        ],
    }
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")

    out_path = service.rescore_existing_report(str(report_path))
    rescored = json.loads(Path(out_path).read_text(encoding="utf-8"))

    assert rescored["scorer"] == "adapter_routing_benchmark_service_v1"
    assert rescored["n_cases"] == 2
    assert "summary" in rescored
    assert "lenient" in rescored["summary"]["systems"]["base"]
    assert "strict" in rescored["summary"]["systems"]["routed_min_kl"]

