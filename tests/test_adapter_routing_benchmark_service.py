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

import pytest

from modelcypher.core.use_cases.adapter_routing_benchmark_service import (
    AdapterRoutingBenchmarkService,
)


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
    assert "dominance_gate" in rescored["summary"]
    assert rescored["summary"]["dominance_gate"]["recommended_mode"] == "base"


def test_dominance_gate_prefers_best_strict_system(any_backend: Any) -> None:
    service = _make_service(any_backend)
    systems = ["base", "adapter", "adapter_2", "routed_min_kl"]
    systems_summary = {
        "base": {"strict": {"correct": 6, "total": 20, "accuracy": 0.3}},
        "adapter": {"strict": {"correct": 9, "total": 20, "accuracy": 0.45}},
        "adapter_2": {"strict": {"correct": 5, "total": 20, "accuracy": 0.25}},
        "routed_min_kl": {"strict": {"correct": 8, "total": 20, "accuracy": 0.4}},
    }

    gate = service._build_dominance_gate(systems=systems, systems_summary=systems_summary)

    assert gate["recommended_mode"] == "single_adapter"
    assert gate["recommended_system"] == "adapter"
    assert gate["best_single_system"] == "adapter"
    assert gate["best_routed_system"] == "routed_min_kl"
    assert gate["dominant_single_adapter"]
    assert gate["strict_margins"]["routed_minus_best_single_correct"] == -1


def test_dominance_gate_marks_ties(any_backend: Any) -> None:
    service = _make_service(any_backend)
    systems = ["base", "adapter", "routed_min_kl"]
    systems_summary = {
        "base": {"strict": {"correct": 6, "total": 20, "accuracy": 0.3}},
        "adapter": {"strict": {"correct": 8, "total": 20, "accuracy": 0.4}},
        "routed_min_kl": {"strict": {"correct": 8, "total": 20, "accuracy": 0.4}},
    }

    gate = service._build_dominance_gate(systems=systems, systems_summary=systems_summary)

    assert gate["recommended_mode"] == "tie"
    assert gate["recommended_system"] is None
    assert gate["tied_systems"] == ["adapter", "routed_min_kl"]


def test_sweep_selectors_runs_all_and_compares(
    any_backend: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _make_service(any_backend)
    selectors = ["min_kl", "max_kl", "max_cosine", "min_cosine"]
    canned = {
        "min_kl": {
            "systems": ["base", "adapter", "routed_min_kl"],
            "summary": {
                "systems": {
                    "base": {"strict": {"correct": 6, "total": 20, "accuracy": 0.3}},
                    "adapter": {"strict": {"correct": 7, "total": 20, "accuracy": 0.35}},
                    "routed_min_kl": {"strict": {"correct": 9, "total": 20, "accuracy": 0.45}},
                },
                "dominance_gate": {
                    "recommended_mode": "routed",
                    "recommended_system": "routed_min_kl",
                },
            },
        },
        "max_kl": {
            "systems": ["base", "adapter", "routed_max_kl"],
            "summary": {
                "systems": {
                    "base": {"strict": {"correct": 6, "total": 20, "accuracy": 0.3}},
                    "adapter": {"strict": {"correct": 7, "total": 20, "accuracy": 0.35}},
                    "routed_max_kl": {"strict": {"correct": 4, "total": 20, "accuracy": 0.2}},
                },
                "dominance_gate": {
                    "recommended_mode": "single_adapter",
                    "recommended_system": "adapter",
                },
            },
        },
        "max_cosine": {
            "systems": ["base", "adapter", "routed_max_cosine"],
            "summary": {
                "systems": {
                    "base": {"strict": {"correct": 6, "total": 20, "accuracy": 0.3}},
                    "adapter": {"strict": {"correct": 7, "total": 20, "accuracy": 0.35}},
                    "routed_max_cosine": {"strict": {"correct": 9, "total": 20, "accuracy": 0.45}},
                },
                "dominance_gate": {
                    "recommended_mode": "tie",
                    "recommended_system": None,
                },
            },
        },
        "min_cosine": {
            "systems": ["base", "adapter", "routed_min_cosine"],
            "summary": {
                "systems": {
                    "base": {"strict": {"correct": 6, "total": 20, "accuracy": 0.3}},
                    "adapter": {"strict": {"correct": 7, "total": 20, "accuracy": 0.35}},
                    "routed_min_cosine": {"strict": {"correct": 1, "total": 20, "accuracy": 0.05}},
                },
                "dominance_gate": {
                    "recommended_mode": "single_adapter",
                    "recommended_system": "adapter",
                },
            },
        },
    }

    seen_selectors: list[str] = []

    def _fake_run(**kwargs: Any) -> dict[str, Any]:
        selector = str(kwargs["selection_method"])
        seen_selectors.append(selector)
        return canned[selector]

    monkeypatch.setattr(service, "run", _fake_run)

    output_dir = tmp_path / "selector_sweep"
    summary = service.sweep_selectors(
        base_model_path="/tmp/base",
        adapter_paths=["/tmp/a", "/tmp/b"],
        eval_suite_path="/tmp/eval.jsonl",
        selectors=selectors,
        output_directory=str(output_dir),
    )

    assert seen_selectors == selectors
    assert summary["selectors_run"] == selectors
    assert set(summary["per_selector"]) == set(selectors)
    assert summary["comparison"]["min_kl"]["strict_correct"]["routed_min_kl"] == 9
    assert summary["comparison"]["min_kl"]["recommended_mode"] == "routed"
    # Tie on routed strict between max_cosine and min_kl resolves alphabetically.
    assert summary["best_selector"] == "max_cosine"

    for selector in selectors:
        assert (output_dir / f"{selector}.json").exists()
    assert (output_dir / "sweep_summary.json").exists()


def test_dominance_gate_base_wins(any_backend: Any) -> None:
    service = _make_service(any_backend)
    systems = ["base", "adapter", "routed_min_kl"]
    systems_summary = {
        "base": {"strict": {"correct": 10, "total": 20, "accuracy": 0.5}},
        "adapter": {"strict": {"correct": 7, "total": 20, "accuracy": 0.35}},
        "routed_min_kl": {"strict": {"correct": 8, "total": 20, "accuracy": 0.4}},
    }

    gate = service._build_dominance_gate(systems=systems, systems_summary=systems_summary)

    assert gate["recommended_mode"] == "base"
    assert gate["recommended_system"] == "base"
    assert not gate["dominant_single_adapter"]
    assert not gate["dominant_routed"]


def test_dominance_gate_all_zero_scores(any_backend: Any) -> None:
    service = _make_service(any_backend)
    systems = ["base", "adapter", "routed_min_kl"]
    systems_summary = {
        "base": {"strict": {"correct": 0, "total": 20, "accuracy": 0.0}},
        "adapter": {"strict": {"correct": 0, "total": 20, "accuracy": 0.0}},
        "routed_min_kl": {"strict": {"correct": 0, "total": 20, "accuracy": 0.0}},
    }

    gate = service._build_dominance_gate(systems=systems, systems_summary=systems_summary)

    assert gate["recommended_mode"] == "tie"
    assert gate["recommended_system"] is None
    assert gate["tied_systems"] == ["adapter", "base", "routed_min_kl"]
    assert not gate["dominant_single_adapter"]
    assert not gate["dominant_routed"]


def test_dominance_gate_only_routed_systems(any_backend: Any) -> None:
    service = _make_service(any_backend)
    systems = ["base", "routed_min_kl", "routed_max_cosine"]
    systems_summary = {
        "base": {"strict": {"correct": 5, "total": 20, "accuracy": 0.25}},
        "routed_min_kl": {"strict": {"correct": 9, "total": 20, "accuracy": 0.45}},
        "routed_max_cosine": {"strict": {"correct": 7, "total": 20, "accuracy": 0.35}},
    }

    gate = service._build_dominance_gate(systems=systems, systems_summary=systems_summary)

    assert gate["recommended_mode"] == "routed"
    assert gate["recommended_system"] == "routed_min_kl"
    assert gate["best_single_system"] is None
    assert gate["best_routed_system"] == "routed_min_kl"
    assert not gate["dominant_routed"]
    assert "routed_minus_best_single_correct" not in gate["strict_margins"]


def test_dominance_gate_routed_wins(any_backend: Any) -> None:
    service = _make_service(any_backend)
    systems = ["base", "adapter", "routed_min_kl"]
    systems_summary = {
        "base": {"strict": {"correct": 5, "total": 20, "accuracy": 0.25}},
        "adapter": {"strict": {"correct": 7, "total": 20, "accuracy": 0.35}},
        "routed_min_kl": {"strict": {"correct": 11, "total": 20, "accuracy": 0.55}},
    }

    gate = service._build_dominance_gate(systems=systems, systems_summary=systems_summary)

    assert gate["recommended_mode"] == "routed"
    assert gate["recommended_system"] == "routed_min_kl"
    assert gate["dominant_routed"]
    assert not gate["dominant_single_adapter"]
    assert gate["strict_margins"]["routed_minus_best_single_correct"] == 4
