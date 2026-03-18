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

"""Tests for training comparison service.

Pure-domain tests — no GPU, no MLX, no model loading.
Tests cover dataclass serialization, metric comparison, winner determination,
result file comparison, and envelope construction.
"""

from __future__ import annotations

import json

import pytest

from modelcypher.core.use_cases.training_comparison_service import (
    ComparisonResult,
    MetricDelta,
    TrainingComparisonService,
)


# ---------------------------------------------------------------------------
# MetricDelta dataclass
# ---------------------------------------------------------------------------


class TestMetricDelta:
    def test_to_dict(self):
        md = MetricDelta(
            metric="post_loss",
            value_a=1.5,
            value_b=1.2,
            delta=-0.3,
            better="b",
        )
        d = md.to_dict()
        assert d["metric"] == "post_loss"
        assert d["delta"] == -0.3
        assert d["better"] == "b"

    def test_none_values(self):
        md = MetricDelta(
            metric="min_cka",
            value_a=None,
            value_b=None,
            delta=None,
            better=None,
        )
        d = md.to_dict()
        assert d["value_a"] is None
        assert d["delta"] is None


# ---------------------------------------------------------------------------
# ComparisonResult dataclass
# ---------------------------------------------------------------------------


class TestComparisonResult:
    def test_to_dict(self):
        result = ComparisonResult(
            label_a="run1",
            label_b="run2",
            metrics=[
                MetricDelta("post_loss", 1.5, 1.2, -0.3, "b"),
            ],
            winner="b",
            winner_reason="Lower post-training loss",
        )
        d = result.to_dict()
        assert d["label_a"] == "run1"
        assert d["winner"] == "b"
        assert len(d["metrics"]) == 1
        assert d["commensurable"] is True

    def test_no_winner(self):
        result = ComparisonResult(
            label_a="run1",
            label_b="run2",
            winner=None,
            winner_reason="No clear winner",
        )
        d = result.to_dict()
        assert d["winner"] is None

    def test_is_json_serializable(self):
        result = ComparisonResult(
            label_a="a",
            label_b="b",
            metrics=[MetricDelta("post_loss", 1.5, 1.2, -0.3, "b")],
            winner="b",
            winner_reason="test",
        )
        serialized = json.dumps(result.to_dict())
        parsed = json.loads(serialized)
        assert parsed["winner"] == "b"


# ---------------------------------------------------------------------------
# _compare_dicts
# ---------------------------------------------------------------------------


class TestCompareDicts:
    @pytest.fixture()
    def service(self):
        return TrainingComparisonService()

    def test_lower_is_better_b_wins(self, service):
        a = {"post_loss": 1.5}
        b = {"post_loss": 1.2}
        metrics = service._compare_dicts(a, b)
        assert len(metrics) == 1
        assert metrics[0].metric == "post_loss"
        assert metrics[0].delta == pytest.approx(-0.3)
        assert metrics[0].better == "b"

    def test_lower_is_better_a_wins(self, service):
        a = {"post_loss": 1.2}
        b = {"post_loss": 1.5}
        metrics = service._compare_dicts(a, b)
        assert metrics[0].better == "a"

    def test_higher_is_better_b_wins(self, service):
        a = {"min_cka": 0.90}
        b = {"min_cka": 0.95}
        metrics = service._compare_dicts(a, b)
        assert metrics[0].better == "b"

    def test_higher_is_better_a_wins(self, service):
        a = {"min_cka": 0.95}
        b = {"min_cka": 0.90}
        metrics = service._compare_dicts(a, b)
        assert metrics[0].better == "a"

    def test_tie(self, service):
        a = {"post_loss": 1.5}
        b = {"post_loss": 1.5}
        metrics = service._compare_dicts(a, b)
        assert metrics[0].better == "tie"

    def test_skips_missing_from_both(self, service):
        a = {"something_else": 1}
        b = {"something_else": 2}
        metrics = service._compare_dicts(a, b)
        assert len(metrics) == 0

    def test_skips_missing_from_one(self, service):
        a = {"post_loss": 1.5}
        b = {}
        metrics = service._compare_dicts(a, b)
        assert len(metrics) == 0

    def test_skips_non_numeric(self, service):
        a = {"post_loss": "not a number"}
        b = {"post_loss": 1.2}
        metrics = service._compare_dicts(a, b)
        assert len(metrics) == 0

    def test_uncategorized_metric_no_better(self, service):
        # training_time_seconds is in _COMPARE_KEYS but not in
        # _LOWER_IS_BETTER or _HIGHER_IS_BETTER
        a = {"training_time_seconds": 100.0}
        b = {"training_time_seconds": 200.0}
        metrics = service._compare_dicts(a, b)
        assert len(metrics) == 1
        assert metrics[0].better is None

    def test_multiple_metrics(self, service):
        a = {"post_loss": 1.5, "min_cka": 0.90, "final_loss": 1.3}
        b = {"post_loss": 1.2, "min_cka": 0.95, "final_loss": 1.4}
        metrics = service._compare_dicts(a, b)
        assert len(metrics) == 3
        metric_map = {m.metric: m for m in metrics}
        assert metric_map["post_loss"].better == "b"
        assert metric_map["min_cka"].better == "b"
        assert metric_map["final_loss"].better == "a"

    def test_pipeline_gate_compared(self, service):
        a = {"pipeline_gate_passed": 0}
        b = {"pipeline_gate_passed": 1}
        metrics = service._compare_dicts(a, b)
        assert len(metrics) == 1
        assert metrics[0].metric == "pipeline_gate_passed"
        assert metrics[0].better == "b"


# ---------------------------------------------------------------------------
# _determine_winner
# ---------------------------------------------------------------------------


class TestDetermineWinner:
    @pytest.fixture()
    def service(self):
        return TrainingComparisonService()

    def test_primary_post_loss_b_wins(self, service):
        metrics = [MetricDelta("post_loss", 1.5, 1.2, -0.3, "b")]
        winner, reason = service._determine_winner(metrics, {}, {})
        assert winner == "b"
        assert "post-training loss" in reason.lower()

    def test_primary_post_loss_a_wins(self, service):
        metrics = [MetricDelta("post_loss", 1.2, 1.5, 0.3, "a")]
        winner, reason = service._determine_winner(metrics, {}, {})
        assert winner == "a"

    def test_secondary_adapted_loss(self, service):
        # No post_loss → fall through to adapted_loss
        metrics = [MetricDelta("adapted_loss", 2.0, 1.5, -0.5, "b")]
        winner, reason = service._determine_winner(metrics, {}, {})
        assert winner == "b"
        assert "adapted loss" in reason.lower()

    def test_tertiary_min_cka(self, service):
        # No post_loss, no adapted_loss → fall through to min_cka
        metrics = [MetricDelta("min_cka", 0.85, 0.95, 0.10, "b")]
        winner, reason = service._determine_winner(metrics, {}, {})
        assert winner == "b"
        assert "CKA" in reason

    def test_fallback_count_wins(self, service):
        # No primary/secondary/tertiary → count wins
        metrics = [
            MetricDelta("n_improved", 3, 5, 2, "b"),
            MetricDelta("training_time_seconds", 100, 200, 100, None),
        ]
        winner, reason = service._determine_winner(metrics, {}, {})
        assert winner == "b"
        assert "1/" in reason  # "Wins on 1/2 metrics"

    def test_no_clear_winner(self, service):
        metrics = [
            MetricDelta("training_time_seconds", 100, 200, 100, None),
        ]
        winner, reason = service._determine_winner(metrics, {}, {})
        assert winner is None
        assert "no clear winner" in reason.lower()

    def test_post_loss_within_tolerance(self, service):
        # Delta < 1e-4 → treated as tie for post_loss
        metrics = [
            MetricDelta("post_loss", 1.50000, 1.50005, 0.00005, "a"),
            MetricDelta("min_cka", 0.85, 0.95, 0.10, "b"),
        ]
        winner, reason = service._determine_winner(metrics, {}, {})
        # post_loss within tolerance, falls through to min_cka
        assert winner == "b"
        assert "CKA" in reason


# ---------------------------------------------------------------------------
# compare_results (end-to-end with fixture files)
# ---------------------------------------------------------------------------


class TestCompareResults:
    @pytest.fixture()
    def service(self):
        return TrainingComparisonService()

    def _write_result(self, tmp_path, name: str, data: dict) -> None:
        path = tmp_path / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_b_wins_on_post_loss(self, service, tmp_path):
        path_a = self._write_result(tmp_path, "a.json", {
            "post_loss": 1.5,
            "min_cka": 0.90,
            "adapter_path": "/adapter_a",
        })
        path_b = self._write_result(tmp_path, "b.json", {
            "post_loss": 1.2,
            "min_cka": 0.95,
            "adapter_path": "/adapter_b",
        })
        result = service.compare_results(path_a, path_b)
        assert result.winner == "b"
        assert result.label_a == "/adapter_a"
        assert result.label_b == "/adapter_b"

    def test_a_wins_on_post_loss(self, service, tmp_path):
        path_a = self._write_result(tmp_path, "a.json", {
            "post_loss": 1.0,
        })
        path_b = self._write_result(tmp_path, "b.json", {
            "post_loss": 1.5,
        })
        result = service.compare_results(path_a, path_b)
        assert result.winner == "a"

    def test_unwraps_envelope(self, service, tmp_path):
        """Result files wrapped in AgentEnvelope should be unwrapped."""
        path_a = self._write_result(tmp_path, "a.json", {
            "command": "mc train run",
            "result": {"post_loss": 1.5, "adapter_path": "/a"},
        })
        path_b = self._write_result(tmp_path, "b.json", {
            "command": "mc train run",
            "result": {"post_loss": 1.2, "adapter_path": "/b"},
        })
        result = service.compare_results(path_a, path_b)
        assert result.winner == "b"
        assert result.label_a == "/a"

    def test_label_fallback_to_stem(self, service, tmp_path):
        path_a = self._write_result(tmp_path, "run1.json", {"post_loss": 1.5})
        path_b = self._write_result(tmp_path, "run2.json", {"post_loss": 1.2})
        result = service.compare_results(path_a, path_b)
        assert result.label_a == "run1"
        assert result.label_b == "run2"

    def test_no_comparable_metrics(self, service, tmp_path):
        path_a = self._write_result(tmp_path, "a.json", {"custom_field": 42})
        path_b = self._write_result(tmp_path, "b.json", {"custom_field": 99})
        result = service.compare_results(path_a, path_b)
        assert result.winner is None
        assert len(result.metrics) == 0


# ---------------------------------------------------------------------------
# Envelope construction
# ---------------------------------------------------------------------------


class TestComparisonEnvelope:
    @pytest.fixture()
    def service(self):
        return TrainingComparisonService()

    def test_winner_envelope(self, service):
        result = ComparisonResult(
            label_a="run1",
            label_b="run2",
            metrics=[MetricDelta("post_loss", 1.5, 1.2, -0.3, "b")],
            winner="b",
            winner_reason="Lower post-training loss",
            _meta_b={"model": "/model", "adapter_path": "/adapter-b"},
        )
        envelope = service.make_envelope(result)
        assert envelope.command == "mc train compare"
        assert envelope.status == "success"
        d = envelope.to_dict()
        assert "diagnostics" in d
        recs = d["diagnostics"]["recommendations"]
        assert len(recs) == 1
        assert recs[0]["action"] == "use_b"
        assert d["next_actions"][0]["name"] == "export"

    def test_no_winner_envelope(self, service):
        result = ComparisonResult(
            label_a="a",
            label_b="b",
            metrics=[],
            winner=None,
            winner_reason="No clear winner",
        )
        envelope = service.make_envelope(result)
        d = envelope.to_dict()
        assert "no clear winner" in d["diagnostics"]["summary"].lower()
        assert len(d["diagnostics"]["recommendations"]) == 0

    def test_envelope_observations(self, service):
        result = ComparisonResult(
            label_a="a",
            label_b="b",
            metrics=[
                MetricDelta("post_loss", 1.5, 1.2, -0.3, "b"),
                MetricDelta("min_cka", 0.90, 0.95, 0.05, "b"),
            ],
            winner="b",
            winner_reason="Better overall",
        )
        envelope = service.make_envelope(result)
        d = envelope.to_dict()
        obs = d["diagnostics"]["observations"]
        assert len(obs) == 2
        assert any("post_loss" in o for o in obs)
        assert any("min_cka" in o for o in obs)

    def test_gate_status_in_observations(self, service):
        result = ComparisonResult(
            label_a="run1",
            label_b="run2",
            metrics=[MetricDelta("post_loss", 1.5, 1.2, -0.3, "b")],
            winner="b",
            winner_reason="Lower post-training loss",
            _raw_a={"pipeline_gate_passed": True},
            _raw_b={"pipeline_gate_passed": False},
        )
        envelope = service.make_envelope(result)
        d = envelope.to_dict()
        obs = d["diagnostics"]["observations"]
        assert any("Run A: pipeline gate passed" in o for o in obs)
        assert any("Run B: pipeline gate failed" in o for o in obs)

    def test_gate_status_missing_no_observation(self, service):
        result = ComparisonResult(
            label_a="run1",
            label_b="run2",
            winner=None,
            winner_reason="No clear winner",
            _raw_a={"post_loss": 1.5},
            _raw_b={"post_loss": 1.2},
        )
        envelope = service.make_envelope(result)
        d = envelope.to_dict()
        obs = d["diagnostics"]["observations"]
        assert not any("pipeline gate" in o for o in obs)

    def test_envelope_is_json_serializable(self, service):
        result = ComparisonResult(
            label_a="a",
            label_b="b",
            winner=None,
            winner_reason="tie",
        )
        envelope = service.make_envelope(result)
        serialized = json.dumps(envelope.to_dict())
        parsed = json.loads(serialized)
        assert parsed["command"] == "mc train compare"


# ---------------------------------------------------------------------------
# Commensurability (C3)
# ---------------------------------------------------------------------------


class TestCommensurability:
    @pytest.fixture()
    def service(self):
        return TrainingComparisonService()

    def test_commensurable_by_default(self):
        result = ComparisonResult(label_a="a", label_b="b")
        assert result.commensurable is True
        assert result.to_dict()["commensurable"] is True

    def test_matching_metadata_commensurable(self, service, tmp_path):
        envelope_a = {
            "command": "mc train run",
            "result": {"post_loss": 1.5, "adapter_path": "/a"},
            "metadata": {"model_id": "abc123", "data_hash": "hash1"},
        }
        envelope_b = {
            "command": "mc train run",
            "result": {"post_loss": 1.2, "adapter_path": "/b"},
            "metadata": {"model_id": "abc123", "data_hash": "hash1"},
        }
        (tmp_path / "a.json").write_text(json.dumps(envelope_a))
        (tmp_path / "b.json").write_text(json.dumps(envelope_b))
        result = service.compare_results(tmp_path / "a.json", tmp_path / "b.json")
        assert result.commensurable is True

    def test_mismatched_model_id(self, service, tmp_path):
        envelope_a = {
            "command": "mc train run",
            "result": {"post_loss": 1.5, "adapter_path": "/a"},
            "metadata": {"model_id": "abc123"},
        }
        envelope_b = {
            "command": "mc train run",
            "result": {"post_loss": 1.2, "adapter_path": "/b"},
            "metadata": {"model_id": "xyz789"},
        }
        (tmp_path / "a.json").write_text(json.dumps(envelope_a))
        (tmp_path / "b.json").write_text(json.dumps(envelope_b))
        result = service.compare_results(tmp_path / "a.json", tmp_path / "b.json")
        assert result.commensurable is False

    def test_mismatched_data_hash(self, service, tmp_path):
        envelope_a = {
            "command": "mc train run",
            "result": {"post_loss": 1.5},
            "metadata": {"data_hash": "hash_a"},
        }
        envelope_b = {
            "command": "mc train run",
            "result": {"post_loss": 1.2},
            "metadata": {"data_hash": "hash_b"},
        }
        (tmp_path / "a.json").write_text(json.dumps(envelope_a))
        (tmp_path / "b.json").write_text(json.dumps(envelope_b))
        result = service.compare_results(tmp_path / "a.json", tmp_path / "b.json")
        assert result.commensurable is False

    def test_missing_metadata_backward_compat(self, service, tmp_path):
        # Old results without metadata → commensurable (no info to contradict)
        (tmp_path / "a.json").write_text(json.dumps({"post_loss": 1.5}))
        (tmp_path / "b.json").write_text(json.dumps({"post_loss": 1.2}))
        result = service.compare_results(tmp_path / "a.json", tmp_path / "b.json")
        assert result.commensurable is True

    def test_mismatch_observation_in_envelope(self, service, tmp_path):
        envelope_a = {
            "command": "mc train run",
            "result": {"post_loss": 1.5},
            "metadata": {"model_id": "abc"},
        }
        envelope_b = {
            "command": "mc train run",
            "result": {"post_loss": 1.2},
            "metadata": {"model_id": "xyz"},
        }
        (tmp_path / "a.json").write_text(json.dumps(envelope_a))
        (tmp_path / "b.json").write_text(json.dumps(envelope_b))
        result = service.compare_results(tmp_path / "a.json", tmp_path / "b.json")
        envelope = service.make_envelope(result)
        d = envelope.to_dict()
        obs = d["diagnostics"]["observations"]
        assert any("different model architecture" in o for o in obs)

    def test_one_side_missing_field_still_commensurable(self, service, tmp_path):
        # If only one side has the field, can't determine mismatch
        envelope_a = {
            "command": "mc train run",
            "result": {"post_loss": 1.5},
            "metadata": {"model_id": "abc"},
        }
        envelope_b = {
            "command": "mc train run",
            "result": {"post_loss": 1.2},
            "metadata": {},
        }
        (tmp_path / "a.json").write_text(json.dumps(envelope_a))
        (tmp_path / "b.json").write_text(json.dumps(envelope_b))
        result = service.compare_results(tmp_path / "a.json", tmp_path / "b.json")
        assert result.commensurable is True

    def test_non_commensurable_suppresses_winner_recommendation(self, service, tmp_path):
        """Mismatched identity metadata must suppress use_X recommendations.

        When runs are not commensurable, the envelope must:
        - set status to "partial" (fail closed)
        - produce zero recommendations
        - explain in the summary that runs are not comparable
        """
        envelope_a = {
            "command": "mc train run",
            "result": {"post_loss": 1.5, "adapter_path": "/a"},
            "metadata": {"model_id": "model_abc", "data_hash": "same"},
        }
        envelope_b = {
            "command": "mc train run",
            "result": {"post_loss": 1.2, "adapter_path": "/b"},
            "metadata": {"model_id": "model_xyz", "data_hash": "same"},
        }
        (tmp_path / "a.json").write_text(json.dumps(envelope_a))
        (tmp_path / "b.json").write_text(json.dumps(envelope_b))
        result = service.compare_results(tmp_path / "a.json", tmp_path / "b.json")
        # B has lower post_loss, so _determine_winner picks B
        assert result.winner == "b"
        assert result.commensurable is False

        envelope = service.make_envelope(result)
        d = envelope.to_dict()
        # Fail closed: status="partial", no winner recommendation
        assert d["status"] == "partial"
        assert len(d["diagnostics"]["recommendations"]) == 0
        assert "not commensurable" in d["diagnostics"]["summary"].lower()

    def test_mismatched_benchmark_suite_non_commensurable(self, service, tmp_path):
        """Mismatched benchmark_suite must mark comparison as non-commensurable."""
        envelope_a = {
            "command": "mc train run",
            "result": {"post_loss": 1.5},
            "metadata": {"benchmark_suite": "quick"},
        }
        envelope_b = {
            "command": "mc train run",
            "result": {"post_loss": 1.2},
            "metadata": {"benchmark_suite": "full"},
        }
        (tmp_path / "a.json").write_text(json.dumps(envelope_a))
        (tmp_path / "b.json").write_text(json.dumps(envelope_b))
        result = service.compare_results(tmp_path / "a.json", tmp_path / "b.json")
        assert result.commensurable is False
        # Envelope should note the mismatch
        envelope = service.make_envelope(result)
        obs = envelope.to_dict()["diagnostics"]["observations"]
        assert any("benchmark suite" in o for o in obs)
