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

"""Tests for agent protocol and training diagnostics.

Pure-domain tests — no GPU, no MLX, no model loading.
"""

from __future__ import annotations

import json

import pytest

from modelcypher.core.domain.agent_protocol import (
    AgentDiagnostics,
    AgentEnvelope,
    AgentMetadata,
    AgentRecommendation,
    derived_eval_hash,
    file_hash,
    make_metadata,
    model_id,
)
from modelcypher.core.domain.training.diagnostics import (
    diagnose_training_result,
    interpret_pipeline_gate,
    interpret_stop_reason,
    suggest_next_steps,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic training results
# ---------------------------------------------------------------------------


def _make_result(**overrides) -> dict:
    """Build a minimal synthetic DatasetTrainResult.to_dict() payload."""
    base = {
        "train_iters": 100,
        "initial_loss": 2.5,
        "final_loss": 1.2,
        "stop_reason": "certificate (‖g‖=8.17e-01, Δmax=0.00e+00<CI=3.75e-02, epoch=8)",
        "baseline_loss": 2.5,
        "baseline_perplexity": 12.18,
        "post_loss": 1.2,
        "post_perplexity": 3.32,
        "n_lora_layers": 12,
        "n_trainable_params": 500_000,
        "adapter_path": "/tmp/adapter",
        "spectral_bounds_ok": True,
        "max_spectral_ratio": 0.45,
        "training_time_seconds": 120.0,
        "min_cka": 0.96,
        "mean_cka": 0.98,
        "adapter_saturation_median_ratio": 0.87,
        "pipeline_gate_passed": True,
        "pipeline_gate_failure_modes": [],
        "pipeline_gate_checks": {},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# AgentProtocol dataclass tests
# ---------------------------------------------------------------------------


class TestAgentProtocol:
    def test_recommendation_to_dict_with_command(self):
        rec = AgentRecommendation(
            action="evaluate",
            reason="Check inference quality",
            command="mc train evaluate -m /model -a /adapter",
        )
        d = rec.to_dict()
        assert d["action"] == "evaluate"
        assert d["command"] == "mc train evaluate -m /model -a /adapter"
        assert d["reason"] == "Check inference quality"

    def test_recommendation_to_dict_without_command(self):
        rec = AgentRecommendation(
            action="note_saturation",
            reason="Near geometric limit",
        )
        d = rec.to_dict()
        assert "command" not in d
        assert d["action"] == "note_saturation"

    def test_diagnostics_to_dict(self):
        diag = AgentDiagnostics(
            summary="Training completed.",
            observations=["Loss improved", "Gate passed"],
            recommendations=[
                AgentRecommendation("evaluate", "Check quality"),
            ],
        )
        d = diag.to_dict()
        assert d["summary"] == "Training completed."
        assert len(d["observations"]) == 2
        assert len(d["recommendations"]) == 1
        assert d["recommendations"][0]["action"] == "evaluate"

    def test_metadata_to_dict_sparse(self):
        meta = AgentMetadata(timestamp="2026-03-11T00:00:00Z")
        d = meta.to_dict()
        assert d["timestamp"] == "2026-03-11T00:00:00Z"
        assert "model" not in d
        assert "seed" not in d

    def test_metadata_to_dict_full(self):
        meta = AgentMetadata(
            timestamp="2026-03-11T00:00:00Z",
            model="/path/to/model",
            adapter_path="/path/to/adapter",
            duration_seconds=120.0,
            seed=42,
        )
        d = meta.to_dict()
        assert d["model"] == "/path/to/model"
        assert d["seed"] == 42

    def test_envelope_to_dict_is_json_serializable(self):
        envelope = AgentEnvelope(
            command="mc train run",
            status="success",
            result={"train_iters": 100, "final_loss": 1.2},
            diagnostics=AgentDiagnostics(summary="Done."),
            metadata=AgentMetadata(timestamp="2026-03-11T00:00:00Z"),
        )
        d = envelope.to_dict()
        serialized = json.dumps(d)
        parsed = json.loads(serialized)
        assert parsed["command"] == "mc train run"
        assert parsed["result"]["train_iters"] == 100
        assert parsed["diagnostics"]["summary"] == "Done."

    def test_make_metadata_timestamp(self):
        meta = make_metadata(model="/model", seed=42)
        assert meta.model == "/model"
        assert meta.seed == 42
        assert "T" in meta.timestamp  # ISO format


# ---------------------------------------------------------------------------
# Stop reason interpretation
# ---------------------------------------------------------------------------


class TestInterpretStopReason:
    def test_certificate(self):
        explanation = interpret_stop_reason(
            "certificate (‖g‖=8.17e-01, Δmax=0.00e+00<CI=3.75e-02, epoch=8)"
        )
        assert "noise floor" in explanation.lower()

    def test_adapter_saturation_exhausted(self):
        explanation = interpret_stop_reason(
            "adapter_saturation_exhausted (Weyl crossing, median_ratio=0.9500, epoch=6)"
        )
        assert "spectral capacity" in explanation.lower()

    def test_degeneration_exceeded(self):
        explanation = interpret_stop_reason(
            "degeneration_exceeded (max_ngram(3)=0.450 > baseline=0.100+eps, epoch=2)"
        )
        assert "degenerat" in explanation.lower()

    def test_safety_cap(self):
        explanation = interpret_stop_reason("safety_cap (1500 iters)")
        assert "safety" in explanation.lower()
        assert "machine precision" in explanation.lower()

    def test_loss_stable(self):
        explanation = interpret_stop_reason("loss_stable (|Δ_epoch| < SE = 1.2e-04)")
        assert "stabilized" in explanation.lower()

    def test_online_eval_degraded(self):
        explanation = interpret_stop_reason(
            "online_eval_degraded_significant (stage=pre_outcome, 15/25 correct, epoch=0)"
        )
        assert "degradation" in explanation.lower()

    def test_val_loss(self):
        explanation = interpret_stop_reason(
            "val_loss_converged (threshold=1.5e-04, epoch=4)"
        )
        assert "validation loss" in explanation.lower()

    def test_unknown_reason(self):
        explanation = interpret_stop_reason("something_totally_new")
        assert "something_totally_new" in explanation

    def test_none_reason(self):
        explanation = interpret_stop_reason(None)
        assert "completed" in explanation.lower()


# ---------------------------------------------------------------------------
# Pipeline gate interpretation
# ---------------------------------------------------------------------------


class TestInterpretPipelineGate:
    def test_pass_with_value(self):
        checks = {
            "spectral_bounds": {
                "status": "pass",
                "value": True,
                "message": None,
            }
        }
        obs = interpret_pipeline_gate(checks)
        assert len(obs) == 1
        assert "passed" in obs[0].lower()

    def test_fail_with_message(self):
        checks = {
            "cka_preservation": {
                "status": "fail",
                "message": "min_cka=0.82 < bound=0.90",
                "failure_mode": "cka_bound_violation",
            }
        }
        obs = interpret_pipeline_gate(checks)
        assert len(obs) == 1
        assert "FAILED" in obs[0]
        assert "min_cka" in obs[0]

    def test_unresolved(self):
        checks = {
            "mode_connectivity": {
                "status": "unresolved",
            }
        }
        obs = interpret_pipeline_gate(checks)
        assert "unresolved" in obs[0].lower()

    def test_empty_checks(self):
        assert interpret_pipeline_gate(None) == []
        assert interpret_pipeline_gate({}) == []


# ---------------------------------------------------------------------------
# Full diagnostic generation
# ---------------------------------------------------------------------------


class TestDiagnoseTrainingResult:
    def test_successful_training(self):
        result = _make_result()
        diag = diagnose_training_result(result, model_path="/model")
        assert isinstance(diag, AgentDiagnostics)
        assert len(diag.summary) > 0
        assert len(diag.observations) > 0
        # Should have an evaluate recommendation (adapter was saved)
        actions = [r.action for r in diag.recommendations]
        assert "evaluate" in actions

    def test_failed_gate_recommends_action(self):
        result = _make_result(
            pipeline_gate_passed=False,
            pipeline_gate_failure_modes=["degeneration"],
        )
        diag = diagnose_training_result(result)
        actions = [r.action for r in diag.recommendations]
        assert "try_different_data" in actions

    def test_loss_not_improved_recommends_check(self):
        result = _make_result(
            baseline_loss=2.5,
            post_loss=2.7,
        )
        diag = diagnose_training_result(result)
        actions = [r.action for r in diag.recommendations]
        assert "check_data_quality" in actions

    def test_saturated_adapter_noted(self):
        result = _make_result(adapter_saturation_median_ratio=0.98)
        diag = diagnose_training_result(result)
        actions = [r.action for r in diag.recommendations]
        assert "note_saturation" in actions

    def test_safety_cap_recommends_investigation(self):
        result = _make_result(stop_reason="safety_cap (1500 iters)")
        diag = diagnose_training_result(result)
        actions = [r.action for r in diag.recommendations]
        assert "investigate_convergence" in actions

    def test_degeneration_stop_recommends_different_data(self):
        result = _make_result(
            stop_reason="degeneration_exceeded (max_ngram(3)=0.450 > baseline=0.100+eps, epoch=2)"
        )
        diag = diagnose_training_result(result)
        actions = [r.action for r in diag.recommendations]
        assert "try_different_data" in actions

    def test_summary_contains_key_facts(self):
        result = _make_result()
        diag = diagnose_training_result(result)
        assert "100" in diag.summary  # train_iters
        assert "improved" in diag.summary.lower() or "noise floor" in diag.summary.lower()

    def test_observations_include_loss(self):
        result = _make_result(baseline_loss=2.5, post_loss=1.2)
        diag = diagnose_training_result(result)
        loss_obs = [o for o in diag.observations if "Loss" in o]
        assert len(loss_obs) > 0
        assert "improved" in loss_obs[0].lower()

    def test_observations_include_cka(self):
        result = _make_result(min_cka=0.96, mean_cka=0.98)
        diag = diagnose_training_result(result)
        cka_obs = [o for o in diag.observations if "CKA" in o]
        assert len(cka_obs) > 0
        assert "0.96" in cka_obs[0]

    def test_evaluate_command_includes_paths_and_mode(self):
        result = _make_result(adapter_path="/tmp/adapter")
        diag = diagnose_training_result(
            result, model_path="/model", adapter_path="/tmp/adapter"
        )
        eval_recs = [r for r in diag.recommendations if r.action == "evaluate"]
        assert len(eval_recs) > 0
        assert "/model" in eval_recs[0].command
        assert "/tmp/adapter" in eval_recs[0].command
        assert "--benchmark quick" in eval_recs[0].command

    def test_no_adapter_path_skips_evaluate(self):
        result = _make_result(adapter_path=None)
        diag = diagnose_training_result(result, model_path=None)
        actions = [r.action for r in diag.recommendations]
        assert "evaluate" not in actions


# ---------------------------------------------------------------------------
# Commensurability infrastructure (C1+C2)
# ---------------------------------------------------------------------------


class TestFileHash:
    def test_computes_sha256(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"text": "hello"}')
        h = file_hash(f)
        assert h is not None
        assert len(h) == 64  # SHA-256 hex

    def test_same_content_same_hash(self, tmp_path):
        f1 = tmp_path / "a.jsonl"
        f2 = tmp_path / "b.jsonl"
        content = '{"text": "identical"}'
        f1.write_text(content)
        f2.write_text(content)
        assert file_hash(f1) == file_hash(f2)

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.jsonl"
        f2 = tmp_path / "b.jsonl"
        f1.write_text("aaa")
        f2.write_text("bbb")
        assert file_hash(f1) != file_hash(f2)

    def test_nonexistent_returns_none(self, tmp_path):
        assert file_hash(tmp_path / "nope.jsonl") is None

    def test_directory_returns_none(self, tmp_path):
        assert file_hash(tmp_path) is None


class TestModelId:
    def test_from_config(self, tmp_path):
        config = {
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "vocab_size": 32000,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        mid = model_id(tmp_path)
        assert mid is not None
        assert len(mid) == 16  # Truncated SHA-256

    def test_same_config_same_id(self, tmp_path):
        config = {
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "vocab_size": 32000,
        }
        d1 = tmp_path / "m1"
        d2 = tmp_path / "m2"
        d1.mkdir()
        d2.mkdir()
        (d1 / "config.json").write_text(json.dumps(config))
        (d2 / "config.json").write_text(json.dumps(config))
        assert model_id(d1) == model_id(d2)

    def test_different_architecture_different_id(self, tmp_path):
        d1 = tmp_path / "m1"
        d2 = tmp_path / "m2"
        d1.mkdir()
        d2.mkdir()
        (d1 / "config.json").write_text(json.dumps({
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "vocab_size": 32000,
        }))
        (d2 / "config.json").write_text(json.dumps({
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "vocab_size": 32000,
        }))
        assert model_id(d1) != model_id(d2)

    def test_missing_config_returns_none(self, tmp_path):
        assert model_id(tmp_path) is None

    def test_invalid_json_returns_none(self, tmp_path):
        (tmp_path / "config.json").write_text("not json")
        assert model_id(tmp_path) is None


class TestMetadataIdentityFields:
    def test_to_dict_includes_identity(self):
        meta = AgentMetadata(
            timestamp="2026-03-11T00:00:00Z",
            model="/model",
            model_id="abc123",
            data_hash="def456",
            eval_data_hash="ghi789",
            benchmark_suite="quick",
        )
        d = meta.to_dict()
        assert d["model_id"] == "abc123"
        assert d["data_hash"] == "def456"
        assert d["eval_data_hash"] == "ghi789"
        assert d["benchmark_suite"] == "quick"

    def test_to_dict_omits_none_identity(self):
        meta = AgentMetadata(timestamp="2026-03-11T00:00:00Z")
        d = meta.to_dict()
        assert "model_id" not in d
        assert "data_hash" not in d
        assert "eval_data_hash" not in d
        assert "benchmark_suite" not in d

    def test_make_metadata_hashes_file(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"text": "hello"}')
        meta = make_metadata(data_path=str(f))
        assert meta.data_hash is not None
        assert len(meta.data_hash) == 64

    def test_make_metadata_nonexistent_file(self):
        meta = make_metadata(data_path="/nonexistent/data.jsonl")
        assert meta.data_hash is None


class TestDerivedEvalHash:
    """Auto-derived eval splits must produce a stable identity."""

    def test_deterministic(self):
        h1 = derived_eval_hash("abc123", 42, 5)
        h2 = derived_eval_hash("abc123", 42, 5)
        assert h1 == h2
        assert len(h1) == 64  # Full SHA-256 hex

    def test_different_seed_different_hash(self):
        h1 = derived_eval_hash("abc123", 42, 5)
        h2 = derived_eval_hash("abc123", 99, 5)
        assert h1 != h2

    def test_different_data_different_hash(self):
        h1 = derived_eval_hash("abc123", 42, 5)
        h2 = derived_eval_hash("xyz789", 42, 5)
        assert h1 != h2

    def test_different_split_size_different_hash(self):
        h1 = derived_eval_hash("abc123", 42, 5)
        h2 = derived_eval_hash("abc123", 42, 10)
        assert h1 != h2

    def test_make_metadata_precomputed_eval_hash(self, tmp_path):
        """Pre-computed eval_data_hash takes priority over eval_data_path."""
        f = tmp_path / "eval.jsonl"
        f.write_text('{"text": "eval data"}')
        precomputed = derived_eval_hash("data_hash", 42, 5)
        meta = make_metadata(
            eval_data_path=str(f),
            eval_data_hash=precomputed,
        )
        # Pre-computed hash wins over file hash
        assert meta.eval_data_hash == precomputed
        assert meta.eval_data_hash != file_hash(f)

    def test_make_metadata_falls_back_to_file_hash(self, tmp_path):
        """Without pre-computed hash, eval_data_path is hashed."""
        f = tmp_path / "eval.jsonl"
        f.write_text('{"text": "eval data"}')
        meta = make_metadata(eval_data_path=str(f))
        assert meta.eval_data_hash == file_hash(f)
