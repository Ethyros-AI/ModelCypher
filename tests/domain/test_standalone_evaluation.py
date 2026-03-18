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

"""Tests for standalone evaluation service.

Pure-domain tests — no GPU, no MLX, no model loading.
Tests cover dataclass serialization, verdict logic, prompt loading,
summary generation, and envelope construction.
"""

from __future__ import annotations

import json

import pytest

from modelcypher.core.use_cases.standalone_evaluation_service import (
    InferenceComparison,
    StandaloneEvalResult,
    StandaloneEvaluationService,
)


# ---------------------------------------------------------------------------
# InferenceComparison dataclass
# ---------------------------------------------------------------------------


class TestInferenceComparison:
    def test_to_dict_basic(self):
        ic = InferenceComparison(
            prompt="What is 2+2?",
            base_response="4",
            adapted_response="The answer is 4.",
            reference=None,
            verdict="improved",
        )
        d = ic.to_dict()
        assert d["prompt"] == "What is 2+2?"
        assert d["verdict"] == "improved"
        assert "reference" not in d

    def test_to_dict_with_reference(self):
        ic = InferenceComparison(
            prompt="Capital of France?",
            base_response="London",
            adapted_response="Paris",
            reference="Paris",
            verdict="improved",
        )
        d = ic.to_dict()
        assert d["reference"] == "Paris"
        assert d["adapted_response"] == "Paris"


# ---------------------------------------------------------------------------
# StandaloneEvalResult dataclass
# ---------------------------------------------------------------------------


class TestStandaloneEvalResult:
    def test_inference_mode_to_dict(self):
        result = StandaloneEvalResult(
            model_path="/model",
            adapter_path="/adapter",
            mode="inference",
            n_prompts=10,
            n_improved=6,
            n_degraded=2,
            n_unchanged=1,
            n_degenerated=1,
            n_unmeasured=0,
            overall_verdict="improved",
            per_prompt=[
                InferenceComparison(
                    prompt="test",
                    base_response="a",
                    adapted_response="b",
                    reference=None,
                    verdict="improved",
                ),
            ],
        )
        d = result.to_dict()
        assert d["mode"] == "inference"
        assert d["n_prompts"] == 10
        assert d["n_improved"] == 6
        assert d["n_unmeasured"] == 0
        assert len(d["per_prompt"]) == 1
        assert "base_perplexity" not in d

    def test_loss_mode_to_dict(self):
        result = StandaloneEvalResult(
            model_path="/model",
            adapter_path="/adapter",
            mode="loss",
            base_loss=2.5,
            base_perplexity=12.18,
            adapted_loss=1.8,
            adapted_perplexity=6.05,
            overall_verdict="improved",
        )
        d = result.to_dict()
        assert d["mode"] == "loss"
        assert d["base_loss"] == 2.5
        assert d["adapted_loss"] == 1.8
        assert d["base_perplexity"] == 12.18
        assert "n_prompts" not in d

    def test_benchmark_mode_to_dict(self):
        result = StandaloneEvalResult(
            model_path="/model",
            adapter_path=None,
            mode="benchmark",
            benchmark_results={"base": {"gsm8k": 0.45}},
            overall_verdict="neutral",
        )
        d = result.to_dict()
        assert d["mode"] == "benchmark"
        assert d["benchmark_results"]["base"]["gsm8k"] == 0.45
        assert d["adapter_path"] is None

    def test_minimal_result(self):
        result = StandaloneEvalResult(
            model_path="/model",
            adapter_path=None,
            mode="inference",
        )
        d = result.to_dict()
        assert d["overall_verdict"] == "neutral"
        assert d["n_prompts"] == 0
        assert d["n_unmeasured"] == 0

    def test_to_dict_is_json_serializable(self):
        result = StandaloneEvalResult(
            model_path="/model",
            adapter_path="/adapter",
            mode="loss",
            base_loss=2.5,
            adapted_loss=1.8,
            overall_verdict="improved",
        )
        serialized = json.dumps(result.to_dict())
        parsed = json.loads(serialized)
        assert parsed["base_loss"] == 2.5


# ---------------------------------------------------------------------------
# Verdict determination (static method)
# ---------------------------------------------------------------------------


class TestDetermineVerdict:
    def test_improved(self):
        verdict = StandaloneEvaluationService._determine_verdict(5, 2, 1, 0)
        assert verdict == "improved"

    def test_degraded(self):
        verdict = StandaloneEvaluationService._determine_verdict(2, 5, 1, 0)
        assert verdict == "degraded"

    def test_neutral_tie(self):
        verdict = StandaloneEvaluationService._determine_verdict(3, 3, 2, 0)
        assert verdict == "neutral"

    def test_degeneration_dominates_when_exceeds_improved(self):
        # 4 degenerated >= 2 improved → degenerated
        verdict = StandaloneEvaluationService._determine_verdict(2, 1, 1, 4)
        assert verdict == "degenerated"

    def test_degeneration_equal_to_improved(self):
        # 3 degenerated >= 3 improved → degenerated (degenerated wins ties)
        verdict = StandaloneEvaluationService._determine_verdict(3, 1, 1, 3)
        assert verdict == "degenerated"

    def test_degeneration_below_improved(self):
        # 2 degenerated < 5 improved → improved wins
        verdict = StandaloneEvaluationService._determine_verdict(5, 2, 1, 2)
        assert verdict == "improved"

    def test_all_zero(self):
        verdict = StandaloneEvaluationService._determine_verdict(0, 0, 0, 0)
        assert verdict == "neutral"

    def test_all_unchanged(self):
        verdict = StandaloneEvaluationService._determine_verdict(0, 0, 5, 0)
        assert verdict == "neutral"

    def test_unmeasured_excluded(self):
        # 3 improved + 7 unmeasured → "improved" (unmeasured doesn't dilute)
        verdict = StandaloneEvaluationService._determine_verdict(3, 1, 0, 0, unmeasured=7)
        assert verdict == "improved"

    def test_all_unmeasured(self):
        # 0 measured → "neutral"
        verdict = StandaloneEvaluationService._determine_verdict(0, 0, 0, 0, unmeasured=10)
        assert verdict == "neutral"

    def test_single_degeneration_no_improvement(self):
        # 1 degenerated, 0 improved → degenerated (1 >= 0)
        verdict = StandaloneEvaluationService._determine_verdict(0, 0, 5, 1)
        assert verdict == "degenerated"


# ---------------------------------------------------------------------------
# Summary builder (static method)
# ---------------------------------------------------------------------------


class TestBuildSummary:
    def test_inference_summary(self):
        result = StandaloneEvalResult(
            model_path="/model",
            adapter_path="/adapter",
            mode="inference",
            n_prompts=10,
            n_improved=6,
            n_degraded=2,
            n_degenerated=1,
            overall_verdict="improved",
        )
        summary = StandaloneEvaluationService._build_summary(result)
        assert "10" in summary
        assert "6 improved" in summary
        assert "improved" in summary.lower()

    def test_loss_summary(self):
        result = StandaloneEvalResult(
            model_path="/model",
            adapter_path="/adapter",
            mode="loss",
            base_loss=2.5,
            adapted_loss=1.8,
            overall_verdict="improved",
        )
        summary = StandaloneEvaluationService._build_summary(result)
        assert "2.5000" in summary
        assert "1.8000" in summary

    def test_benchmark_summary(self):
        result = StandaloneEvalResult(
            model_path="/model",
            adapter_path=None,
            mode="benchmark",
            overall_verdict="neutral",
        )
        summary = StandaloneEvaluationService._build_summary(result)
        assert "benchmark" in summary.lower()
        assert "neutral" in summary.lower()


# ---------------------------------------------------------------------------
# Prompt loading (static method, uses filesystem)
# ---------------------------------------------------------------------------


class TestLoadPrompts:
    def test_valid_prompts(self, tmp_path):
        f = tmp_path / "prompts.jsonl"
        f.write_text(
            '{"prompt": "What is 2+2?", "reference": "4"}\n'
            '{"prompt": "Capital of France?"}\n'
        )
        prompts = StandaloneEvaluationService._load_prompts(f)
        assert len(prompts) == 2
        assert prompts[0]["prompt"] == "What is 2+2?"
        assert prompts[0]["reference"] == "4"
        assert "reference" not in prompts[1]

    def test_skips_empty_lines(self, tmp_path):
        f = tmp_path / "prompts.jsonl"
        f.write_text('{"prompt": "test"}\n\n{"prompt": "test2"}\n  \n')
        prompts = StandaloneEvaluationService._load_prompts(f)
        assert len(prompts) == 2

    def test_skips_invalid_json(self, tmp_path):
        f = tmp_path / "prompts.jsonl"
        f.write_text('{"prompt": "valid"}\nnot json\n')
        prompts = StandaloneEvaluationService._load_prompts(f)
        assert len(prompts) == 1

    def test_skips_missing_prompt_field(self, tmp_path):
        f = tmp_path / "prompts.jsonl"
        f.write_text('{"prompt": "valid"}\n{"question": "no prompt key"}\n')
        prompts = StandaloneEvaluationService._load_prompts(f)
        assert len(prompts) == 1

    def test_empty_file(self, tmp_path):
        f = tmp_path / "prompts.jsonl"
        f.write_text("")
        prompts = StandaloneEvaluationService._load_prompts(f)
        assert len(prompts) == 0


# ---------------------------------------------------------------------------
# Envelope construction (requires a pre-built result, no backend needed)
# ---------------------------------------------------------------------------


class _FakeBackend:
    """Minimal stub for Backend to construct the service."""

    def load_model(self, *args, **kwargs):
        raise NotImplementedError

    def generate(self, *args, **kwargs):
        raise NotImplementedError

    def finfo(self):
        raise NotImplementedError


class TestMakeEnvelope:
    @pytest.fixture()
    def service(self):
        return StandaloneEvaluationService(backend=_FakeBackend())

    def test_inference_envelope(self, service):
        result = StandaloneEvalResult(
            model_path="/model",
            adapter_path="/adapter",
            mode="inference",
            n_prompts=5,
            n_improved=3,
            n_degraded=1,
            n_unchanged=1,
            n_degenerated=0,
            overall_verdict="improved",
        )
        envelope = service.make_envelope(result)
        assert envelope.command == "mc train evaluate"
        assert envelope.status == "success"
        d = envelope.to_dict()
        assert "diagnostics" in d
        # Inference mode recommends benchmark evaluation, not direct deploy
        actions = [r["action"] for r in d["diagnostics"]["recommendations"]]
        assert "evaluate_benchmark" in actions
        assert "deploy" not in actions

    def test_loss_envelope(self, service):
        result = StandaloneEvalResult(
            model_path="/model",
            adapter_path="/adapter",
            mode="loss",
            base_loss=2.5,
            base_perplexity=12.18,
            adapted_loss=1.8,
            adapted_perplexity=6.05,
            overall_verdict="improved",
        )
        envelope = service.make_envelope(result)
        d = envelope.to_dict()
        obs = d["diagnostics"]["observations"]
        assert any("Loss" in o for o in obs)
        assert any("Perplexity" in o for o in obs)

    def test_degraded_envelope_recommends_different_data(self, service):
        result = StandaloneEvalResult(
            model_path="/model",
            adapter_path="/adapter",
            mode="loss",
            base_loss=2.5,
            adapted_loss=3.0,
            overall_verdict="degraded",
        )
        envelope = service.make_envelope(result)
        d = envelope.to_dict()
        actions = [r["action"] for r in d["diagnostics"]["recommendations"]]
        assert "try_different_data" in actions

    def test_degeneration_envelope(self, service):
        result = StandaloneEvalResult(
            model_path="/model",
            adapter_path="/adapter",
            mode="inference",
            n_prompts=10,
            n_improved=2,
            n_degraded=1,
            n_unchanged=1,
            n_degenerated=6,
            overall_verdict="degenerated",
        )
        envelope = service.make_envelope(result)
        d = envelope.to_dict()
        actions = [r["action"] for r in d["diagnostics"]["recommendations"]]
        assert "investigate_degeneration" in actions

    def test_loss_improved_recommends_deploy(self, service):
        result = StandaloneEvalResult(
            model_path="/model",
            adapter_path="/adapter",
            mode="loss",
            base_loss=2.5,
            adapted_loss=1.8,
            overall_verdict="improved",
        )
        envelope = service.make_envelope(result, eval_data_path="/eval.jsonl")
        d = envelope.to_dict()
        actions = [r["action"] for r in d["diagnostics"]["recommendations"]]
        assert "deploy" in actions
        next_steps = [r["name"] for r in d["next_actions"]]
        assert "compare" in next_steps
        assert "export" in next_steps

    def test_benchmark_improved_recommends_deploy(self, service):
        result = StandaloneEvalResult(
            model_path="/model",
            adapter_path="/adapter",
            mode="benchmark",
            overall_verdict="improved",
        )
        envelope = service.make_envelope(result)
        d = envelope.to_dict()
        actions = [r["action"] for r in d["diagnostics"]["recommendations"]]
        assert "deploy" in actions

    def test_inference_envelope_next_actions_include_benchmark(self, service):
        result = StandaloneEvalResult(
            model_path="/model",
            adapter_path="/adapter",
            mode="inference",
            n_prompts=3,
            n_improved=2,
            overall_verdict="improved",
        )
        envelope = service.make_envelope(result)
        next_steps = [r["name"] for r in envelope.to_dict()["next_actions"]]
        assert "benchmark" in next_steps

    def test_unmeasured_in_observation(self, service):
        result = StandaloneEvalResult(
            model_path="/model",
            adapter_path="/adapter",
            mode="inference",
            n_prompts=5,
            n_improved=2,
            n_unmeasured=3,
            overall_verdict="improved",
        )
        envelope = service.make_envelope(result)
        d = envelope.to_dict()
        obs = " ".join(d["diagnostics"]["observations"])
        assert "unmeasured" in obs

    def test_envelope_is_json_serializable(self, service):
        result = StandaloneEvalResult(
            model_path="/model",
            adapter_path=None,
            mode="inference",
            overall_verdict="neutral",
        )
        envelope = service.make_envelope(result)
        serialized = json.dumps(envelope.to_dict())
        parsed = json.loads(serialized)
        assert parsed["command"] == "mc train evaluate"

    def test_envelope_metadata(self, service):
        result = StandaloneEvalResult(
            model_path="/model",
            adapter_path="/adapter",
            mode="loss",
            overall_verdict="neutral",
        )
        envelope = service.make_envelope(result)
        d = envelope.to_dict()
        assert d["metadata"]["model"] == "/model"
        assert d["metadata"]["adapter_path"] == "/adapter"
