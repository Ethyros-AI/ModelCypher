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

"""Validation harness for derived dataset training.

Runs repeated training trials with derived settings and captures
counterexamples where post-training metrics fail to improve over baseline.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.domain.star.problem_generator import StarProblem, StarProblemGenerator
from modelcypher.core.domain.training.online_eval import (
    OnlineEvalResult,
    create_eval_problem_set,
    evaluate_correctness,
)


@dataclass(frozen=True)
class _Phase5Metrics:
    baseline_n_correct: int
    baseline_n_total: int
    adapted_n_correct: int
    adapted_n_total: int
    baseline_max_ngram_repeat: float | None
    adapted_max_ngram_repeat: float | None
    baseline_margins: dict[str, float] | None = None
    adapted_margins: dict[str, float] | None = None
    max_logit_delta_inf: float | None = None
    argmax_cert_gap: float | None = None
    argmax_preservation_certified: bool | None = None
    argmax_n_correct_flipped: int | None = None


@dataclass
class _Phase5Context:
    enabled: bool
    probe_count: int
    probe_seed: int
    artifact_root: Path
    keep_all_artifacts: bool
    probe_problems: list[StarProblem]
    baseline_eval: OnlineEvalResult | None = None
    baseline_max_ngram_repeat: float | None = None
    max_tokens: int | None = None
    baseline_margins: dict[str, float] | None = None
    baseline_logits: dict[str, Any] | None = None
    ngram_order: int | None = None


@dataclass(frozen=True)
class DerivedTrainingTrial:
    """Single repeated training trial result."""

    trial_index: int
    seed: int
    baseline_loss: float
    post_loss: float
    loss_delta: float
    baseline_perplexity: float
    post_perplexity: float
    perplexity_delta: float
    spectral_bounds_ok: bool
    max_spectral_ratio: float
    stop_reason: str
    train_iters: int
    training_time_seconds: float
    min_cka: float | None
    mean_cka: float | None
    min_cka_layer: int | None
    per_layer_cka: dict[int, float] | None
    per_layer_gram_epsilon: dict[int, float] | None
    per_layer_cka_bound: dict[int, float] | None
    per_layer_null_observability: dict[int, dict[str, float | int]] | None
    per_layer_null_accessibility: dict[int, dict[str, float | int]] | None
    per_module_null_accessibility: dict[str, dict[str, float | int]] | None
    cka_margin_to_bound: float | None
    null_access_min_behavioral_preserved_fraction: float | None
    null_access_min_behavioral_preserved_layer: int | None
    null_observability_max_condition_number: float | None
    null_observability_max_condition_layer: int | None
    online_eval_first_pre_degraded_epoch: int | None
    online_eval_first_post_degraded_epoch: int | None
    epoch_geometry_trace: list[dict[str, Any]] | None
    adapter_saturation_median_ratio: float | None
    max_effective_gain_ratio: float | None
    online_eval_baseline_correct: int | None
    online_eval_baseline_total: int | None
    online_eval_adapted_correct: int | None
    online_eval_adapted_total: int | None
    online_eval_delta_correct: int | None
    baseline_max_ngram_repeat: float | None
    adapted_max_ngram_repeat: float | None
    max_ngram_repeat_delta: float | None
    ngram_order: int | None
    # Dual-manifold CKA diagnostics
    inference_min_cka: float | None
    inference_mean_cka: float | None
    inference_min_cka_layer: int | None
    cka_blindness_ratio: float | None
    cka_blindness_worst_layer: int | None
    cka_delta_gap: float | None
    # Decision-boundary margin diagnostics
    margin_mean_baseline: float | None
    margin_mean_adapted: float | None
    margin_mean_delta: float | None
    margin_n_near_zero_baseline: int | None
    margin_n_near_zero_adapted: int | None
    margin_n_flipped_sign: int | None
    max_logit_delta_inf: float | None
    argmax_cert_gap: float | None
    argmax_preservation_certified: bool | None
    argmax_n_correct_flipped: int | None
    moe_router_stability: float | None
    structural_passed: bool
    inference_passed: bool
    cooccurrence_class: str
    passed: bool
    failure_modes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_index": self.trial_index,
            "seed": self.seed,
            "baseline_loss": self.baseline_loss,
            "post_loss": self.post_loss,
            "loss_delta": self.loss_delta,
            "baseline_perplexity": self.baseline_perplexity,
            "post_perplexity": self.post_perplexity,
            "perplexity_delta": self.perplexity_delta,
            "spectral_bounds_ok": self.spectral_bounds_ok,
            "max_spectral_ratio": self.max_spectral_ratio,
            "stop_reason": self.stop_reason,
            "train_iters": self.train_iters,
            "training_time_seconds": self.training_time_seconds,
            "min_cka": self.min_cka,
            "mean_cka": self.mean_cka,
            "min_cka_layer": self.min_cka_layer,
            "per_layer_cka": self.per_layer_cka,
            "per_layer_gram_epsilon": self.per_layer_gram_epsilon,
            "per_layer_cka_bound": self.per_layer_cka_bound,
            "per_layer_null_observability": self.per_layer_null_observability,
            "per_layer_null_accessibility": self.per_layer_null_accessibility,
            "per_module_null_accessibility": self.per_module_null_accessibility,
            "cka_margin_to_bound": self.cka_margin_to_bound,
            "null_access_min_behavioral_preserved_fraction": (
                self.null_access_min_behavioral_preserved_fraction
            ),
            "null_access_min_behavioral_preserved_layer": (
                self.null_access_min_behavioral_preserved_layer
            ),
            "null_observability_max_condition_number": (
                self.null_observability_max_condition_number
            ),
            "null_observability_max_condition_layer": (
                self.null_observability_max_condition_layer
            ),
            "online_eval_first_pre_degraded_epoch": self.online_eval_first_pre_degraded_epoch,
            "online_eval_first_post_degraded_epoch": self.online_eval_first_post_degraded_epoch,
            "epoch_geometry_trace": self.epoch_geometry_trace,
            "adapter_saturation_median_ratio": self.adapter_saturation_median_ratio,
            "max_effective_gain_ratio": self.max_effective_gain_ratio,
            "online_eval_baseline_correct": self.online_eval_baseline_correct,
            "online_eval_baseline_total": self.online_eval_baseline_total,
            "online_eval_adapted_correct": self.online_eval_adapted_correct,
            "online_eval_adapted_total": self.online_eval_adapted_total,
            "online_eval_delta_correct": self.online_eval_delta_correct,
            "baseline_max_ngram_repeat": self.baseline_max_ngram_repeat,
            "adapted_max_ngram_repeat": self.adapted_max_ngram_repeat,
            "max_ngram_repeat_delta": self.max_ngram_repeat_delta,
            "ngram_order": self.ngram_order,
            "inference_min_cka": self.inference_min_cka,
            "inference_mean_cka": self.inference_mean_cka,
            "inference_min_cka_layer": self.inference_min_cka_layer,
            "cka_blindness_ratio": self.cka_blindness_ratio,
            "cka_blindness_worst_layer": self.cka_blindness_worst_layer,
            "cka_delta_gap": self.cka_delta_gap,
            "margin_mean_baseline": self.margin_mean_baseline,
            "margin_mean_adapted": self.margin_mean_adapted,
            "margin_mean_delta": self.margin_mean_delta,
            "margin_n_near_zero_baseline": self.margin_n_near_zero_baseline,
            "margin_n_near_zero_adapted": self.margin_n_near_zero_adapted,
            "margin_n_flipped_sign": self.margin_n_flipped_sign,
            "max_logit_delta_inf": self.max_logit_delta_inf,
            "argmax_cert_gap": self.argmax_cert_gap,
            "argmax_preservation_certified": self.argmax_preservation_certified,
            "argmax_n_correct_flipped": self.argmax_n_correct_flipped,
            "moe_router_stability": self.moe_router_stability,
            "structural_passed": self.structural_passed,
            "inference_passed": self.inference_passed,
            "cooccurrence_class": self.cooccurrence_class,
            "passed": self.passed,
            "failure_modes": list(self.failure_modes),
        }


@dataclass(frozen=True)
class DerivedTrainingValidationResult:
    """Aggregate validation result across repeated trials."""

    model_path: str
    dataset_path: str
    eval_dataset_path: str | None
    trials_requested: int
    phase5_inference_enabled: bool
    phase5_probe_count: int | None
    phase5_probe_seed: int | None
    trial_results: tuple[DerivedTrainingTrial, ...]
    counterexamples: tuple[DerivedTrainingTrial, ...]
    pass_count: int
    fail_count: int
    structural_pass_count: int
    structural_fail_count: int
    inference_pass_count: int
    inference_fail_count: int
    all_passed: bool
    seed_source: str
    seeds: tuple[int, ...]
    min_loss_delta: float
    mean_loss_delta: float
    min_perplexity_delta: float
    mean_perplexity_delta: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "dataset_path": self.dataset_path,
            "eval_dataset_path": self.eval_dataset_path,
            "trials_requested": self.trials_requested,
            "phase5_inference_enabled": self.phase5_inference_enabled,
            "phase5_probe_count": self.phase5_probe_count,
            "phase5_probe_seed": self.phase5_probe_seed,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "structural_pass_count": self.structural_pass_count,
            "structural_fail_count": self.structural_fail_count,
            "inference_pass_count": self.inference_pass_count,
            "inference_fail_count": self.inference_fail_count,
            "all_passed": self.all_passed,
            "seed_source": self.seed_source,
            "seeds": list(self.seeds),
            "min_loss_delta": self.min_loss_delta,
            "mean_loss_delta": self.mean_loss_delta,
            "min_perplexity_delta": self.min_perplexity_delta,
            "mean_perplexity_delta": self.mean_perplexity_delta,
            "trial_results": [trial.to_dict() for trial in self.trial_results],
            "counterexamples": [trial.to_dict() for trial in self.counterexamples],
        }


class DerivedTrainingValidationService:
    """Run repeated trials to validate derived training behavior."""

    def __init__(self, dataset_training_service: Any, backend: Any) -> None:
        self._dataset_training_service = dataset_training_service
        self._backend = backend

    def validate(
        self,
        model_path: str | Path,
        dataset_path: str | Path,
        eval_dataset_path: str | Path | None,
        trials: int,
        base_seed: int | None = None,
        seq_length: int | None = None,
        enable_phase5_inference: bool = False,
        phase5_probe_count: int | None = None,
        phase5_probe_seed: int | None = None,
        artifact_root: str | Path | None = None,
    ) -> DerivedTrainingValidationResult:
        if trials <= 0:
            raise ValueError("trials must be positive")

        resolved_model_path = Path(model_path).expanduser().resolve()
        resolved_dataset_path = Path(dataset_path).expanduser().resolve()
        resolved_eval_path = (
            Path(eval_dataset_path).expanduser().resolve()
            if eval_dataset_path is not None
            else None
        )

        if base_seed is None:
            seed_root = self._derive_seed_root(
                model_path=resolved_model_path,
                dataset_path=resolved_dataset_path,
            )
            seed_source = "derived_from_model_dataset_hash"
        else:
            seed_root = int(base_seed)
            seed_source = "user_supplied_base_seed"

        seeds = tuple(seed_root + idx for idx in range(trials))
        phase5_ctx = self._build_phase5_context(
            enabled=enable_phase5_inference,
            probe_count=phase5_probe_count,
            probe_seed=phase5_probe_seed,
            artifact_root=artifact_root,
            model_path=resolved_model_path,
            dataset_path=resolved_dataset_path,
            eval_dataset_path=resolved_eval_path,
            seed_root=seed_root,
        )

        trials_out: list[DerivedTrainingTrial] = []
        loss_deltas: list[float] = []
        ppl_deltas: list[float] = []

        for index, seed in enumerate(seeds):
            trial_output_dir = None
            train_kwargs: dict[str, Any] = {
                "model_path": resolved_model_path,
                "dataset_path": resolved_dataset_path,
                "eval_dataset_path": resolved_eval_path,
                "seed": seed,
                "seq_length": seq_length,
                "auto_regime": True,
                "no_save": (not phase5_ctx.enabled),
            }
            if phase5_ctx.enabled:
                trial_output_dir = phase5_ctx.artifact_root / (
                    f"trial_{index:03d}_seed_{seed}"
                )
                trial_output_dir.mkdir(parents=True, exist_ok=True)
                train_kwargs["output_path"] = trial_output_dir
                train_kwargs["no_save"] = False

            result = self._dataset_training_service.train_from_dataset_research(
                **train_kwargs,
            )

            loss_delta = float(result.baseline_loss - result.post_loss)
            perplexity_delta = float(result.baseline_perplexity - result.post_perplexity)
            loss_deltas.append(loss_delta)
            ppl_deltas.append(perplexity_delta)

            phase5_metrics = None
            if phase5_ctx.enabled:
                phase5_metrics = self._run_phase5_for_trial(
                    context=phase5_ctx,
                    model_path=resolved_model_path,
                    train_result=result,
                    epoch_index=index,
                )

            trial = self._build_trial(
                index=index,
                seed=seed,
                train_result=result,
                loss_delta=loss_delta,
                perplexity_delta=perplexity_delta,
                phase5_metrics=phase5_metrics,
                ngram_order=phase5_ctx.ngram_order if phase5_ctx.enabled else None,
            )
            trials_out.append(trial)

            if (
                phase5_ctx.enabled
                and not phase5_ctx.keep_all_artifacts
                and trial_output_dir is not None
                and trial.passed
                and trial_output_dir.exists()
            ):
                shutil.rmtree(trial_output_dir, ignore_errors=True)

        pass_count = sum(1 for trial in trials_out if trial.passed)
        fail_count = len(trials_out) - pass_count
        structural_pass_count = sum(
            1 for trial in trials_out if trial.structural_passed
        )
        structural_fail_count = len(trials_out) - structural_pass_count
        inference_pass_count = sum(1 for trial in trials_out if trial.inference_passed)
        inference_fail_count = len(trials_out) - inference_pass_count
        counterexamples = tuple(trial for trial in trials_out if not trial.passed)

        return DerivedTrainingValidationResult(
            model_path=str(resolved_model_path),
            dataset_path=str(resolved_dataset_path),
            eval_dataset_path=str(resolved_eval_path) if resolved_eval_path else None,
            trials_requested=trials,
            phase5_inference_enabled=phase5_ctx.enabled,
            phase5_probe_count=phase5_ctx.probe_count if phase5_ctx.enabled else None,
            phase5_probe_seed=phase5_ctx.probe_seed if phase5_ctx.enabled else None,
            trial_results=tuple(trials_out),
            counterexamples=counterexamples,
            pass_count=pass_count,
            fail_count=fail_count,
            structural_pass_count=structural_pass_count,
            structural_fail_count=structural_fail_count,
            inference_pass_count=inference_pass_count,
            inference_fail_count=inference_fail_count,
            all_passed=fail_count == 0,
            seed_source=seed_source,
            seeds=seeds,
            min_loss_delta=min(loss_deltas),
            mean_loss_delta=sum(loss_deltas) / len(loss_deltas),
            min_perplexity_delta=min(ppl_deltas),
            mean_perplexity_delta=sum(ppl_deltas) / len(ppl_deltas),
        )

    def _build_trial(
        self,
        index: int,
        seed: int,
        train_result: Any,
        loss_delta: float,
        perplexity_delta: float,
        phase5_metrics: _Phase5Metrics | None,
        ngram_order: int | None = None,
    ) -> DerivedTrainingTrial:
        eps = machine_epsilon(
            self._backend,
            self._backend.array(
                [
                    train_result.baseline_loss,
                    train_result.post_loss,
                    train_result.baseline_perplexity,
                    train_result.post_perplexity,
                ]
            ),
        )
        loss_improved = loss_delta > eps
        perplexity_improved = perplexity_delta > eps
        bounds_ok = bool(train_result.spectral_bounds_ok)
        sqrt_eps = math.sqrt(eps)

        structural_failure_modes: list[str] = []
        if not loss_improved:
            structural_failure_modes.append("loss_not_improved")
        if not perplexity_improved:
            structural_failure_modes.append("perplexity_not_improved")
        if not bounds_ok:
            structural_failure_modes.append("spectral_bounds_violation")
        if str(train_result.stop_reason).startswith("safety_cap"):
            structural_failure_modes.append("safety_cap_hit")
        min_cka = (
            float(train_result.min_cka)
            if getattr(train_result, "min_cka", None) is not None
            else None
        )
        # CKA shift is a diagnostic signal for co-occurrence tracking,
        # NOT a structural fail criterion.  The structural fail criterion
        # is bound violation: actual CKA below the perturbation-theory
        # lower bound (1-ε)/(1+ε) minus numerical tolerance.
        cka_shift = min_cka is not None and min_cka < 1.0 - sqrt_eps

        per_layer_cka_bound_raw = getattr(train_result, "per_layer_cka_bound", None)
        per_layer_cka_raw = getattr(train_result, "per_layer_cka", None)
        cka_margin_to_bound: float | None = None
        if per_layer_cka_raw and per_layer_cka_bound_raw:
            margins = []
            for layer, actual in per_layer_cka_raw.items():
                bound = per_layer_cka_bound_raw.get(layer)
                if bound is not None:
                    margins.append(float(actual) - float(bound))
            if margins:
                cka_margin_to_bound = min(margins)
                if cka_margin_to_bound < -sqrt_eps:
                    structural_failure_modes.append("cka_bound_violation")

        sat = getattr(train_result, "adapter_saturation_median_ratio", None)
        sat = float(sat) if sat is not None else None
        if sat is not None and sat >= 1.0:
            structural_failure_modes.append("adapter_saturation_exceeded")

        # Bounded-gain stability (Sahraee-Ardakan et al. 2026):
        # gain_ratio = max(eta_step / eta_ceiling) should stay <= 1.0.
        max_gain_ratio = getattr(train_result, "max_effective_gain_ratio", None)
        max_gain_ratio = float(max_gain_ratio) if max_gain_ratio is not None else None
        if max_gain_ratio is not None and max_gain_ratio > 1.0 + sqrt_eps:
            structural_failure_modes.append("gain_divergence")

        inference_failure_modes: list[str] = []
        online_eval_baseline_correct = None
        online_eval_baseline_total = None
        online_eval_adapted_correct = None
        online_eval_adapted_total = None
        online_eval_delta_correct = None
        baseline_max_ngram_repeat = None
        adapted_max_ngram_repeat = None
        max_ngram_repeat_delta = None
        if phase5_metrics is not None:
            online_eval_baseline_correct = int(phase5_metrics.baseline_n_correct)
            online_eval_baseline_total = int(phase5_metrics.baseline_n_total)
            online_eval_adapted_correct = int(phase5_metrics.adapted_n_correct)
            online_eval_adapted_total = int(phase5_metrics.adapted_n_total)
            online_eval_delta_correct = (
                online_eval_adapted_correct - online_eval_baseline_correct
            )
            if (
                phase5_metrics.baseline_max_ngram_repeat is not None
                and phase5_metrics.adapted_max_ngram_repeat is not None
            ):
                baseline_max_ngram_repeat = float(phase5_metrics.baseline_max_ngram_repeat)
                adapted_max_ngram_repeat = float(phase5_metrics.adapted_max_ngram_repeat)
                max_ngram_repeat_delta = (
                    adapted_max_ngram_repeat - baseline_max_ngram_repeat
                )
            if online_eval_adapted_correct < online_eval_baseline_correct:
                inference_failure_modes.append("online_eval_degraded")
            if (
                adapted_max_ngram_repeat is not None
                and baseline_max_ngram_repeat is not None
                and adapted_max_ngram_repeat > baseline_max_ngram_repeat + sqrt_eps
            ):
                inference_failure_modes.append("ngram_degenerated")
            # Argmax preservation certificate gate (fail-closed on explicit False;
            # fail-open when measurement is unavailable / None).
            if phase5_metrics.argmax_preservation_certified is False:
                inference_failure_modes.append("argmax_not_certified")

        # Dual-manifold CKA diagnostics (diagnostic-only, no gating)
        inference_min_cka_val = getattr(train_result, "inference_min_cka", None)
        inference_min_cka_val = (
            float(inference_min_cka_val)
            if inference_min_cka_val is not None
            else None
        )
        inference_mean_cka_val = getattr(train_result, "inference_mean_cka", None)
        inference_mean_cka_val = (
            float(inference_mean_cka_val)
            if inference_mean_cka_val is not None
            else None
        )
        inference_min_cka_layer_val = getattr(
            train_result, "inference_min_cka_layer", None,
        )
        inference_min_cka_layer_val = (
            int(inference_min_cka_layer_val)
            if inference_min_cka_layer_val is not None
            else None
        )

        cka_blindness_ratio: float | None = None
        cka_blindness_worst_layer: int | None = None
        cka_delta_gap: float | None = None
        eval_gram_eps_raw = getattr(train_result, "per_layer_gram_epsilon", None)
        inference_per_layer_gram_epsilon_raw = getattr(
            train_result, "inference_per_layer_gram_epsilon", None,
        )
        if (
            eval_gram_eps_raw is not None
            and inference_per_layer_gram_epsilon_raw is not None
        ):
            ratios: dict[int, float] = {}
            for layer_key, eps_inf in dict(
                inference_per_layer_gram_epsilon_raw
            ).items():
                layer_int = int(layer_key)
                eps_eval = eval_gram_eps_raw.get(layer_key)
                if eps_eval is None:
                    eps_eval = eval_gram_eps_raw.get(layer_int)
                if eps_eval is not None and float(eps_eval) > eps:
                    ratios[layer_int] = float(eps_inf) / float(eps_eval)
            if ratios:
                cka_blindness_worst_layer = max(ratios, key=ratios.get)
                cka_blindness_ratio = ratios[cka_blindness_worst_layer]
        if min_cka is not None and inference_min_cka_val is not None:
            cka_delta_gap = float(min_cka) - inference_min_cka_val

        # Decision-boundary margin diagnostics (diagnostic-only)
        margin_mean_baseline: float | None = None
        margin_mean_adapted: float | None = None
        margin_mean_delta: float | None = None
        margin_n_near_zero_baseline: int | None = None
        margin_n_near_zero_adapted: int | None = None
        margin_n_flipped_sign: int | None = None
        if phase5_metrics is not None:
            bm = phase5_metrics.baseline_margins
            am = phase5_metrics.adapted_margins
            if bm:
                margin_vals_b = list(bm.values())
                margin_mean_baseline = sum(margin_vals_b) / len(margin_vals_b)
                margin_n_near_zero_baseline = sum(
                    1 for m in margin_vals_b if abs(m) < sqrt_eps
                )
            if am:
                margin_vals_a = list(am.values())
                margin_mean_adapted = sum(margin_vals_a) / len(margin_vals_a)
                margin_n_near_zero_adapted = sum(
                    1 for m in margin_vals_a if abs(m) < sqrt_eps
                )
            if bm is not None and am is not None:
                if margin_mean_baseline is not None and margin_mean_adapted is not None:
                    margin_mean_delta = margin_mean_adapted - margin_mean_baseline
                # Count problems where the top-1 margin changed sign
                shared_ids = set(bm.keys()) & set(am.keys())
                margin_n_flipped_sign = sum(
                    1
                    for pid in shared_ids
                    if (bm[pid] > 0 and am[pid] <= 0) or (bm[pid] <= 0 and am[pid] > 0)
                )

        structural_passed = len(structural_failure_modes) == 0
        inference_passed = len(inference_failure_modes) == 0
        failure_modes = tuple(structural_failure_modes + inference_failure_modes)
        passed = len(failure_modes) == 0

        # cka_shift already computed above (bool); co-occurrence tracks
        # whether CKA moved at all, regardless of bound violation.
        inference_degraded = len(inference_failure_modes) > 0
        if cka_shift and inference_degraded:
            cooccurrence_class = "cka_shift_and_inference_degraded"
        elif cka_shift and not inference_degraded:
            cooccurrence_class = "cka_shift_without_inference_degradation"
        elif (not cka_shift) and inference_degraded:
            cooccurrence_class = "inference_degraded_without_cka_shift"
        else:
            cooccurrence_class = "no_shift_no_inference_degradation"

        def _normalize_metric_value(value: Any) -> float | int:
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, int):
                return int(value)
            return float(value)

        per_layer_cka = getattr(train_result, "per_layer_cka", None)
        if per_layer_cka is not None:
            per_layer_cka = {
                int(layer): float(score)
                for layer, score in dict(per_layer_cka).items()
            }
        per_layer_gram_epsilon_raw = getattr(train_result, "per_layer_gram_epsilon", None)
        per_layer_gram_epsilon = None
        if per_layer_gram_epsilon_raw is not None:
            per_layer_gram_epsilon = {
                int(layer): float(val)
                for layer, val in dict(per_layer_gram_epsilon_raw).items()
            }
        per_layer_cka_bound_out = None
        if per_layer_cka_bound_raw is not None:
            per_layer_cka_bound_out = {
                int(layer): float(val)
                for layer, val in dict(per_layer_cka_bound_raw).items()
            }
        per_layer_null_observability_raw = getattr(
            train_result,
            "per_layer_null_observability",
            None,
        )
        per_layer_null_observability = None
        if per_layer_null_observability_raw is not None:
            per_layer_null_observability = {
                int(layer): {
                    str(metric): _normalize_metric_value(metric_val)
                    for metric, metric_val in dict(layer_metrics).items()
                }
                for layer, layer_metrics in dict(per_layer_null_observability_raw).items()
            }
        per_layer_null_accessibility_raw = getattr(
            train_result,
            "per_layer_null_accessibility",
            None,
        )
        per_layer_null_accessibility = None
        if per_layer_null_accessibility_raw is not None:
            per_layer_null_accessibility = {
                int(layer): {
                    str(metric): _normalize_metric_value(metric_val)
                    for metric, metric_val in dict(layer_metrics).items()
                }
                for layer, layer_metrics in dict(per_layer_null_accessibility_raw).items()
            }
        per_module_null_accessibility_raw = getattr(
            train_result,
            "per_module_null_accessibility",
            None,
        )
        per_module_null_accessibility = None
        if per_module_null_accessibility_raw is not None:
            per_module_null_accessibility = {
                str(module): {
                    str(metric): _normalize_metric_value(metric_val)
                    for metric, metric_val in dict(module_metrics).items()
                }
                for module, module_metrics in dict(per_module_null_accessibility_raw).items()
            }

        null_access_min_behavioral_preserved_fraction: float | None = None
        null_access_min_behavioral_preserved_layer: int | None = None
        if per_layer_null_accessibility:
            for layer_idx, metrics in per_layer_null_accessibility.items():
                preserved = metrics.get("behavioral_preserved_fraction")
                if preserved is None:
                    continue
                preserved_val = float(preserved)
                if (
                    null_access_min_behavioral_preserved_fraction is None
                    or preserved_val < null_access_min_behavioral_preserved_fraction
                ):
                    null_access_min_behavioral_preserved_fraction = preserved_val
                    null_access_min_behavioral_preserved_layer = int(layer_idx)

        null_observability_max_condition_number: float | None = None
        null_observability_max_condition_layer: int | None = None
        if per_layer_null_observability:
            for layer_idx, metrics in per_layer_null_observability.items():
                condition_number = metrics.get("condition_number")
                if condition_number is None:
                    continue
                condition_val = float(condition_number)
                if (
                    null_observability_max_condition_number is None
                    or condition_val > null_observability_max_condition_number
                ):
                    null_observability_max_condition_number = condition_val
                    null_observability_max_condition_layer = int(layer_idx)

        epoch_metrics_raw = getattr(train_result, "epoch_metrics", None)
        online_eval_first_pre_degraded_epoch: int | None = None
        online_eval_first_post_degraded_epoch: int | None = None
        epoch_geometry_trace: list[dict[str, Any]] | None = None
        if isinstance(epoch_metrics_raw, list):
            epoch_geometry_trace = []
            for epoch_idx, metric_obj in enumerate(epoch_metrics_raw):
                if not isinstance(metric_obj, dict):
                    continue
                epoch_field = metric_obj.get("epoch")
                if isinstance(epoch_field, (int, float)):
                    epoch_num = int(epoch_field)
                else:
                    epoch_num = int(epoch_idx)

                pre_degraded = metric_obj.get("online_eval_pre_degraded")
                post_degraded = metric_obj.get("online_eval_post_degraded")
                if pre_degraded is True and online_eval_first_pre_degraded_epoch is None:
                    online_eval_first_pre_degraded_epoch = epoch_num
                if post_degraded is True and online_eval_first_post_degraded_epoch is None:
                    online_eval_first_post_degraded_epoch = epoch_num

                epoch_geometry_trace.append(
                    {
                        "epoch": epoch_num,
                        "adapter_saturation_median_ratio": metric_obj.get(
                            "adapter_saturation_median_ratio",
                        ),
                        "dim_final_used_fraction": metric_obj.get("dim_final_used_fraction"),
                        "dim_final_null_fraction": metric_obj.get("dim_final_null_fraction"),
                        "dim_null_recruitment_from_baseline": metric_obj.get(
                            "dim_null_recruitment_from_baseline",
                        ),
                        "online_eval_pre_n_correct": metric_obj.get("online_eval_pre_n_correct"),
                        "online_eval_pre_n_total": metric_obj.get("online_eval_pre_n_total"),
                        "online_eval_pre_degraded": metric_obj.get("online_eval_pre_degraded"),
                        "online_eval_post_n_correct": metric_obj.get("online_eval_post_n_correct"),
                        "online_eval_post_n_total": metric_obj.get("online_eval_post_n_total"),
                        "online_eval_post_degraded": metric_obj.get("online_eval_post_degraded"),
                    }
                )
            if len(epoch_geometry_trace) == 0:
                epoch_geometry_trace = None
        min_cka_layer = getattr(train_result, "min_cka_layer", None)
        if min_cka_layer is None and per_layer_cka:
            min_cka_layer = int(min(per_layer_cka, key=per_layer_cka.get))
        elif min_cka_layer is not None:
            min_cka_layer = int(min_cka_layer)

        mean_cka = (
            float(train_result.mean_cka)
            if getattr(train_result, "mean_cka", None) is not None
            else None
        )

        return DerivedTrainingTrial(
            trial_index=index,
            seed=seed,
            baseline_loss=float(train_result.baseline_loss),
            post_loss=float(train_result.post_loss),
            loss_delta=loss_delta,
            baseline_perplexity=float(train_result.baseline_perplexity),
            post_perplexity=float(train_result.post_perplexity),
            perplexity_delta=perplexity_delta,
            spectral_bounds_ok=bounds_ok,
            max_spectral_ratio=float(train_result.max_spectral_ratio),
            stop_reason=str(train_result.stop_reason),
            train_iters=int(train_result.train_iters),
            training_time_seconds=float(train_result.training_time_seconds),
            min_cka=min_cka,
            mean_cka=mean_cka,
            min_cka_layer=min_cka_layer,
            per_layer_cka=per_layer_cka,
            per_layer_gram_epsilon=per_layer_gram_epsilon,
            per_layer_cka_bound=per_layer_cka_bound_out,
            per_layer_null_observability=per_layer_null_observability,
            per_layer_null_accessibility=per_layer_null_accessibility,
            per_module_null_accessibility=per_module_null_accessibility,
            cka_margin_to_bound=cka_margin_to_bound,
            null_access_min_behavioral_preserved_fraction=(
                null_access_min_behavioral_preserved_fraction
            ),
            null_access_min_behavioral_preserved_layer=(
                null_access_min_behavioral_preserved_layer
            ),
            null_observability_max_condition_number=(
                null_observability_max_condition_number
            ),
            null_observability_max_condition_layer=(
                null_observability_max_condition_layer
            ),
            online_eval_first_pre_degraded_epoch=online_eval_first_pre_degraded_epoch,
            online_eval_first_post_degraded_epoch=online_eval_first_post_degraded_epoch,
            epoch_geometry_trace=epoch_geometry_trace,
            adapter_saturation_median_ratio=sat,
            max_effective_gain_ratio=max_gain_ratio,
            online_eval_baseline_correct=online_eval_baseline_correct,
            online_eval_baseline_total=online_eval_baseline_total,
            online_eval_adapted_correct=online_eval_adapted_correct,
            online_eval_adapted_total=online_eval_adapted_total,
            online_eval_delta_correct=online_eval_delta_correct,
            baseline_max_ngram_repeat=baseline_max_ngram_repeat,
            adapted_max_ngram_repeat=adapted_max_ngram_repeat,
            max_ngram_repeat_delta=max_ngram_repeat_delta,
            ngram_order=ngram_order,
            inference_min_cka=inference_min_cka_val,
            inference_mean_cka=inference_mean_cka_val,
            inference_min_cka_layer=inference_min_cka_layer_val,
            cka_blindness_ratio=cka_blindness_ratio,
            cka_blindness_worst_layer=cka_blindness_worst_layer,
            cka_delta_gap=cka_delta_gap,
            margin_mean_baseline=margin_mean_baseline,
            margin_mean_adapted=margin_mean_adapted,
            margin_mean_delta=margin_mean_delta,
            margin_n_near_zero_baseline=margin_n_near_zero_baseline,
            margin_n_near_zero_adapted=margin_n_near_zero_adapted,
            margin_n_flipped_sign=margin_n_flipped_sign,
            max_logit_delta_inf=(
                phase5_metrics.max_logit_delta_inf
                if phase5_metrics is not None
                else None
            ),
            argmax_cert_gap=(
                phase5_metrics.argmax_cert_gap
                if phase5_metrics is not None
                else None
            ),
            argmax_preservation_certified=(
                phase5_metrics.argmax_preservation_certified
                if phase5_metrics is not None
                else None
            ),
            argmax_n_correct_flipped=(
                phase5_metrics.argmax_n_correct_flipped
                if phase5_metrics is not None
                else None
            ),
            moe_router_stability=(
                float(getattr(train_result, "moe_router_stability", None))
                if getattr(train_result, "moe_router_stability", None) is not None
                else None
            ),
            structural_passed=structural_passed,
            inference_passed=inference_passed,
            cooccurrence_class=cooccurrence_class,
            passed=passed,
            failure_modes=failure_modes,
        )

    def _build_phase5_context(
        self,
        *,
        enabled: bool,
        probe_count: int | None,
        probe_seed: int | None,
        artifact_root: str | Path | None,
        model_path: Path,
        dataset_path: Path,
        eval_dataset_path: Path | None,
        seed_root: int,
    ) -> _Phase5Context:
        if not enabled:
            return _Phase5Context(
                enabled=False,
                probe_count=0,
                probe_seed=0,
                artifact_root=Path("."),
                keep_all_artifacts=False,
                probe_problems=[],
            )

        effective_probe_count = (
            int(probe_count)
            if probe_count is not None
            else self._derive_phase5_probe_count()
        )
        if effective_probe_count <= 0:
            raise ValueError("phase5 probe count must be positive")

        effective_probe_seed = (
            int(probe_seed)
            if probe_seed is not None
            else self._derive_phase5_probe_seed(
                model_path=model_path,
                dataset_path=dataset_path,
                eval_dataset_path=eval_dataset_path,
            )
        )

        if artifact_root is not None:
            resolved_artifact_root = Path(artifact_root).expanduser().resolve()
            keep_all_artifacts = True
        else:
            resolved_artifact_root = self._derive_phase5_artifact_root(
                model_path=model_path,
                dataset_path=dataset_path,
                eval_dataset_path=eval_dataset_path,
                seed_root=seed_root,
            )
            keep_all_artifacts = False
        resolved_artifact_root.mkdir(parents=True, exist_ok=True)

        return _Phase5Context(
            enabled=True,
            probe_count=effective_probe_count,
            probe_seed=effective_probe_seed,
            artifact_root=resolved_artifact_root,
            keep_all_artifacts=keep_all_artifacts,
            probe_problems=create_eval_problem_set(
                n_problems=effective_probe_count,
                seed=effective_probe_seed,
            ),
        )

    def _run_phase5_for_trial(
        self,
        *,
        context: _Phase5Context,
        model_path: Path,
        train_result: Any,
        epoch_index: int,
    ) -> _Phase5Metrics:
        seq_length_used = getattr(train_result, "seq_length_used", None)
        if seq_length_used is None:
            raise ValueError(
                "Phase 5 inference requires seq_length_used in training result.",
            )
        max_tokens = int(seq_length_used)
        if context.max_tokens is None:
            context.max_tokens = max_tokens
        elif context.max_tokens != max_tokens:
            raise ValueError(
                "Phase 5 requires a fixed max token budget across trials; "
                f"observed {max_tokens} vs baseline {context.max_tokens}.",
            )

        if context.baseline_eval is None or context.baseline_max_ngram_repeat is None:
            (
                baseline_eval,
                baseline_max_ngram,
                baseline_margins,
                _,
                baseline_logits,
                probe_ngram_order,
            ) = self._run_probe_eval(
                model_path=model_path,
                adapter_path=None,
                problems=context.probe_problems,
                max_tokens=context.max_tokens,
                baseline_correct_ids=None,
                epoch=epoch_index,
                collect_logits_for_delta=True,
            )
            context.baseline_eval = baseline_eval
            context.baseline_max_ngram_repeat = baseline_max_ngram
            context.baseline_margins = baseline_margins
            context.baseline_logits = baseline_logits
            context.ngram_order = probe_ngram_order

        adapter_path = getattr(train_result, "adapter_path", None)
        if not adapter_path:
            raise ValueError(
                "Phase 5 inference requires a saved adapter_path; "
                "train run returned no adapter artifact.",
            )

        (
            adapted_eval,
            adapted_max_ngram,
            adapted_margins,
            max_logit_delta_inf,
            _,
            _adapted_ngram_order,
        ) = self._run_probe_eval(
            model_path=model_path,
            adapter_path=Path(adapter_path),
            problems=context.probe_problems,
            max_tokens=context.max_tokens,
            baseline_correct_ids=context.baseline_eval.correct_ids,
            epoch=epoch_index,
            baseline_logits=context.baseline_logits,
        )

        # Argmax preservation certificate: did the adapted model preserve
        # the argmax on every baseline-correct probe?  Direct measurement —
        # check whether adapted_margin > 0 for each correct probe.
        argmax_cert_gap: float | None = None
        argmax_preservation_certified: bool | None = None
        argmax_n_correct_flipped: int | None = None
        if (
            adapted_margins
            and context.baseline_eval is not None
            and context.baseline_eval.correct_ids
        ):
            correct_ids = context.baseline_eval.correct_ids
            correct_adapted_margins = [
                float(adapted_margins[pid])
                for pid in correct_ids
                if pid in adapted_margins
            ]
            if correct_adapted_margins:
                argmax_cert_gap = min(correct_adapted_margins)
                argmax_n_correct_flipped = sum(
                    1 for m in correct_adapted_margins if m <= 0.0
                )
                argmax_preservation_certified = argmax_n_correct_flipped == 0

        return _Phase5Metrics(
            baseline_n_correct=context.baseline_eval.n_correct,
            baseline_n_total=context.baseline_eval.n_total,
            adapted_n_correct=adapted_eval.n_correct,
            adapted_n_total=adapted_eval.n_total,
            baseline_max_ngram_repeat=context.baseline_max_ngram_repeat,
            adapted_max_ngram_repeat=adapted_max_ngram,
            baseline_margins=context.baseline_margins,
            adapted_margins=adapted_margins,
            max_logit_delta_inf=max_logit_delta_inf,
            argmax_cert_gap=argmax_cert_gap,
            argmax_preservation_certified=argmax_preservation_certified,
            argmax_n_correct_flipped=argmax_n_correct_flipped,
        )

    def _run_probe_eval(
        self,
        *,
        model_path: Path,
        adapter_path: Path | None,
        problems: list[StarProblem],
        max_tokens: int,
        baseline_correct_ids: frozenset[str] | None,
        epoch: int,
        baseline_logits: dict[str, Any] | None = None,
        collect_logits_for_delta: bool = False,
    ) -> tuple[OnlineEvalResult, float | None, dict[str, float], float | None, dict[str, Any] | None, int | None]:
        """Run probe evaluation and margin measurement.

        Returns (eval_result, max_ngram_repeat, margins, max_logit_delta_inf,
        collected_logits, ngram_order).
        max_ngram_repeat is None when readout erank derivation fails (fail-closed).

        max_logit_delta_inf is only computed when baseline_logits is provided
        (adapted run), allowing per-probe ||Δlogits||_∞ measurement.
        collected_logits is populated when collect_logits_for_delta=True.
        """
        from modelcypher.core.domain.geometry.perturbation_bound import (
            compute_readout_effective_rank,
        )
        from modelcypher.core.domain.training.degeneration import (
            derive_ngram_order,
            ngram_repetition_rate,
        )

        model, tokenizer = self._backend.load_model(
            str(model_path),
            str(adapter_path) if adapter_path is not None else None,
        )

        # Derive n-gram order from readout geometry (fail-closed: 0.0 on failure).
        ngram_order: int | None = None
        try:
            readout_erank = compute_readout_effective_rank(model, self._backend)
            ngram_order = derive_ngram_order(readout_erank, generation_length=max_tokens)
        except Exception:
            logger.debug("Readout erank unavailable in probe eval", exc_info=True)

        def _generate(prompt: str, max_toks: int) -> str:
            return self._backend.generate(model, tokenizer, prompt, max_tokens=max_toks)

        eval_result = evaluate_correctness(
            problems=problems,
            generate_fn=_generate,
            epoch=epoch,
            baseline_correct_ids=baseline_correct_ids,
            max_tokens=max_tokens,
        )

        prompts = self._build_probe_prompts(problems)
        responses = [_generate(prompt, max_tokens) for prompt in prompts]
        if ngram_order is not None:
            max_repetition: float | None = max(
                (ngram_repetition_rate(text, ngram_order) for text in responses),
                default=0.0,
            )
        else:
            max_repetition = None

        # Decision-boundary margin diagnostics
        from modelcypher.core.domain.training.online_eval import compute_answer_margin

        def _collect_logits(prompt: str):
            return self._backend.collect_logits(model, tokenizer, prompt)

        margins = compute_answer_margin(problems, _collect_logits, self._backend)

        # Collect per-probe logit vectors for delta measurement in adapted run
        collected_logits: dict[str, Any] | None = None
        if collect_logits_for_delta:
            collected_logits = {}
            for problem in problems:
                pid = problem.problem_id
                prompt = self._build_probe_prompts([problem])[0]
                try:
                    logits = self._backend.collect_logits(
                        model, tokenizer, prompt,
                    )
                    self._backend.eval(logits)
                    collected_logits[pid] = logits
                except Exception:
                    logger.debug(
                        "Logit collection failed for %s", pid,
                        exc_info=True,
                    )

        # Measured logit perturbation: ||Δlogits||_∞ over baseline-correct probes.
        # Only baseline-correct probes matter for preservation: probes the
        # baseline already got wrong have margin ≈ 0 and SHOULD see large
        # perturbation (that's training working).
        max_logit_delta_inf: float | None = None
        if baseline_logits is not None:
            max_delta = 0.0
            for problem in problems:
                pid = problem.problem_id
                if pid not in baseline_logits:
                    continue
                if baseline_correct_ids is not None and pid not in baseline_correct_ids:
                    continue
                prompt = self._build_probe_prompts([problem])[0]
                try:
                    adapted_logits = self._backend.collect_logits(
                        model, tokenizer, prompt,
                    )
                    delta = adapted_logits - baseline_logits[pid]
                    self._backend.eval(delta)
                    abs_delta = self._backend.abs(delta)
                    self._backend.eval(abs_delta)
                    delta_inf = float(
                        self._backend.to_scalar(self._backend.max(abs_delta))
                    )
                    max_delta = max(max_delta, delta_inf)
                except Exception:
                    logger.debug(
                        "Logit delta measurement failed for %s", pid,
                        exc_info=True,
                    )
            max_logit_delta_inf = max_delta

        del model
        del tokenizer
        gc.collect()
        self._clear_backend_cache()

        return eval_result, max_repetition, margins, max_logit_delta_inf, collected_logits, ngram_order

    @staticmethod
    def _build_probe_prompts(problems: list[StarProblem]) -> list[str]:
        from modelcypher.core.domain.star.prompting import (
            build_forward_prompt,
            default_few_shot_examples,
        )

        n_demonstrations = len(default_few_shot_examples())
        return [
            build_forward_prompt(problem, demonstrations=n_demonstrations)
            for problem in problems
        ]

    def _derive_phase5_probe_count(self) -> int:
        derive_fn = getattr(self._dataset_training_service, "_derive_regime_n_from_ci", None)
        if not callable(derive_fn):
            raise ValueError(
                "Phase 5 probe count derivation requires dataset training service "
                "method _derive_regime_n_from_ci().",
            )
        ci_n = int(derive_fn())
        n_types = len(StarProblemGenerator.PROBLEM_TYPES)
        return ci_n * n_types

    @staticmethod
    def _derive_phase5_probe_seed(
        *,
        model_path: Path,
        dataset_path: Path,
        eval_dataset_path: Path | None,
    ) -> int:
        payload = (
            "phase5_probe_seed:"
            f"{model_path}:{dataset_path}:{eval_dataset_path}"
        )
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big")

    @staticmethod
    def _derive_phase5_artifact_root(
        *,
        model_path: Path,
        dataset_path: Path,
        eval_dataset_path: Path | None,
        seed_root: int,
    ) -> Path:
        payload = (
            "phase5_artifact_root:"
            f"{model_path}:{dataset_path}:{eval_dataset_path}:{seed_root}"
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return (
            Path("results")
            / "derived_training_validation_artifacts"
            / digest
        ).resolve()

    def _clear_backend_cache(self) -> None:
        clear_fn = getattr(self._backend, "clear_cache", None)
        if callable(clear_fn):
            try:
                clear_fn()
            except Exception:
                pass

    @staticmethod
    def _derive_seed_root(model_path: Path, dataset_path: Path) -> int:
        payload = f"{model_path}:{dataset_path}"
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big")
