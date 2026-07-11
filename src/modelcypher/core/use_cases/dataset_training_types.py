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

"""Serializable dataset-training result and plan types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.training.identity import (
    GEOMETRIC_LORA_CONTROLLER,
    GEOMETRIC_LORA_INIT_METHOD,
    GEOMETRIC_LORA_METHOD,
    GEOMETRIC_LORA_OPTIMIZER,
    GEOMETRIC_LORA_STOPPING,
    resolve_geometric_lora_optimizer_name,
)
from modelcypher.core.domain.training.mass_step_size import (
    OPTIMIZER_MODE_ADAMW_GEOMETRIC,
)

if TYPE_CHECKING:
    from modelcypher.core.domain.moe.expert_selection import ExpertTargetSelection

@dataclass
class DatasetTrainResult:
    """Result of dataset-driven geometric LoRA training."""

    train_iters: int
    initial_loss: float
    final_loss: float
    stop_reason: str
    baseline_loss: float
    baseline_perplexity: float
    post_loss: float
    post_perplexity: float
    n_lora_layers: int
    n_trainable_params: int
    adapter_path: str | None
    spectral_bounds_ok: bool
    max_spectral_ratio: float
    training_time_seconds: float
    target_module_count: int | None = None
    target_modules: list[str] | None = None
    rank_overrides: dict[str, int] | None = None
    rank_ceiling_source: str | None = None
    sigma_k_min: float | None = None
    sigma_max: float | None = None
    resolved_batch_size: int | None = None
    epoch_metrics: list[dict[str, Any]] | None = None
    # G4: Capability preservation (CKA alignment to base model)
    min_cka: float | None = None
    mean_cka: float | None = None
    per_layer_cka: dict[int, float] | None = None
    per_layer_gram_epsilon: dict[int, float] | None = None
    per_layer_cka_bound: dict[int, float] | None = None
    per_layer_null_observability: dict[int, dict[str, float | int]] | None = None
    per_layer_null_accessibility: dict[int, dict[str, float | int]] | None = None
    per_module_null_accessibility: dict[str, dict[str, float | int]] | None = None
    min_cka_layer: int | None = None
    # G3: Weyl adapter saturation monitoring (not model-space capacity)
    adapter_saturation_median_ratio: float | None = None
    # Effective sequence length used by training/eval (derived from data unless overridden).
    seq_length_used: int | None = None
    # Model-space dimensional recruitment (null-space utilization over training)
    dim_final_used_fraction: float | None = None
    dim_final_null_fraction: float | None = None
    dim_null_recruitment_from_baseline: float | None = None
    # Canonical training identity for the shipped geometry-derived LoRA path.
    method: str = GEOMETRIC_LORA_METHOD
    init_method: str = GEOMETRIC_LORA_INIT_METHOD
    optimizer: str = GEOMETRIC_LORA_OPTIMIZER
    controller: str = GEOMETRIC_LORA_CONTROLLER
    stopping: str = GEOMETRIC_LORA_STOPPING
    # Outer similarity (final epoch, when rss_monitor=True)
    rss_final_cosine: float | None = None
    rss_final_spearman: float | None = None
    rss_final_top1: float | None = None
    quantization_frontier_precheck: dict[str, Any] | None = None
    # Derived validation split diagnostics (when eval split is auto-derived)
    validation_split: dict[str, Any] | None = None
    # Number of retention samples auto-collected from training prompts.
    auto_retention_samples_collected: int = 0
    # Bounded-gain stability certificate (Sahraee-Ardakan et al. 2026)
    max_effective_gain_ratio: float | None = None
    # Inference-manifold CKA (diagnostic): CKA on StarProblem prompts, not eval samples
    inference_min_cka: float | None = None
    inference_mean_cka: float | None = None
    inference_per_layer_cka: dict[int, float] | None = None
    inference_per_layer_gram_epsilon: dict[int, float] | None = None
    inference_min_cka_layer: int | None = None
    # RMT signal-rank ceiling diagnostics (per-layer intrinsic signal dimensionality)
    per_layer_signal_ranks: dict[int, dict[str, float | int]] | None = None
    # G4: Mode connectivity (barrier between base and adapted activation spaces)
    mode_connectivity_barrier: float | None = None
    mode_connectivity_normalized_barrier: float | None = None
    mode_connectivity_method: str | None = None
    # G4: Degeneration gate — n-gram order derived from readout effective rank
    degeneration_max_ngram_repeat: float | None = None
    degeneration_mean_ngram_repeat: float | None = None
    degeneration_ngram_order: int | None = None
    # Standard benchmark eval (pre/post training, optional via --benchmark)
    benchmark_baseline: dict[str, float] | None = None
    benchmark_post: dict[str, float] | None = None
    # MoE expert-targeting diagnostics (optional)
    moe_targets: "ExpertTargetSelection | None" = None
    moe_saturated_during_training: list[str] | None = None
    moe_router_stability: float | None = None
    # Pipeline promotability gate (shared with derived validation)
    pipeline_gate_operator: str | None = None
    pipeline_gate_passed: bool | None = None
    pipeline_gate_failure_modes: list[str] | None = None
    pipeline_gate_checks: dict[str, Any] | None = None
    # Adapter provenance: which CE variant produced this adapter
    training_objective: str = "ce"
    controller_mode: str | None = None
    optimizer_research_mode: str | None = None
    controller_trace: list[dict[str, Any]] | None = None
    offline_replay: dict[str, Any] | None = None
    closed_loop_control_law: dict[str, Any] | None = None
    derived_plan: dict[str, Any] | None = None
    artifacts: dict[str, str] | None = None
    capability_manifest: dict[str, Any] | None = None
    runtime_status: dict[str, Any] | None = None
    next_actions: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a JSON-serializable dictionary."""
        result = {
            "train_iters": self.train_iters,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "stop_reason": self.stop_reason,
            "baseline_loss": self.baseline_loss,
            "baseline_perplexity": self.baseline_perplexity,
            "post_loss": self.post_loss,
            "post_perplexity": self.post_perplexity,
            "n_lora_layers": self.n_lora_layers,
            "n_trainable_params": self.n_trainable_params,
            "adapter_path": self.adapter_path,
            "target_module_count": self.target_module_count,
            "target_modules": self.target_modules,
            "rank_overrides": self.rank_overrides,
            "rank_ceiling_source": self.rank_ceiling_source,
            "sigma_k_min": self.sigma_k_min,
            "sigma_max": self.sigma_max,
            "resolved_batch_size": self.resolved_batch_size,
            "spectral_bounds_ok": self.spectral_bounds_ok,
            "max_spectral_ratio": self.max_spectral_ratio,
            "training_time_seconds": self.training_time_seconds,
            "method": self.method,
            "init_method": self.init_method,
            "optimizer": self.optimizer,
            "controller": self.controller,
            "stopping": self.stopping,
            "auto_retention_samples_collected": self.auto_retention_samples_collected,
        }
        if self.epoch_metrics is not None:
            result["epoch_metrics"] = self.epoch_metrics
        if self.min_cka is not None:
            result["min_cka"] = self.min_cka
        if self.mean_cka is not None:
            result["mean_cka"] = self.mean_cka
        if self.per_layer_cka is not None:
            result["per_layer_cka"] = self.per_layer_cka
        if self.per_layer_gram_epsilon is not None:
            result["per_layer_gram_epsilon"] = self.per_layer_gram_epsilon
        if self.per_layer_cka_bound is not None:
            result["per_layer_cka_bound"] = self.per_layer_cka_bound
        if self.per_layer_null_observability is not None:
            result["per_layer_null_observability"] = self.per_layer_null_observability
        if self.per_layer_null_accessibility is not None:
            result["per_layer_null_accessibility"] = self.per_layer_null_accessibility
        if self.per_module_null_accessibility is not None:
            result["per_module_null_accessibility"] = self.per_module_null_accessibility
        if self.min_cka_layer is not None:
            result["min_cka_layer"] = self.min_cka_layer
        if self.adapter_saturation_median_ratio is not None:
            result["adapter_saturation_median_ratio"] = self.adapter_saturation_median_ratio
        if self.seq_length_used is not None:
            result["seq_length_used"] = self.seq_length_used
        if self.dim_final_used_fraction is not None:
            result["dim_final_used_fraction"] = self.dim_final_used_fraction
        if self.dim_final_null_fraction is not None:
            result["dim_final_null_fraction"] = self.dim_final_null_fraction
        if self.dim_null_recruitment_from_baseline is not None:
            result["dim_null_recruitment_from_baseline"] = self.dim_null_recruitment_from_baseline
        if self.rss_final_cosine is not None:
            result["rss_final_cosine"] = self.rss_final_cosine
        if self.rss_final_spearman is not None:
            result["rss_final_spearman"] = self.rss_final_spearman
        if self.rss_final_top1 is not None:
            result["rss_final_top1"] = self.rss_final_top1
        if self.quantization_frontier_precheck is not None:
            result["quantization_frontier_precheck"] = self.quantization_frontier_precheck
        if self.validation_split is not None:
            result["validation_split"] = self.validation_split
        if self.max_effective_gain_ratio is not None:
            result["max_effective_gain_ratio"] = self.max_effective_gain_ratio
        if self.inference_min_cka is not None:
            result["inference_min_cka"] = self.inference_min_cka
        if self.inference_mean_cka is not None:
            result["inference_mean_cka"] = self.inference_mean_cka
        if self.inference_per_layer_cka is not None:
            result["inference_per_layer_cka"] = self.inference_per_layer_cka
        if self.inference_per_layer_gram_epsilon is not None:
            result["inference_per_layer_gram_epsilon"] = self.inference_per_layer_gram_epsilon
        if self.inference_min_cka_layer is not None:
            result["inference_min_cka_layer"] = self.inference_min_cka_layer
        if self.per_layer_signal_ranks is not None:
            result["per_layer_signal_ranks"] = self.per_layer_signal_ranks
        if self.mode_connectivity_barrier is not None:
            result["mode_connectivity_barrier"] = self.mode_connectivity_barrier
        if self.mode_connectivity_normalized_barrier is not None:
            result["mode_connectivity_normalized_barrier"] = self.mode_connectivity_normalized_barrier
        if self.mode_connectivity_method is not None:
            result["mode_connectivity_method"] = self.mode_connectivity_method
        if self.degeneration_max_ngram_repeat is not None:
            result["degeneration_max_ngram_repeat"] = self.degeneration_max_ngram_repeat
        if self.degeneration_mean_ngram_repeat is not None:
            result["degeneration_mean_ngram_repeat"] = self.degeneration_mean_ngram_repeat
        if self.degeneration_ngram_order is not None:
            result["degeneration_ngram_order"] = self.degeneration_ngram_order
        if self.benchmark_baseline is not None:
            result["benchmark_baseline"] = self.benchmark_baseline
        if self.benchmark_post is not None:
            result["benchmark_post"] = self.benchmark_post
        if self.moe_targets is not None:
            result["moe_targets"] = {
                "n_targets": self.moe_targets.n_trainable_experts,
                "target_module_keys": self.moe_targets.target_module_keys,
                "estimated_params": self.moe_targets.estimated_params,
                "saturated": list(self.moe_targets.saturated),
                "skipped": list(self.moe_targets.skipped),
            }
        if self.moe_saturated_during_training is not None:
            result["moe_saturated_during_training"] = self.moe_saturated_during_training
        if self.moe_router_stability is not None:
            result["moe_router_stability"] = self.moe_router_stability
        if self.pipeline_gate_operator is not None:
            result["pipeline_gate_operator"] = self.pipeline_gate_operator
        if self.pipeline_gate_passed is not None:
            result["pipeline_gate_passed"] = self.pipeline_gate_passed
        if self.pipeline_gate_failure_modes is not None:
            result["pipeline_gate_failure_modes"] = self.pipeline_gate_failure_modes
        if self.pipeline_gate_checks is not None:
            result["pipeline_gate_checks"] = self.pipeline_gate_checks
        if self.controller_mode is not None:
            result["controller_mode"] = self.controller_mode
        if self.optimizer_research_mode is not None:
            result["optimizer_research_mode"] = self.optimizer_research_mode
        if self.controller_trace is not None:
            result["controller_trace"] = self.controller_trace
        if self.offline_replay is not None:
            result["offline_replay"] = self.offline_replay
        if self.closed_loop_control_law is not None:
            result["closed_loop_control_law"] = self.closed_loop_control_law
        if self.derived_plan is not None:
            result["derived_plan"] = self.derived_plan
        if self.artifacts is not None:
            result["artifacts"] = self.artifacts
        if self.capability_manifest is not None:
            result["capability_manifest"] = self.capability_manifest
        if self.runtime_status is not None:
            result["runtime_status"] = self.runtime_status
        if self.next_actions is not None:
            result["next_actions"] = self.next_actions
        if self.benchmark_baseline is not None and self.benchmark_post is not None:
            result["benchmark_delta"] = {
                k: self.benchmark_post[k] - self.benchmark_baseline.get(k, 0.0)
                for k in self.benchmark_post
            }
        return result


@dataclass
class NBTargetSurface:
    """Resolved geometric LoRA adaptation surface.

    Captures the exact modules, per-module ranks, and spectral bounds
    from the production geometry pipeline.  Reusable by any method that
    wants to train on the same surface for controlled comparison.
    """

    target_keys: list[str]
    rank_overrides: dict[str, int]
    rank_ceiling_source: str
    sigma_k_min: float
    sigma_max: float
    geometry_info: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_keys": self.target_keys,
            "rank_overrides": self.rank_overrides,
            "rank_ceiling_source": self.rank_ceiling_source,
            "sigma_k_min": self.sigma_k_min,
            "sigma_max": self.sigma_max,
            "n_modules": len(self.target_keys),
            "rank_range": [
                min(self.rank_overrides.values()),
                max(self.rank_overrides.values()),
            ],
        }


@dataclass
class DerivedTrainingPlan:
    """Resolved pre-training plan for the canonical geometric LoRA path.

    The plan carries both the user-facing derivation summary and the internal
    resolved artifacts needed to run training without re-deriving the target
    surface.
    """

    model_path: Path
    dataset_path: Path
    eval_dataset_path: Path | None
    output_path: Path | None
    seed: int
    seed_source: str
    seq_length: int
    seq_length_source: str
    validation_split: dict[str, Any]
    train_samples: list[dict[str, Any]]
    eval_samples: list[dict[str, Any]]
    geometries: dict[str, Any]
    target_modules: list[str]
    rank_overrides: dict[str, int]
    rank_ceiling_source: str
    signal_rank_results: dict[int, Any]
    rank_capacity_sample_count: int
    sigma_k_min: float
    sigma_max: float
    estimated_trainable_params: int
    optimizer_geometry_config: Any
    quantization_frontier_precheck: dict[str, Any] | None
    controller_mode: str
    optimizer_research_mode: str
    controller_law: dict[str, Any] | None = None

    def _module_geometry_payload(self) -> dict[str, dict[str, Any]]:
        payload: dict[str, dict[str, Any]] = {}
        for module in sorted(self.target_modules):
            geom = self.geometries[module]
            payload[module] = {
                "shape": [int(geom.shape[0]), int(geom.shape[1])],
                "sigma_max": float(geom.sigma_max),
                "sigma_k": float(geom.sigma_k),
                "effective_rank": int(geom.effective_rank),
                "full_rank": int(geom.full_rank),
                "tail_dims": int(geom.tail_dims),
                "shannon_effective_rank": float(geom.shannon_effective_rank),
                "spectral_gap": float(geom.spectral_gap),
                "recommended_rank": int(getattr(geom, "recommended_rank", 1)),
            }
        return payload

    def _signal_rank_payload(self) -> dict[int, dict[str, Any]] | None:
        if not self.signal_rank_results:
            return None
        return {
            int(layer_idx): {
                "signal_rank": int(result.signal_rank),
                "effective_rank": float(getattr(result, "effective_rank", 0.0)),
                "noise_rank": int(result.noise_rank),
                "mp_upper_edge": float(result.mp_upper_edge),
                "signal_variance_fraction": float(result.signal_variance_fraction),
            }
            for layer_idx, result in sorted(self.signal_rank_results.items())
        }

    def to_dict(self) -> dict[str, Any]:
        rank_values = list(self.rank_overrides.values())
        rank_range = [
            int(min(rank_values)) if rank_values else 0,
            int(max(rank_values)) if rank_values else 0,
        ]
        if self.optimizer_research_mode == OPTIMIZER_MODE_ADAMW_GEOMETRIC:
            learning_rate_policy = (
                "Canonical AdamW: fixed lr=2e-4 with cosine decay over 6 "
                "data-epochs, calibrated from the R1 frozen-tuple winner."
            )
            batch_size_policy = (
                "Logical batch size is derived from gradient noise after LoRA "
                "injection, then reduced to a memory-safe micro-batch only if "
                "gradient accumulation is required."
            )
        else:
            learning_rate_policy = (
                "No fixed scalar LR is derived upfront. MASS chooses eta_step "
                "= min(eta_ceiling, eta_sps, eta_weyl) online."
            )
            batch_size_policy = (
                "Batch size is derived during training from gradient noise "
                "scale after LoRA injection."
            )
        return {
            "inputs": {
                "model_path": str(self.model_path),
                "dataset_path": str(self.dataset_path),
                "eval_dataset_path": (
                    str(self.eval_dataset_path) if self.eval_dataset_path else None
                ),
                "seed": int(self.seed),
                "seed_source": self.seed_source,
                "output_path": str(self.output_path) if self.output_path else None,
            },
            "data_plan": {
                "seq_length": int(self.seq_length),
                "seq_length_source": self.seq_length_source,
                "split_method": self.validation_split.get("method"),
                "n_train": int(len(self.train_samples)),
                "rank_capacity_sample_count": int(self.rank_capacity_sample_count),
                "n_eval": int(len(self.eval_samples)),
                "validation_split": self.validation_split,
            },
            "adaptation_surface": {
                "target_module_count": int(len(self.target_modules)),
                "target_modules": list(sorted(self.target_modules)),
                "per_module_ranks": {
                    module: int(self.rank_overrides[module])
                    for module in sorted(self.rank_overrides)
                },
                "rank_range": rank_range,
                "rank_ceiling_source": self.rank_ceiling_source,
                "estimated_trainable_params": int(self.estimated_trainable_params),
                "sigma_k_min": float(self.sigma_k_min),
                "sigma_max": float(self.sigma_max),
                "module_geometry": self._module_geometry_payload(),
                "signal_rank_summary": self._signal_rank_payload(),
            },
            "controller_plan": {
                "method": GEOMETRIC_LORA_METHOD,
                "init_method": GEOMETRIC_LORA_INIT_METHOD,
                "optimizer": (
                    resolve_geometric_lora_optimizer_name(
                        self.optimizer_research_mode,
                    )
                ),
                "controller": GEOMETRIC_LORA_CONTROLLER,
                "stopping": GEOMETRIC_LORA_STOPPING,
                "controller_mode": self.controller_mode,
                "optimizer_research_mode": self.optimizer_research_mode,
                "closed_loop_control_law": self.controller_law,
                "learning_rate_policy": learning_rate_policy,
                "batch_size_policy": batch_size_policy,
            },
            "derived_now": {
                "seed": int(self.seed),
                "resolved_output_path": (
                    str(self.output_path) if self.output_path is not None else None
                ),
                "sequence_length": int(self.seq_length),
                "validation_split": dict(self.validation_split),
                "target_surface": {
                    "target_module_count": int(len(self.target_modules)),
                    "rank_range": rank_range,
                    "rank_ceiling_source": self.rank_ceiling_source,
                },
                "optimizer_geometry_config": {
                    "n_layers": int(getattr(self.optimizer_geometry_config, "n_layers", 0)),
                    "base_lr": float(getattr(self.optimizer_geometry_config, "base_lr", 0.0)),
                },
                "quantization_frontier_precheck": self.quantization_frontier_precheck,
            },
            "measured_during_training": {
                "controller_terms": [
                    "eta_ceiling",
                    "eta_sps",
                    "eta_weyl",
                    "eta_step",
                    "effective_gain_ratio",
                ],
                "runtime_signals": [
                    "gradient_noise_scale",
                    "behavioral_transport_norm",
                    "spectral_budget_ratio",
                    "remaining_budget",
                    "margin_mean_delta",
                    "cka_blindness_ratio",
                    "null_accessibility",
                ],
                "stopping_signals": [
                    "loss_stable",
                    "adapter_saturation_exhausted",
                    "certificate",
                ],
            },
            "verified_after_training": {
                "post_training_gates": [
                    "spectral_bounds_ok",
                    "min_cka",
                    "mean_cka",
                    "degeneration_max_ngram_repeat",
                    "mode_connectivity_barrier",
                    "pipeline_gate_passed",
                ],
                "optional_outputs": [
                    "benchmark_delta (when --benchmark is enabled)",
                ],
            },
            "removed_user_knobs": [
                "learning_rate",
                "warmup",
                "lr_schedule",
                "gradient_clipping",
                "manual_rank",
                "manual_target_module_selection",
                "patience_early_stopping",
                "manual_dropout_default",
            ],
        }

    def to_text_summary(self) -> str:
        rank_values = list(self.rank_overrides.values())
        rank_min = min(rank_values) if rank_values else 0
        rank_max = max(rank_values) if rank_values else 0
        split_method = self.validation_split.get("method", "unknown")
        if self.optimizer_research_mode == OPTIMIZER_MODE_ADAMW_GEOMETRIC:
            controller_summary = (
                "Controller: canonical AdamW with fixed lr=2e-4 and cosine "
                "decay over 6 data-epochs"
            )
        else:
            controller_summary = (
                "Controller: no fixed scalar LR; MASS will choose "
                "eta_step = min(eta_ceiling, eta_sps, eta_weyl) online"
            )
        lines = [
            "Resolved training plan",
            f"Model: {self.model_path}",
            f"Dataset: {self.dataset_path}",
            (
                f"Eval: {self.eval_dataset_path}"
                if self.eval_dataset_path is not None
                else f"Eval: derived split ({split_method})"
            ),
            f"Seed: {self.seed} ({self.seed_source})",
            (
                f"Output: {self.output_path}"
                if self.output_path is not None
                else "Output: no adapter will be saved"
            ),
            f"Seq length: {self.seq_length} ({self.seq_length_source})",
            (
                f"Split: {split_method} | train={len(self.train_samples)} "
                f"eval={len(self.eval_samples)}"
            ),
            (
                f"Target surface: {len(self.target_modules)} modules | "
                f"ranks={rank_min}-{rank_max} | "
                f"params~{self.estimated_trainable_params:,}"
            ),
            (
                f"Spectral bounds: sigma_k_min={self.sigma_k_min:.4e} | "
                f"sigma_max={self.sigma_max:.4e} | ceiling={self.rank_ceiling_source}"
            ),
            controller_summary,
            (
                "Closed-loop law: "
                + str(self.controller_law.get("law_id"))
                if self.controller_law is not None
                else "Closed-loop law: none"
            ),
            (
                "Measured during training: eta_sps, eta_weyl, eta_step, "
                "gradient-noise batch size, stopping certificate, preservation telemetry"
            ),
            (
                "Verified after training: spectral bounds, CKA, degeneration, "
                "pipeline gate, optional benchmark delta"
            ),
            "Benchmark: opt-in only; add --benchmark quick for pre/post task scores",
        ]
        return "\n".join(lines)


__all__ = ["DatasetTrainResult", "DerivedTrainingPlan", "NBTargetSurface"]
