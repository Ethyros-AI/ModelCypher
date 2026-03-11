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

"""Dataset-driven geometric LoRA training via NB-LoRA.

One training method. Geometry derives everything:
- Which layers to target (tail_dims > 0)
- Rank per layer (min(tail_dims, n_train_samples))
- Scale bound (sigma_k / 2 * (1 - sqrt(eps)), IEEE 754 derived)
- Learning rate (MASS: min(eta_ceiling, eta_sps, eta_weyl) — spectral bounds)
- Optimizer (Cayley-Stiefel retraction on rank-r Stiefel manifold)
- Batch size (B_crit = 1/SNR from gradient noise)
- When to stop (val loss convergence, Weyl adapter saturation exhaustion, loss stability)
- Post-training verification (CKA alignment, spectral bounds)
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.use_cases._dataset_training_service_helpers_mixin import (
    _DatasetTrainingServiceHelperMixin,
)
from modelcypher.core.domain.dataset_loading import (
    load_jsonl_dataset,
    merge_datasets_with_fraction,
)
from modelcypher.core.domain.training.exceptions import TrainingDerivationError
from modelcypher.core.domain.training.geometric_lora import (
    analyze_weight_geometries,
    apply_data_rank_ceiling,
    apply_signal_rank_ceiling,
    compute_coupled_ranks,
    compute_per_layer_signal_ranks,
    estimate_nb_lora_parameter_count,
    select_target_modules,
)
from modelcypher.core.domain.training.quantization_frontier_precheck import (
    make_quantization_frontier_precheck_payload_v1,
    run_quantization_frontier_precheck_v1,
)
from modelcypher.core.domain.training.quantization_weyl_precheck import (
    run_quantization_weyl_precheck,
)
from modelcypher.core.domain.training.geometric_optimizer import (
    derive_optimizer_geometry_config,
)
from modelcypher.core.domain.training.pipeline_gate import (
    PipelineGateInput,
    evaluate_pipeline_gate,
)
from modelcypher.core.domain.training.mass_step_size import (
    CONTROLLER_MODE_BEHAVIORAL_PROBE,
    CONTROLLER_MODE_STRUCTURAL_OBSERVE,
    OPTIMIZER_MODE_ADAMW_MATCHED_TRACE,
    OPTIMIZER_MODE_CAYLEY_STIEFEL_MASS,
    replay_controller_trace,
    validate_controller_mode,
    validate_optimizer_research_mode,
)

if TYPE_CHECKING:
    from modelcypher.core.domain.moe.expert_selection import ExpertTargetSelection
    from modelcypher.core.domain.training.constraint_config import ConstraintState
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)

# Apple Metal SIMD group width (hardware constant, not a hyperparameter).
_MLX_SIMD_WIDTH = 32

# PRNG stream offsets: distinct integers → non-overlapping streams.
# Values are arbitrary-but-fixed for reproducibility (Knuth TAOCP §3.2.1).
_EVAL_SEED_OFFSET = 1


@dataclass
class DatasetTrainResult:
    """Result of dataset-driven NB-LoRA training."""

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
    # G6: Optimizer type used
    optimizer_type: str = "cayley_stiefel"
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
    derived_plan: dict[str, Any] | None = None

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
            "optimizer_type": self.optimizer_type,
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
        if self.derived_plan is not None:
            result["derived_plan"] = self.derived_plan
        if self.benchmark_baseline is not None and self.benchmark_post is not None:
            result["benchmark_delta"] = {
                k: self.benchmark_post[k] - self.benchmark_baseline.get(k, 0.0)
                for k in self.benchmark_post
            }
        return result


@dataclass
class NBTargetSurface:
    """Resolved NB-LoRA adaptation surface (geometry-derived).

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
    """Resolved pre-training plan for the canonical NB-LoRA path.

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
    sigma_k_min: float
    sigma_max: float
    estimated_trainable_params: int
    optimizer_geometry_config: Any
    quantization_frontier_precheck: dict[str, Any] | None
    controller_mode: str
    optimizer_research_mode: str

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
            }
        return payload

    def _signal_rank_payload(self) -> dict[int, dict[str, Any]] | None:
        if not self.signal_rank_results:
            return None
        return {
            int(layer_idx): {
                "signal_rank": int(result.signal_rank),
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
        optimizer_type = (
            "adamw"
            if self.optimizer_research_mode == OPTIMIZER_MODE_ADAMW_MATCHED_TRACE
            else "cayley_stiefel"
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
                "optimizer_type": optimizer_type,
                "controller_mode": self.controller_mode,
                "optimizer_research_mode": self.optimizer_research_mode,
                "learning_rate_policy": (
                    "No fixed scalar LR is derived upfront. MASS chooses "
                    "eta_step = min(eta_ceiling, eta_sps, eta_weyl) online."
                ),
                "batch_size_policy": (
                    "Batch size is derived during training from gradient noise "
                    "scale after NB-LoRA injection."
                ),
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
            (
                "Controller: no fixed scalar LR; MASS will choose "
                "eta_step = min(eta_ceiling, eta_sps, eta_weyl) online"
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


class DatasetTrainingService(_DatasetTrainingServiceHelperMixin):
    """Train LoRA adapters from text datasets using NB-LoRA.

    One method. Geometry decides everything. Bounds by construction.
    ScaledGD preconditioning. Weyl adapter-saturation monitoring. CKA verification.
    """

    def __init__(self, adapter: Any, backend: "Backend"):
        self._adapter = adapter
        self._backend = backend
        self._progress_reporter: Any | None = None

    def _load_training_model(
        self,
        model_path: Path,
    ) -> tuple[Any, Any, bool]:
        """Load model + tokenizer with the same path semantics used for training."""
        return self._adapter.load_training_model(str(model_path), backend=self._backend)

    def derive_nb_target_surface(
        self,
        model_path: str | Path,
        dataset_path: str | Path,
        eval_dataset_path: str | Path | None = None,
        seed: int = 42,
    ) -> "NBTargetSurface":
        """Run the canonical derivation path and return the resolved NB surface."""
        plan = self.build_training_plan(
            model_path=model_path,
            dataset_path=dataset_path,
            eval_dataset_path=eval_dataset_path,
            seed=seed,
            no_save=True,
        )
        geometry_info = {}
        for key in plan.target_modules:
            geom = plan.geometries[key]
            geometry_info[key] = {
                "sigma_max": geom.sigma_max,
                "sigma_k": geom.sigma_k,
                "effective_rank": geom.effective_rank,
                "tail_dims": geom.tail_dims,
                "shannon_effective_rank": geom.shannon_effective_rank,
                "spectral_gap": geom.spectral_gap,
                "shape": geom.shape,
            }
        return NBTargetSurface(
            target_keys=list(plan.target_modules),
            rank_overrides=dict(plan.rank_overrides),
            rank_ceiling_source=plan.rank_ceiling_source,
            sigma_k_min=plan.sigma_k_min,
            sigma_max=plan.sigma_max,
            geometry_info=geometry_info,
        )

    def build_training_plan(
        self,
        model_path: str | Path,
        dataset_path: str | Path,
        output_path: str | Path | None = None,
        eval_dataset_path: str | Path | None = None,
        seq_length: int | None = None,
        seed: int | None = None,
        retention_dataset_path: str | Path | None = None,
        quantization_reference_model_path: str | Path | None = None,
        no_save: bool = False,
        target_experts: list[str] | str | None = None,
        controller_mode: str = CONTROLLER_MODE_STRUCTURAL_OBSERVE,
        optimizer_research_mode: str = OPTIMIZER_MODE_CAYLEY_STIEFEL_MASS,
    ) -> DerivedTrainingPlan:
        """Resolve the exact canonical training plan without mutating model state."""
        model_path = Path(model_path).expanduser().resolve()
        dataset_path = Path(dataset_path).expanduser().resolve()
        eval_path = (
            Path(eval_dataset_path).expanduser().resolve()
            if eval_dataset_path
            else None
        )

        if seed is None:
            resolved_seed = self._derive_training_seed(
                model_path=model_path,
                dataset_path=dataset_path,
            )
            seed_source = "derived_from_model_dataset_hash"
        else:
            resolved_seed = int(seed)
            seed_source = "user_supplied_override"

        controller_mode = validate_controller_mode(controller_mode)
        optimizer_research_mode = validate_optimizer_research_mode(
            optimizer_research_mode,
        )

        if no_save:
            output_dir = None
        elif output_path is not None:
            output_dir = Path(output_path).expanduser().resolve()
        else:
            output_dir = (
                model_path.parent
                / "adapters"
                / f"{model_path.name}-nblora-{resolved_seed}"
            )

        random.seed(resolved_seed)
        self._backend.random_seed(resolved_seed)

        logger.info("build_training_plan: loading model from %s", model_path)
        model, tokenizer, _ = self._load_training_model(model_path)
        try:
            all_samples = load_jsonl_dataset(dataset_path)
            explicit_retention_samples: list[dict[str, Any]] | None = None
            if retention_dataset_path is not None:
                retention_path = Path(retention_dataset_path).expanduser().resolve()
                explicit_retention_samples = load_jsonl_dataset(retention_path)

            eval_samples_early = (
                load_jsonl_dataset(eval_path) if eval_path is not None else None
            )

            if seq_length is None:
                token_lengths = []
                token_source_samples = list(all_samples)
                if explicit_retention_samples is not None:
                    token_source_samples.extend(explicit_retention_samples)
                if eval_samples_early is not None:
                    token_source_samples.extend(eval_samples_early)
                for sample in token_source_samples:
                    text = sample.get("text")
                    if isinstance(text, str) and text:
                        n_tokens = len(self._backend.encode_tokens(tokenizer, text))
                        if n_tokens > 0:
                            token_lengths.append(n_tokens)
                if not token_lengths:
                    raise TrainingDerivationError(
                        failure_class="unavailable_measurement",
                        detail="seq_length derivation requires tokenizable text samples.",
                        diagnostics={"n_samples": len(all_samples)},
                    )
                max_tokens = max(token_lengths)
                max_tokens_with_eos = max_tokens + 1
                resolved_seq_length = (
                    (max_tokens_with_eos + _MLX_SIMD_WIDTH - 1) // _MLX_SIMD_WIDTH
                ) * _MLX_SIMD_WIDTH
                seq_length_source = "data_derived_max_token_length"
            else:
                resolved_seq_length = int(seq_length)
                seq_length_source = "user_supplied_override"

            if eval_path is not None:
                train_samples = list(all_samples)
                eval_samples = list(eval_samples_early or [])
                validation_split_info = {
                    "method": "explicit_eval_dataset",
                    "n_train": len(train_samples),
                    "n_eval": len(eval_samples),
                }
            else:
                shuffled_samples = list(all_samples)
                random.Random(resolved_seed).shuffle(shuffled_samples)
                split_index, validation_split_info = (
                    self._derive_validation_split_from_pilot(
                        model=model,
                        tokenizer=tokenizer,
                        samples=shuffled_samples,
                        seq_length=resolved_seq_length,
                    )
                )
                eval_samples = shuffled_samples[:split_index]
                train_samples = shuffled_samples[split_index:]

            quantization_frontier_precheck_result: dict[str, Any] | None = None
            if quantization_reference_model_path is not None:
                fp_reference_path = Path(
                    quantization_reference_model_path,
                ).expanduser().resolve()
                if not fp_reference_path.exists():
                    raise FileNotFoundError(
                        "quantization_reference_model_path does not exist: "
                        f"{fp_reference_path}",
                    )
                probe_texts = self._derive_probe_texts(
                    eval_samples,
                    tokenizer,
                    int(resolved_seq_length),
                )
                quantization_frontier_precheck_result = (
                    self._run_quantization_frontier_precheck(
                        model=model,
                        tokenizer=tokenizer,
                        model_path=model_path,
                        fp_reference_path=fp_reference_path,
                        probe_texts=probe_texts,
                    )
                )
                if not bool(
                    quantization_frontier_precheck_result.get("valid", False),
                ):
                    raise TrainingDerivationError(
                        failure_class="quantization_frontier_unavailable",
                        detail=(
                            "Quantization frontier precheck could not measure "
                            "activation-aware centered-Gram diagnostics; "
                            "training is blocked."
                        ),
                        diagnostics={
                            "reference_model_path": str(fp_reference_path),
                            "candidate_model_path": str(model_path),
                            "failure_modes": list(
                                quantization_frontier_precheck_result.get(
                                    "failure_modes",
                                    [],
                                )
                            ),
                            "n_probes": int(
                                quantization_frontier_precheck_result.get(
                                    "n_probes",
                                    0,
                                )
                            ),
                            "raw_weyl": quantization_frontier_precheck_result.get(
                                "raw_weyl",
                            ),
                        },
                    )

            if hasattr(self._adapter, "analyze_model_geometry_streaming"):
                geometries = self._adapter.analyze_model_geometry_streaming(
                    model,
                    use_randomized=True,
                )
                weights = {}
            else:
                weights = self._adapter.extract_weight_matrices(model)
                geometries = analyze_weight_geometries(weights, self._backend)

            opt_config = derive_optimizer_geometry_config(
                weights,
                self._backend,
                geometries=geometries,
            )
            base_activations = self._collect_probe_activations(
                model,
                tokenizer,
                eval_samples,
                seq_length=resolved_seq_length,
            )

            target_modules = select_target_modules(geometries, include_zero_tail=True)
            manual_target_expert_keys = self._parse_target_expert_keys(target_experts)
            if manual_target_expert_keys:
                filtered = [
                    key for key in target_modules if key in manual_target_expert_keys
                ]
                if not filtered:
                    raise ValueError(
                        "No manually targeted experts were geometrically targetable "
                        "(tail_dims > 0).",
                    )
                target_modules = filtered
            if not target_modules:
                raise ValueError("No targetable layers found from geometric analysis")

            coupled_ranks = compute_coupled_ranks(geometries, target_modules)
            signal_rank_results = compute_per_layer_signal_ranks(
                base_activations,
                self._backend,
            )
            if signal_rank_results:
                final_ranks = apply_signal_rank_ceiling(
                    coupled_ranks,
                    signal_rank_results,
                )
                ceiling_label = "RMT signal-rank"
            else:
                final_ranks = apply_data_rank_ceiling(
                    coupled_ranks,
                    n_samples=len(train_samples),
                )
                ceiling_label = "data-rank (fallback)"

            sigma_k_min = min(
                geom.sigma_k
                for module_key, geom in geometries.items()
                if module_key in target_modules
            )
            sigma_max = max(
                geom.sigma_max
                for module_key, geom in geometries.items()
                if module_key in target_modules
            )
            estimated_trainable_params = estimate_nb_lora_parameter_count(
                geometries,
                final_ranks,
            )

            return DerivedTrainingPlan(
                model_path=model_path,
                dataset_path=dataset_path,
                eval_dataset_path=eval_path,
                output_path=output_dir,
                seed=resolved_seed,
                seed_source=seed_source,
                seq_length=resolved_seq_length,
                seq_length_source=seq_length_source,
                validation_split=dict(validation_split_info or {}),
                train_samples=list(train_samples),
                eval_samples=list(eval_samples),
                geometries=geometries,
                target_modules=list(target_modules),
                rank_overrides={
                    module: int(rank)
                    for module, rank in sorted(final_ranks.items())
                },
                rank_ceiling_source=ceiling_label,
                signal_rank_results=signal_rank_results,
                sigma_k_min=float(sigma_k_min),
                sigma_max=float(sigma_max),
                estimated_trainable_params=int(estimated_trainable_params),
                optimizer_geometry_config=opt_config,
                quantization_frontier_precheck=quantization_frontier_precheck_result,
                controller_mode=controller_mode,
                optimizer_research_mode=optimizer_research_mode,
            )
        finally:
            del model, tokenizer
            self._backend.clear_cache()

    def _run_quantization_frontier_precheck(
        self,
        *,
        model: Any,
        tokenizer: Any,
        model_path: Path,
        fp_reference_path: Path,
        probe_texts: list[str],
    ) -> dict[str, Any]:
        """Measure activation-aware frontier telemetry on base quantized vs FP."""
        logger.info(
            "Running quantization frontier precheck: reference=%s, candidate=%s, probes=%d",
            fp_reference_path,
            model_path,
            len(probe_texts),
        )
        fp_model, fp_tokenizer, _ = self._load_training_model(fp_reference_path)
        try:
            try:
                fp_weights = self._adapter.extract_weight_matrices(fp_model)
                candidate_weights = self._adapter.extract_weight_matrices(model)
                raw_weyl = run_quantization_weyl_precheck(
                    fp_weights=fp_weights,
                    quantized_weights=candidate_weights,
                    backend=self._backend,
                )
            except Exception as exc:
                raw_weyl = {
                    "measurement_error": {
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    },
                }

            if len(probe_texts) < 2:
                result = run_quantization_frontier_precheck_v1(
                    fp_activations={},
                    quantized_activations={},
                    n_probes=len(probe_texts),
                    backend=self._backend,
                    raw_weyl=raw_weyl,
                )
            else:
                try:
                    fp_activations = self._collect_probe_activations_from_texts(
                        fp_model,
                        fp_tokenizer,
                        probe_texts,
                    )
                    quantized_activations = self._collect_probe_activations_from_texts(
                        model,
                        tokenizer,
                        probe_texts,
                    )
                except TrainingDerivationError as exc:
                    result = make_quantization_frontier_precheck_payload_v1(
                        n_probes=len(probe_texts),
                        raw_weyl=raw_weyl,
                        failure_modes=["activation_collection_failed"],
                    )
                    result["collection_error"] = exc.to_dict()
                else:
                    result = run_quantization_frontier_precheck_v1(
                        fp_activations=self._stack_probe_activations(fp_activations),
                        quantized_activations=self._stack_probe_activations(quantized_activations),
                        n_probes=len(probe_texts),
                        backend=self._backend,
                        raw_weyl=raw_weyl,
                    )
            result["reference_model_path"] = str(fp_reference_path)
            result["candidate_model_path"] = str(model_path)
            return result
        finally:
            del fp_model, fp_tokenizer
            self._backend.clear_cache()

    def train_from_dataset(
        self,
        model_path: str | Path,
        dataset_path: str | Path,
        output_path: str | Path | None = None,
        init_adapter_path: str | Path | None = None,
        eval_dataset_path: str | Path | None = None,
        seq_length: int | None = None,
        seed: int | None = None,
        topo_monitor: bool = False,
        dim_monitor: bool = False,
        paired: bool | None = None,
        format_projection: bool = False,
        narrow_dataset_path: str | Path | None = None,
        augmented_dataset_path: str | Path | None = None,
        online_eval: bool = False,
        online_eval_n_problems: int | None = None,
        entropy_regularization: bool = False,
        # Answer-span masked CE training
        answer_mask: bool = False,
        retention_dataset_path: str | Path | None = None,
        # Sub-epoch evaluation interval
        eval_interval: int | None = None,
        # Global EOS exclusion from CE
        eos_exclude: bool = False,
        # Outer similarity monitoring (Kucukahmetler et al. 2026)
        rss_monitor: bool = False,
        # Ablation experiment params (research only, not CLI-exposed)
        entropy_floor_fraction: float | None = None,  # Research only — NOT in strict CLI
        research_online_eval_problem_set_path: str | Path | None = None,
        quantization_reference_model_path: str | Path | None = None,
        no_save: bool = False,
        max_iters_cap: int | None = None,
        benchmark_suite: str | None = None,
        target_experts: list[str] | str | None = None,
        controller_mode: str = CONTROLLER_MODE_STRUCTURAL_OBSERVE,
        optimizer_research_mode: str = OPTIMIZER_MODE_CAYLEY_STIEFEL_MASS,
        enable_offline_replay: bool = True,
        # External gradient hook — composed with any internal format-projection hook
        gradient_hook: "Callable | None" = None,
        # AdamW-decoupled weight decay (research variable, default 0.0)
        weight_decay: float = 0.0,
        plan: DerivedTrainingPlan | None = None,
    ) -> DatasetTrainResult:
        """Train an NB-LoRA adapter from a JSONL dataset.

        All hyperparameters are geometry-derived.  The user selects a model
        and dataset; everything else is math.
        """
        model_path = Path(model_path).expanduser().resolve()
        dataset_path = Path(dataset_path).expanduser().resolve()
        eval_path = Path(eval_dataset_path).expanduser().resolve() if eval_dataset_path else None
        init_adapter = (
            Path(init_adapter_path).expanduser().resolve() if init_adapter_path else None
        )
        requested_output_dir = (
            None
            if no_save
            else (
                Path(output_path).expanduser().resolve()
                if output_path is not None
                else None
            )
        )
        controller_mode = validate_controller_mode(controller_mode)
        optimizer_research_mode = validate_optimizer_research_mode(
            optimizer_research_mode,
        )

        # Emit training started progress event
        rp = self._progress_reporter
        if rp is not None:
            rp.training_started(str(model_path), str(dataset_path))

        if plan is None:
            plan = self.build_training_plan(
                model_path=model_path,
                dataset_path=dataset_path,
                output_path=output_path,
                eval_dataset_path=eval_dataset_path,
                seq_length=seq_length,
                seed=seed,
                retention_dataset_path=retention_dataset_path,
                quantization_reference_model_path=quantization_reference_model_path,
                no_save=no_save,
                target_experts=target_experts,
                controller_mode=controller_mode,
                optimizer_research_mode=optimizer_research_mode,
            )
        else:
            if plan.model_path != model_path or plan.dataset_path != dataset_path:
                raise ValueError(
                    "Provided plan does not match the requested model_path/dataset_path.",
                )
            if eval_path != plan.eval_dataset_path:
                raise ValueError(
                    "Provided plan does not match the requested eval_dataset_path.",
                )
            if plan.controller_mode != controller_mode:
                raise ValueError(
                    "Provided plan does not match controller_mode.",
                )
            if plan.optimizer_research_mode != optimizer_research_mode:
                raise ValueError(
                    "Provided plan does not match optimizer_research_mode.",
                )
            if no_save and plan.output_path is not None:
                raise ValueError("Provided plan does not match no_save=True.")
            if not no_save:
                if requested_output_dir is not None and plan.output_path != requested_output_dir:
                    raise ValueError("Provided plan does not match output_path.")
                if requested_output_dir is None:
                    expected_output_dir = (
                        model_path.parent / "adapters" / f"{model_path.name}-nblora-{plan.seed}"
                    )
                    if plan.output_path != expected_output_dir:
                        raise ValueError(
                            "Provided plan does not match the derived output path.",
                        )

        model_path = plan.model_path
        dataset_path = plan.dataset_path
        eval_path = plan.eval_dataset_path
        output_dir = plan.output_path
        seed = plan.seed
        seq_length = plan.seq_length
        validation_split_info = dict(plan.validation_split)
        train_samples = list(plan.train_samples)
        eval_samples = list(plan.eval_samples)
        quantization_frontier_precheck_result = plan.quantization_frontier_precheck
        sigma_k_min = float(plan.sigma_k_min)
        sigma_max = float(plan.sigma_max)
        logger.info(
            "Training plan resolved: seed=%d, seq_length=%d, target_modules=%d, "
            "rank_range=%s, split=%s",
            seed,
            seq_length,
            len(plan.target_modules),
            (
                [
                    min(plan.rank_overrides.values()),
                    max(plan.rank_overrides.values()),
                ]
                if plan.rank_overrides
                else [0, 0]
            ),
            validation_split_info.get("method"),
        )

        # 1. Load model + tokenizer
        logger.info("Loading model from %s", model_path)
        model, tokenizer, vl_model = self._load_training_model(model_path)
        random.seed(seed)
        self._backend.random_seed(seed)
        logger.info("RNG seeded for training execution: seed=%d", seed)

        # 2. Load dataset-dependent auxiliary surfaces
        logger.info("Loading dataset from %s", dataset_path)
        explicit_retention_samples: list[dict[str, Any]] | None = None
        if retention_dataset_path is not None:
            retention_path = Path(retention_dataset_path).expanduser().resolve()
            explicit_retention_samples = load_jsonl_dataset(retention_path)

        if init_adapter is not None:
            if not hasattr(self._adapter, "apply_standard_lora_adapter"):
                raise ValueError(
                    "Adapter does not support initialization from an existing LoRA adapter",
                )
            merged_layers = self._adapter.apply_standard_lora_adapter(model, init_adapter)
            logger.info(
                "Initialized model from adapter %s (merged_layers=%d)",
                init_adapter,
                merged_layers,
            )

        # 2.1. Unpack packed MoE expert tensors for per-expert geometry + training.
        # Lossless identity operation: same weights, different layout.
        # Split derivation and quantization frontier checks run first on the base model.
        n_unpacked = self._backend.prepare_model_for_training(model, str(model_path))
        if n_unpacked > 0:
            logger.info(
                "MoE model: %d layers unpacked for per-expert training",
                n_unpacked,
            )

        # 2.2. Pre-training benchmark (optional, when --benchmark is set)
        benchmark_baseline_scores: dict[str, float] | None = None
        _benchmark_service = None
        _benchmark_generate_fn = None
        if benchmark_suite is not None:
            from modelcypher.core.use_cases.benchmark_service import BenchmarkService

            logger.info("Running pre-training benchmark suite: %s", benchmark_suite)
            _benchmark_service = BenchmarkService()
            _backend_ref = self._backend

            def _benchmark_generate_fn(m, t, prompt, max_tokens, verbose=False):
                return _backend_ref.generate(m, t, prompt, max_tokens=max_tokens)

            try:
                baseline_suite = _benchmark_service.run_suite(
                    model=model,
                    tokenizer=tokenizer,
                    suite_name=benchmark_suite,
                    generate_fn=_benchmark_generate_fn,
                    limit_per_benchmark=10,
                    max_failures=5,
                )
                benchmark_baseline_scores = {
                    r.benchmark: r.accuracy for r in baseline_suite.benchmarks
                }
                benchmark_baseline_scores["overall"] = baseline_suite.overall_accuracy
                logger.info(
                    "Pre-training benchmark: %s",
                    ", ".join(
                        f"{k}={v:.1%}" for k, v in benchmark_baseline_scores.items()
                    ),
                )
            except Exception:
                logger.warning(
                    "Pre-training benchmark failed — continuing without",
                    exc_info=True,
                )

        # 2.5. Retention replay: merge retention samples into training data
        auto_retention_samples_collected = 0
        if explicit_retention_samples is not None:
            retention_samples = explicit_retention_samples
            # Maximum entropy principle: absent per-sample importance signal,
            # uniform weighting over available data maximizes sample entropy.
            # Mix fraction = data ratio = n_ret / (n_ret + n_train).
            n_ret = len(retention_samples)
            n_trn = len(train_samples)
            retention_fraction = n_ret / (n_ret + n_trn) if (n_ret + n_trn) > 0 else 0.0
            logger.info(
                "Derived retention_fraction=%.6f from data ratio (%d ret / %d total)",
                retention_fraction, n_ret, n_ret + n_trn,
            )
            train_samples = merge_datasets_with_fraction(
                train_samples, retention_samples, retention_fraction,
            )
        else:
            # Auto-retention disabled by default. The Cayley-Stiefel spectral
            # bound already provides preservation by construction (||ΔW||₂ bounded).
            # Auto-retention duplicates training samples at 50% mix, diluting
            # gradient signal by 2× and doubling iterations per epoch with no
            # new information. Use explicit retention_dataset_path when needed.
            logger.info("Auto-retention disabled (Cayley-Stiefel bound provides preservation)")

        # 2.6. Answer-masked dataset preparation
        answer_masked_train = None
        answer_masked_val = None
        vl_samples_present = any(
            isinstance(sample.get("image_path"), str) and bool(sample.get("image_path"))
            for sample in (train_samples + eval_samples)
        )
        if answer_mask and vl_samples_present:
            raise ValueError(
                "--answer-mask is not yet supported for image-conditioned VL training."
            )
        if answer_mask:
            missing_train = sum(
                1 for sample in train_samples if "answer_start" not in sample
            )
            missing_eval = sum(
                1 for sample in eval_samples if "answer_start" not in sample
            )
            if missing_train > 0 or missing_eval > 0:
                raise TrainingDerivationError(
                    failure_class="insufficient_answer_mask_metadata",
                    detail=(
                        "--answer-mask requested but dataset is missing required "
                        "answer_start metadata. Add answer_start to masked dataset schema."
                    ),
                    diagnostics={
                        "missing_answer_start_count": missing_train + missing_eval,
                        "missing_train_answer_start_count": missing_train,
                        "missing_eval_answer_start_count": missing_eval,
                        "total_train_samples": len(train_samples),
                        "total_eval_samples": len(eval_samples),
                        "hint": "Add answer_start to masked dataset schema for every sample.",
                    },
                )

            answer_masked_train = self._adapter.prepare_masked_dataset(
                train_samples, tokenizer,
            )
            answer_masked_val = self._adapter.prepare_masked_dataset(
                eval_samples, tokenizer,
            )
            if not answer_masked_train:
                raise ValueError("No valid masked training samples after tokenization")
            logger.info(
                "Answer-masked training: %d train / %d eval masked samples",
                len(answer_masked_train),
                len(answer_masked_val) if answer_masked_val else 0,
            )

        if vl_samples_present:
            if not vl_model:
                raise ValueError(
                    "Dataset contains image_path entries but model has no vision_config."
                )
            train_dataset = self._adapter.prepare_vl_dataset(
                train_samples,
                tokenizer,
                str(model_path),
            )
            eval_dataset = self._adapter.prepare_vl_dataset(
                eval_samples,
                tokenizer,
                str(model_path),
            )
            logger.info(
                "Prepared VL datasets: %d train / %d eval (image-conditioned)",
                len(train_dataset), len(eval_dataset),
            )
        else:
            train_dataset = self._adapter.prepare_dataset(train_samples, tokenizer)
            eval_dataset = self._adapter.prepare_dataset(eval_samples, tokenizer)
        if not train_dataset:
            raise ValueError("No valid training samples after tokenization")
        if not eval_dataset:
            raise ValueError("No valid eval samples after tokenization")

        # For VL, tokenization can expand <|image_pad|> to many visual-token slots.
        # If derived seq_length underestimates this expanded length, bump to the
        # nearest SIMD width boundary to avoid silent truncation.
        if vl_samples_present:
            vl_max_tokens = max(
                int(sample["tokens"].shape[0]) for sample in (train_dataset + eval_dataset)
            )
            if vl_max_tokens > seq_length:
                old_seq = seq_length
                seq_length = (
                    (vl_max_tokens + _MLX_SIMD_WIDTH - 1) // _MLX_SIMD_WIDTH
                ) * _MLX_SIMD_WIDTH
                logger.info(
                    "Adjusted seq_length for VL token expansion: %d -> %d "
                    "(max_vl_tokens=%d, SIMD_width=%d)",
                    old_seq,
                    seq_length,
                    vl_max_tokens,
                    _MLX_SIMD_WIDTH,
                )

        # Eval uses all available data. batch_size=1 eliminates padding waste
        # and computes exact per-sample loss (no approximation from batching).
        eval_batches = len(eval_dataset)
        eval_batch_size = 1

        if rp is not None:
            rp.training_dataset_loaded(
                n_train=len(train_dataset),
                n_eval=len(eval_dataset),
                seq_length=seq_length,
            )

        # 3. Baseline eval
        baseline_loss, baseline_ppl = self._adapter.evaluate_loss(
            model=model, dataset=eval_dataset, tokenizer=tokenizer,
            batch_size=eval_batch_size, seq_length=seq_length, n_batches=eval_batches,
        )

        # 4. Analyze geometry — this IS the configuration
        if rp is not None:
            rp.training_geometry_started()

        geometries = plan.geometries
        opt_config = plan.optimizer_geometry_config
        logger.info(
            "Geometric optimizer config: %d layers, base_lr=%.2e",
            opt_config.n_layers, opt_config.base_lr,
        )

        # 4.6. Collect base model activations for verification only.
        # The target surface and ranks are already fixed by the shared plan.
        base_activations = self._collect_probe_activations(
            model, tokenizer, eval_samples, seq_length=seq_length,
        )

        # 4.7. Constrained training setup (paired data detection + baseline measurement)
        from modelcypher.core.domain.dataset_loading import (
            build_pair_groups,
            is_paired_dataset,
        )

        # Detect structured data (logic_id + template_id) for geometric reshaping.
        # Geometric reshaping: constructive loss (expand erank, contrastive trajectories).
        # Old constraints (--paired): conservative penalties, experimental, ablation-failed.
        use_constraints = paired is True  # Old system: explicit opt-in only
        use_geometric_reshape = (
            not use_constraints
            and is_paired_dataset(train_samples)
        )
        constraint_config = None
        constraint_state = None
        paired_train_dataset = None
        logic_groups = None
        template_groups = None

        if use_geometric_reshape:
            logger.info(
                "Structured data detected — using geometric reshaping loss "
                "(expand effective rank + contrastive trajectory separation)",
            )
            paired_train_dataset = self._adapter.prepare_paired_dataset(
                train_samples, tokenizer,
            )
            if not paired_train_dataset:
                raise ValueError("No valid paired samples after tokenization")
            logic_groups, template_groups = build_pair_groups(train_samples)

        if use_constraints:
            from modelcypher.core.domain.training.constraint_config import (
                ConstraintState,
                derive_constraint_thresholds,
            )

            logger.warning(
                "EXPERIMENTAL: constrained geometric training enabled via --paired. "
                "Ablation (2026-02-17) showed constraints monotonically hurt on 350M.",
            )

            # Prepare paired dataset with answer masks
            paired_train_dataset = self._adapter.prepare_paired_dataset(
                train_samples, tokenizer,
            )
            if not paired_train_dataset:
                raise ValueError("No valid paired samples after tokenization")

            logic_groups, template_groups = build_pair_groups(train_samples)

            # Sample ALL transformer layers — no guessing about which layers
            # concentrate reasoning.  Baseline measurement is one-time; cost
            # is negligible vs. training.
            n_model_layers = self._backend.get_num_layers(model)
            target_layers = list(range(n_model_layers))

            # Measure baseline constraints on clean base model (before NB-LoRA)
            inv_distances, sep_distances, layer_entropies, layer_entropy_stds = (
                self._adapter.measure_baseline_constraints(
                    model, tokenizer, paired_train_dataset,
                    logic_groups, template_groups,
                    target_layers, max_seq_length=seq_length,
                )
            )

            constraint_config = derive_constraint_thresholds(
                inv_distances, sep_distances, layer_entropies, layer_entropy_stds,
            )
            constraint_state = ConstraintState()
            logger.info(
                "Constraint config: %s", constraint_config.to_dict(),
            )
            if constraint_state.frozen:
                logger.info(
                    "Frozen multipliers: %s (ablation mode)",
                    sorted(constraint_state.frozen),
                )

        target_modules = list(plan.target_modules)
        moe_topology = self._load_moe_topology(model_path)
        if not target_modules:
            raise ValueError("No targetable layers found from geometric analysis")

        final_ranks = dict(plan.rank_overrides)
        signal_rank_results = dict(plan.signal_rank_results)
        ceiling_label = plan.rank_ceiling_source

        moe_targets = self._build_moe_target_selection(
            target_modules=target_modules,
            geometries=geometries,
            rank_overrides=final_ranks,
            topology=moe_topology,
            num_layers=self._backend.get_num_layers(model),
        )

        logger.info(
            "Resolved target surface from shared plan: %d modules, ranks=%s, "
            "ceiling=%s, params~%d",
            len(target_modules),
            sorted(set(final_ranks.values())),
            ceiling_label,
            plan.estimated_trainable_params,
        )

        # 5.9. Pre-training routing snapshot (MoE only).
        # Collect routing decisions BEFORE LoRA injection so the post-training
        # comparison measures indirect routing shift from changed expert outputs.
        routing_sample_texts: list[str] | None = None
        pre_routing_profile: "RoutingProfile | None" = None
        # 20 texts matches degeneration diagnostic budget (line ~1613).
        # ~20 texts * ~100 tokens = ~2000 routing decisions/layer,
        # yielding SE(frequency) ~ 1/sqrt(2000) ~ 0.02.
        _ROUTING_SAMPLE_N = 20

        if moe_topology is not None and hasattr(
            self._backend, "collect_routing_decisions"
        ):
            from modelcypher.core.domain.moe.routing_analysis import RoutingProfile

            routing_sample_texts = [
                s["text"]
                for s in eval_samples[:_ROUTING_SAMPLE_N]
                if s.get("text")
            ]
            if routing_sample_texts:
                try:
                    pre_decisions = self._backend.collect_routing_decisions(
                        model, tokenizer, routing_sample_texts,
                    )
                    pre_routing_profile = RoutingProfile.from_routing_decisions(
                        pre_decisions, moe_topology,
                    )
                    logger.info(
                        "Pre-training routing snapshot: %d layers, %d tokens",
                        pre_routing_profile.num_layers,
                        pre_routing_profile.total_tokens,
                    )
                except Exception:
                    logger.debug(
                        "Pre-training routing collection failed", exc_info=True,
                    )

        # 6. Inject NB-LoRA (bounds by construction)
        logger.info(
            "Injecting NB-LoRA into %d target modules...",
            len(target_modules),
        )
        # Keep logging numerically aligned with adapter injection when caller does
        # Safety margin derived from IEEE 754: 1 - sqrt(eps).
        effective_safety_margin = max(0.0, 1.0 - math.sqrt(float(self._backend.finfo().eps)))
        n_lora_layers = self._adapter.inject_nb_lora(
            model, geometries, target_modules,
            safety_margin=None,
            rank_overrides=final_ranks,
        )
        if n_lora_layers <= 0:
            raise ValueError("No NB-LoRA layers were injected")
        logger.info("NB-LoRA injection complete: %d layers", n_lora_layers)

        # 6b. Log per-layer capacity at injection time
        for mod_name in target_modules:
            geom = geometries.get(mod_name)
            if geom is None:
                continue
            actual_rank = final_ranks.get(mod_name, geom.tail_dims)
            logger.info(
                "Injected %s: rank=%d (tail_dims=%d), shannon_eff_rank=%.1f, "
                "sigma_k=%.6f, scale_bound=%.6f, capacity_util=%.3f",
                mod_name,
                actual_rank,
                geom.tail_dims,
                geom.shannon_effective_rank,
                geom.sigma_k,
                geom.sigma_k / 2.0 * effective_safety_margin,
                geom.shannon_effective_rank / float(geom.full_rank),
            )

        # Freeze base, unfreeze NB-LoRA params
        self._adapter.freeze_and_apply_lora(model)

        n_trainable_params = int(
            sum(param.size for _, param in self._backend.tree_flatten(model.trainable_parameters()))
        )

        # 7. Derive batch size from gradient noise: B_crit = 1/SNR
        batch_size = self._adapter.derive_critical_batch_size(
            model, train_dataset, seq_length,
        )
        # Constrained training: minimum batch size derived from pair group
        # structure via pigeonhole principle.  A batch of size max(G,T)+1
        # guarantees at least one invariance pair and one counterfactual pair.
        if (use_constraints or use_geometric_reshape) and logic_groups and template_groups:
            n_logic = len(logic_groups)
            n_template = len(template_groups)
            # Pigeonhole surplus: max(groups) + 1 guarantees at least one group
            # has ≥2 representatives, enabling a contrastive pair per group.
            min_constrained_batch = max(n_logic, n_template) + 1
            batch_size = max(batch_size, min_constrained_batch)
        logger.info("Geometry-derived batch size: %d", batch_size)

        # Derive gradient accumulation from measured memory fit.
        # Logical batch remains geometry-derived; only micro-batch is reduced.
        grad_accum_steps = 1
        if (
            answer_masked_train is None
            and not use_constraints
            and not use_geometric_reshape
            and hasattr(self._adapter, "derive_memory_safe_micro_batch_size")
        ):
            try:
                safe_micro_batch = self._adapter.derive_memory_safe_micro_batch_size(
                    model=model,
                    train_dataset=train_dataset,
                    seq_length=seq_length,
                    logical_batch_size=batch_size,
                    seed=seed,
                )
                if safe_micro_batch < batch_size:
                    grad_accum_steps = math.ceil(batch_size / safe_micro_batch)
                    logger.info(
                        "Memory-bound micro batching: logical_batch=%d micro_batch=%d accum_steps=%d",
                        batch_size,
                        safe_micro_batch,
                        grad_accum_steps,
                    )
                else:
                    logger.info(
                        "Memory-bound micro batching: logical_batch=%d fits without accumulation",
                        batch_size,
                    )
            except Exception:
                logger.warning(
                    "Memory-safe micro-batch probe failed; running without accumulation",
                    exc_info=True,
                )

        if rp is not None:
            rp.training_geometry_complete(
                n_target_modules=len(target_modules),
                n_trainable_params=n_trainable_params,
                batch_size=batch_size,
                rank_min=min(final_ranks.values()) if final_ranks else 0,
                rank_max=max(final_ranks.values()) if final_ranks else 0,
                split_method=str(validation_split_info.get("method", "unknown")),
            )

        derived_max_iters_cap = self._derive_training_safety_cap(
            n_samples=len(train_dataset),
            batch_size=batch_size,
        )
        if max_iters_cap is not None:
            if max_iters_cap <= 0:
                raise ValueError("max_iters_cap must be positive")
            resolved_max_iters_cap = max_iters_cap
        else:
            resolved_max_iters_cap = derived_max_iters_cap
        logger.info(
            "Training safety cap: max_iters=%d (derived=%d, override=%s)",
            resolved_max_iters_cap,
            derived_max_iters_cap,
            max_iters_cap is not None,
        )

        # 8. MASS: reuse the canonical spectral ceilings from the shared plan.
        logger.info(
            "MASS geometry: sigma_max=%.4e, sigma_k_min=%.4e, ceiling=%.4e",
            sigma_max, sigma_k_min, sigma_k_min / sigma_max,
        )

        # 8.5. Format bias projection hook (optional)
        _external_hook = gradient_hook  # from parameter — preserve before local assignment
        gradient_hook = None
        if format_projection:
            gradient_hook = self._build_format_projection_hook(
                model, tokenizer,
                narrow_dataset_path=narrow_dataset_path,
                augmented_dataset_path=augmented_dataset_path,
            )
        # Compose: format hook runs first, external hook runs second
        if _external_hook is not None:
            if gradient_hook is not None:
                _fmt = gradient_hook
                def _composed(g, _f=_fmt, _e=_external_hook):
                    return _e(_f(g))
                gradient_hook = _composed
            else:
                gradient_hook = _external_hook

        # 8.10. Online eval: create problems + measure baseline (optional)
        eval_problems = None
        eval_baseline_correct_ids: frozenset[str] = frozenset()
        baseline_result = None
        baseline_margins: dict[str, float] | None = None
        if online_eval:
            from modelcypher.core.domain.training.online_eval import (
                compute_answer_margin,
                create_eval_problem_set,
                evaluate_correctness,
            )
            from modelcypher.core.domain.star.problem_generator import StarProblem

            if research_online_eval_problem_set_path is not None:
                eval_problem_path = Path(
                    research_online_eval_problem_set_path,
                ).expanduser().resolve()
                if not eval_problem_path.exists():
                    raise FileNotFoundError(
                        f"research_online_eval_problem_set_path does not exist: {eval_problem_path}",
                    )
                raw_payload = json.loads(eval_problem_path.read_text(encoding="utf-8"))
                if isinstance(raw_payload, dict):
                    records = raw_payload.get("problems", [])
                elif isinstance(raw_payload, list):
                    records = raw_payload
                else:
                    raise ValueError(
                        "research_online_eval_problem_set_path must contain a JSON list "
                        "or an object with a 'problems' list.",
                    )
                if not isinstance(records, list):
                    raise ValueError(
                        "research_online_eval_problem_set_path: 'problems' must be a list",
                    )
                eval_problems = [
                    StarProblem.from_problem_record(record)
                    for record in records
                ]
                if not eval_problems:
                    raise ValueError(
                        "research_online_eval_problem_set_path contains no problems",
                    )
                if (
                    online_eval_n_problems is not None
                    and online_eval_n_problems != len(eval_problems)
                ):
                    raise ValueError(
                        "online_eval_n_problems does not match the loaded research "
                        "online eval problem set length",
                    )
                online_eval_n_problems = len(eval_problems)
                logger.info(
                    "Loaded %d online eval problems from %s",
                    len(eval_problems),
                    eval_problem_path,
                )
            else:
                if online_eval_n_problems is None:
                    raise ValueError(
                        "--online-eval requires --online-eval-n <N> "
                        "(number of eval problems is a compute budget choice, not derivable)"
                    )
                eval_seed = seed + _EVAL_SEED_OFFSET
                eval_problems = create_eval_problem_set(
                    n_problems=online_eval_n_problems,
                    seed=eval_seed,
                )
                logger.info(
                    "Created %d online eval problems (seed=%d, derived from training seed+1)",
                    len(eval_problems), eval_seed,
                )

            # Measure baseline: greedy decoding is deterministic,
            # so this is an exact measurement, not a sample estimate
            def _baseline_gen(prompt: str, max_toks: int) -> str:
                return self._backend.generate(model, tokenizer, prompt, max_toks)

            baseline_result = evaluate_correctness(
                problems=eval_problems,
                generate_fn=_baseline_gen,
                epoch=0,
                baseline_correct_ids=None,  # first measurement = baseline
                max_tokens=seq_length,
            )
            eval_baseline_correct_ids = baseline_result.correct_ids
            if controller_mode == CONTROLLER_MODE_BEHAVIORAL_PROBE:
                def _collect_baseline_logits(prompt: str):
                    return self._backend.collect_logits(
                        model,
                        tokenizer,
                        prompt,
                    )

                baseline_margins = compute_answer_margin(
                    eval_problems,
                    _collect_baseline_logits,
                    self._backend,
                )
            logger.info(
                "Online eval baseline: %d/%d (%.1f%%)",
                baseline_result.n_correct,
                baseline_result.n_total,
                baseline_result.accuracy * 100,
            )

        # 8.10.1. Derive n-gram order from readout geometry (birthday paradox).
        # n = ceil(2 * log(T) / log(r_eff)) where T ~ 400 words and r_eff =
        # Shannon effective rank of the readout weight matrix.
        # Fail-closed: if derivation fails, degen_ngram_order stays None and
        # the degeneration gate is disabled (no underived magic numbers).
        degen_ngram_order: int | None = None
        readout_erank: float | None = None
        try:
            from modelcypher.core.domain.geometry.perturbation_bound import (
                compute_readout_effective_rank,
            )
            from modelcypher.core.domain.training.degeneration import (
                derive_ngram_order,
            )

            readout_erank = compute_readout_effective_rank(model, self._backend)
            degen_ngram_order = derive_ngram_order(
                readout_erank, generation_length=400,
            )
            logger.info(
                "Readout effective rank=%.1f -> n-gram order=%d",
                readout_erank, degen_ngram_order,
            )
        except Exception:
            logger.info(
                "Readout erank unavailable — degeneration gate disabled (no underived fallback)"
            )
            logger.debug("Readout erank error details", exc_info=True)

        # 8.10.2. Baseline degeneration measurement: few-shot prompted generation.
        # Uses same prompt format and token budget as G5 validation and the
        # post-training diagnostic. Measured here so train_loop can compare
        # per-epoch and stop if degeneration exceeds baseline + sqrt(eps).
        degen_baseline_max: float | None = None
        degen_prompts_for_training: list[str] | None = None
        if eval_problems and degen_ngram_order is not None:
            try:
                from modelcypher.core.domain.training.degeneration import (
                    ngram_repetition_rate,
                )
                from modelcypher.core.domain.star.prompting import (
                    build_forward_prompt,
                    default_few_shot_examples,
                )

                n_demos = len(default_few_shot_examples())
                _DEGEN_N_PROMPTS = 20
                degen_subset = eval_problems[:_DEGEN_N_PROMPTS]
                degen_prompts_for_training = [
                    build_forward_prompt(p, demonstrations=n_demos)
                    for p in degen_subset
                ]
                baseline_degen_rates: list[float] = []
                for prompt_text in degen_prompts_for_training:
                    try:
                        response = self._backend.generate(
                            model, tokenizer, prompt_text, max_tokens=512,
                        )
                        rate = ngram_repetition_rate(response, degen_ngram_order)
                        baseline_degen_rates.append(rate)
                    except Exception:
                        logger.debug(
                            "Baseline degeneration generation failed", exc_info=True,
                        )
                if baseline_degen_rates:
                    degen_baseline_max = max(baseline_degen_rates)
                    logger.info(
                        "Baseline degeneration: max_ngram(%d)=%.3f, mean=%.3f (%d prompts)",
                        degen_ngram_order,
                        degen_baseline_max,
                        sum(baseline_degen_rates) / len(baseline_degen_rates),
                        len(baseline_degen_rates),
                    )
            except Exception:
                logger.debug("Baseline degeneration unavailable", exc_info=True)

        # 8.10.2. Collect inference probe activations for dual-manifold CKA
        # (diagnostic-only). Model is post-injection but NB-LoRA starts at
        # identity, so activations are mathematically equivalent to base.
        inference_base_activations: dict[int, list] | None = None
        if eval_problems:
            inference_base_activations = self._collect_inference_probe_activations(
                model, tokenizer, eval_problems,
            )

        # 9. Train — ScaledGD + Weyl adapter saturation + validation loss stopping
        if rp is not None:
            rp.training_loop_started(max_iters=resolved_max_iters_cap)
        train_start = time.time()
        losses, stop_reason, epoch_metrics = self._adapter.train_loop(
            model=model,
            train_dataset=train_dataset,
            batch_size=batch_size,
            seq_length=seq_length,
            max_iters=resolved_max_iters_cap,
            seed=seed,
            sigma_max=sigma_max,
            eval_dataset=eval_dataset,
            eval_batches=eval_batches,
            adaptive_lr=True,
            lr_monotonic=False,
            sigma_k_min=sigma_k_min,
            tokenizer=tokenizer,
            opt_config=opt_config,
            topo_monitor=topo_monitor,
            topo_probe_texts=self._derive_probe_texts(eval_samples, tokenizer, seq_length) if topo_monitor else None,
            dim_monitor=dim_monitor,
            dim_probe_texts=self._derive_probe_texts(eval_samples, tokenizer, seq_length) if dim_monitor else None,
            constraint_config=constraint_config,
            constraint_state=constraint_state,
            paired_dataset=paired_train_dataset,
            logic_groups=logic_groups,
            template_groups=template_groups,
            geometric_reshape=use_geometric_reshape,
            gradient_hook=gradient_hook,
            entropy_regularization=entropy_regularization,
            online_eval_problems=eval_problems,
            online_eval_baseline_ids=eval_baseline_correct_ids,
            answer_masked_dataset=answer_masked_train,
            answer_masked_eval=answer_masked_val,
            eval_interval=eval_interval,
            eos_exclude=eos_exclude,
            rss_monitor=rss_monitor,
            base_activations=(
                base_activations
                if rss_monitor or controller_mode == CONTROLLER_MODE_BEHAVIORAL_PROBE
                else None
            ),
            entropy_floor_fraction=entropy_floor_fraction,
            degen_prompts=degen_prompts_for_training,
            degen_baseline_max=degen_baseline_max,
            degen_ngram_order=degen_ngram_order,
            readout_erank=readout_erank,
            grad_accum_steps=grad_accum_steps,
            controller_mode=controller_mode,
            optimizer_research_mode=optimizer_research_mode,
            baseline_margins=baseline_margins,
            weight_decay=weight_decay,
        )
        training_time_seconds = time.time() - train_start

        if losses:
            initial_loss = losses[0][1]
            final_loss = losses[-1][1]
            train_iters = len(losses)
        else:
            initial_loss = baseline_loss
            final_loss = baseline_loss
            train_iters = 0

        if rp is not None:
            rp.training_loop_complete(
                train_iters=train_iters,
                initial_loss=initial_loss,
                final_loss=final_loss,
                stop_reason=stop_reason,
                training_time_seconds=training_time_seconds,
            )

        # 10. Post-training eval
        logger.info("Starting post-training evaluation...")
        post_loss, post_ppl = self._adapter.evaluate_loss(
            model=model, dataset=eval_dataset, tokenizer=tokenizer,
            batch_size=eval_batch_size, seq_length=seq_length, n_batches=eval_batches,
        )
        logger.info("Post-training eval complete: loss=%.4f, ppl=%.4f", post_loss, post_ppl)

        # 11. Verify bounds (should always pass — by construction)
        if rp is not None:
            rp.training_verification_started(
                gates=[
                    "spectral_bounds_ok",
                    "min_cka",
                    "mean_cka",
                    "degeneration_max_ngram_repeat",
                    "mode_connectivity_barrier",
                    "pipeline_gate_passed",
                ],
            )
        logger.info("Verifying spectral bounds...")
        spectral_bounds_ok, max_spectral_ratio, _ = self._adapter.verify_bounds(model)
        logger.info("Spectral bounds verified: ok=%s, max_ratio=%.4f", spectral_bounds_ok, max_spectral_ratio)

        # 11.5. CKA verification — does the adapted model preserve base representations?
        min_cka = None
        mean_cka = None
        per_layer_cka = None
        per_layer_gram_epsilon = None
        per_layer_cka_bound = None
        per_layer_null_observability = None
        per_layer_null_accessibility = None
        per_module_null_accessibility = None
        min_cka_layer = None
        inference_min_cka = None
        inference_mean_cka = None
        inference_per_layer_cka = None
        inference_per_layer_gram_epsilon = None
        inference_min_cka_layer = None
        mode_connectivity_barrier = None
        mode_connectivity_normalized_barrier = None
        mode_connectivity_method = None
        if base_activations:
            logger.info("Starting CKA verification...")
            cka_result = self._verify_capability_preservation(
                model, tokenizer, base_activations, eval_samples,
                seq_length=seq_length,
                inference_base_activations=inference_base_activations,
                inference_problems=eval_problems,
            )
            min_cka = cka_result.get("min_cka")
            mean_cka = cka_result.get("mean_cka")
            per_layer_cka = cka_result.get("per_layer_cka")
            per_layer_gram_epsilon = cka_result.get("per_layer_gram_epsilon")
            per_layer_cka_bound = cka_result.get("per_layer_cka_bound")
            per_layer_null_observability = cka_result.get("per_layer_null_observability")
            per_layer_null_accessibility = cka_result.get("per_layer_null_accessibility")
            per_module_null_accessibility = cka_result.get("per_module_null_accessibility")
            if per_layer_cka:
                min_cka_layer = min(per_layer_cka, key=per_layer_cka.get)
            if min_cka is not None and mean_cka is not None:
                logger.info(
                    "CKA verification: min=%.4f, mean=%.4f (%d probes, %d layers)",
                    min_cka, mean_cka,
                    cka_result.get("n_probes", 0),
                    len(cka_result.get("per_layer_cka", {})),
                )
            else:
                logger.info(
                    "CKA verification: skipped (%s)",
                    cka_result.get("skipped_reason", "unknown"),
                )
            # Inference-manifold CKA (diagnostic)
            inference_min_cka = cka_result.get("inference_min_cka")
            inference_mean_cka = cka_result.get("inference_mean_cka")
            inference_per_layer_cka = cka_result.get("inference_per_layer_cka")
            inference_per_layer_gram_epsilon = cka_result.get(
                "inference_per_layer_gram_epsilon",
            )
            inference_min_cka_layer = cka_result.get("inference_min_cka_layer")
            # Mode connectivity
            mode_connectivity_barrier = cka_result.get("mode_connectivity_barrier")
            mode_connectivity_normalized_barrier = cka_result.get("mode_connectivity_normalized_barrier")
            mode_connectivity_method = cka_result.get("mode_connectivity_method")

        if rp is not None:
            rp.training_verification_complete(
                spectral_bounds_ok=spectral_bounds_ok,
                min_cka=min_cka,
                mean_cka=mean_cka,
                gates=[
                    "spectral_bounds_ok",
                    "min_cka",
                    "mean_cka",
                    "degeneration_max_ngram_repeat",
                    "mode_connectivity_barrier",
                    "pipeline_gate_passed",
                ],
            )

        # Extract adapter saturation ratio from last epoch metrics
        adapter_saturation_median_ratio = None
        dim_final_used_fraction = None
        dim_final_null_fraction = None
        dim_null_recruitment_from_baseline = None
        if epoch_metrics:
            last = epoch_metrics[-1]
            if hasattr(last, "adapter_saturation_median_ratio"):
                adapter_saturation_median_ratio = last.adapter_saturation_median_ratio
            if hasattr(last, "dim_final_used_fraction"):
                dim_final_used_fraction = last.dim_final_used_fraction
            if hasattr(last, "dim_final_null_fraction"):
                dim_final_null_fraction = last.dim_final_null_fraction
            if hasattr(last, "dim_null_recruitment_from_baseline"):
                dim_null_recruitment_from_baseline = last.dim_null_recruitment_from_baseline

        # Extract RSS final values from last epoch metrics
        rss_final_cosine = None
        rss_final_spearman = None
        rss_final_top1 = None
        if epoch_metrics:
            last = epoch_metrics[-1]
            if hasattr(last, "rss_cosine") and last.rss_cosine is not None:
                rss_final_cosine = last.rss_cosine
            if hasattr(last, "rss_spearman") and last.rss_spearman is not None:
                rss_final_spearman = last.rss_spearman
            if hasattr(last, "rss_top1_agreement") and last.rss_top1_agreement is not None:
                rss_final_top1 = last.rss_top1_agreement

        # 11.7. Degeneration diagnostic — measures n-gram repetition on few-shot
        # prompted generation. Uses same prompt format and token budget as G5
        # validation (build_forward_prompt + 512 max_tokens) to ensure the
        # measurement matches what validation will check.
        degeneration_max_ngram_repeat = None
        degeneration_mean_ngram_repeat = None
        if eval_problems and degen_ngram_order is not None:
            try:
                from modelcypher.core.domain.training.degeneration import (
                    ngram_repetition_rate,
                )
                from modelcypher.core.domain.star.prompting import (
                    build_forward_prompt,
                    default_few_shot_examples,
                )

                n_demos = len(default_few_shot_examples())
                # 20 prompts matches G5 validation protocol. SE ≤ 0.112 at
                # worst-case σ=0.5, sufficient to detect moderate degeneration
                # (0.3-0.4 range) that 5 prompts missed on seed 42.
                _DEGEN_N_PROMPTS = 20
                degen_prompts = eval_problems[:_DEGEN_N_PROMPTS]
                degen_rates: list[float] = []
                for problem in degen_prompts:
                    try:
                        prompt_text = build_forward_prompt(
                            problem, demonstrations=n_demos,
                        )
                        response = self._backend.generate(
                            model, tokenizer, prompt_text, max_tokens=512,
                        )
                        rate = ngram_repetition_rate(response, degen_ngram_order)
                        degen_rates.append(rate)
                    except Exception:
                        logger.debug("Degeneration generation failed for prompt", exc_info=True)

                if degen_rates:
                    degeneration_max_ngram_repeat = max(degen_rates)
                    degeneration_mean_ngram_repeat = sum(degen_rates) / len(degen_rates)
                    logger.info(
                        "Degeneration diagnostic: max_ngram(%d)=%.3f, mean=%.3f (%d prompts)",
                        degen_ngram_order,
                        degeneration_max_ngram_repeat,
                        degeneration_mean_ngram_repeat,
                        len(degen_rates),
                    )
            except Exception:
                logger.debug("Degeneration diagnostic unavailable", exc_info=True)

        # 11.8. Post-training benchmark (optional, when --benchmark is set)
        benchmark_post_scores: dict[str, float] | None = None
        if (
            benchmark_suite is not None
            and _benchmark_service is not None
            and _benchmark_generate_fn is not None
        ):
            logger.info("Running post-training benchmark suite: %s", benchmark_suite)
            try:
                post_suite = _benchmark_service.run_suite(
                    model=model,
                    tokenizer=tokenizer,
                    suite_name=benchmark_suite,
                    generate_fn=_benchmark_generate_fn,
                    limit_per_benchmark=10,
                    max_failures=5,
                )
                benchmark_post_scores = {
                    r.benchmark: r.accuracy for r in post_suite.benchmarks
                }
                benchmark_post_scores["overall"] = post_suite.overall_accuracy
                logger.info(
                    "Post-training benchmark: %s",
                    ", ".join(
                        f"{k}={v:.1%}" for k, v in benchmark_post_scores.items()
                    ),
                )
                if benchmark_baseline_scores is not None:
                    for k in benchmark_post_scores:
                        delta = benchmark_post_scores[k] - benchmark_baseline_scores.get(k, 0.0)
                        logger.info("  %s delta: %+.1f%%", k, delta * 100)
            except Exception:
                logger.warning(
                    "Post-training benchmark failed", exc_info=True,
                )

        # Extract max gain ratio across all epochs for stability certificate.
        max_gain_ratio: float | None = None
        if epoch_metrics:
            gain_ratios = [
                m.max_effective_gain_ratio
                for m in epoch_metrics
                if m.max_effective_gain_ratio is not None
            ]
            if gain_ratios:
                max_gain_ratio = max(gain_ratios)

        epoch_metrics_payload = [m.to_dict() for m in epoch_metrics] if epoch_metrics else None
        controller_trace_payload = None
        if epoch_metrics_payload is not None:
            controller_trace_payload = [
                metric["controller_trace"]
                for metric in epoch_metrics_payload
                if isinstance(metric, dict) and metric.get("controller_trace") is not None
            ] or None
        offline_replay = (
            replay_controller_trace(epoch_metrics_payload)
            if enable_offline_replay and epoch_metrics_payload is not None
            else None
        )

        # 11.10. Pipeline promotability gate (shared with derived validation).
        gate_eps = float(self._backend.finfo().eps)
        gate_input = PipelineGateInput(
            spectral_bounds_ok=bool(spectral_bounds_ok),
            stop_reason=stop_reason,
            per_layer_cka=per_layer_cka,
            per_layer_cka_bound=per_layer_cka_bound,
            adapter_saturation_median_ratio=adapter_saturation_median_ratio,
            max_effective_gain_ratio=max_gain_ratio,
            epoch_metrics=epoch_metrics_payload,
            strict_fail_closed_core=True,
        )
        pipeline_gate_verdict = evaluate_pipeline_gate(gate_input, eps=gate_eps)
        if not pipeline_gate_verdict.passed:
            verdict_dict = pipeline_gate_verdict.to_dict()
            raise TrainingDerivationError(
                failure_class="pipeline_gate_failed",
                detail=(
                    "Pipeline gate failed: "
                    + ", ".join(pipeline_gate_verdict.failure_modes)
                ),
                diagnostics=verdict_dict,
            )

        # Detect training objective from data format flags
        training_objective = "ce"
        if eos_exclude:
            training_objective = "ce_eos_excluded"
        if answer_masked_train:
            training_objective = "ce_answer_masked"

        # 12. Save if requested
        saved_adapter_path: str | None = None
        if output_dir is not None:
            logger.info("Saving adapter to %s...", output_dir)
            metadata = {
                "base_model_path": str(model_path),
                "stop_reason": stop_reason,
                "n_lora_layers": str(n_lora_layers),
                "train_iters": str(train_iters),
                "method": "nb_lora_cayley",
                "safety_margin": str(effective_safety_margin),
                "optimizer": "cayley_stiefel",
                "training_objective": training_objective,
                "capability_transfer": "true",
            }
            if min_cka is not None:
                metadata["min_cka"] = f"{min_cka:.4f}"
            if mean_cka is not None:
                metadata["mean_cka"] = f"{mean_cka:.4f}"
            saved_path = self._adapter.save_adapter(
                model=model,
                output_path=output_dir,
                metadata=metadata,
            )
            self._write_geometry_manifest(
                adapter_dir=saved_path,
                model_path=model_path,
                target_modules=target_modules,
                geometries=geometries,
                rank_overrides=final_ranks,
                rank_ceiling_source=ceiling_label,
            )
            self._write_training_plan(
                adapter_dir=saved_path,
                derived_plan=plan.to_dict(),
            )
            saved_adapter_path = str(saved_path)

        moe_saturated_during_training: list[str] | None = None
        moe_saturation_threshold = max(
            0.0,
            1.0 - math.sqrt(float(self._backend.finfo().eps)),
        )
        if epoch_metrics:
            saturated_keys: set[str] = set()
            for metric in epoch_metrics:
                saturation_map = metric.expert_saturation_map
                if not saturation_map:
                    continue
                for expert_key, ratio in saturation_map.items():
                    if ratio >= moe_saturation_threshold:
                        saturated_keys.add(expert_key)
            if saturated_keys:
                moe_saturated_during_training = sorted(saturated_keys)

        # 11.9. MoE routing stability (KL divergence pre/post training).
        moe_router_stability: float | None = None
        if pre_routing_profile is not None and routing_sample_texts is not None:
            try:
                from modelcypher.core.domain.moe.routing_analysis import (
                    RoutingProfile,
                    routing_kl_divergence,
                )

                post_decisions = self._backend.collect_routing_decisions(
                    model, tokenizer, routing_sample_texts,
                )
                post_routing_profile = RoutingProfile.from_routing_decisions(
                    post_decisions, moe_topology,
                )
                moe_router_stability = routing_kl_divergence(
                    pre_routing_profile, post_routing_profile,
                )
                logger.info(
                    "MoE routing stability: KL(post||pre) = %.6f nats",
                    moe_router_stability,
                )
            except Exception:
                logger.debug(
                    "Post-training routing collection failed", exc_info=True,
                )

        if rp is not None:
            rp.training_complete(
                adapter_path=saved_adapter_path,
                training_time_seconds=training_time_seconds,
                final_loss=final_loss,
                post_loss=post_loss,
            )

        return DatasetTrainResult(
            train_iters=train_iters,
            initial_loss=initial_loss,
            final_loss=final_loss,
            stop_reason=stop_reason,
            baseline_loss=baseline_loss,
            baseline_perplexity=baseline_ppl,
            post_loss=post_loss,
            post_perplexity=post_ppl,
            n_lora_layers=n_lora_layers,
            n_trainable_params=n_trainable_params,
            adapter_path=saved_adapter_path,
            target_module_count=int(len(target_modules)),
            target_modules=list(sorted(target_modules)),
            rank_overrides={key: int(value) for key, value in sorted(final_ranks.items())},
            rank_ceiling_source=ceiling_label,
            sigma_k_min=float(sigma_k_min),
            sigma_max=float(sigma_max),
            resolved_batch_size=int(batch_size),
            spectral_bounds_ok=spectral_bounds_ok,
            max_spectral_ratio=max_spectral_ratio,
            training_time_seconds=training_time_seconds,
            epoch_metrics=epoch_metrics_payload,
            min_cka=min_cka,
            mean_cka=mean_cka,
            per_layer_cka=per_layer_cka,
            per_layer_gram_epsilon=per_layer_gram_epsilon,
            per_layer_cka_bound=per_layer_cka_bound,
            per_layer_null_observability=per_layer_null_observability,
            per_layer_null_accessibility=per_layer_null_accessibility,
            per_module_null_accessibility=per_module_null_accessibility,
            min_cka_layer=min_cka_layer,
            adapter_saturation_median_ratio=adapter_saturation_median_ratio,
            seq_length_used=int(seq_length),
            dim_final_used_fraction=dim_final_used_fraction,
            dim_final_null_fraction=dim_final_null_fraction,
            dim_null_recruitment_from_baseline=dim_null_recruitment_from_baseline,
            optimizer_type=(
                "adamw"
                if optimizer_research_mode == OPTIMIZER_MODE_ADAMW_MATCHED_TRACE
                else "cayley_stiefel"
            ),
            rss_final_cosine=rss_final_cosine,
            rss_final_spearman=rss_final_spearman,
            rss_final_top1=rss_final_top1,
            quantization_frontier_precheck=quantization_frontier_precheck_result,
            validation_split=validation_split_info,
            auto_retention_samples_collected=auto_retention_samples_collected,
            max_effective_gain_ratio=max_gain_ratio,
            inference_min_cka=inference_min_cka,
            inference_mean_cka=inference_mean_cka,
            inference_per_layer_cka=inference_per_layer_cka,
            inference_per_layer_gram_epsilon=inference_per_layer_gram_epsilon,
            inference_min_cka_layer=inference_min_cka_layer,
            per_layer_signal_ranks={
                layer_idx: {
                    "signal_rank": sr.signal_rank,
                    "noise_rank": sr.noise_rank,
                    "mp_upper_edge": sr.mp_upper_edge,
                    "signal_variance_fraction": sr.signal_variance_fraction,
                }
                for layer_idx, sr in signal_rank_results.items()
            } if signal_rank_results else None,
            mode_connectivity_barrier=mode_connectivity_barrier,
            mode_connectivity_normalized_barrier=mode_connectivity_normalized_barrier,
            mode_connectivity_method=mode_connectivity_method,
            degeneration_max_ngram_repeat=degeneration_max_ngram_repeat,
            degeneration_mean_ngram_repeat=degeneration_mean_ngram_repeat,
            degeneration_ngram_order=degen_ngram_order,
            benchmark_baseline=benchmark_baseline_scores,
            benchmark_post=benchmark_post_scores,
            moe_targets=moe_targets,
            moe_saturated_during_training=moe_saturated_during_training,
            moe_router_stability=moe_router_stability,
            pipeline_gate_operator=pipeline_gate_verdict.operator,
            pipeline_gate_passed=pipeline_gate_verdict.passed,
            pipeline_gate_failure_modes=list(pipeline_gate_verdict.failure_modes),
            pipeline_gate_checks={
                name: check.to_dict()
                for name, check in pipeline_gate_verdict.checks.items()
            },
            training_objective=training_objective,
            controller_mode=controller_mode,
            optimizer_research_mode=optimizer_research_mode,
            controller_trace=controller_trace_payload,
            offline_replay=offline_replay,
            derived_plan=plan.to_dict(),
        )

__all__ = ["DatasetTrainResult", "DatasetTrainingService", "DerivedTrainingPlan"]
