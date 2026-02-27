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

import hashlib
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from modelcypher.core.domain.training.quantization_weyl_precheck import (
    run_quantization_weyl_precheck,
)
from modelcypher.core.domain.training.geometric_optimizer import (
    derive_optimizer_geometry_config,
)

if TYPE_CHECKING:
    from modelcypher.core.domain.training.constraint_config import ConstraintState
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)

# Apple Metal SIMD group width (hardware constant, not a hyperparameter).
_MLX_SIMD_WIDTH = 32

# PRNG stream offsets: distinct integers → non-overlapping streams.
# Values are arbitrary-but-fixed for reproducibility (Knuth TAOCP §3.2.1).
_EVAL_SEED_OFFSET = 1
_OUTCOME_SEED_OFFSET = 2


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
    # Auto-regime selection metadata (when auto_regime=True)
    regime_global: str | None = None
    regime_per_type: dict[str, Any] | None = None
    regime_headroom: dict[str, Any] | None = None
    quantization_precheck: dict[str, Any] | None = None
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
    # G4 diagnostic: Degeneration measurement (not a gate until n derived — TODO: G1)
    degeneration_max_4gram_repeat: float | None = None
    degeneration_mean_4gram_repeat: float | None = None

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
        if self.regime_global is not None:
            result["regime_global"] = self.regime_global
        if self.regime_per_type is not None:
            result["regime_per_type"] = self.regime_per_type
        if self.regime_headroom is not None:
            result["regime_headroom"] = self.regime_headroom
        if self.quantization_precheck is not None:
            result["quantization_precheck"] = self.quantization_precheck
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
        if self.degeneration_max_4gram_repeat is not None:
            result["degeneration_max_4gram_repeat"] = self.degeneration_max_4gram_repeat
        if self.degeneration_mean_4gram_repeat is not None:
            result["degeneration_mean_4gram_repeat"] = self.degeneration_mean_4gram_repeat
        return result


class DatasetTrainingService:
    """Train LoRA adapters from text datasets using NB-LoRA.

    One method. Geometry decides everything. Bounds by construction.
    ScaledGD preconditioning. Weyl adapter-saturation monitoring. CKA verification.
    """

    def __init__(self, adapter: Any, backend: "Backend"):
        self._adapter = adapter
        self._backend = backend
        self._progress_reporter: Any | None = None

    def train_from_dataset_strict(
        self,
        model_path: str | Path,
        dataset_path: str | Path,
        output_path: str | Path | None = None,
        eval_dataset_path: str | Path | None = None,
    ) -> DatasetTrainResult:
        """Strict training entrypoint: model+dataset only."""
        resolved_model_path = Path(model_path).expanduser().resolve()
        resolved_dataset_path = Path(dataset_path).expanduser().resolve()
        derived_seed = self._derive_strict_seed(
            model_path=resolved_model_path,
            dataset_path=resolved_dataset_path,
        )
        logger.info(
            "Strict training seed derived from model+dataset hashes: seed=%d",
            derived_seed,
        )
        return self.train_from_dataset(
            model_path=resolved_model_path,
            dataset_path=resolved_dataset_path,
            output_path=output_path,
            eval_dataset_path=eval_dataset_path,
            seed=derived_seed,
            auto_regime=True,
        )

    def train_from_dataset_research(
        self,
        model_path: str | Path,
        dataset_path: str | Path,
        **kwargs: Any,
    ) -> DatasetTrainResult:
        """Research entrypoint exposing non-strict controls."""
        return self.train_from_dataset(
            model_path=model_path,
            dataset_path=dataset_path,
            **kwargs,
        )

    def train_from_dataset(
        self,
        model_path: str | Path,
        dataset_path: str | Path,
        output_path: str | Path | None = None,
        init_adapter_path: str | Path | None = None,
        eval_dataset_path: str | Path | None = None,
        seq_length: int | None = None,
        lr_override: float | None = None,
        seed: int = 42,
        topo_monitor: bool = False,
        dim_monitor: bool = False,
        paired: bool | None = None,
        constraint_state_override: ConstraintState | None = None,
        format_projection: bool = False,
        narrow_dataset_path: str | Path | None = None,
        augmented_dataset_path: str | Path | None = None,
        online_eval: bool = False,
        online_eval_n_problems: int | None = None,
        entropy_regularization: bool = False,
        outcome_training: bool = False,
        outcome_n_problems: int | None = None,
        auto_regime: bool = True,
        regime_n_problems: int | None = None,
        # Answer-span masked CE training
        answer_mask: bool = False,
        retention_dataset_path: str | Path | None = None,
        # Sub-epoch evaluation interval
        eval_interval: int | None = None,
        # Global EOS exclusion from CE
        eos_exclude: bool = False,
        # Research: override geometric scale bound (bypasses spectral safety)
        scale_bound_override: float | None = None,
        # Outer similarity monitoring (Kucukahmetler et al. 2026)
        rss_monitor: bool = False,
        # Ablation experiment params (research only, not CLI-exposed)
        entropy_floor_fraction: float | None = None,  # Research only — NOT in strict CLI
        kl_reference_penalty: bool = False,
        outcome_signal_density_gate: float = 0.0,
        outcome_post_eval: bool = False,
        outcome_rollback_on_degradation: bool = True,
        research_online_eval_stop_stage: str = "pre_outcome",
        research_outcome_selector: str = "all",
        research_online_eval_problem_set_path: str | Path | None = None,
        quantization_reference_model_path: str | Path | None = None,
        research_allow_quantization_crossing: bool = False,
        no_save: bool = False,
        max_iters_cap: int | None = None,
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
        if no_save:
            output_dir = None
        elif output_path is not None:
            output_dir = Path(output_path).expanduser().resolve()
        else:
            # Auto-derive: <model_parent>/adapters/<model_name>-nblora-<seed>
            # Encodes provenance (model), method (nblora), uniqueness (seed).
            output_dir = model_path.parent / "adapters" / f"{model_path.name}-nblora-{seed}"
            logger.info("Auto-derived output path: %s", output_dir)

        if research_online_eval_stop_stage not in {"pre_outcome", "post_outcome"}:
            raise ValueError(
                "research_online_eval_stop_stage must be "
                "'pre_outcome' or 'post_outcome'",
            )
        if research_outcome_selector not in {"all", "lost_only"}:
            raise ValueError(
                "research_outcome_selector must be 'all' or 'lost_only'",
            )

        # Emit training started progress event
        rp = self._progress_reporter
        if rp is not None:
            rp.training_started(str(model_path), str(dataset_path))

        # Deterministic training state for reproducible experiments.
        random.seed(seed)
        self._backend.random_seed(seed)
        logger.info("RNG seeded: seed=%d", seed)

        # 1. Load model + tokenizer
        logger.info("Loading model from %s", model_path)
        model, tokenizer = self._backend.load_model(str(model_path))

        quantization_precheck_result: dict[str, Any] | None = None
        if quantization_reference_model_path is not None:
            fp_reference_path = Path(quantization_reference_model_path).expanduser().resolve()
            if not fp_reference_path.exists():
                raise FileNotFoundError(
                    f"quantization_reference_model_path does not exist: {fp_reference_path}",
                )
            logger.info(
                "Running quantization Weyl precheck: reference=%s, candidate=%s",
                fp_reference_path,
                model_path,
            )
            fp_model, _ = self._backend.load_model(str(fp_reference_path))
            fp_weights = self._adapter.extract_weight_matrices(fp_model)
            candidate_weights = self._adapter.extract_weight_matrices(model)
            quantization_precheck_result = run_quantization_weyl_precheck(
                fp_weights=fp_weights,
                quantized_weights=candidate_weights,
                backend=self._backend,
            )
            crossing_detected = not bool(
                quantization_precheck_result.get("all_non_crossing", False),
            )
            logger.info(
                "Quantization precheck: layers=%d, crossing=%d, max(error/(gap/2))=%.6f",
                int(quantization_precheck_result.get("n_layers", 0)),
                int(quantization_precheck_result.get("n_crossing", 0)),
                float(quantization_precheck_result.get("max_error_over_gap_half", 0.0)),
            )
            if crossing_detected and not research_allow_quantization_crossing:
                raise TrainingDerivationError(
                    failure_class="quantization_crossing_detected",
                    detail=(
                        "Quantization Weyl precheck measured spectral-gap crossing; "
                        "training is blocked unless research_allow_quantization_crossing=True."
                    ),
                    diagnostics={
                        "reference_model_path": str(fp_reference_path),
                        "candidate_model_path": str(model_path),
                        "n_layers": int(quantization_precheck_result.get("n_layers", 0)),
                        "n_crossing": int(quantization_precheck_result.get("n_crossing", 0)),
                        "crossing_layers": list(
                            quantization_precheck_result.get("crossing_layers", []),
                        ),
                    },
                )

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

        # 2. Load + split dataset
        logger.info("Loading dataset from %s", dataset_path)
        all_samples = load_jsonl_dataset(dataset_path)
        explicit_retention_samples: list[dict[str, Any]] | None = None
        if retention_dataset_path is not None:
            retention_path = Path(retention_dataset_path).expanduser().resolve()
            explicit_retention_samples = load_jsonl_dataset(retention_path)

        # Load eval data early so seq_length derivation covers ALL splits.
        # Without this, eval samples longer than train max get truncated silently.
        eval_samples_early: list[dict[str, Any]] | None = None
        if eval_path is not None:
            eval_samples_early = load_jsonl_dataset(eval_path)

        # Derive seq_length from data: max token length rounded up to SIMD width.
        # Max preserves ALL training signal — zero truncation by construction.
        if seq_length is None:
            token_lengths = []
            token_source_samples = list(all_samples)
            if explicit_retention_samples is not None:
                token_source_samples.extend(explicit_retention_samples)
            if eval_samples_early is not None:
                token_source_samples.extend(eval_samples_early)
            for s in token_source_samples:
                text = s.get("text")
                if isinstance(text, str) and text:
                    n = len(self._backend.encode_tokens(tokenizer, text))
                    if n > 0:
                        token_lengths.append(n)
            if not token_lengths:
                raise TrainingDerivationError(
                    failure_class="unavailable_measurement",
                    detail="seq_length derivation requires tokenizable text samples.",
                    diagnostics={"n_samples": len(all_samples)},
                )
            max_tokens = max(token_lengths)
            # Round up to SIMD width boundary for Metal kernel alignment.
            seq_length = (
                (max_tokens + _MLX_SIMD_WIDTH - 1) // _MLX_SIMD_WIDTH
            ) * _MLX_SIMD_WIDTH
            logger.info(
                "Derived seq_length=%d from data (max_tokens=%d, n_primary=%d, "
                "n_retention=%d, n_eval=%d, SIMD_width=%d)",
                seq_length,
                max_tokens,
                len(all_samples),
                len(explicit_retention_samples) if explicit_retention_samples is not None else 0,
                len(eval_samples_early) if eval_samples_early is not None else 0,
                _MLX_SIMD_WIDTH,
            )

        validation_split_info: dict[str, Any] | None = None
        if eval_path is not None:
            train_samples = all_samples
            eval_samples = eval_samples_early  # already loaded above
            logger.info(
                "Using explicit eval split: %d train / %d eval",
                len(train_samples), len(eval_samples),
            )
            validation_split_info = {
                "method": "explicit_eval_dataset",
                "n_train": len(train_samples),
                "n_eval": len(eval_samples),
            }
        else:
            shuffled_samples = list(all_samples)
            random.Random(seed).shuffle(shuffled_samples)
            split_index, validation_split_info = self._derive_validation_split_from_pilot(
                model=model,
                tokenizer=tokenizer,
                samples=shuffled_samples,
                seq_length=seq_length,
            )
            eval_samples = shuffled_samples[:split_index]
            train_samples = shuffled_samples[split_index:]
            logger.info(
                "Using derived split from pilot loss variance: %d train / %d eval",
                len(train_samples), len(eval_samples),
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
            auto_retention_samples = self._collect_auto_retention(
                model=model,
                tokenizer=tokenizer,
                train_samples=train_samples,
                seq_length=seq_length,
                seed=seed,
                n_retention=None,
            )
            auto_retention_samples_collected = len(auto_retention_samples)
            if auto_retention_samples_collected > 0:
                n_train = len(train_samples)
                auto_fraction = auto_retention_samples_collected / (
                    auto_retention_samples_collected + n_train
                )
                train_samples = merge_datasets_with_fraction(
                    train_samples,
                    auto_retention_samples,
                    auto_fraction,
                )
                logger.info(
                    "Auto-retention replay enabled: n_train=%d n_retention=%d fraction=%.6f",
                    n_train,
                    auto_retention_samples_collected,
                    auto_fraction,
                )
            else:
                logger.info("Auto-retention replay skipped: no valid retention samples collected")

        # 2.6. Answer-masked dataset preparation
        answer_masked_train = None
        answer_masked_val = None
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

        train_dataset = self._adapter.prepare_dataset(train_samples, tokenizer)
        eval_dataset = self._adapter.prepare_dataset(eval_samples, tokenizer)
        if not train_dataset:
            raise ValueError("No valid training samples after tokenization")
        if not eval_dataset:
            raise ValueError("No valid eval samples after tokenization")

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

        # Use streaming analysis when available — processes one layer at a
        # time, releasing weights immediately.  Falls back to batch analysis
        # for adapters that don't implement the streaming method.
        if hasattr(self._adapter, "analyze_model_geometry_streaming"):
            geometries = self._adapter.analyze_model_geometry_streaming(
                model, use_randomized=True,
            )
            weights = {}  # not needed — optimizer config uses precomputed geometries
        else:
            weights = self._adapter.extract_weight_matrices(model)
            geometries = analyze_weight_geometries(weights, self._backend)

        # 4.5. Derive optimizer geometry config (ScaledGD epsilon, decay per layer)
        opt_config = derive_optimizer_geometry_config(
            weights,
            self._backend,
            geometries=geometries,
        )
        logger.info(
            "Geometric optimizer config: %d layers, base_lr=%.2e",
            opt_config.n_layers, opt_config.base_lr,
        )

        # 4.6. Collect base model activations for CKA verification (before injection)
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
            constraint_state = (
                constraint_state_override
                if constraint_state_override is not None
                else ConstraintState()
            )
            logger.info(
                "Constraint config: %s", constraint_config.to_dict(),
            )
            if constraint_state.frozen:
                logger.info(
                    "Frozen multipliers: %s (ablation mode)",
                    sorted(constraint_state.frozen),
                )

        # 5. Select targets (geometry decides: layers with tail_dims > 0)
        target_modules = select_target_modules(geometries)
        if not target_modules:
            raise ValueError("No targetable layers found from geometric analysis")

        # 5.5. Cross-projection rank coupling (geometry-derived)
        # Caps q_proj rank at k_proj tail_dims per attention layer.
        # Prevents query-space overshoot beyond key discriminability.
        coupled_ranks = compute_coupled_ranks(geometries, target_modules)

        # 5.6. Signal-rank ceiling via RMT Marchenko-Pastur separation.
        # Caps ranks at the intrinsic signal dimensionality of the training
        # data's activation space (typically 10–50, not tail_dims ~959).
        # Uses base_activations already collected for CKA (zero extra passes).
        signal_rank_results = compute_per_layer_signal_ranks(
            base_activations, self._backend,
        )
        if signal_rank_results:
            final_ranks = apply_signal_rank_ceiling(coupled_ranks, signal_rank_results)
            ceiling_label = "RMT signal-rank"
        else:
            # Fallback: data-rank ceiling if signal rank computation returned empty
            final_ranks = apply_data_rank_ceiling(
                coupled_ranks, n_samples=len(train_dataset),
            )
            ceiling_label = "data-rank (fallback)"

        uncapped_params = estimate_nb_lora_parameter_count(geometries, coupled_ranks)
        capped_params = estimate_nb_lora_parameter_count(geometries, final_ranks)
        n_rank_capped = sum(
            1 for key, rank in coupled_ranks.items()
            if final_ranks.get(key, rank) < rank
        )
        if n_rank_capped > 0 and uncapped_params > 0:
            logger.info(
                "%s ceiling applied: %d layers capped, "
                "params %d -> %d (%.2fx reduction)",
                ceiling_label,
                n_rank_capped,
                uncapped_params,
                capped_params,
                uncapped_params / max(capped_params, 1),
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
            scale_bound_override=scale_bound_override,
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

        if rp is not None:
            rp.training_geometry_complete(
                n_target_modules=len(target_modules),
                n_trainable_params=n_trainable_params,
                batch_size=batch_size,
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

        # 8. MASS: sigma_max and sigma_k_min for spectral ceiling (Weyl 1912)
        sigma_max = max(g.sigma_max for g in geometries.values() if g.sigma_max > 0)
        sigma_k_vals = [g.sigma_k for g in geometries.values() if g.sigma_k > 0]
        if not sigma_k_vals:
            raise TrainingDerivationError(
                failure_class="insufficient_adapter_geometry",
                detail="No positive sigma_k found across adapted layers.",
                diagnostics={"n_geometries": len(geometries)},
            )
        sigma_k_min = min(sigma_k_vals)
        logger.info(
            "MASS geometry: sigma_max=%.4e, sigma_k_min=%.4e, ceiling=%.4e",
            sigma_max, sigma_k_min, sigma_k_min / sigma_max,
        )

        # 8.5. Format bias projection hook (optional)
        gradient_hook = None
        if format_projection:
            gradient_hook = self._build_format_projection_hook(
                model, tokenizer,
                narrow_dataset_path=narrow_dataset_path,
                augmented_dataset_path=augmented_dataset_path,
            )

        # 8.9. Regime-selection setup: auto mode derives online-eval + outcome flags
        effective_online_eval = online_eval
        effective_online_eval_n_problems = online_eval_n_problems
        effective_entropy_regularization = entropy_regularization
        effective_outcome_training = outcome_training
        effective_outcome_n_problems = outcome_n_problems

        if auto_regime:
            # Derive regime_n_problems from Clopper-Pearson CI resolution when
            # not explicitly provided.  We find the minimum N such that the CI
            # half-width at the chance operating point is < chance_rate for every
            # problem type.  This is fully determined by the binomial distribution
            # and the per-type chance rates (data-derived from problem structure).
            if regime_n_problems is None:
                regime_n_problems = self._derive_regime_n_from_ci()
                logger.info(
                    "Derived regime_n_problems=%d from Clopper-Pearson CI resolution",
                    regime_n_problems,
                )
            if regime_n_problems <= 1:
                raise ValueError(
                    "auto_regime requires regime_n_problems > 1 "
                    "(confidence derivation uses alpha=1/N)."
                )
            if not online_eval:
                effective_online_eval = True
            if (
                online_eval_n_problems is None
                and research_online_eval_problem_set_path is None
            ):
                from modelcypher.core.domain.star.problem_generator import (
                    StarProblemGenerator,
                )

                effective_online_eval_n_problems = (
                    regime_n_problems * len(StarProblemGenerator.PROBLEM_TYPES)
                )
            logger.info(
                "Auto regime enabled: deriving online_eval/outcome/entropy flags "
                "from baseline with N=%d problems",
                regime_n_problems,
            )

        # 8.10. Online eval: create problems + measure baseline (optional)
        eval_problems = None
        eval_baseline_correct_ids: frozenset[str] = frozenset()
        baseline_result = None
        regime_result = None
        if effective_online_eval:
            from modelcypher.core.domain.training.online_eval import (
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
                    effective_online_eval_n_problems is not None
                    and effective_online_eval_n_problems != len(eval_problems)
                ):
                    raise ValueError(
                        "online_eval_n_problems does not match the loaded research "
                        "online eval problem set length",
                    )
                effective_online_eval_n_problems = len(eval_problems)
                logger.info(
                    "Loaded %d online eval problems from %s",
                    len(eval_problems),
                    eval_problem_path,
                )
            else:
                if effective_online_eval_n_problems is None:
                    raise ValueError(
                        "--online-eval requires --online-eval-n <N> "
                        "(number of eval problems is a compute budget choice, not derivable)"
                    )
                eval_seed = seed + _EVAL_SEED_OFFSET
                eval_problems = create_eval_problem_set(
                    n_problems=effective_online_eval_n_problems,
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
            logger.info(
                "Online eval baseline: %d/%d (%.1f%%)",
                baseline_result.n_correct,
                baseline_result.n_total,
                baseline_result.accuracy * 100,
            )

        # 8.10.1. Collect inference probe activations for dual-manifold CKA
        # (diagnostic-only). Model is post-injection but NB-LoRA starts at
        # identity, so activations are mathematically equivalent to base.
        inference_base_activations: dict[int, list] | None = None
        if eval_problems:
            inference_base_activations = self._collect_inference_probe_activations(
                model, tokenizer, eval_problems,
            )

        # 8.11. Auto regime selection: derive outcome/entropy from baseline geometry
        if auto_regime:
            if baseline_result is None:
                raise TrainingDerivationError(
                    failure_class="insufficient_baseline_measurement",
                    detail=(
                        "Auto-regime requires baseline online evaluation, but "
                        "no baseline result was available."
                    ),
                    diagnostics={
                        "auto_regime": True,
                        "effective_online_eval": effective_online_eval,
                        "effective_online_eval_n_problems": effective_online_eval_n_problems,
                    },
                )
            from modelcypher.core.domain.training.regime_selection import (
                select_training_regime,
            )

            try:
                regime_result = select_training_regime(baseline_result)
            except ValueError as exc:
                raise TrainingDerivationError(
                    failure_class="insufficient_baseline_measurement",
                    detail=(
                        "Auto-regime could not derive a valid training regime "
                        "from baseline measurements."
                    ),
                    diagnostics={
                        "reason": str(exc),
                        "baseline_n_total": baseline_result.n_total,
                        "baseline_n_correct": baseline_result.n_correct,
                        "baseline_per_type_total": dict(baseline_result.per_type_total),
                    },
                ) from exc

            if not outcome_training:
                effective_outcome_training = regime_result.use_outcome_training
                if effective_outcome_training and outcome_n_problems is None:
                    effective_outcome_n_problems = regime_n_problems

            if not entropy_regularization:
                effective_entropy_regularization = (
                    regime_result.use_entropy_regularization
                )

            if regime_result.ceiling_bound:
                effective_outcome_training = False
                effective_entropy_regularization = False

            logger.info(
                "Auto regime decision: global=%s confidence=%.4f "
                "(use_ce=%s, use_outcome_training=%s, "
                "use_entropy_regularization=%s)",
                regime_result.global_regime,
                regime_result.confidence_level,
                regime_result.use_ce,
                regime_result.use_outcome_training,
                regime_result.use_entropy_regularization,
            )
            logger.info(
                "Auto regime headroom: baseline_ci=[%.6f, %.6f], "
                "headroom_upper=%.6f, resolution=%.6f, ceiling_bound=%s",
                regime_result.baseline_ci_lower,
                regime_result.baseline_ci_upper,
                regime_result.headroom_upper,
                regime_result.headroom_resolution,
                regime_result.ceiling_bound,
            )
            if regime_result.ceiling_bound:
                logger.info(
                    "Auto regime ceiling override active: %s",
                    regime_result.ceiling_rationale,
                )
            for problem_type in sorted(regime_result.per_type):
                per_type = regime_result.per_type[problem_type]
                logger.info(
                    "  %s: %d/%d (acc=%.3f, ci=[%.3f, %.3f], chance=%.3f) -> %s",
                    problem_type,
                    per_type.n_correct,
                    per_type.n_total,
                    per_type.observed_accuracy,
                    per_type.ci_lower,
                    per_type.ci_upper,
                    per_type.chance_rate,
                    per_type.regime,
                )

        # 8.12. Outcome training: create problem set for REINFORCE (optional)
        outcome_problems = None
        if effective_outcome_training:
            if effective_outcome_n_problems is None:
                raise ValueError(
                    "--outcome requires --outcome-n <N> "
                    "(number of outcome problems is a compute budget choice, not derivable)"
                )
            from modelcypher.core.domain.training.online_eval import (
                create_eval_problem_set,
            )

            outcome_seed = seed + _OUTCOME_SEED_OFFSET
            all_outcome_problems = create_eval_problem_set(
                n_problems=effective_outcome_n_problems,
                seed=outcome_seed,
            )
            outcome_problems = all_outcome_problems
            if auto_regime and regime_result is not None:
                outcome_problems, ce_filtered_counts = (
                    self._filter_outcome_problems_by_regime(
                        all_outcome_problems,
                        regime_result.per_type,
                    )
                )
                logger.info(
                    "Auto regime outcome filter: retained %d/%d problems for REINFORCE "
                    "(dropped CE-only types: %s)",
                    len(outcome_problems),
                    len(all_outcome_problems),
                    ce_filtered_counts if ce_filtered_counts else "{}",
                )
            logger.info(
                "Created %d outcome problems for REINFORCE (seed=%d, derived from training seed+2)",
                len(outcome_problems), outcome_seed,
            )

        # 9. Train — ScaledGD + Weyl adapter saturation + validation loss stopping
        if rp is not None:
            rp.training_loop_started(max_iters=resolved_max_iters_cap)
        if (
            research_online_eval_stop_stage != "pre_outcome"
            or research_outcome_selector != "all"
        ):
            logger.info(
                "Research controls enabled: online_eval_stop_stage=%s, outcome_selector=%s",
                research_online_eval_stop_stage,
                research_outcome_selector,
            )
        train_start = time.time()
        losses, stop_reason, epoch_metrics = self._adapter.train_loop(
            model=model,
            train_dataset=train_dataset,
            batch_size=batch_size,
            seq_length=seq_length,
            max_iters=resolved_max_iters_cap,
            seed=seed,
            sigma_max=sigma_max,
            lr_override=lr_override,
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
            entropy_regularization=effective_entropy_regularization,
            online_eval_problems=eval_problems,
            online_eval_baseline_ids=eval_baseline_correct_ids,
            outcome_training=effective_outcome_training,
            outcome_problems=outcome_problems,
            answer_masked_dataset=answer_masked_train,
            answer_masked_eval=answer_masked_val,
            eval_interval=eval_interval,
            eos_exclude=eos_exclude,
            rss_monitor=rss_monitor,
            base_activations=base_activations if rss_monitor else None,
            entropy_floor_fraction=entropy_floor_fraction,
            kl_reference_penalty=kl_reference_penalty,
            outcome_signal_density_gate=outcome_signal_density_gate,
            outcome_post_eval=(
                outcome_post_eval or outcome_rollback_on_degradation
            ),
            outcome_rollback_on_degradation=outcome_rollback_on_degradation,
            research_online_eval_stop_stage=research_online_eval_stop_stage,
            research_outcome_selector=research_outcome_selector,
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
            rp.training_verification_started()
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
            logger.info(
                "CKA verification: min=%.4f, mean=%.4f (%d probes, %d layers)",
                min_cka, mean_cka,
                cka_result.get("n_probes", 0),
                len(cka_result.get("per_layer_cka", {})),
            )
            # Inference-manifold CKA (diagnostic)
            inference_min_cka = cka_result.get("inference_min_cka")
            inference_mean_cka = cka_result.get("inference_mean_cka")
            inference_per_layer_cka = cka_result.get("inference_per_layer_cka")
            inference_per_layer_gram_epsilon = cka_result.get(
                "inference_per_layer_gram_epsilon",
            )
            inference_min_cka_layer = cka_result.get("inference_min_cka_layer")

        if rp is not None:
            rp.training_verification_complete(
                spectral_bounds_ok=spectral_bounds_ok,
                min_cka=min_cka,
                mean_cka=mean_cka,
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
            )
            saved_adapter_path = str(saved_path)

        regime_global = None
        regime_per_type = None
        regime_headroom = None
        if regime_result is not None:
            regime_global = regime_result.global_regime
            regime_per_type = {
                problem_type: {
                    "problem_type": per_type.problem_type,
                    "n_correct": per_type.n_correct,
                    "n_total": per_type.n_total,
                    "observed_accuracy": per_type.observed_accuracy,
                    "ci_lower": per_type.ci_lower,
                    "ci_upper": per_type.ci_upper,
                    "chance_rate": per_type.chance_rate,
                    "regime": per_type.regime,
                    "rationale": per_type.rationale,
                }
                for problem_type, per_type in regime_result.per_type.items()
            }
            regime_headroom = {
                "baseline_ci_lower": regime_result.baseline_ci_lower,
                "baseline_ci_upper": regime_result.baseline_ci_upper,
                "headroom_upper": regime_result.headroom_upper,
                "headroom_resolution": regime_result.headroom_resolution,
                "ceiling_bound": regime_result.ceiling_bound,
                "ceiling_rationale": regime_result.ceiling_rationale,
                "confidence_level": regime_result.confidence_level,
                "n_total": regime_result.n_total,
            }

        # Extract max gain ratio across all epochs for stability certificate
        max_gain_ratio: float | None = None
        if epoch_metrics:
            gain_ratios = [
                m.max_effective_gain_ratio
                for m in epoch_metrics
                if m.max_effective_gain_ratio is not None
            ]
            if gain_ratios:
                max_gain_ratio = max(gain_ratios)

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
            spectral_bounds_ok=spectral_bounds_ok,
            max_spectral_ratio=max_spectral_ratio,
            training_time_seconds=training_time_seconds,
            epoch_metrics=[m.to_dict() for m in epoch_metrics] if epoch_metrics else None,
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
            optimizer_type="cayley_stiefel",
            rss_final_cosine=rss_final_cosine,
            rss_final_spearman=rss_final_spearman,
            rss_final_top1=rss_final_top1,
            regime_global=regime_global,
            regime_per_type=regime_per_type,
            regime_headroom=regime_headroom,
            quantization_precheck=quantization_precheck_result,
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
        )

    def _derive_strict_seed(self, model_path: Path, dataset_path: Path) -> int:
        """Derive deterministic seed from model artifacts and dataset bytes."""
        model_hash = self._hash_model_artifacts(model_path)
        dataset_hash = self._hash_file(dataset_path)
        digest = hashlib.sha256(
            f"{model_hash}:{dataset_hash}".encode("utf-8"),
        ).digest()
        return int.from_bytes(digest[:4], "big", signed=False)

    def _derive_training_safety_cap(self, *, n_samples: int, batch_size: int) -> int:
        """Derive max-iteration cap from batch geometry and machine precision."""
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        eps = float(self._backend.finfo().eps)
        sqrt_eps = math.sqrt(eps)
        iters_per_epoch = math.ceil(n_samples / batch_size)

        # Loss-change resolvability floor from IEEE-754: once changes are below
        # sqrt(eps), additional epochs cannot be distinguished from roundoff.
        epochs_until_precision_floor = math.ceil(1.0 / sqrt_eps)
        return iters_per_epoch * epochs_until_precision_floor

    def _collect_auto_retention(
        self,
        model: Any,
        tokenizer: Any,
        train_samples: list[dict[str, Any]],
        seq_length: int,
        seed: int,
        n_retention: int | None = None,
    ) -> list[dict[str, str]]:
        """Collect retention samples by greedily decoding sampled train prompts.

        Prompt extraction is deterministic:
        1. ``text`` up to first newline
        2. Prompt tokens capped to ``seq_length`` so prompts are bounded only by
           the active sequence window, without additional arbitrary truncation.

        Generation is greedy (temp=0.0, top_p=1.0) with a backend-compat fallback
        that omits unsupported kwargs.

        n_retention defaults to len(train_samples) — one retention per training
        prompt, maximally complete.
        """
        if n_retention is None:
            target_count = len(train_samples)
        else:
            target_count = min(len(train_samples), max(int(n_retention), 0))

        if target_count <= 0:
            return []

        sampled = random.Random(seed).sample(train_samples, target_count)
        retention_samples: list[dict[str, str]] = []
        eos_id = getattr(tokenizer, "eos_token_id", None)

        def _encode_text(value: str) -> list[int | str]:
            if hasattr(tokenizer, "encode") and callable(getattr(tokenizer, "encode")):
                return list(tokenizer.encode(value))
            return list(self._backend.encode_tokens(tokenizer, value))

        def _decode_tokens(token_ids: list[int | str]) -> str:
            if hasattr(tokenizer, "decode") and callable(getattr(tokenizer, "decode")):
                return str(tokenizer.decode(token_ids))
            return str(self._backend.decode_tokens(tokenizer, token_ids))

        for sample in sampled:
            text = sample.get("text")
            if not isinstance(text, str) or not text:
                continue

            prompt = text.split("\n", 1)[0]
            if not prompt:
                continue

            # Bound prompt only by active sequence window.
            # Reserve one slot for EOS when the tokenizer uses one.
            content_cap = max(1, int(seq_length) - (1 if eos_id is not None else 0))
            prompt_tokens = _encode_text(prompt)[:content_cap]
            if not prompt_tokens:
                continue

            prompt = _decode_tokens(prompt_tokens)
            if not isinstance(prompt, str) or not prompt:
                continue

            try:
                generated = self._backend.generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=max(1, content_cap - len(prompt_tokens)),
                    temp=0.0,
                    top_p=1.0,
                )
            except TypeError:
                generated = self._backend.generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=max(1, content_cap - len(prompt_tokens)),
                )

            if not isinstance(generated, str):
                continue

            completion = generated[len(prompt):] if generated.startswith(prompt) else generated
            retention_text = prompt + completion
            retention_tokens = _encode_text(retention_text)
            if not retention_tokens:
                continue
            if len(retention_tokens) > content_cap:
                retention_tokens = retention_tokens[:content_cap]
                retention_text = _decode_tokens(retention_tokens)
            if isinstance(retention_text, str) and retention_text:
                retention_samples.append({"text": retention_text})

        return retention_samples

    def _build_format_projection_hook(
        self,
        model: Any,
        tokenizer: Any,
        narrow_dataset_path: str | Path | None,
        augmented_dataset_path: str | Path | None,
    ) -> Any:
        """Build a gradient hook that projects out format bias.

        Computes mean gradients on narrow and augmented samples, derives the
        format bias direction, and returns a hook that removes it from each
        gradient step.

        Sample count for mean gradient estimation: uses all available samples
        from both datasets (matched to the smaller set for balanced estimation).

        Requires both narrow and augmented dataset paths.
        """
        if narrow_dataset_path is None or augmented_dataset_path is None:
            raise ValueError(
                "--format-projection requires both --narrow-data and "
                "--augmented-data to derive the format bias direction"
            )

        from modelcypher.core.domain.training.format_bias_projection import (
            compute_format_bias,
        )

        narrow_path = Path(narrow_dataset_path).expanduser().resolve()
        aug_path = Path(augmented_dataset_path).expanduser().resolve()

        narrow_samples = load_jsonl_dataset(narrow_path)
        augmented_samples = load_jsonl_dataset(aug_path)

        # Use all available samples, matched to the smaller set
        n_samples = min(len(narrow_samples), len(augmented_samples))

        logger.info(
            "Format projection: computing bias from %d narrow + %d augmented samples "
            "(matched to smaller set: %d)",
            len(narrow_samples), len(augmented_samples), n_samples,
        )

        # Compute mean gradients using the adapter
        mu_narrow = self._adapter.compute_mean_gradient(
            model, tokenizer, narrow_samples[:n_samples],
        )
        mu_augmented = self._adapter.compute_mean_gradient(
            model, tokenizer, augmented_samples[:n_samples],
        )

        decomp = compute_format_bias(mu_narrow, mu_augmented)

        logger.info(
            "Format bias decomposition: "
            "‖μ_format‖=%.6f, ‖μ_invariant‖=%.6f, "
            "cos(narrow,aug)=%.4f, format_fraction=%.4f, α_crit=%.4f",
            decomp.norm_format,
            decomp.norm_invariant,
            decomp.cos_narrow_aug,
            decomp.format_fraction,
            decomp.alpha_crit,
        )

        # Build the hook — adapter converts numpy v_format to framework array
        hook = self._adapter.build_projection_hook(decomp.v_format)
        return hook

    def _derive_probe_texts(
        self,
        eval_samples: list[dict[str, Any]],
        tokenizer: Any,
        seq_length: int,
    ) -> list[str]:
        """Derive monitor probe texts from eval samples.

        Uses ALL available eval samples.  Per-sample character truncation is
        derived from ``seq_length * median_chars_per_token`` measured on the
        actual dataset — no guessed constant.
        """
        # Measure chars/token from the actual eval data.
        ratios: list[float] = []
        for s in eval_samples:
            text = s.get("text")
            if not isinstance(text, str) or not text:
                continue
            n_tokens = len(self._backend.encode_tokens(tokenizer, text))
            if n_tokens > 0:
                ratios.append(len(text) / float(n_tokens))
        if not ratios:
            # Fallback: use full text (no truncation).
            return [s["text"] for s in eval_samples if isinstance(s.get("text"), str) and s["text"]]
        ratios.sort()
        mid = len(ratios) // 2
        median_cpt = ratios[mid] if len(ratios) % 2 == 1 else (ratios[mid - 1] + ratios[mid]) / 2.0

        char_budget = max(1, int(math.ceil(seq_length * median_cpt)))
        probes: list[str] = []
        for s in eval_samples:
            text = s.get("text")
            if isinstance(text, str) and text:
                probes.append(text[:char_budget])
        return probes

    def _collect_probe_activations(
        self,
        model: Any,
        tokenizer: Any,
        eval_samples: list[dict[str, Any]],
        seq_length: int | None = None,
    ) -> dict[int, list]:
        """Collect per-layer activations on probe texts for CKA comparison.

        Uses ALL eval samples. Per-sample truncation derived from seq_length
        and measured chars/token (no guessed constant).
        Raises TrainingDerivationError when verification probes are unavailable.
        """
        try:
            # Derive char truncation from data when seq_length is provided.
            if seq_length is not None:
                probe_texts = self._derive_probe_texts(eval_samples, tokenizer, seq_length)
            else:
                probe_texts = [
                    s["text"] for s in eval_samples
                    if isinstance(s.get("text"), str) and s["text"]
                ]

            if not probe_texts:
                raise TrainingDerivationError(
                    failure_class="unavailable_measurement",
                    detail="CKA verification requested but no probe texts are available.",
                    diagnostics={
                        "measurement": "cka_base_activations",
                        "n_eval_samples": len(eval_samples),
                    },
                )

            # Backend returns dict[layer_idx, Array[batch, seq, hidden]]
            # We collect one text at a time and mean-pool over seq dim
            activations: dict[int, list] = {}
            for text in probe_texts:
                acts = self._backend.collect_hidden_activations(
                    model, tokenizer, [text],
                )
                for layer_idx, act in acts.items():
                    # act: [1, seq, hidden] → mean over seq → [hidden]
                    pooled = self._backend.mean(act, axis=1)  # [1, hidden]
                    pooled = self._backend.reshape(pooled, (-1,))  # [hidden]
                    self._backend.eval(pooled)
                    activations.setdefault(layer_idx, []).append(pooled)

            if not activations:
                raise TrainingDerivationError(
                    failure_class="unavailable_measurement",
                    detail="CKA verification requested but hidden activations were unavailable.",
                    diagnostics={
                        "measurement": "cka_base_activations",
                        "n_probe_texts": len(probe_texts),
                    },
                )

            logger.info(
                "Collected base activations: %d probes, %d layers",
                len(probe_texts), len(activations),
            )
            return activations
        except TrainingDerivationError:
            raise
        except Exception as exc:
            raise TrainingDerivationError(
                failure_class="unavailable_measurement",
                detail="CKA verification failed while collecting base activations.",
                diagnostics={
                    "measurement": "cka_base_activations",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            ) from exc

    def _collect_inference_probe_activations(
        self,
        model: Any,
        tokenizer: Any,
        problems: list,
    ) -> dict[int, list]:
        """Collect per-layer activations on inference probe texts (StarProblem prompts).

        Uses the same prompt construction as evaluate_correctness so the
        activation geometry matches what the model sees during inference eval.
        Returns dict[layer_idx, list[pooled_activation]].
        """
        from modelcypher.core.domain.star.prompting import (
            build_forward_prompt,
            default_few_shot_examples,
        )

        n_demonstrations = len(default_few_shot_examples())
        prompts = [
            build_forward_prompt(p, demonstrations=n_demonstrations)
            for p in problems
        ]

        activations: dict[int, list] = {}
        for prompt in prompts:
            acts = self._backend.collect_hidden_activations(
                model, tokenizer, [prompt],
            )
            for layer_idx, act in acts.items():
                pooled = self._backend.mean(act, axis=1)
                pooled = self._backend.reshape(pooled, (-1,))
                self._backend.eval(pooled)
                activations.setdefault(layer_idx, []).append(pooled)

        logger.info(
            "Collected inference probe activations: %d prompts, %d layers",
            len(prompts), len(activations),
        )
        return activations

    def _verify_capability_preservation(
        self,
        model: Any,
        tokenizer: Any,
        base_activations: dict[int, list],
        eval_samples: list[dict[str, Any]],
        seq_length: int | None = None,
        inference_base_activations: dict[int, list] | None = None,
        inference_problems: list | None = None,
    ) -> dict[str, Any]:
        """CKA verification: does the adapted model preserve base representations?

        Computes linear CKA per-layer between base (pre-injection) and adapted
        (post-training) model activations on the same probe texts.  Uses ALL
        eval samples with data-derived character truncation.

        When inference_base_activations and inference_problems are provided,
        also computes inference-manifold CKA separately on StarProblem prompts.

        Returns dict with min_cka, mean_cka, per_layer_cka, n_probes,
        and optionally inference_* keys for dual-manifold diagnostics.
        """
        try:
            from modelcypher.core.domain.geometry.cka import (
                compute_gram_perturbation_ratio,
                compute_linear_cka_from_activations,
            )
            from modelcypher.core.domain.geometry.null_space_accessibility import (
                aggregate_layer_accessibility,
                analyze_layer_null_observability,
                analyze_module_null_accessibility,
            )

            if seq_length is not None:
                probe_texts = self._derive_probe_texts(eval_samples, tokenizer, seq_length)
            else:
                probe_texts = [
                    s["text"] for s in eval_samples
                    if isinstance(s.get("text"), str) and s["text"]
                ]

            # Collect adapted model activations via Backend (port, not adapter)
            adapted_acts: dict[int, list] = {}
            for text in probe_texts:
                acts = self._backend.collect_hidden_activations(
                    model, tokenizer, [text],
                )
                for layer_idx, act in acts.items():
                    pooled = self._backend.mean(act, axis=1)
                    pooled = self._backend.reshape(pooled, (-1,))
                    self._backend.eval(pooled)
                    adapted_acts.setdefault(layer_idx, []).append(pooled)

            # Compute CKA and Gram perturbation per layer
            cka_scores: dict[int, float] = {}
            gram_epsilons: dict[int, float] = {}
            cka_bounds: dict[int, float] = {}
            base_stacks: dict[int, Any] = {}
            for layer_idx in base_activations:
                if layer_idx not in adapted_acts:
                    continue
                base_list = base_activations[layer_idx]
                adapted_list = adapted_acts[layer_idx]
                if len(base_list) != len(adapted_list) or len(base_list) < 2:
                    continue

                base_stack = self._backend.stack(base_list)
                adapted_stack = self._backend.stack(adapted_list)
                self._backend.eval(base_stack, adapted_stack)
                base_stacks[layer_idx] = base_stack

                cka = compute_linear_cka_from_activations(
                    base_stack, adapted_stack, self._backend,
                )
                cka_scores[layer_idx] = cka

                eps_layer, bound_layer = compute_gram_perturbation_ratio(
                    base_stack, adapted_stack, self._backend,
                )
                gram_epsilons[layer_idx] = eps_layer
                cka_bounds[layer_idx] = bound_layer

            per_layer_null_observability: dict[int, dict[str, float | int]] = {}
            for layer_idx, base_stack in base_stacks.items():
                per_layer_null_observability[layer_idx] = analyze_layer_null_observability(
                    base_stack,
                    self._backend,
                )

            per_module_null_accessibility: dict[str, dict[str, float | int]] | None = None
            per_layer_null_accessibility: dict[int, dict[str, float | int]] | None = None
            extract_delta_fn = getattr(self._adapter, "extract_lora_weight_deltas", None)
            if callable(extract_delta_fn) and base_stacks:
                try:
                    module_metrics: dict[str, dict[str, float | int]] = {}
                    delta_by_module = extract_delta_fn(model) or {}
                    for module_key, delta_w in dict(delta_by_module).items():
                        parts = module_key.split(".")
                        layer_idx: int | None = None
                        for idx, part in enumerate(parts):
                            if part == "layers" and idx + 1 < len(parts):
                                try:
                                    layer_idx = int(parts[idx + 1])
                                except ValueError:
                                    layer_idx = None
                                break
                        if layer_idx is None:
                            continue
                        layer_acts = base_stacks.get(layer_idx)
                        if layer_acts is None:
                            continue
                        module_metrics[module_key] = analyze_module_null_accessibility(
                            delta_w,
                            layer_acts,
                            self._backend,
                        )
                    if module_metrics:
                        per_module_null_accessibility = module_metrics
                        per_layer_null_accessibility = aggregate_layer_accessibility(
                            module_metrics,
                        )
                except Exception:
                    logger.exception("Null-space accessibility diagnostics unavailable")

            if not cka_scores:
                raise TrainingDerivationError(
                    failure_class="unavailable_measurement",
                    detail=(
                        "CKA verification requested but no comparable layers were available "
                        "between base and adapted activations."
                    ),
                    diagnostics={
                        "measurement": "cka_alignment",
                        "n_probes_used": len(probe_texts),
                        "n_base_layers": len(base_activations),
                        "n_adapted_layers": len(adapted_acts),
                    },
                )

            min_cka = min(cka_scores.values())
            mean_cka = sum(cka_scores.values()) / len(cka_scores)

            result_dict: dict[str, Any] = {
                "min_cka": min_cka,
                "mean_cka": mean_cka,
                "per_layer_cka": cka_scores,
                "per_layer_gram_epsilon": gram_epsilons,
                "per_layer_cka_bound": cka_bounds,
                "per_layer_null_observability": per_layer_null_observability,
                "per_layer_null_accessibility": per_layer_null_accessibility,
                "per_module_null_accessibility": per_module_null_accessibility,
                "n_probes": len(probe_texts),
            }

            # Inference-manifold CKA (diagnostic): separate CKA on StarProblem
            # prompts to measure geometry that eval probes may not span.
            if inference_base_activations and inference_problems:
                inf_adapted = self._collect_inference_probe_activations(
                    model, tokenizer, inference_problems,
                )
                inf_cka_scores: dict[int, float] = {}
                inf_gram_eps: dict[int, float] = {}
                for layer_idx in inference_base_activations:
                    if layer_idx not in inf_adapted:
                        continue
                    inf_base_list = inference_base_activations[layer_idx]
                    inf_adapted_list = inf_adapted[layer_idx]
                    if (
                        len(inf_base_list) != len(inf_adapted_list)
                        or len(inf_base_list) < 2
                    ):
                        continue
                    inf_base_stack = self._backend.stack(inf_base_list)
                    inf_adapted_stack = self._backend.stack(inf_adapted_list)
                    self._backend.eval(inf_base_stack, inf_adapted_stack)

                    inf_cka_scores[layer_idx] = compute_linear_cka_from_activations(
                        inf_base_stack, inf_adapted_stack, self._backend,
                    )
                    eps_inf, _ = compute_gram_perturbation_ratio(
                        inf_base_stack, inf_adapted_stack, self._backend,
                    )
                    inf_gram_eps[layer_idx] = eps_inf

                if inf_cka_scores:
                    result_dict["inference_per_layer_cka"] = inf_cka_scores
                    result_dict["inference_per_layer_gram_epsilon"] = inf_gram_eps
                    result_dict["inference_min_cka"] = min(inf_cka_scores.values())
                    result_dict["inference_mean_cka"] = (
                        sum(inf_cka_scores.values()) / len(inf_cka_scores)
                    )
                    result_dict["inference_min_cka_layer"] = min(
                        inf_cka_scores, key=inf_cka_scores.get,
                    )
                    logger.info(
                        "Inference-manifold CKA: min=%.4f, mean=%.4f (%d layers)",
                        result_dict["inference_min_cka"],
                        result_dict["inference_mean_cka"],
                        len(inf_cka_scores),
                    )

            return result_dict
        except TrainingDerivationError:
            raise
        except Exception as exc:
            raise TrainingDerivationError(
                failure_class="unavailable_measurement",
                detail="CKA verification failed during aligned-score computation.",
                diagnostics={
                    "measurement": "cka_alignment",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            ) from exc

    def _derive_validation_split_from_pilot(
        self,
        *,
        model: Any,
        tokenizer: Any,
        samples: list[dict[str, Any]],
        seq_length: int,
    ) -> tuple[int, dict[str, Any]]:
        """Derive validation split directly from streaming pilot loss measurements."""
        if not hasattr(self._adapter, "measure_sample_losses"):
            raise TrainingDerivationError(
                failure_class="unavailable_measurement",
                detail=(
                    "Pilot validation split requires per-sample loss measurement support "
                    "from the active training adapter."
                ),
                diagnostics={
                    "measurement": "pilot_loss_variance",
                    "adapter_type": type(self._adapter).__name__,
                },
            )

        n_total = len(samples)
        if n_total < 2:
            raise TrainingDerivationError(
                failure_class="insufficient_validation_resolution",
                detail="Validation split derivation needs at least two total samples.",
                diagnostics={"n_total": n_total},
            )

        finfo = self._backend.finfo()
        eps = float(getattr(finfo, "eps", math.ldexp(1.0, -23)))
        if not math.isfinite(eps) or eps <= 0.0:
            eps = math.ldexp(1.0, -23)
        sqrt_eps = math.sqrt(eps)

        mean_loss = 0.0
        m2 = 0.0
        # Structural minimum: 1 validation sample.  Welford loop raises
        # this when measured variance demands more.
        n_val_upper = 1
        final_variance = 0.0
        final_target_se = sqrt_eps
        pilot_steps = 0
        max_pilot = n_total - 1

        for i in range(1, max_pilot + 1):
            measured = self._adapter.measure_sample_losses(
                model=model,
                tokenizer=tokenizer,
                samples=[samples[i - 1]],
                seq_length=seq_length,
            )
            if len(measured) != 1:
                raise TrainingDerivationError(
                    failure_class="unavailable_measurement",
                    detail="Pilot loss measurement returned incomplete sample coverage.",
                    diagnostics={
                        "measurement": "pilot_loss_variance",
                        "expected_losses": 1,
                        "observed_losses": len(measured),
                        "pilot_step": i,
                    },
                )
            loss = measured[0]
            if not math.isfinite(loss):
                raise TrainingDerivationError(
                    failure_class="unavailable_measurement",
                    detail="Pilot loss measurement returned non-finite values.",
                    diagnostics={
                        "measurement": "pilot_loss_variance",
                        "pilot_step": i,
                        "loss_value": loss,
                    },
                )

            pilot_steps = i
            delta = loss - mean_loss
            mean_loss += delta / float(i)
            delta2 = loss - mean_loss
            m2 += delta * delta2

            variance = m2 / float(i - 1) if i > 1 else 0.0
            # Relative precision: resolve mean_loss to sqrt(eps) relative error.
            # When |mean_loss| < sqrt(eps) (near-zero = already converged),
            # use absolute precision eps (machine epsilon = irreducible floor).
            target_se = sqrt_eps * abs(mean_loss) if abs(mean_loss) > sqrt_eps else sqrt_eps * sqrt_eps
            n_val_req = max(1, int(math.ceil(variance / (target_se * target_se))))
            n_val_upper = max(n_val_upper, n_val_req)

            final_variance = variance
            final_target_se = target_se

            if i >= n_val_upper:
                break

        if n_val_upper >= n_total:
            raise TrainingDerivationError(
                failure_class="insufficient_validation_resolution",
                detail=(
                    "Pilot loss variance requires a validation split that leaves no "
                    "training samples."
                ),
                diagnostics={
                    "n_total": n_total,
                    "n_val_required": n_val_upper,
                    "pilot_steps": pilot_steps,
                    "pilot_mean_loss": mean_loss,
                    "pilot_variance": final_variance,
                    "target_se": final_target_se,
                },
            )

        return n_val_upper, {
            "method": "pilot_loss_variance_welford",
            "n_eval": n_val_upper,
            "n_train": n_total - n_val_upper,
            "pilot_steps": pilot_steps,
            "pilot_mean_loss": mean_loss,
            "pilot_variance": final_variance,
            "target_se": final_target_se,
            "sqrt_eps": sqrt_eps,
        }

    def _filter_outcome_problems_by_regime(
        self,
        outcome_problems: list[Any],
        per_type_regime: dict[str, Any],
    ) -> tuple[list[Any], dict[str, int]]:
        """Keep REINFORCE-capable problem types; drop CE-only types."""
        filtered: list[Any] = []
        dropped_counts: dict[str, int] = {}

        for problem in outcome_problems:
            problem_type = str(getattr(problem, "problem_type", "unknown"))
            per_type = per_type_regime.get(problem_type)
            regime = "ce" if per_type is None else str(getattr(per_type, "regime", "ce"))
            if regime == "ce":
                dropped_counts[problem_type] = dropped_counts.get(problem_type, 0) + 1
                continue
            filtered.append(problem)

        return filtered, dropped_counts

    def _derive_validation_split_from_losses(
        self,
        *,
        sample_losses: list[float],
        n_total: int,
    ) -> tuple[int, dict[str, Any]]:
        """Derive validation-set size from pilot loss variance (bounded one pass)."""
        if n_total < 2:
            raise TrainingDerivationError(
                failure_class="insufficient_validation_resolution",
                detail="Validation split derivation needs at least two total samples.",
                diagnostics={"n_total": n_total},
            )
        if len(sample_losses) != n_total:
            raise TrainingDerivationError(
                failure_class="unavailable_measurement",
                detail="Pilot-loss derivation received mismatched sample/loss lengths.",
                diagnostics={
                    "n_total": n_total,
                    "n_losses": len(sample_losses),
                },
            )

        finfo = self._backend.finfo()
        eps = float(getattr(finfo, "eps", math.ldexp(1.0, -23)))
        if not math.isfinite(eps) or eps <= 0.0:
            eps = math.ldexp(1.0, -23)
        sqrt_eps = math.sqrt(eps)

        mean_loss = 0.0
        m2 = 0.0
        # Structural minimum: 1 validation sample.  Welford loop raises
        # this when measured variance demands more.
        n_val_upper = 1
        final_variance = 0.0
        final_target_se = sqrt_eps
        pilot_steps = 0
        max_pilot = n_total - 1  # Keep at least one training sample.

        for i in range(1, max_pilot + 1):
            loss = sample_losses[i - 1]
            pilot_steps = i
            delta = loss - mean_loss
            mean_loss += delta / float(i)
            delta2 = loss - mean_loss
            m2 += delta * delta2

            variance = m2 / float(i - 1) if i > 1 else 0.0
            # Relative precision: resolve mean_loss to sqrt(eps) relative error.
            # When |mean_loss| < sqrt(eps) (near-zero = already converged),
            # use absolute precision eps (machine epsilon = irreducible floor).
            target_se = sqrt_eps * abs(mean_loss) if abs(mean_loss) > sqrt_eps else sqrt_eps * sqrt_eps
            n_val_req = max(1, int(math.ceil(variance / (target_se * target_se))))
            n_val_upper = max(n_val_upper, n_val_req)

            final_variance = variance
            final_target_se = target_se

            if i >= n_val_upper:
                break

        if n_val_upper >= n_total:
            raise TrainingDerivationError(
                failure_class="insufficient_validation_resolution",
                detail=(
                    "Pilot loss variance requires a validation split that leaves no "
                    "training samples."
                ),
                diagnostics={
                    "n_total": n_total,
                    "n_val_required": n_val_upper,
                    "pilot_steps": pilot_steps,
                    "pilot_mean_loss": mean_loss,
                    "pilot_variance": final_variance,
                    "target_se": final_target_se,
                },
            )

        return n_val_upper, {
            "method": "pilot_loss_variance_welford",
            "n_eval": n_val_upper,
            "n_train": n_total - n_val_upper,
            "pilot_steps": pilot_steps,
            "pilot_mean_loss": mean_loss,
            "pilot_variance": final_variance,
            "target_se": final_target_se,
            "sqrt_eps": sqrt_eps,
        }

    def _write_geometry_manifest(
        self,
        *,
        adapter_dir: Path,
        model_path: Path,
        target_modules: list[str],
        geometries: dict[str, Any],
    ) -> None:
        """Persist geometry prerequisites needed by strict STaR composition."""
        module_geometry: dict[str, dict[str, Any]] = {}
        for module in sorted(target_modules):
            geom = geometries.get(module)
            if geom is None:
                raise TrainingDerivationError(
                    failure_class="insufficient_adapter_geometry",
                    detail="Missing layer geometry while writing adapter manifest.",
                    diagnostics={"missing_module": module},
                )
            module_geometry[module] = {
                "sigma_k": float(geom.sigma_k),
                "sigma_max": float(geom.sigma_max),
                "tail_dims": int(geom.tail_dims),
                "spectral_gap": float(geom.spectral_gap),
            }

        manifest = {
            "base_model_hash": self._hash_model_artifacts(model_path),
            "target_modules": sorted(target_modules),
            "sigma_k_by_module": {
                module: values["sigma_k"] for module, values in module_geometry.items()
            },
            "module_geometry": module_geometry,
            "derivation": {
                "source": "analyze_weight_geometries",
                "sigma_k_definition": (
                    "Structural-rank boundary singular value from Shannon effective rank."
                ),
                "created_unix_seconds": int(time.time()),
            },
        }
        manifest_path = adapter_dir / "geometry_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

    def _hash_model_artifacts(self, model_path: Path) -> str:
        """Compute SHA256 over model config + safetensors for adapter provenance."""
        digest = hashlib.sha256()
        files: list[Path] = []
        config = model_path / "config.json"
        if config.exists():
            files.append(config)
        index_file = model_path / "model.safetensors.index.json"
        if index_file.exists():
            files.append(index_file)
        files.extend(sorted(model_path.glob("*.safetensors")))

        if not files:
            for file_path in sorted(p for p in model_path.rglob("*") if p.is_file()):
                digest.update(str(file_path.relative_to(model_path)).encode("utf-8"))
                with file_path.open("rb") as handle:
                    while True:
                        chunk = handle.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
            return digest.hexdigest()

        for file_path in files:
            digest.update(str(file_path.relative_to(model_path)).encode("utf-8"))
            with file_path.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
        return digest.hexdigest()

    def _hash_file(self, file_path: Path) -> str:
        """Compute SHA256 over a single file."""
        digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _derive_regime_n_from_ci(self) -> int:
        """Derive regime problem count from Clopper-Pearson CI resolution.

        For each problem type, finds the minimum N such that the CI half-width
        at the chance operating point (k = round(chance * N)) is less than the
        chance rate itself.  This ensures the CI can resolve whether the model
        is above chance for that type.

        The CI is Clopper-Pearson exact binomial (1934, Biometrika 26(4))
        with alpha = 1/N (data-derived confidence level).

        Returns the max across all types — the strictest requirement.
        """
        from scipy.stats import beta as beta_dist

        from modelcypher.core.domain.training.regime_selection import (
            DEFAULT_PROBLEM_TYPE_CHANCE_RATES,
        )

        eps_f64 = math.ulp(1.0)
        machine_limit_n = int(math.ceil(1.0 / eps_f64))

        per_type_n: dict[str, int] = {}
        for problem_type, chance_rate in DEFAULT_PROBLEM_TYPE_CHANCE_RATES.items():
            chance = float(chance_rate)

            def ci_resolves(n: int) -> bool:
                alpha = 1.0 / n
                if chance <= 0.0:
                    # Exact-match: derive at k=n (perfect signal). Any single
                    # correct answer distinguishes from chance=0.
                    k = n
                else:
                    k = max(0, min(n, int(round(chance * n))))
                lower = (
                    0.0
                    if k == 0
                    else float(beta_dist.ppf(alpha / 2.0, k, n - k + 1))
                )
                upper = (
                    1.0
                    if k == n
                    else float(beta_dist.ppf(1.0 - alpha / 2.0, k + 1, n - k))
                )
                if not (math.isfinite(lower) and math.isfinite(upper)):
                    return False
                half_width = (upper - lower) / 2.0
                # CI must resolve: half-width < max(chance, 1/n) to be useful.
                # When chance=0 (no-guess baseline), the smallest detectable
                # effect at sample size n is 1/n (one correct out of n).
                # Clopper-Pearson: CI_upper for k=0 ≈ 1-(α)^(1/n) ≈ -ln(α)/n.
                # With α ≈ 1/n, target ≈ 1/n.
                target = chance if chance > 0.0 else 1.0 / n
                return half_width <= target

            lower_n = 1
            upper_n = 2
            while upper_n < machine_limit_n and not ci_resolves(upper_n):
                lower_n = upper_n
                upper_n = min(machine_limit_n, upper_n * 2)

            if not ci_resolves(upper_n):
                raise TrainingDerivationError(
                    failure_class="unavailable_measurement",
                    detail=(
                        f"Regime-N CI derivation failed for type={problem_type} "
                        f"(chance={chance:.3f}): CI did not resolve before "
                        f"machine precision limit (n={machine_limit_n})."
                    ),
                    diagnostics={
                        "problem_type": problem_type,
                        "chance_rate": chance,
                        "machine_limit_n": machine_limit_n,
                    },
                )

            while upper_n - lower_n > 1:
                mid_n = lower_n + (upper_n - lower_n) // 2
                if ci_resolves(mid_n):
                    upper_n = mid_n
                else:
                    lower_n = mid_n

            per_type_n[problem_type] = upper_n

        if not per_type_n:
            raise TrainingDerivationError(
                failure_class="unavailable_measurement",
                detail="No problem types available for regime-N CI derivation.",
                diagnostics={},
            )
        return max(per_type_n.values())


__all__ = ["DatasetTrainResult", "DatasetTrainingService"]
