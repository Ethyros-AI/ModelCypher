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
- Rank per layer (tail_dims = null-space capacity)
- Scale bound (sigma_k / 2 * safety_margin)
- Learning rate (1/L where L = λ_max(Hessian), fallback 1/σ_max)
- Optimizer preconditioning (ScaledGD: condition-number-free on rank-r manifold)
- Batch size (B_crit = 1/SNR from gradient noise)
- When to stop (val loss convergence, Weyl budget exhaustion, loss stability)
- Post-training verification (CKA alignment, spectral bounds)
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.dataset_loading import load_jsonl_dataset
from modelcypher.core.domain.training.geometric_lora import (
    analyze_weight_geometries,
    select_target_modules,
)
from modelcypher.core.domain.training.geometric_optimizer import (
    derive_optimizer_geometry_config,
)

if TYPE_CHECKING:
    from modelcypher.core.domain.training.constraint_config import ConstraintState
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)


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
    # G3: Weyl budget monitoring
    budget_median_ratio: float | None = None
    # G6: Optimizer type used
    optimizer_type: str = "cayley_riemannian"

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
        }
        if self.epoch_metrics is not None:
            result["epoch_metrics"] = self.epoch_metrics
        if self.min_cka is not None:
            result["min_cka"] = self.min_cka
        if self.mean_cka is not None:
            result["mean_cka"] = self.mean_cka
        if self.budget_median_ratio is not None:
            result["budget_median_ratio"] = self.budget_median_ratio
        return result


class DatasetTrainingService:
    """Train LoRA adapters from text datasets using NB-LoRA.

    One method. Geometry decides everything. Bounds by construction.
    ScaledGD preconditioning. Weyl budget monitoring. CKA verification.
    """

    def __init__(self, adapter: Any, backend: "Backend"):
        self._adapter = adapter
        self._backend = backend

    def train_from_dataset(
        self,
        model_path: str | Path,
        dataset_path: str | Path,
        output_path: str | Path | None = None,
        eval_dataset_path: str | Path | None = None,
        max_iters: int = 10000,
        seq_length: int = 256,
        lr_override: float | None = None,
        deep: bool = False,
        safety_margin: float = 0.9,
        seed: int = 42,
        eval_batches: int = 10,
        adaptive_lr: bool = True,
        lr_monotonic: bool = False,
        lipschitz_batches: int = 3,
        topo_monitor: bool = False,
        dim_monitor: bool = False,
        paired: bool | None = None,
        constraint_state_override: ConstraintState | None = None,
        format_projection: bool = False,
        narrow_dataset_path: str | Path | None = None,
        augmented_dataset_path: str | Path | None = None,
    ) -> DatasetTrainResult:
        """Train an NB-LoRA adapter from a JSONL dataset.

        All parameters are geometry-derived except the safety cap (max_iters).
        If paired=True (or auto-detected from data), uses constrained training
        with answer-only CE, invariance, separation, and geodesic constraints.
        """
        model_path = Path(model_path).expanduser().resolve()
        dataset_path = Path(dataset_path).expanduser().resolve()
        eval_path = Path(eval_dataset_path).expanduser().resolve() if eval_dataset_path else None
        output_dir = Path(output_path).expanduser().resolve() if output_path else None

        # Deterministic training state for reproducible experiments.
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except Exception:
            pass
        self._backend.random_seed(seed)
        logger.info("RNG seeded: seed=%d", seed)

        # 1. Load model + tokenizer
        logger.info("Loading model from %s", model_path)
        model, tokenizer = self._backend.load_model(str(model_path))

        # 2. Load + split dataset
        logger.info("Loading dataset from %s", dataset_path)
        all_samples = load_jsonl_dataset(dataset_path)

        if eval_path is not None:
            train_samples = all_samples
            eval_samples = load_jsonl_dataset(eval_path)
            logger.info(
                "Using explicit eval split: %d train / %d eval",
                len(train_samples), len(eval_samples),
            )
        else:
            split_index = int(len(all_samples) * 0.8)
            train_samples = all_samples[:split_index]
            eval_samples = all_samples[split_index:]
            logger.info(
                "Using 80/20 split: %d train / %d eval",
                len(train_samples), len(eval_samples),
            )

        train_dataset = self._adapter.prepare_dataset(train_samples, tokenizer)
        eval_dataset = self._adapter.prepare_dataset(eval_samples, tokenizer)
        if not train_dataset:
            raise ValueError("No valid training samples after tokenization")
        if not eval_dataset:
            raise ValueError("No valid eval samples after tokenization")

        # Eval batch size: data-derived
        eval_batch_size = max(1, min(4, len(eval_dataset) // max(1, eval_batches)))

        # 3. Baseline eval
        baseline_loss, baseline_ppl = self._adapter.evaluate_loss(
            model=model, dataset=eval_dataset, tokenizer=tokenizer,
            batch_size=eval_batch_size, seq_length=seq_length, n_batches=eval_batches,
        )

        # 4. Analyze geometry — this IS the configuration
        weights = self._adapter.extract_weight_matrices(model)
        geometries = analyze_weight_geometries(weights, self._backend)

        # 4.5. Derive optimizer geometry config (ScaledGD epsilon, decay per layer)
        opt_config = derive_optimizer_geometry_config(weights, self._backend)
        logger.info(
            "Geometric optimizer config: %d layers, base_lr=%.2e",
            opt_config.n_layers, opt_config.base_lr,
        )

        # 4.6. Collect base model activations for CKA verification (before injection)
        base_activations = self._collect_probe_activations(
            model, tokenizer, eval_samples,
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
            from modelcypher.core.domain.training.loop_preservation import (
                select_layers_to_sample,
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

            # Determine target layers for hidden state collection
            n_model_layers = self._backend.get_num_layers(model)
            target_layers = select_layers_to_sample(n_model_layers)
            # Also add layers 11-15 for reasoning tail guardrail (if model has them)
            tail_layers = [i for i in range(11, min(16, n_model_layers))]
            target_layers = sorted(set(target_layers + tail_layers))

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

        # 5. Select targets (geometry decides)
        if deep:
            target_modules = list(geometries.keys())
        else:
            target_modules = select_target_modules(geometries)
        if not target_modules:
            raise ValueError("No targetable layers found from geometric analysis")

        # 6. Inject NB-LoRA (bounds by construction)
        n_lora_layers = self._adapter.inject_nb_lora(
            model, geometries, target_modules, safety_margin=safety_margin,
        )
        if n_lora_layers <= 0:
            raise ValueError("No NB-LoRA layers were injected")

        # 6b. Log per-layer capacity at injection time
        for mod_name in target_modules:
            geom = geometries.get(mod_name)
            if geom is None:
                continue
            logger.info(
                "Injected %s: rank=%d, tail_dims=%d, shannon_eff_rank=%.1f, "
                "sigma_k=%.6f, scale_bound=%.6f, capacity_util=%.3f",
                mod_name,
                geom.tail_dims,
                geom.tail_dims,
                geom.shannon_effective_rank,
                geom.sigma_k,
                geom.sigma_k / 2.0 * safety_margin,
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
        # Constrained training needs ≥8 samples per batch for
        # both invariance pairs (same logic) and counterfactual pairs (same template)
        if use_constraints or use_geometric_reshape:
            batch_size = max(batch_size, 8)
        logger.info("Geometry-derived batch size: %d", batch_size)

        # 8. sigma_max for spectral LR fallback (train_loop measures Hessian first)
        sigma_max = max(g.sigma_max for g in geometries.values() if g.sigma_max > 0)

        # 8.5. Format bias projection hook (optional)
        gradient_hook = None
        if format_projection:
            gradient_hook = self._build_format_projection_hook(
                model, tokenizer,
                narrow_dataset_path=narrow_dataset_path,
                augmented_dataset_path=augmented_dataset_path,
                n_samples=40,
            )

        # 9. Train — ScaledGD + Weyl budget + validation loss stopping
        train_start = time.time()
        losses, stop_reason, epoch_metrics = self._adapter.train_loop(
            model=model,
            train_dataset=train_dataset,
            batch_size=batch_size,
            seq_length=seq_length,
            max_iters=max_iters,
            seed=seed,
            sigma_max=sigma_max,
            lr_override=lr_override,
            eval_dataset=eval_dataset,
            eval_batches=eval_batches,
            adaptive_lr=adaptive_lr,
            lr_monotonic=lr_monotonic,
            lipschitz_batches=lipschitz_batches,
            tokenizer=tokenizer,
            opt_config=opt_config,
            topo_monitor=topo_monitor,
            topo_probe_texts=[s["text"][:200] for s in eval_samples[:5]] if topo_monitor else None,
            dim_monitor=dim_monitor,
            dim_probe_texts=[s["text"][:300] for s in eval_samples[:10]] if dim_monitor else None,
            constraint_config=constraint_config,
            constraint_state=constraint_state,
            paired_dataset=paired_train_dataset,
            logic_groups=logic_groups,
            template_groups=template_groups,
            geometric_reshape=use_geometric_reshape,
            gradient_hook=gradient_hook,
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

        # 10. Post-training eval
        post_loss, post_ppl = self._adapter.evaluate_loss(
            model=model, dataset=eval_dataset, tokenizer=tokenizer,
            batch_size=eval_batch_size, seq_length=seq_length, n_batches=eval_batches,
        )

        # 11. Verify bounds (should always pass — by construction)
        spectral_bounds_ok, max_spectral_ratio, _ = self._adapter.verify_bounds(model)

        # 11.5. CKA verification — does the adapted model preserve base representations?
        min_cka = None
        mean_cka = None
        if base_activations:
            cka_result = self._verify_capability_preservation(
                model, tokenizer, base_activations, eval_samples,
            )
            min_cka = cka_result.get("min_cka")
            mean_cka = cka_result.get("mean_cka")
            logger.info(
                "CKA verification: min=%.4f, mean=%.4f (%d probes, %d layers)",
                min_cka or 0.0, mean_cka or 0.0,
                cka_result.get("n_probes", 0),
                len(cka_result.get("per_layer_cka", {})),
            )

        # Extract budget median ratio from last epoch metrics
        budget_median_ratio = None
        if epoch_metrics:
            last = epoch_metrics[-1]
            if hasattr(last, "budget_median_ratio"):
                budget_median_ratio = last.budget_median_ratio

        # 12. Save if requested
        saved_adapter_path: str | None = None
        if output_dir is not None:
            metadata = {
                "base_model_path": str(model_path),
                "stop_reason": stop_reason,
                "n_lora_layers": str(n_lora_layers),
                "train_iters": str(train_iters),
                "method": "nb_lora_cayley",
                "safety_margin": str(safety_margin),
                "optimizer": "cayley_riemannian",
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
            saved_adapter_path = str(saved_path)

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
            budget_median_ratio=budget_median_ratio,
            optimizer_type="cayley_riemannian",
        )

    def _build_format_projection_hook(
        self,
        model: Any,
        tokenizer: Any,
        narrow_dataset_path: str | Path | None,
        augmented_dataset_path: str | Path | None,
        n_samples: int = 40,
    ) -> Any:
        """Build a gradient hook that projects out format bias.

        Computes mean gradients on narrow and augmented samples, derives the
        format bias direction, and returns a hook that removes it from each
        gradient step.

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

        logger.info(
            "Format projection: computing bias from %d narrow + %d augmented samples",
            min(n_samples, len(narrow_samples)),
            min(n_samples, len(augmented_samples)),
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

    def _collect_probe_activations(
        self,
        model: Any,
        tokenizer: Any,
        eval_samples: list[dict[str, Any]],
        n_probes: int = 20,
    ) -> dict[int, list]:
        """Collect per-layer activations on probe texts for CKA comparison.

        Uses Backend.collect_hidden_activations() directly (port, not adapter).
        Returns dict[layer_idx, list[activation_array]] or empty dict on failure.
        """
        try:
            probe_texts = [s["text"][:200] for s in eval_samples[:n_probes]]

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

            logger.info(
                "Collected base activations: %d probes, %d layers",
                len(probe_texts), len(activations),
            )
            return activations
        except Exception:
            logger.warning("Failed to collect base activations for CKA", exc_info=True)
            return {}

    def _verify_capability_preservation(
        self,
        model: Any,
        tokenizer: Any,
        base_activations: dict[int, list],
        eval_samples: list[dict[str, Any]],
        n_probes: int = 20,
    ) -> dict[str, Any]:
        """CKA verification: does the adapted model preserve base representations?

        Computes linear CKA per-layer between base (pre-injection) and adapted
        (post-training) model activations on the same probe texts.

        Returns dict with min_cka, mean_cka, per_layer_cka, n_probes.
        """
        try:
            from modelcypher.core.domain.geometry.cka import compute_linear_cka_from_activations

            probe_texts = [s["text"][:200] for s in eval_samples[:n_probes]]

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

            # Compute CKA per layer
            cka_scores: dict[int, float] = {}
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

                cka = compute_linear_cka_from_activations(
                    base_stack, adapted_stack, self._backend,
                )
                cka_scores[layer_idx] = cka

            if not cka_scores:
                return {"min_cka": None, "mean_cka": None, "per_layer_cka": {}, "n_probes": 0}

            min_cka = min(cka_scores.values())
            mean_cka = sum(cka_scores.values()) / len(cka_scores)

            return {
                "min_cka": min_cka,
                "mean_cka": mean_cka,
                "per_layer_cka": cka_scores,
                "n_probes": len(probe_texts),
            }
        except Exception:
            logger.warning("CKA verification failed", exc_info=True)
            return {"min_cka": None, "mean_cka": None, "per_layer_cka": {}, "n_probes": 0}


__all__ = ["DatasetTrainResult", "DatasetTrainingService"]
