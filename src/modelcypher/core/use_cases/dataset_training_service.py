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
- Batch size (B_crit = 1/SNR from gradient noise)
- When to stop (loss stability)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.dataset_loading import load_jsonl_dataset
from modelcypher.core.domain.training.geometric_lora import (
    analyze_weight_geometries,
    select_target_modules,
)

if TYPE_CHECKING:
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

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a JSON-serializable dictionary."""
        return {
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
        }


class DatasetTrainingService:
    """Train LoRA adapters from text datasets using NB-LoRA.

    One method. Geometry decides everything. Bounds by construction.
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
    ) -> DatasetTrainResult:
        """Train an NB-LoRA adapter from a JSONL dataset.

        All parameters are geometry-derived except the safety cap (max_iters).
        """
        model_path = Path(model_path).expanduser().resolve()
        dataset_path = Path(dataset_path).expanduser().resolve()
        eval_path = Path(eval_dataset_path).expanduser().resolve() if eval_dataset_path else None
        output_dir = Path(output_path).expanduser().resolve() if output_path else None

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

        # Eval uses a fixed small batch size (doesn't affect training dynamics)
        eval_batch_size = 2

        # 3. Baseline eval
        baseline_loss, baseline_ppl = self._adapter.evaluate_loss(
            model=model, dataset=eval_dataset, tokenizer=tokenizer,
            batch_size=eval_batch_size, seq_length=seq_length, n_batches=eval_batches,
        )

        # 4. Analyze geometry — this IS the configuration
        weights = self._adapter.extract_weight_matrices(model)
        geometries = analyze_weight_geometries(weights, self._backend)

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

        # Freeze base, unfreeze NB-LoRA params
        self._adapter.freeze_and_apply_lora(model)

        n_trainable_params = int(
            sum(param.size for _, param in self._backend.tree_flatten(model.trainable_parameters()))
        )

        # 7. Derive batch size from gradient noise: B_crit = 1/SNR
        batch_size = self._adapter.derive_critical_batch_size(
            model, train_dataset, seq_length,
        )
        logger.info("Geometry-derived batch size: %d", batch_size)

        # 8. sigma_max for spectral LR fallback (train_loop measures Hessian first)
        sigma_max = max(g.sigma_max for g in geometries.values() if g.sigma_max > 0)

        # 9. Train — one loop, validation loss decides when to stop
        train_start = time.time()
        losses, stop_reason = self._adapter.train_loop(
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
            }
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
        )


__all__ = ["DatasetTrainResult", "DatasetTrainingService"]
