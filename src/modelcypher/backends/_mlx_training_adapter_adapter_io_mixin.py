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

# ruff: noqa: F403,F405

"""Adapter IO and verification methods for :class:`MLXTrainingAdapter`."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from modelcypher.backends.mlx_training_adapter_core import *  # noqa: F403
from modelcypher.core.domain.training.identity import (
    GEOMETRIC_LORA_CONTROLLER,
    GEOMETRIC_LORA_INIT_METHOD_CAYLEY,
    GEOMETRIC_LORA_INIT_METHOD,
    GEOMETRIC_LORA_METHOD,
    GEOMETRIC_LORA_OPTIMIZER,
    GEOMETRIC_LORA_STOPPING,
)


class _MLXTrainingAdapterAdapterIOMixin:
    def verify_bounds(self, model) -> tuple[bool, float, list[dict[str, Any]]]:
        """Verify spectral bounds post-training.

        Should ALWAYS pass. If it doesn't, there's a mathematical bug.

        Returns (all_ok, max_ratio, details).
        """
        details: list[dict[str, Any]] = []
        max_ratio = 0.0

        for name, module in self._iter_nb_lora_modules(model):
            spectral_norm = module.get_spectral_norm()
            theoretical_max = 2.0 * module.scale_bound
            ratio = spectral_norm / theoretical_max if theoretical_max > 0 else float("inf")

            # SVD error bound (Demmel & Kahan 1990): relative error in
            # computed singular values ≤ sqrt(max(m,n)) * eps.
            _eps_f32 = math.ldexp(1.0, -23)
            _max_dim = max(int(module.A_tilde.shape[1]),
                           int(module.B_tilde.shape[1]))
            _svd_tol = math.sqrt(_max_dim) * _eps_f32
            details.append({
                "layer": name,
                "spectral_norm": spectral_norm,
                "theoretical_max": theoretical_max,
                "ratio": ratio,
                "ok": ratio <= 1.0 + _svd_tol,
            })
            max_ratio = max(max_ratio, ratio)

        all_ok = all(d["ok"] for d in details)

        if not all_ok:
            logger.error(
                "SPECTRAL BOUND VIOLATION: max_ratio=%.4f (should be <= 1.0). "
                "This is a mathematical bug in the Cayley transform.",
                max_ratio,
            )
        else:
            logger.info(
                "Spectral bounds verified: %d layers, max_ratio=%.4f (by construction)",
                len(details), max_ratio,
            )

        return all_ok, max_ratio, details

    def save_adapter(
        self,
        model,
        output_path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save LoRA adapter weights.

        Supports two adapter modes:
        - PiSSA-LoRA: saves lora_a, lora_b, and modified base weights directly.
          PiSSA modifies linear.weight (W_residual), so we must save those too.
        - NB-LoRA (legacy): converts Cayley-parameterized (A_tilde, B_tilde, S_raw)
          to standard (lora_a, lora_b) pairs with scale=1.0.
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self._has_pissa_lora(model):
            return self._save_pissa_adapter(model, output_dir, metadata)
        return self._save_nb_lora_adapter(model, output_dir, metadata)

    def _save_pissa_adapter(
        self,
        model,
        output_dir: Path,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save PiSSA-LoRA adapter: fuse weights and save full model.

        PiSSA modifies both the LoRA factors and the base linear.weight
        (W_residual). The cleanest output is fused weights — add the trained
        LoRA product back into the residual base weight.
        """
        from mlx_lm.tuner.lora import LoRALinear  # noqa: PLC0415

        adapter_weights: dict[str, Any] = {}
        target_modules: set[str] = set()
        discovered_ranks: list[int] = []
        per_layer_rank_map: dict[str, int] = {}

        for name, lora in self._iter_pissa_lora_modules(model):
            rank = int(lora.lora_a.shape[1])
            discovered_ranks.append(rank)

            key_base = name.replace(".weight", "")
            adapter_weights[f"{key_base}.lora_a"] = lora.lora_a
            adapter_weights[f"{key_base}.lora_b"] = lora.lora_b
            # PiSSA residual base weight — needed for correct reconstruction
            adapter_weights[f"{key_base}.linear.weight"] = lora.linear.weight
            target_modules.add(self._module_name_from_layer_key(name))
            per_layer_rank_map[name] = rank

        if not adapter_weights:
            raise ValueError("No PiSSA-LoRA layers found to export")

        global_rank = max(discovered_ranks)
        # Pad LoRA factors to global rank for compatibility
        for key in list(adapter_weights.keys()):
            arr = adapter_weights[key]
            if key.endswith(".lora_a"):
                if int(arr.shape[1]) < global_rank:
                    pad = mx.zeros((int(arr.shape[0]), global_rank - int(arr.shape[1])), dtype=arr.dtype)
                    adapter_weights[key] = mx.concatenate([arr, pad], axis=1)
            elif key.endswith(".lora_b") and ".linear." not in key:
                if int(arr.shape[0]) < global_rank:
                    pad = mx.zeros((global_rank - int(arr.shape[0]), int(arr.shape[1])), dtype=arr.dtype)
                    adapter_weights[key] = mx.concatenate([arr, pad], axis=0)

        mx.eval(*adapter_weights.values())

        metadata_str: dict[str, str] | None = None
        if metadata:
            metadata_str = {str(k): str(v) for k, v in metadata.items()}

        weights_path = output_dir / "adapters.safetensors"
        self._backend.save_safetensors(str(weights_path), adapter_weights, metadata=metadata_str)

        config = {
            "fine_tune_type": "lora",
            "type": GEOMETRIC_LORA_METHOD,
            "num_layers": int(self._backend.get_num_layers(model)),
            "lora_parameters": {
                "rank": int(global_rank),
                "scale": 1.0,
                "dropout": 0.0,
                "keys": sorted(target_modules),
            },
            "target_modules": sorted(target_modules),
            "rank": int(global_rank),
            "per_layer_ranks": per_layer_rank_map,
            "method": GEOMETRIC_LORA_METHOD,
            "init_method": GEOMETRIC_LORA_INIT_METHOD,
            "optimizer": GEOMETRIC_LORA_OPTIMIZER,
            "controller": GEOMETRIC_LORA_CONTROLLER,
            "stopping": GEOMETRIC_LORA_STOPPING,
        }
        if metadata:
            config["metadata"] = metadata

        config_path = output_dir / "adapter_config.json"
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)

        logger.info(
            "Saved PiSSA-LoRA adapter: %d layers, rank=%d, path=%s",
            len(discovered_ranks), global_rank, output_dir,
        )
        return output_dir

    def _save_nb_lora_adapter(
        self,
        model,
        output_dir: Path,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save NB-LoRA adapter in standard LoRA format.

        Converts Cayley-parameterized (A_tilde, B_tilde, S_raw) to standard
        (lora_a, lora_b) pairs with scale=1.0.
        """
        adapter_weights: dict[str, Any] = {}
        target_modules: set[str] = set()
        discovered_ranks: list[int] = []
        per_layer_rank_map: dict[str, int] = {}

        for name, module in self._iter_nb_lora_modules(model):
            lora_a, lora_b = module.to_standard_lora()
            rank = int(lora_a.shape[1])
            discovered_ranks.append(rank)

            key_base = name.replace(".weight", "")
            adapter_weights[f"{key_base}.lora_a"] = lora_a
            adapter_weights[f"{key_base}.lora_b"] = lora_b
            target_modules.add(self._module_name_from_layer_key(name))
            per_layer_rank_map[name] = rank

        if not adapter_weights:
            raise ValueError("No NB-LoRA layers found to export")

        global_rank = max(discovered_ranks)
        for key in list(adapter_weights.keys()):
            arr = adapter_weights[key]
            if key.endswith(".lora_a"):
                if int(arr.shape[1]) < global_rank:
                    pad = mx.zeros((int(arr.shape[0]), global_rank - int(arr.shape[1])), dtype=arr.dtype)
                    adapter_weights[key] = mx.concatenate([arr, pad], axis=1)
            elif key.endswith(".lora_b"):
                if int(arr.shape[0]) < global_rank:
                    pad = mx.zeros((global_rank - int(arr.shape[0]), int(arr.shape[1])), dtype=arr.dtype)
                    adapter_weights[key] = mx.concatenate([arr, pad], axis=0)

        mx.eval(*adapter_weights.values())

        metadata_str: dict[str, str] | None = None
        if metadata:
            metadata_str = {str(k): str(v) for k, v in metadata.items()}

        weights_path = output_dir / "adapters.safetensors"
        self._backend.save_safetensors(str(weights_path), adapter_weights, metadata=metadata_str)

        config = {
            "fine_tune_type": "lora",
            "type": GEOMETRIC_LORA_METHOD,
            "num_layers": int(self._backend.get_num_layers(model)),
            "lora_parameters": {
                "rank": int(global_rank),
                "scale": 1.0,
                "dropout": 0.0,
                "keys": sorted(target_modules),
            },
            "target_modules": sorted(target_modules),
            "rank": int(global_rank),
            "per_layer_ranks": per_layer_rank_map,
            "method": GEOMETRIC_LORA_METHOD,
            "init_method": GEOMETRIC_LORA_INIT_METHOD_CAYLEY,
            "optimizer": GEOMETRIC_LORA_OPTIMIZER,
            "controller": GEOMETRIC_LORA_CONTROLLER,
            "stopping": GEOMETRIC_LORA_STOPPING,
        }
        if metadata:
            config["metadata"] = metadata

        config_path = output_dir / "adapter_config.json"
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)

        logger.info(
            "Saved NB-LoRA adapter: %d layers, rank=%d, path=%s",
            len(discovered_ranks), global_rank, output_dir,
        )
        return output_dir

    def apply_standard_lora_adapter(self, model, adapter_path: str | Path) -> int:
        """Merge a saved standard LoRA adapter into model weights.

        This applies delta_W = lora_b^T @ lora_a^T to each target layer weight.
        Used for cumulative STaR rounds that continue from prior adapter state.
        """
        adapter_dir = Path(adapter_path).expanduser().resolve()
        weights_path = adapter_dir / "adapters.safetensors"
        if not weights_path.exists():
            weights_path = adapter_dir / "adapter.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(f"No adapter weights found at {adapter_dir}")

        adapter_weights = self._backend.load_safetensors(str(weights_path))
        merged_layers = 0

        for key in sorted(adapter_weights.keys()):
            if not key.endswith(".lora_a"):
                continue
            key_base = key[:-7]
            key_b = f"{key_base}.lora_b"
            if key_b not in adapter_weights:
                continue

            layer_key = f"{key_base}.weight"
            try:
                parent, attr_name = self._resolve_parent_and_attr(model, layer_key)
                linear = getattr(parent, attr_name)
            except Exception:
                logger.warning("Skipping adapter merge for unresolved layer %s", layer_key)
                continue

            if not hasattr(linear, "weight"):
                logger.warning("Skipping adapter merge for non-linear layer %s", layer_key)
                continue

            lora_a = adapter_weights[key]
            lora_b = adapter_weights[key_b]

            # LoRA forward: x @ lora_a @ lora_b
            # Weight delta for [out, in] weight layout: lora_b^T @ lora_a^T
            delta = mx.matmul(mx.transpose(lora_b), mx.transpose(lora_a))
            delta = mx.astype(delta, linear.weight.dtype)
            linear.weight = linear.weight + delta
            mx.eval(linear.weight)
            merged_layers += 1

        logger.info(
            "Applied prior adapter: %d layers merged from %s",
            merged_layers,
            adapter_dir,
        )
        return merged_layers

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _clamp_all_scales(self, model) -> None:
        """Clamp S_raw in all NBLoRALinear modules after optimizer step."""
        for _, module in self._iter_nb_lora_modules(model):
            module.clamp_scale()
            mx.eval(module.S_raw)

    # ── Certificate computation methods ─────────────────────────────

    def _compute_val_gradient(
        self,
        model,
        eval_dataset,
        batch_size: int,
        seq_length: int,
        n_batches: int,
    ) -> dict[str, Any] | None:
        """Compute flat gradient of validation loss at current params.

        Averages gradients across ``n_batches`` validation batches.

        Returns:
            Flat dict {param_key: gradient_array}, or None on failure.
        """
        from mlx.utils import tree_flatten as mlx_flatten
        from mlx_lm.tuner.trainer import default_loss, iterate_batches

        loss_vg = nn.value_and_grad(model, default_loss)
        accum: dict[str, Any] | None = None
        count = 0

        try:
            for batch, lengths in iterate_batches(
                eval_dataset, batch_size, seq_length, loop=False,
            ):
                if count >= n_batches:
                    break
                (loss, _), grads = loss_vg(model, batch, lengths)
                mx.eval(loss)
                flat = dict(mlx_flatten(grads))
                if accum is None:
                    accum = {k: mx.zeros_like(v) for k, v in flat.items()}
                    mx.eval(*accum.values())
                for k in accum:
                    if k in flat:
                        accum[k] = accum[k] + flat[k]
                mx.eval(*accum.values())
                count += 1
        except Exception:
            logger.debug("Val gradient computation failed", exc_info=True)
            return None

        if accum is None or count == 0:
            return None

        for k in accum:
            accum[k] = accum[k] * (1.0 / count)
        mx.eval(*accum.values())
        return accum

    def _compute_val_hvp(
        self,
        model,
        eval_dataset,
        batch_size: int,
        seq_length: int,
        n_batches: int,
        direction: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Compute H_val @ d via central-difference HVP on validation data.

        H_val @ d ≈ (∇L_val(θ+εd) - ∇L_val(θ-εd)) / 2ε

        ε = (3·ε_mach)^(1/3) × max(||params||, 1.0) (Nocedal & Wright 2006, §8.1).

        Cost: 2 × n_batches backward passes.

        Returns:
            Flat dict {param_key: hvp_array}, or None on failure.
        """
        from mlx.utils import tree_flatten as mlx_flatten
        from mlx.utils import tree_unflatten

        trainable = dict(mlx_flatten(model.trainable_parameters()))
        original = {k: mx.array(v) for k, v in trainable.items()}
        mx.eval(*original.values())

        # Central-difference optimal perturbation (Nocedal & Wright 2006, §8.1):
        # Minimizing truncation (h²) + roundoff (eps_f/h) gives h = (3*eps_f)^(1/3).
        # Scale by ||θ|| to make relative to parameter magnitude.
        param_norm = math.sqrt(
            sum(float(mx.sum(v * v)) for v in trainable.values())
        )
        _eps_f32 = math.ldexp(1.0, -23)
        eps = (3.0 * _eps_f32) ** (1.0 / 3.0) * max(param_norm, 1.0)

        try:
            # θ + ε d
            plus_p = {k: trainable[k] + eps * direction[k]
                       for k in trainable if k in direction}
            model.update(tree_unflatten(plus_p))
            mx.eval(model.parameters())
            g_plus = self._compute_val_gradient(
                model, eval_dataset, batch_size, seq_length, n_batches,
            )

            # θ - ε d
            minus_p = {k: trainable[k] - eps * direction[k]
                        for k in trainable if k in direction}
            model.update(tree_unflatten(minus_p))
            mx.eval(model.parameters())
            g_minus = self._compute_val_gradient(
                model, eval_dataset, batch_size, seq_length, n_batches,
            )

            if g_plus is None or g_minus is None:
                return None

            hvp = {
                k: (g_plus[k] - g_minus[k]) * (1.0 / (2.0 * eps))
                for k in g_plus if k in g_minus
            }
            mx.eval(*hvp.values())
            return hvp

        except Exception:
            logger.debug("Val HVP computation failed", exc_info=True)
            return None
        finally:
            model.update(tree_unflatten(original))
            mx.eval(model.parameters())

    def _compute_per_batch_val_losses(
        self,
        model,
        eval_dataset,
        batch_size: int,
        seq_length: int,
        n_batches: int,
    ) -> list[float]:
        """Compute per-batch validation losses (forward-only, no grad).

        Returns:
            List of per-batch average loss values.
        """
        from mlx_lm.tuner.trainer import default_loss, iterate_batches

        from modelcypher.backends.mlx_training_adapter_core import (
            iterate_vl_batches,
            make_vl_loss,
        )

        is_vl = (
            isinstance(eval_dataset, list)
            and len(eval_dataset) > 0
            and isinstance(eval_dataset[0], dict)
            and "tokens" in eval_dataset[0]
            and "pixel_values" in eval_dataset[0]
        )

        if is_vl:
            vl_loss_fn = make_vl_loss(
                image_token_id=eval_dataset[0].get("image_token_id"),
                video_token_id=eval_dataset[0].get("video_token_id"),
            )
            batch_iter = iterate_vl_batches(
                eval_dataset, batch_size, seq_length, loop=False,
            )
        else:
            batch_iter = iterate_batches(
                eval_dataset, batch_size, seq_length, loop=False,
            )

        per_batch: list[float] = []
        for batch_item in batch_iter:
            if len(per_batch) >= n_batches:
                break
            if is_vl:
                batch, lengths, pixel_values_batch, position_ids_batch = batch_item
                loss, ntoks = vl_loss_fn(
                    model, batch, lengths,
                    pixel_values_batch, position_ids_batch,
                )
            else:
                batch, lengths = batch_item
                loss, ntoks = default_loss(model, batch, lengths)
            mx.eval(loss, ntoks)
            n = float(ntoks)
            if n > 0:
                per_batch.append(float(loss))
        return per_batch


__all__ = ["_MLXTrainingAdapterAdapterIOMixin"]
