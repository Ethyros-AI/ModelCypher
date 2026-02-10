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

"""MLX-specific adapter for dataset-driven geometric LoRA training."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.geometry.spectral_init import (
    spectral_normalized_lora_init,
)
from modelcypher.core.domain.training.geometric_early_stopping import check_loss_stable
from modelcypher.core.domain.training.geometric_optimizer import OptimizerGeometryConfig
from modelcypher.core.domain.training.hessian_estimator import (
    Config as HessianConfig,
    top_eigenvalue,
)
from modelcypher.core.domain.training.scaled_gd import precondition_lora_gradients
from modelcypher.core.domain.training.spectral_budget import (
    DTYPE_THRESHOLD_F32,
    compute_budget_ratios,
    is_budget_exhausted,
)
from modelcypher.ports.training import LoRALayerConfig

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)


class MLXTrainingAdapter:
    """MLX-specific training operations for geometric LoRA training."""

    def __init__(self, backend: "Backend"):
        self._backend = backend

    def prepare_dataset(self, samples: list[dict[str, Any]], tokenizer) -> list[tuple[Any, int]]:
        """Tokenize samples into the format expected by mlx-lm iterate_batches."""
        import mlx.core as mx

        dataset: list[tuple[Any, int]] = []
        for sample in samples:
            text = sample.get("text")
            if not isinstance(text, str):
                continue
            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue
            dataset.append((mx.array(tokens, dtype=mx.int32), 0))
        return dataset

    def extract_weight_matrices(self, model) -> dict[str, Any]:
        """Extract 2D projection weights from the model."""
        import mlx.core as mx

        weights: dict[str, Any] = {}
        base = getattr(model, "model", model)

        if not hasattr(base, "layers"):
            raise ValueError("Model has no .layers attribute — unsupported architecture")

        for layer_idx, layer in enumerate(base.layers):
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    proj = getattr(attn, proj_name, None)
                    if proj is not None and hasattr(proj, "weight"):
                        key = f"model.layers.{layer_idx}.self_attn.{proj_name}.weight"
                        weights[key] = proj.weight
                        mx.eval(proj.weight)

            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                for proj_name in ("up_proj", "down_proj", "gate_proj"):
                    proj = getattr(mlp, proj_name, None)
                    if proj is not None and hasattr(proj, "weight"):
                        key = f"model.layers.{layer_idx}.mlp.{proj_name}.weight"
                        weights[key] = proj.weight
                        mx.eval(proj.weight)

        logger.info(
            "Extracted %d weight matrices from %d layers",
            len(weights),
            len(base.layers),
        )
        return weights

    def inject_geometric_lora(self, model, configs: list[LoRALayerConfig]) -> int:
        """Inject LoRA linears with spectral-normalized initialization."""
        import mlx.core as mx
        from mlx_lm.tuner.lora import LoRALinear

        injected = 0

        for cfg in configs:
            if cfg.rank <= 0:
                continue

            try:
                parent, attr_name = self._resolve_parent_and_attr(model, cfg.layer_key)
                linear = getattr(parent, attr_name)

                lora = LoRALinear.from_base(
                    linear,
                    r=cfg.rank,
                    dropout=cfg.dropout,
                    scale=1.0,
                )

                # spectral_init returns A:[rank,in], B:[out,rank].
                # LoRALinear stores lora_a:[in,rank], lora_b:[rank,out].
                a_init, b_init = spectral_normalized_lora_init(
                    in_features=cfg.in_features,
                    out_features=cfg.out_features,
                    rank=cfg.rank,
                    sigma_k=cfg.sigma_k,
                    backend=self._backend,
                )
                lora.lora_a = a_init.T
                lora.lora_b = b_init.T
                mx.eval(lora.lora_a, lora.lora_b)

                setattr(parent, attr_name, lora)
                injected += 1
            except Exception as exc:
                logger.warning("Failed to inject LoRA at %s: %s", cfg.layer_key, exc)

        return injected

    def freeze_and_unfreeze_lora(self, model) -> None:
        """Freeze all existing model parameters before LoRA injection."""
        model.freeze()

    def evaluate_loss(
        self,
        model,
        dataset,
        tokenizer,
        batch_size: int,
        seq_length: int,
        n_batches: int,
    ) -> tuple[float, float]:
        """Compute average loss and perplexity over a dataset."""
        del tokenizer
        import mlx.core as mx
        from mlx_lm.tuner.trainer import default_loss, iterate_batches

        total_loss = 0.0
        total_tokens = 0.0
        n_evaluated = 0

        for batch, lengths in iterate_batches(dataset, batch_size, seq_length, loop=False):
            loss, ntoks = default_loss(model, batch, lengths)
            mx.eval(loss, ntoks)
            total_loss += float(loss) * float(ntoks)
            total_tokens += float(ntoks)
            n_evaluated += 1
            if n_evaluated >= n_batches:
                break

        if total_tokens == 0:
            return float("inf"), float("inf")

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(min(avg_loss, 100.0))
        return avg_loss, perplexity

    def measure_lipschitz_constant(
        self,
        model,
        dataset,
        batch_size: int,
        seq_length: int,
        seed: int,
        power_iterations: int = 10,
    ) -> tuple[float, float]:
        """Measure eta=1/L where L=max-batch top Hessian eigenvalue."""
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten, tree_unflatten
        from mlx_lm.tuner.trainer import default_loss, iterate_batches

        batches = list(
            iterate_batches(
                dataset,
                batch_size,
                seq_length,
                loop=False,
                seed=seed,
            )
        )
        if not batches:
            raise ValueError("Dataset has no batches for Lipschitz measurement")

        logger.info(
            "Measuring L on %d batches (%d power iterations each)",
            len(batches),
            power_iterations,
        )

        base_config = HessianConfig.moderate()
        if power_iterations == base_config.power_iterations:
            config = base_config
        else:
            config = HessianConfig(
                hutchinson_vectors=base_config.hutchinson_vectors,
                power_iterations=power_iterations,
                finite_difference_epsilon=base_config.finite_difference_epsilon,
                power_iteration_tolerance=base_config.power_iteration_tolerance,
            )

        original_trainable = dict(tree_flatten(model.trainable_parameters()))
        trainable = dict(original_trainable)

        l_values: list[float] = []
        try:
            for batch, lengths in batches:
                def loss_and_grad_fn(params_dict, _batch=batch, _lengths=lengths):
                    model.update(tree_unflatten(list(params_dict.items())))
                    mx.eval(model.parameters())
                    loss_grad_fn = nn.value_and_grad(model, default_loss)
                    (loss, _), grad = loss_grad_fn(model, _batch, _lengths)
                    grad_flat = dict(tree_flatten(grad))
                    mx.eval(loss)
                    return loss, grad_flat

                l_batch = top_eigenvalue(loss_and_grad_fn, trainable, config)
                if l_batch is not None and l_batch > 0:
                    l_values.append(l_batch)
        finally:
            model.update(tree_unflatten(list(original_trainable.items())))
            mx.eval(model.parameters())

        if not l_values:
            raise ValueError("Lipschitz constant measurement failed on all batches")

        l_value = max(l_values)
        eta = 1.0 / l_value
        logger.info(
            "Measured L over %d/%d batches: max=%.4e, min=%.4e, range=%.1fx, eta=%.4e",
            len(l_values),
            len(batches),
            max(l_values),
            min(l_values),
            max(l_values) / min(l_values),
            eta,
        )
        return eta, l_value

    def train_loop(
        self,
        model,
        train_dataset,
        batch_size: int,
        seq_length: int,
        max_iters: int,
        seed: int,
        lora_configs: list[LoRALayerConfig],
        opt_config: OptimizerGeometryConfig,
        lr_override: float | None = None,
    ) -> tuple[list[tuple[int, float, float]], str]:
        """Run training with ScaledGD and geometric stopping."""
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as opt
        from mlx_lm.tuner.trainer import default_loss, iterate_batches

        use_geometric = lr_override is None
        if use_geometric:
            eta, _ = self.measure_lipschitz_constant(
                model=model,
                dataset=train_dataset,
                batch_size=batch_size,
                seq_length=seq_length,
                seed=seed,
                power_iterations=HessianConfig.moderate().power_iterations,
            )
            optimizer = opt.SGD(learning_rate=eta, momentum=0.0)
        else:
            eta = float(lr_override)
            optimizer = opt.SGD(learning_rate=eta, momentum=0.0)
            logger.info("Using override LR: %.2e (flat SGD, no ScaledGD)", eta)

        loss_value_and_grad = nn.value_and_grad(model, default_loss)

        losses: list[tuple[int, float, float]] = []
        stop_reason: str | None = None

        batch_iter = iterate_batches(
            train_dataset,
            batch_size,
            seq_length,
            loop=True,
            seed=seed,
        )

        n_batches_per_epoch = len(
            list(
                iterate_batches(
                    train_dataset,
                    batch_size,
                    seq_length,
                    loop=False,
                    seed=seed,
                )
            )
        )
        if n_batches_per_epoch <= 0:
            raise ValueError("Training dataset produced zero batches")

        log_interval = max(1, n_batches_per_epoch)
        check_interval = max(1, n_batches_per_epoch)

        logger.info(
            "Training until geometry says stop (safety cap: %d, epoch: %d batches)",
            max_iters,
            n_batches_per_epoch,
        )

        for it in range(max_iters):
            t_step = time.time()
            batch, lengths = next(batch_iter)

            (loss, ntoks), grad = loss_value_and_grad(model, batch, lengths)
            if use_geometric:
                scaled_grad = self._apply_scaled_gd(model, grad, lora_configs, opt_config)
                optimizer.update(model, scaled_grad)
            else:
                optimizer.update(model, grad)

            mx.eval(model.parameters(), optimizer.state)

            loss_val = float(loss)
            ntoks_val = float(ntoks)
            elapsed = time.time() - t_step
            tps = float("inf") if elapsed <= 0 else ntoks_val / elapsed

            losses.append((it, loss_val, tps))

            if (it + 1) % log_interval == 0 or it == 0:
                epoch = (it + 1) / n_batches_per_epoch
                lr_info = f" | eta={eta:.2e}" if use_geometric else ""
                logger.info(
                    "Iter %d (epoch %.1f) | loss=%.4f | tokens/sec=%.1f%s",
                    it + 1,
                    epoch,
                    loss_val,
                    tps,
                    lr_info,
                )

            if use_geometric and (it + 1) % check_interval == 0 and it >= 6 * n_batches_per_epoch:
                stable, threshold = check_loss_stable(losses, window=3 * n_batches_per_epoch)
                if stable:
                    stop_reason = f"loss_stable (|Δ_epoch| < SE = {threshold:.4e})"
                    logger.info("Geometry stop at iter %d: %s", it + 1, stop_reason)
                    break

                exhausted, median_ratio = self._check_budget_exhausted(model, lora_configs, opt_config)
                if exhausted:
                    stop_reason = (
                        f"budget_exhausted (median ratio = {median_ratio:.4f}, Weyl crossing)"
                    )
                    logger.info("Geometry stop at iter %d: %s", it + 1, stop_reason)
                    break
        else:
            stop_reason = f"safety_cap ({max_iters} iters)"
            logger.warning("Hit safety cap at %d iters — geometry did not converge", max_iters)

        return losses, stop_reason

    def save_adapter(
        self,
        model,
        configs: list[LoRALayerConfig],
        output_path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save trained LoRA weights in mlx-lm adapter format."""
        import mlx.core as mx

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        active_configs = [cfg for cfg in configs if cfg.rank > 0]
        if not active_configs:
            raise ValueError("No active LoRA configs to save")

        discovered_ranks: list[int] = []
        for cfg in active_configs:
            layer = self._resolve_lora_layer(model, cfg.layer_key)
            if layer is None or not hasattr(layer, "lora_a") or not hasattr(layer, "lora_b"):
                continue
            discovered_ranks.append(int(layer.lora_a.shape[1]))

        if not discovered_ranks:
            raise ValueError("No injected LoRA layers found to export")

        global_rank = max(discovered_ranks)
        adapter_weights: dict[str, Any] = {}
        target_modules: set[str] = set()

        for cfg in active_configs:
            layer = self._resolve_lora_layer(model, cfg.layer_key)
            if layer is None or not hasattr(layer, "lora_a") or not hasattr(layer, "lora_b"):
                continue

            lora_a = layer.lora_a
            lora_b = layer.lora_b
            rank = int(lora_a.shape[1])

            if rank < global_rank:
                pad_a = mx.zeros((int(lora_a.shape[0]), global_rank - rank), dtype=lora_a.dtype)
                pad_b = mx.zeros((global_rank - rank, int(lora_b.shape[1])), dtype=lora_b.dtype)
                lora_a = mx.concatenate([lora_a, pad_a], axis=1)
                lora_b = mx.concatenate([lora_b, pad_b], axis=0)

            key_base = cfg.layer_key.replace(".weight", "")
            adapter_weights[f"{key_base}.lora_a"] = lora_a
            adapter_weights[f"{key_base}.lora_b"] = lora_b
            target_modules.add(self._module_name_from_layer_key(cfg.layer_key))

        if not adapter_weights:
            raise ValueError("No adapter weights were extracted from injected LoRA layers")

        mx.eval(*adapter_weights.values())

        metadata_str: dict[str, str] | None = None
        if metadata:
            metadata_str = {str(k): str(v) for k, v in metadata.items()}

        weights_path = output_dir / "adapters.safetensors"
        self._backend.save_safetensors(str(weights_path), adapter_weights, metadata=metadata_str)

        config = {
            "fine_tune_type": "lora",
            "num_layers": int(self._backend.get_num_layers(model)),
            "lora_parameters": {
                "rank": int(global_rank),
                "scale": 1.0,
                "dropout": 0.0,
                "keys": sorted(target_modules),
            },
            "target_modules": sorted(target_modules),
            "rank": int(global_rank),
        }
        if metadata:
            config["metadata"] = metadata

        config_path = output_dir / "adapter_config.json"
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)

        return output_dir

    def check_spectral_bounds(
        self,
        model,
        configs: list[LoRALayerConfig],
    ) -> tuple[int, int, float, list[dict[str, Any]]]:
        """Check post-training spectral bound ratios ||BA||/sigma_k."""
        import mlx.core as mx

        within = 0
        total = 0
        max_ratio = 0.0
        details: list[dict[str, Any]] = []

        for cfg in configs:
            if cfg.rank <= 0:
                continue

            layer = self._resolve_lora_layer(model, cfg.layer_key)
            if layer is None or not hasattr(layer, "lora_a") or not hasattr(layer, "lora_b"):
                continue

            try:
                product = layer.scale * (layer.lora_a @ layer.lora_b)
                product_f32 = product.astype(mx.float32)
                mx.eval(product_f32)
                _, singular_values, _ = mx.linalg.svd(product_f32, compute_uv=True, stream=mx.cpu)
                mx.eval(singular_values)
                spectral_norm = float(singular_values[0])
                ratio = spectral_norm / cfg.sigma_k if cfg.sigma_k > 0 else float("inf")
                is_within = ratio <= 1.0
            except Exception as exc:
                logger.debug("Could not check bounds for %s: %s", cfg.layer_key, exc)
                continue

            if is_within:
                within += 1
            total += 1
            max_ratio = max(max_ratio, ratio)
            details.append(
                {
                    "layer": cfg.layer_key,
                    "spectral_norm": spectral_norm,
                    "sigma_k": cfg.sigma_k,
                    "ratio": ratio,
                    "within_bound": is_within,
                }
            )

        return within, total, max_ratio, details

    def _apply_scaled_gd(
        self,
        model,
        grad,
        lora_configs: list[LoRALayerConfig],
        opt_config: OptimizerGeometryConfig,
    ):
        from mlx.utils import tree_flatten, tree_unflatten

        grad_flat = dict(tree_flatten(grad))
        param_flat = dict(tree_flatten(model.trainable_parameters()))
        preconditioned = precondition_lora_gradients(
            grad_flat=grad_flat,
            param_flat=param_flat,
            lora_configs=lora_configs,
            opt_config=opt_config,
            backend=self._backend,
        )
        return tree_unflatten(list(preconditioned.items()))

    def _check_budget_exhausted(
        self,
        model,
        configs: list[LoRALayerConfig],
        opt_config: OptimizerGeometryConfig | None = None,
    ) -> tuple[bool, float]:
        lora_products: list[tuple[float, Any, Any, float]] = []
        ordered_layer_keys: list[str] = []

        for cfg in configs:
            if cfg.rank <= 0 or cfg.sigma_k <= 0:
                continue

            layer = self._resolve_lora_layer(model, cfg.layer_key)
            if layer is None or not hasattr(layer, "lora_a") or not hasattr(layer, "lora_b"):
                continue

            lora_products.append((layer.scale, layer.lora_a, layer.lora_b, cfg.sigma_k))
            ordered_layer_keys.append(cfg.layer_key)

        ratios = compute_budget_ratios(lora_products, self._backend)

        spectral_gaps = None
        sigma_ks = None
        if opt_config is not None:
            spectral_gaps = []
            sigma_ks = []
            for key in ordered_layer_keys:
                layer_cfg = opt_config.layer_configs.get(key)
                if layer_cfg is None:
                    spectral_gaps.append(0.0)
                    sigma_ks.append(0.0)
                else:
                    spectral_gaps.append(layer_cfg.spectral_gap)
                    sigma_ks.append(layer_cfg.sigma_k)

        return is_budget_exhausted(
            ratios=ratios,
            spectral_gaps=spectral_gaps,
            sigma_ks=sigma_ks,
            threshold=DTYPE_THRESHOLD_F32,
        )

    def _resolve_parent_and_attr(self, model, layer_key: str) -> tuple[Any, str]:
        path_parts = layer_key.replace(".weight", "").split(".")
        obj = model
        for part in path_parts[:-1]:
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        return obj, path_parts[-1]

    def _resolve_lora_layer(self, model, layer_key: str):
        try:
            parent, attr_name = self._resolve_parent_and_attr(model, layer_key)
            return getattr(parent, attr_name)
        except Exception:
            return None

    def _module_name_from_layer_key(self, layer_key: str) -> str:
        parts = layer_key.replace(".weight", "").split(".")
        if len(parts) >= 5 and parts[0] == "model" and parts[1] == "layers":
            return ".".join(parts[3:])
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return layer_key.replace(".weight", "")
