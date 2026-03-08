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

"""Core setup and projection methods for :class:`MLXTrainingAdapter`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

from modelcypher.backends.mlx_training_adapter_core import *  # noqa: F403
from modelcypher.core.domain.training.exceptions import TrainingDerivationError  # noqa: F401

if TYPE_CHECKING:
    from modelcypher.core.domain.training.geometric_lora import LayerGeometry
    from modelcypher.ports.backend import Backend


class _VirtualProjection:
    """Lightweight wrapper exposing ``.weight`` for a 2D slice of a packed 3D tensor.

    Zero-copy: the slice is a view into the original packed tensor.
    Downstream ``compute_layer_geometry()`` only reads ``.weight``.
    """

    __slots__ = ("weight",)

    def __init__(self, weight_2d: Any) -> None:
        self.weight = weight_2d


class _MLXTrainingAdapterCoreMixin:
    def __init__(self, backend: "Backend"):
        self._backend = backend

    def load_training_model(self, model_path: str, backend: "Backend" | None = None) -> tuple[Any, Any, bool]:
        """Load training model/tokenizer and report whether it is vision-language."""
        from modelcypher.backends._mlx_qwen35_vl_encoder import (  # noqa: PLC0415
            is_qwen35_vl,
            load_qwen35_vl_model,
        )

        _ = backend
        resolved_model_path = str(model_path)
        if is_qwen35_vl(resolved_model_path):
            logger.info(
                "Detected Qwen3.5-VL model (vision_config present). "
                "Loading text + visual encoder.",
            )
            model, tokenizer = load_qwen35_vl_model(resolved_model_path)
            return model, tokenizer, True

        model, tokenizer = self._backend.load_model(resolved_model_path)
        return model, tokenizer, False

    def prepare_vl_dataset(self, samples: list[dict[str, Any]], tokenizer, model_path: str) -> list[dict]:
        """Prepare a vision-language dataset for MLX training."""
        from modelcypher.backends._mlx_vl_preprocessor import VLPreprocessor  # noqa: PLC0415

        vl_preprocessor = VLPreprocessor.from_model_path(str(model_path))
        return vl_preprocessor.prepare_vl_dataset(samples, tokenizer)

    def prepare_dataset(self, samples: list[dict[str, Any]], tokenizer) -> list[tuple[Any, int]]:
        """Tokenize samples into mlx-lm iterate_batches format.

        Appends EOS token to each sequence so the model learns when to stop.
        """
        eos_id = getattr(tokenizer, "eos_token_id", None)
        dataset: list[tuple[Any, int]] = []
        for sample in samples:
            text = sample.get("text")
            if not isinstance(text, str):
                continue
            tokens = tokenizer.encode(text)
            if eos_id is not None and (not tokens or tokens[-1] != eos_id):
                tokens.append(eos_id)
            if len(tokens) < 2:
                continue
            dataset.append((mx.array(tokens, dtype=mx.int32), 0))
        return dataset

    def prepare_masked_dataset(
        self, samples: list[dict[str, Any]], tokenizer
    ) -> list[tuple[Any, Any, int]]:
        """Tokenize with answer-span masks for answer-only CE training.

        Returns list of (tokens, mask, 0) tuples where:
        - tokens: mx.array of token IDs (with EOS appended)
        - mask: mx.array of floats, 1.0 for answer tokens, 0.0 for prompt and EOS
        - 0: placeholder for compatibility

        The mask is aligned with the full token sequence. When computing loss,
        the caller shifts mask[1:] to align with shifted targets.

        EOS is excluded from the mask (mask=0.0) because the base model already
        has EOS behaviour from pre-training. Training CE on EOS in every answer
        span creates an outsized gradient that biases the adapter toward
        premature termination.

        Samples missing ``answer_start`` get mask=1.0 for all content tokens
        (full-sequence CE) with EOS still excluded.
        """
        eos_id = getattr(tokenizer, "eos_token_id", None)
        dataset: list[tuple[Any, Any, int]] = []

        for sample in samples:
            text = sample.get("text")
            if not isinstance(text, str):
                continue

            tokens = tokenizer.encode(text)
            if eos_id is not None and (not tokens or tokens[-1] != eos_id):
                tokens.append(eos_id)
            if len(tokens) < 2:
                continue

            answer_start_char = sample.get("answer_start")
            if answer_start_char is not None and isinstance(answer_start_char, int):
                # Tokenize prefix to find answer token boundary
                prefix_tokens = tokenizer.encode(text[:answer_start_char])
                answer_token_idx = len(prefix_tokens)
            else:
                # No answer_start — full-sequence CE (e.g. retention samples)
                answer_token_idx = 0

            # Clamp to valid range
            answer_token_idx = min(answer_token_idx, len(tokens))

            mask = [0.0] * answer_token_idx + [1.0] * (len(tokens) - answer_token_idx)

            # Exclude EOS from the answer mask.  The base model already
            # has EOS behaviour from pre-training; training CE on EOS in
            # every answer span produces an outsized gradient that biases
            # the adapter toward premature termination (EOS p ≈ 0.65 after
            # 1-2 answer tokens instead of baseline p ≈ 5e-6).
            if eos_id is not None and tokens[-1] == eos_id:
                mask[-1] = 0.0

            dataset.append((
                mx.array(tokens, dtype=mx.int32),
                mx.array(mask, dtype=mx.float32),
                0,
            ))

        return dataset

    def prepare_paired_dataset(
        self,
        samples: list[dict[str, Any]],
        tokenizer,
    ) -> list[dict[str, Any]]:
        """Tokenize paired samples with answer span masks and pair metadata.

        Returns list of dicts with keys:
            tokens: mx.array of token IDs
            answer_mask: mx.array of 0/1 mask (1 = answer token)
            logic_id: str
            template_id: str
            n_tokens: int
        """
        dataset: list[dict[str, Any]] = []
        for sample in samples:
            text = sample.get("text")
            if not isinstance(text, str):
                continue

            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue

            # Compute answer token mask
            answer_start_str = sample.get("answer_start", "")
            if answer_start_str and answer_start_str in text:
                # Find character offset of answer_start in text
                char_offset = text.index(answer_start_str)
                # Tokenize the prefix to find the token boundary
                prefix = text[:char_offset]
                prefix_tokens = tokenizer.encode(prefix)
                answer_token_start = len(prefix_tokens)
            else:
                # No answer_start or not found — mask everything (full sequence CE)
                answer_token_start = 0

            # answer_mask: 1 for answer tokens, 0 for scaffold tokens
            # Applied to the shifted target sequence (tokens[1:])
            mask = [0] * len(tokens)
            for i in range(answer_token_start, len(tokens)):
                mask[i] = 1

            dataset.append({
                "tokens": mx.array(tokens, dtype=mx.int32),
                "answer_mask": mx.array(mask, dtype=mx.float32),
                "logic_id": sample.get("logic_id", ""),
                "template_id": sample.get("template_id", ""),
                "n_tokens": len(tokens),
            })

        return dataset

    @staticmethod
    def _get_model_base(model) -> tuple[Any, str]:
        """Return (layers_module, key_prefix) for the model architecture.

        Handles two layouts:
        - Standard transformer (Qwen3, LFM2, etc.): model.model.layers
          → prefix = "model.layers"
        - Multimodal language model (Qwen3.5): model.language_model.layers
          → prefix = "model.language_model.layers"

        The key_prefix is used to construct weight keys matching the
        safetensors serialization format (HuggingFace convention).
        """
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            return inner, "model.layers"
        lm = getattr(model, "language_model", None)
        if lm is not None and hasattr(lm, "layers"):
            return lm, "model.language_model.layers"
        if hasattr(model, "layers"):
            return model, "model.layers"
        raise ValueError("Model has no .layers attribute — unsupported architecture")

    @staticmethod
    def _dequantize_weight(proj) -> Any:
        """Return the full-precision weight matrix for a projection.

        For ``nn.QuantizedLinear``, dequantizes packed integer data back
        to float so that SVD and geometry analysis operate on actual
        weight values, not quantization artifacts.  For standard
        ``nn.Linear``, returns the weight unchanged.
        """
        if isinstance(proj, nn.QuantizedLinear):
            w = mx.dequantize(
                proj.weight, proj.scales, proj.biases,
                proj.group_size, proj.bits,
            )
            mx.eval(w)
            return w
        mx.eval(proj.weight)
        return proj.weight

    def _iter_layer_weight_projections(
        self,
        layer_idx: int,
        layer: Any,
        key_prefix: str = "model.layers",
    ) -> Iterator[tuple[str, Any]]:
        """Yield `(weight_key, projection_module)` for analyzable layer weights."""
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                proj = getattr(attn, proj_name, None)
                if proj is not None and hasattr(proj, "weight"):
                    key = f"{key_prefix}.{layer_idx}.self_attn.{proj_name}.weight"
                    yield key, proj

        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            return

        for proj_name in ("up_proj", "down_proj", "gate_proj"):
            proj = getattr(mlp, proj_name, None)
            if proj is not None and hasattr(proj, "weight"):
                key = f"{key_prefix}.{layer_idx}.mlp.{proj_name}.weight"
                yield key, proj

        router_gate = getattr(mlp, "gate", None)
        if router_gate is not None and hasattr(router_gate, "weight"):
            key = f"{key_prefix}.{layer_idx}.mlp.gate.weight"
            yield key, router_gate

        # Packed SwitchGLU experts: mlp.switch_mlp has SwitchLinear modules
        # with 3D weights [num_experts, out_dim, in_dim].
        switch_mlp = getattr(mlp, "switch_mlp", None)
        if switch_mlp is not None:
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                switch_linear = getattr(switch_mlp, proj_name, None)
                if switch_linear is None or not hasattr(switch_linear, "weight"):
                    continue
                packed_w = switch_linear.weight
                if packed_w.ndim != 3:
                    continue
                num_experts = packed_w.shape[0]
                for expert_idx in range(num_experts):
                    expert_slice = packed_w[expert_idx]
                    key = (
                        f"{key_prefix}.{layer_idx}.mlp.experts.{expert_idx}."
                        f"{proj_name}.weight"
                    )
                    yield key, _VirtualProjection(expert_slice)

        # Individual expert modules: mlp.experts is iterable of per-expert modules.
        experts = getattr(mlp, "experts", None)
        if experts is not None and switch_mlp is None:
            if isinstance(experts, dict):
                expert_items = list(experts.items())
            else:
                try:
                    expert_items = list(enumerate(experts))
                except TypeError:
                    expert_items = []
            for raw_expert_idx, expert in expert_items:
                if expert is None:
                    continue
                expert_idx = int(raw_expert_idx)
                for proj_name in ("gate_proj", "up_proj", "down_proj"):
                    proj = getattr(expert, proj_name, None)
                    if proj is None or not hasattr(proj, "weight"):
                        continue
                    key = (
                        f"{key_prefix}.{layer_idx}.mlp.experts.{expert_idx}."
                        f"{proj_name}.weight"
                    )
                    yield key, proj

        shared_expert = getattr(mlp, "shared_expert", None)
        if shared_expert is not None:
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                proj = getattr(shared_expert, proj_name, None)
                if proj is None or not hasattr(proj, "weight"):
                    continue
                key = (
                    f"{key_prefix}.{layer_idx}.mlp.shared_expert."
                    f"{proj_name}.weight"
                )
                yield key, proj

        shared_expert_gate = getattr(mlp, "shared_expert_gate", None)
        if shared_expert_gate is not None and hasattr(shared_expert_gate, "weight"):
            key = f"{key_prefix}.{layer_idx}.mlp.shared_expert_gate.weight"
            yield key, shared_expert_gate

    def _compute_layer_geometry_for_weight(
        self,
        *,
        weight: Any,
        key: str,
        use_randomized: bool,
        rng_kwargs: dict[str, Any],
        compute_layer_geometry: Any,
        compute_layer_geometry_randomized: Any,
    ) -> "LayerGeometry":
        """Compute geometry for one matrix with deterministic per-key seed."""
        if use_randomized:
            seed = rng_kwargs.get("seed")
            if seed is not None:
                rng_kwargs_copy = dict(rng_kwargs)
                rng_kwargs_copy["seed"] = (seed + hash(key)) & 0xFFFFFFFF
            else:
                rng_kwargs_copy = rng_kwargs
            return compute_layer_geometry_randomized(
                weight, key, self._backend, **rng_kwargs_copy,
            )
        return compute_layer_geometry(weight, key, self._backend)

    def _analyze_projection_geometry(
        self,
        *,
        key: str,
        proj: Any,
        use_randomized: bool,
        rng_kwargs: dict[str, Any],
        geometries: dict[str, "LayerGeometry"],
        compute_layer_geometry: Any,
        compute_layer_geometry_randomized: Any,
    ) -> bool:
        """Analyze one projection with safe release semantics."""
        weight = self._dequantize_weight(proj)
        try:
            geom = self._compute_layer_geometry_for_weight(
                weight=weight,
                key=key,
                use_randomized=use_randomized,
                rng_kwargs=rng_kwargs,
                compute_layer_geometry=compute_layer_geometry,
                compute_layer_geometry_randomized=compute_layer_geometry_randomized,
            )
            geometries[key] = geom
            logger.debug(
                "%s: decay=%.1f×, σ_k=%.4f, tail=%d",
                key, geom.decay_ratio, geom.sigma_k, geom.tail_dims,
            )
            return True
        except Exception as exc:
            logger.warning("Failed to analyze %s: %s", key, exc)
            return False
        finally:
            del weight

    def get_weight_matrix_by_key(self, model, layer_key: str) -> Any:
        """Return dequantized/full-precision weight matrix for one key."""
        parent, attr_name = self._resolve_parent_and_attr(model, layer_key)
        proj = getattr(parent, attr_name, None)
        if proj is None or not hasattr(proj, "weight"):
            raise ValueError(f"Layer key does not resolve to a weighted projection: {layer_key}")
        return self._dequantize_weight(proj)

    def extract_weight_matrices(self, model) -> dict[str, Any]:
        """Extract 2D projection weights from the model.

        For quantized models, weights are dequantized so that downstream
        geometry analysis (SVD, Shannon effective rank) operates on the
        actual weight values rather than packed integer representations.

        Note:
            This API accumulates all extracted matrices in memory. For large
            MoE models, prefer streaming analysis:
            ``analyze_model_geometry_streaming()`` for full-model scans or
            ``analyze_weight_geometries_for_keys_streaming()`` for subsets.
        """
        weights: dict[str, Any] = {}
        base, key_prefix = self._get_model_base(model)

        for layer_idx, layer in enumerate(base.layers):
            for key, proj in self._iter_layer_weight_projections(layer_idx, layer, key_prefix):
                weights[key] = self._dequantize_weight(proj)

        logger.info(
            "Extracted %d weight matrices from %d layers",
            len(weights),
            len(base.layers),
        )
        return weights

    def analyze_model_geometry_streaming(
        self,
        model,
        *,
        use_randomized: bool = False,
        randomized_kwargs: dict | None = None,
    ) -> dict[str, "LayerGeometry"]:
        """Analyze weight geometry one layer at a time, releasing weights immediately.

        Unlike ``extract_weight_matrices()`` + ``analyze_weight_geometries()``,
        this method never holds all weight matrices in memory simultaneously.
        Each layer's weight is dequantized (if quantized), analyzed, and then
        released before moving to the next layer. This enables geometry
        analysis of 8B-120B models on a single machine.

        Args:
            model: The model to analyze.
            use_randomized: If True, use ``compute_layer_geometry_randomized``
                (O(mnk) per layer) instead of full SVD. Recommended for 8B+.
            randomized_kwargs: Extra keyword arguments forwarded to
                ``compute_layer_geometry_randomized`` (oversampling,
                max_iters, power_iters, seed).

        Returns:
            Dict mapping layer_key -> LayerGeometry.
        """
        from modelcypher.core.domain.training.geometric_lora import (
            compute_layer_geometry,
            compute_layer_geometry_randomized,
        )

        base, key_prefix = self._get_model_base(model)

        geometries: dict[str, "LayerGeometry"] = {}
        rng_kwargs = randomized_kwargs or {}
        n_layers = len(base.layers)
        progress_interval = max(1, n_layers // 5)
        analyzed = 0

        for layer_idx, layer in enumerate(base.layers):
            for key, proj in self._iter_layer_weight_projections(layer_idx, layer, key_prefix):
                if self._analyze_projection_geometry(
                    key=key,
                    proj=proj,
                    use_randomized=use_randomized,
                    rng_kwargs=rng_kwargs,
                    geometries=geometries,
                    compute_layer_geometry=compute_layer_geometry,
                    compute_layer_geometry_randomized=compute_layer_geometry_randomized,
                ):
                    analyzed += 1

            if (layer_idx + 1) % progress_interval == 0 or layer_idx == n_layers - 1:
                logger.info(
                    "Streaming geometry: analyzed layer %d/%d (%d matrices so far)",
                    layer_idx + 1, n_layers, analyzed,
                )

            # Clear GPU cache between layers to reclaim memory
            try:
                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()
                elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                    mx.metal.clear_cache()
            except Exception:
                pass  # Not on Metal or cache clearing unavailable

        logger.info(
            "Streaming geometry analysis complete: %d matrices from %d layers",
            analyzed, n_layers,
        )

        # Extend to visual encoder merger layers for Qwen3.5-VL models.
        # These are the only visual layers with null space (tail_dims > 0).
        # model.visual is attached by load_qwen35_vl_model() — absent for text models.
        visual = getattr(model, "visual", None)
        if visual is not None:
            merger = getattr(visual, "merger", None)
            if merger is not None:
                for proj_name in ("linear_fc1", "linear_fc2"):
                    proj = getattr(merger, proj_name, None)
                    if proj is None or not hasattr(proj, "weight"):
                        continue
                    key = f"model.visual.merger.{proj_name}.weight"
                    self._analyze_projection_geometry(
                        key=key,
                        proj=proj,
                        use_randomized=use_randomized,
                        rng_kwargs=rng_kwargs,
                        geometries=geometries,
                        compute_layer_geometry=compute_layer_geometry,
                        compute_layer_geometry_randomized=compute_layer_geometry_randomized,
                    )
                    logger.info("Visual merger %s added to geometry map", proj_name)

        return geometries

    def analyze_weight_geometries_for_keys_streaming(
        self,
        model,
        layer_keys: list[str],
        *,
        use_randomized: bool = False,
        randomized_kwargs: dict | None = None,
    ) -> dict[str, "LayerGeometry"]:
        """Analyze selected weight keys without holding all matrices in memory."""
        from modelcypher.core.domain.training.geometric_lora import (
            compute_layer_geometry,
            compute_layer_geometry_randomized,
        )

        geometries: dict[str, "LayerGeometry"] = {}
        rng_kwargs = randomized_kwargs or {}
        for key in sorted(set(layer_keys)):
            try:
                parent, attr_name = self._resolve_parent_and_attr(model, key)
                proj = getattr(parent, attr_name, None)
            except Exception as exc:
                logger.warning("Failed to resolve %s: %s", key, exc)
                continue
            if proj is None or not hasattr(proj, "weight"):
                logger.warning("Skipping %s: not a weighted projection", key)
                continue
            self._analyze_projection_geometry(
                key=key,
                proj=proj,
                use_randomized=use_randomized,
                rng_kwargs=rng_kwargs,
                geometries=geometries,
                compute_layer_geometry=compute_layer_geometry,
                compute_layer_geometry_randomized=compute_layer_geometry_randomized,
            )
        return geometries

    def extract_lora_weight_deltas(self, model) -> dict[str, Any]:
        """Extract current LoRA-induced weight deltas for injected NB-LoRA modules.

        Returns:
            Mapping:
                "model.layers.{idx}.(self_attn|mlp).{proj}.weight" -> delta_W [out, in]
        """
        deltas: dict[str, Any] = {}

        for key, nb_lora in self._iter_nb_lora_modules(model):
            try:
                lora_a, lora_b = nb_lora.to_standard_lora()
                # LoRA delta in model weight layout [out, in].
                delta_w = mx.matmul(mx.transpose(lora_b), mx.transpose(lora_a))
                mx.eval(delta_w)
                deltas[key] = delta_w
            except Exception as exc:
                logger.warning("Failed to extract LoRA delta for %s: %s", key, exc)

        logger.info("Extracted %d LoRA weight deltas", len(deltas))
        return deltas

    def inject_nb_lora(
        self,
        model,
        geometries: dict[str, "LayerGeometry"],
        target_modules: list[str],
        safety_margin: float | None = None,
        rank_overrides: dict[str, int] | None = None,
    ) -> int:
        """Replace target linear layers with NBLoRALinear.

        Scale bound per layer: (sigma_k / 2) * safety_margin
        Rank per layer: tail_dims from geometry (null-space capacity)

        Returns number of layers injected.
        """
        eps = float(self._backend.finfo().eps)
        derived_margin = max(0.0, 1.0 - math.sqrt(eps))
        margin = derived_margin if safety_margin is None else float(safety_margin)
        if not (0.0 < margin <= 1.0):
            raise ValueError(
                f"safety_margin must satisfy 0 < safety_margin <= 1, got {margin}",
            )

        if safety_margin is None:
            logger.info(
                "Derived safety margin from dtype precision: 1-sqrt(eps)=%.8f",
                margin,
            )

        injected = 0

        for key in target_modules:
            geom = geometries.get(key)
            if geom is None:
                continue

            # Determine rank and scale bound based on null-space availability
            if geom.tail_dims > 0:
                # Null-space available: rank from tail_dims, scale from σ_max
                rank = rank_overrides[key] if rank_overrides and key in rank_overrides else geom.tail_dims
                if rank <= 0:
                    logger.warning("Skipping %s: rank_override=%d is non-positive", key, rank)
                    continue
                if rank > geom.tail_dims:
                    logger.warning(
                        "Clamping %s: rank_override=%d exceeds tail_dims=%d",
                        key, rank, geom.tail_dims,
                    )
                    rank = geom.tail_dims
                # Geometry-derived scale bound: 2 * max(S) <= sigma_max
                # Per-step displacement bounded by MASS (eta_weyl = σ_k_min / ||g||).
                scale_bound = (geom.sigma_max / 2.0) * margin
            elif geom.spectral_gap > 0:
                # Full-rank layer: rank-1 adaptation, scale from spectral gap.
                # gap/2 is Weyl crossing threshold, /2 for NB-LoRA factor of 2.
                rank = 1
                scale_bound = (geom.spectral_gap / 4.0) * margin
                logger.info(
                    "Zero-tail module %s: rank=1, spectral_gap=%.6f, scale_bound=%.6f",
                    key, geom.spectral_gap, scale_bound,
                )
            else:
                logger.debug("Skipping %s: tail_dims=0 and spectral_gap=0", key)
                continue

            if scale_bound <= 0:
                logger.warning("Skipping %s: scale_bound=%.6f is non-positive", key, scale_bound)
                continue

            try:
                parent, attr_name = self._resolve_parent_and_attr(model, key)
                linear = getattr(parent, attr_name)

                nb_lora = NBLoRALinear.from_base(
                    linear,
                    rank=rank,
                    scale_bound=scale_bound,
                )

                # Compute k-th singular vectors for projected residual monitoring.
                # If the backend cannot provide the vectors (for example, resource
                # pressure on large weights), continue without this diagnostic.
                structural_rank = geom.full_rank - geom.tail_dims
                try:
                    if isinstance(linear, nn.QuantizedLinear):
                        weight_f32 = mx.dequantize(
                            linear.weight, linear.scales, linear.biases,
                            linear.group_size, linear.bits,
                        )
                    else:
                        weight_f32 = linear.weight
                    u_k, v_k, quality = compute_initialization_vectors(
                        weight_f32, structural_rank, self._backend,
                        seed=hash(key) & 0xFFFFFFFF,
                    )
                    nb_lora.set_initialization_vectors(u_k, v_k)
                    logger.debug(
                        "Init vectors at %s: structural_rank=%d, quality=%.4f",
                        key, structural_rank, quality,
                    )
                except Exception as exc:
                    logger.debug("Skipped init vectors at %s: %s", key, exc)

                setattr(parent, attr_name, nb_lora)
                injected += 1

                logger.debug(
                    "Injected NB-LoRA at %s: rank=%d, σ_k=%.6f, bound=%.6f",
                    key, rank, geom.sigma_k, scale_bound,
                )
            except Exception as exc:
                logger.warning("Failed to inject NB-LoRA at %s: %s", key, exc)

        logger.info("Injected %d NB-LoRA layers (bounds by construction)", injected)
        return injected

    def freeze_and_apply_lora(self, model) -> None:
        """Freeze entire model, then unfreeze only NB-LoRA parameters."""
        model.freeze()
        # Also freeze visual encoder if present (Qwen3.5-VL).
        # model.freeze() only covers model's own parameters. For VL models loaded
        # via load_qwen35_vl_model(), model.visual is attached separately and must
        # be frozen explicitly before selective unfreezing below.
        visual = getattr(model, "visual", None)
        if visual is not None:
            visual.freeze()
        # Walk model tree and unfreeze A_tilde, B_tilde, S_raw in each NBLoRALinear
        for _, nb_lora in self._iter_nb_lora_modules(model):
            nb_lora.unfreeze(keys=["A_tilde", "B_tilde", "S_raw"])
        # Qwen3.5 GatedDeltaNet: mlx-lm loads models with training=False (eval
        # mode), so GatedDeltaNet uses use_kernel=True (Metal kernel) — which has
        # no VJP. Fix: (1) set model.train() so use_kernel=False → ops path,
        # (2) patch the @mx.compile'd ops-path functions to uncompiled equivalents
        # so autograd can differentiate through them.
        from modelcypher.backends._mlx_qwen35_compat import (  # noqa: PLC0415
            _is_qwen35,
            apply_qwen35_training_patch,
        )
        if _is_qwen35(model):
            model.train()  # training=True → use_kernel=False → ops path
            apply_qwen35_training_patch()

    def run_train_probe_step(
        self,
        model,
        tokenizer,
        *,
        prompt: str,
        seed: int = 0,
        use_randomized_geometry: bool = True,
        randomized_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one bounded NB-LoRA probe step for training-memory measurement.

        Probe sequence:
        1. Streaming geometry analysis
        2. NB-LoRA injection (geometry-derived targets/ranks)
        3. Freeze base, unfreeze adapter params
        4. Single train-loop iteration (forward + backward + update)
        5. Spectral bound verification
        """
        from modelcypher.core.domain.training.geometric_lora import (
            apply_data_rank_ceiling,
            compute_coupled_ranks,
            select_target_modules,
        )

        sample_tokens = tokenizer.encode(prompt)
        if len(sample_tokens) < 2:
            raise ValueError("train probe requires at least 2 tokens")
        train_dataset = self.prepare_dataset([{"text": prompt}], tokenizer)
        if not train_dataset:
            raise ValueError("train probe dataset is empty after tokenization")
        seq_length = int(train_dataset[0][0].shape[0])

        geometries = self.analyze_model_geometry_streaming(
            model,
            use_randomized=use_randomized_geometry,
            randomized_kwargs=randomized_kwargs,
        )
        target_modules = select_target_modules(geometries)
        if not target_modules:
            raise ValueError("no targetable modules found for train probe")

        coupled_ranks = compute_coupled_ranks(geometries, target_modules)
        rank_overrides = apply_data_rank_ceiling(
            coupled_ranks,
            n_samples=len(train_dataset),
        )
        n_lora_layers = self.inject_nb_lora(
            model,
            geometries,
            target_modules,
            safety_margin=None,
            rank_overrides=rank_overrides,
        )
        if n_lora_layers <= 0:
            raise ValueError("NB-LoRA injection produced zero adapted layers")

        self.freeze_and_apply_lora(model)

        n_trainable_params = int(
            sum(param.size for _, param in self._backend.tree_flatten(model.trainable_parameters()))
        )

        sigma_candidates = [g.sigma_max for g in geometries.values() if g.sigma_max > 0]
        if not sigma_candidates:
            raise ValueError("unable to derive sigma_max from geometry")
        sigma_max = max(sigma_candidates)
        sigma_k_candidates = [g.sigma_k for g in geometries.values() if g.sigma_k > 0]
        if not sigma_k_candidates:
            raise ValueError("unable to derive sigma_k_min from geometry")
        sigma_k_min = min(sigma_k_candidates)

        losses, stop_reason, _ = self.train_loop(
            model=model,
            train_dataset=train_dataset,
            batch_size=1,
            seq_length=seq_length,
            max_iters=1,
            seed=seed,
            sigma_max=sigma_max,
            sigma_k_min=sigma_k_min,
            eval_dataset=None,
            eval_batches=1,
            adaptive_lr=True,
            tokenizer=tokenizer,
        )
        spectral_bounds_ok, max_spectral_ratio, _ = self.verify_bounds(model)

        last_loss = None
        if losses:
            # losses entries are (iter_idx, train_loss, val_loss)
            last_loss = float(losses[-1][1])

        return {
            "n_lora_layers": int(n_lora_layers),
            "n_trainable_params": int(n_trainable_params),
            "spectral_bounds_ok": bool(spectral_bounds_ok),
            "max_spectral_ratio": float(max_spectral_ratio),
            "train_iters": int(len(losses)),
            "last_loss": last_loss,
            "stop_reason": str(stop_reason),
            "geometry_mode": "randomized" if use_randomized_geometry else "full_svd",
            "seq_length": int(seq_length),
            "target_module_count": int(len(target_modules)),
            "sigma_k_min": float(sigma_k_min),
            "sigma_max": float(sigma_max),
        }

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
        from mlx_lm.tuner.trainer import default_loss, iterate_batches

        is_vl_dataset = (
            isinstance(dataset, list)
            and len(dataset) > 0
            and isinstance(dataset[0], dict)
            and "tokens" in dataset[0]
            and "pixel_values" in dataset[0]
        )
        if is_vl_dataset:
            image_token_id = dataset[0].get("image_token_id")
            video_token_id = dataset[0].get("video_token_id")
            vl_loss = make_vl_loss(
                image_token_id=image_token_id,
                video_token_id=video_token_id,
            )

        total_loss = 0.0
        total_tokens = 0.0
        n_evaluated = 0

        if is_vl_dataset:
            batch_iter = iterate_vl_batches(
                dataset,
                batch_size,
                seq_length,
                loop=False,
            )
        else:
            batch_iter = iterate_batches(
                dataset,
                batch_size,
                seq_length,
                loop=False,
            )

        for batch_item in batch_iter:
            if is_vl_dataset:
                batch, lengths, pixel_values_batch, position_ids_batch = batch_item
                loss, ntoks = vl_loss(
                    model,
                    batch,
                    lengths,
                    pixel_values_batch,
                    position_ids_batch,
                )
            else:
                batch, lengths = batch_item
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
        max_log = math.log(self._backend.finfo().max)
        perplexity = math.exp(min(avg_loss, max_log))
        return avg_loss, perplexity

    def measure_sample_losses(
        self,
        *,
        model,
        tokenizer,
        samples: list[dict[str, Any]],
        seq_length: int,
    ) -> list[float]:
        """Measure per-sample CE losses for pilot validation split derivation."""
        from mlx_lm.tuner.trainer import default_loss, iterate_batches

        losses: list[float] = []
        for sample in samples:
            dataset = self.prepare_dataset([sample], tokenizer)
            if not dataset:
                losses.append(float("inf"))
                continue
            sample_loss = float("inf")
            for batch, lengths in iterate_batches(dataset, 1, seq_length, loop=False):
                loss, _ntoks = default_loss(model, batch, lengths)
                mx.eval(loss)
                sample_loss = float(loss)
                break
            losses.append(sample_loss)
        return losses

    def _evaluate_masked_loss(
        self,
        model,
        masked_dataset: list[tuple[Any, Any, int]],
        batch_size: int,
        seq_length: int,
        n_batches: int,
    ) -> float:
        """Compute average answer-masked loss over a dataset."""
        total_loss = 0.0
        total_answer_tokens = 0.0
        n_evaluated = 0

        for inputs, targets, masks in iterate_masked_batches(
            masked_dataset, batch_size, seq_length, train=False, loop=False,
        ):
            logits = model(inputs)
            logits = logits.astype(mx.float32)
            ce = nn.losses.cross_entropy(logits, targets, reduction="none")
            masked_ce = ce * masks
            ntoks = masks.sum()
            batch_loss = masked_ce.sum()
            mx.eval(batch_loss, ntoks)
            total_loss += float(batch_loss)
            total_answer_tokens += float(ntoks)
            n_evaluated += 1
            if n_evaluated >= n_batches:
                break

        if total_answer_tokens == 0:
            return float("inf")

        return total_loss / total_answer_tokens

    def measure_baseline_constraints(
        self,
        model,
        tokenizer,
        paired_dataset: list[dict[str, Any]],
        logic_groups: dict[str, list[int]],
        template_groups: dict[str, list[int]],
        target_layers: list[int],
        # Default targets short-form reasoning traces; callers should override
        # with measured dataset lengths for longer-context domains.
        max_seq_length: int = 256,
    ) -> tuple[list[float], list[float], dict[int, float], dict[int, float]]:
        """Measure baseline invariance/separation distances and spectral entropy.

        Runs on the BASE model (before NB-LoRA injection) to derive constraint
        thresholds. All thresholds come from geometry, not heuristics.

        Returns:
            (inv_distances, sep_distances, layer_entropies, layer_entropy_stds) where:
            - inv_distances: L2 distances between invariance pairs (same logic)
            - sep_distances: L2 distances between counterfactual pairs (same template)
            - layer_entropies: mean effective rank per target layer
            - layer_entropy_stds: per-layer effective-rank spread across samples
        """
        target_layers_set = set(target_layers)

        # Collect hidden states at target layers for a subset of samples
        n_samples = len(paired_dataset)
        sample_indices = list(range(n_samples))

        # Forward pass each sample, collect hidden states per layer.
        # Store BOTH mean-pooled (for C_inv/C_sep distances) and full token-level
        # (for C_geo spectral entropy) to match training loss computation.
        hidden_states_mean: list[dict[int, Any]] = []  # per sample: {layer: [hidden]}
        hidden_states_full: list[dict[int, Any]] = []  # per sample: {layer: [seq, hidden]}

        base, _ = self._get_model_base(model)
        for idx in sample_indices:
            s = paired_dataset[idx]
            tokens = s["tokens"][:max_seq_length].reshape(1, -1)

            h = base.embed_tokens(tokens)

            layer_h_mean: dict[int, Any] = {}
            layer_h_full: dict[int, Any] = {}
            for layer_idx, layer in enumerate(base.layers):
                # Route masks per layer type (LFM2 hybrid architecture)
                if getattr(layer, "is_attention_layer", True):
                    layer_mask = "causal"
                else:
                    layer_mask = None
                h = layer(h, mask=layer_mask, cache=None)
                if isinstance(h, tuple):
                    h = h[0]
                if layer_idx in target_layers_set:
                    # Mean pool for C_inv/C_sep distance computation
                    mean_h = mx.mean(h, axis=(0, 1))
                    mx.eval(mean_h)
                    layer_h_mean[layer_idx] = mean_h
                    # Full token states for C_geo spectral entropy
                    # h is [1, seq, hidden] -> squeeze to [seq, hidden]
                    full_h = h.reshape(-1, h.shape[-1])
                    mx.eval(full_h)
                    layer_h_full[layer_idx] = full_h

            hidden_states_mean.append(layer_h_mean)
            hidden_states_full.append(layer_h_full)

        # Compute pairwise distances (using mean-pooled hidden states)
        inv_distances: list[float] = []
        sep_distances: list[float] = []

        # Invariance: same logic_id, different template_id
        for lid, members in logic_groups.items():
            active = [i for i in members if i < n_samples]
            for a in range(len(active)):
                for b in range(a + 1, len(active)):
                    ia, ib = active[a], active[b]
                    if paired_dataset[ia]["template_id"] == paired_dataset[ib]["template_id"]:
                        continue  # skip same template (not a true invariance pair)
                    for layer_idx in target_layers:
                        if layer_idx in hidden_states_mean[ia] and layer_idx in hidden_states_mean[ib]:
                            diff = hidden_states_mean[ia][layer_idx] - hidden_states_mean[ib][layer_idx]
                            dist = float(mx.sqrt(mx.sum(diff * diff)).item())
                            inv_distances.append(dist)

        # Separation: same template_id, different logic_id
        for tid, members in template_groups.items():
            active = [i for i in members if i < n_samples]
            for a in range(len(active)):
                for b in range(a + 1, len(active)):
                    ia, ib = active[a], active[b]
                    if paired_dataset[ia]["logic_id"] == paired_dataset[ib]["logic_id"]:
                        continue
                    for layer_idx in target_layers:
                        if layer_idx in hidden_states_mean[ia] and layer_idx in hidden_states_mean[ib]:
                            diff = hidden_states_mean[ia][layer_idx] - hidden_states_mean[ib][layer_idx]
                            dist = float(mx.sqrt(mx.sum(diff * diff)).item())
                            sep_distances.append(dist)

        # Effective rank per target layer (differentiable proxy for spectral entropy).
        # Uses trace(G)²/||G||_F² (Roy & Vetterli 2007) to match the training loss
        # computation which also uses this formula (SVD has no VJP in MLX).
        # We compute per-sample effective rank to estimate local layer variability
        # (mean + std), then derive layer-specific targets from that variability.
        layer_entropies: dict[int, float] = {}
        layer_entropy_stds: dict[int, float] = {}
        for layer_idx in target_layers:
            erank_vals: list[float] = []
            for hs in hidden_states_full:
                if layer_idx in hs:
                    flat = hs[layer_idx].astype(mx.float32)
                    # Gram matrix G = X^T X
                    G = flat.T @ flat
                    trace_G = float(mx.sum(mx.diag(G)).item())
                    frobenius_sq = float(mx.sum(G * G).item())
                    eps_rank = float(division_epsilon(self._backend, G))
                    erank = (trace_G * trace_G) / (frobenius_sq + eps_rank)
                    erank_vals.append(erank)
            if not erank_vals:
                continue
            mean_erank = sum(erank_vals) / len(erank_vals)
            if len(erank_vals) >= 2:
                variance = sum((v - mean_erank) ** 2 for v in erank_vals) / (len(erank_vals) - 1)
                std_erank = math.sqrt(variance)
            else:
                std_erank = 0.0
            layer_entropies[layer_idx] = mean_erank
            layer_entropy_stds[layer_idx] = std_erank

        logger.info(
            "Baseline constraints: %d inv_distances (mean=%.4f), "
            "%d sep_distances (mean=%.4f), %d layer_entropies "
            "(mean erank=%.4f, mean std=%.4f)",
            len(inv_distances),
            sum(inv_distances) / max(1, len(inv_distances)),
            len(sep_distances),
            sum(sep_distances) / max(1, len(sep_distances)),
            len(layer_entropies),
            sum(layer_entropies.values()) / max(1, len(layer_entropies)),
            sum(layer_entropy_stds.values()) / max(1, len(layer_entropy_stds)),
        )

        return inv_distances, sep_distances, layer_entropies, layer_entropy_stds

    def compute_mean_gradient(
        self,
        model,
        tokenizer,
        samples: list[dict],
        n_samples: int | None = None,
    ) -> "Array":
        """Compute mean gradient direction over samples. Returns float32 MLX array.

        Used for format bias decomposition: μ = (1/N) Σ ∇L(x_i).
        Only includes LoRA parameter gradients (A_tilde, B_tilde, lora_a, lora_b).
        """
        from mlx.utils import tree_flatten as mlx_flatten
        from mlx_lm.tuner.trainer import default_loss

        loss_vg = nn.value_and_grad(model, default_loss)

        if n_samples is not None:
            samples = samples[:n_samples]

        # Tokenize
        dataset = []
        for s in samples:
            text = s.get("text", "")
            if not text:
                continue
            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue
            dataset.append(mx.array(tokens, dtype=mx.int32))

        sum_g = None
        count = 0

        for tokens in dataset:
            batch = tokens.reshape(1, -1)
            lengths = mx.array([[0, batch.shape[1]]])
            (loss, ntoks), grad = loss_vg(model, batch, lengths)
            mx.eval(loss)

            flat = []
            for name, arr in mlx_flatten(grad):
                if any(k in name for k in ('A_tilde', 'B_tilde', 'lora_a', 'lora_b')):
                    flat.append(arr.reshape(-1).astype(mx.float32))
            if flat:
                g = mx.concatenate(flat)
                mx.eval(g)
                if sum_g is None:
                    sum_g = g
                else:
                    sum_g += g
                count += 1

        if count == 0:
            raise RuntimeError("No valid gradients computed for format bias")
        return (sum_g / count).astype(mx.float32)

    def build_projection_hook(self, v_format: "Array"):
        """Build a gradient hook that projects out the format bias direction.

        Args:
            v_format: [d] float32 abstract Array (MLX array) — unit format bias direction

        Returns:
            Callable that takes a gradient pytree and returns a decontaminated pytree.
        """
        from mlx.utils import tree_flatten as mlx_flatten
        from mlx.utils import tree_unflatten

        mx.eval(v_format)

        def hook(grad):
            flat = dict(mlx_flatten(grad))
            pieces = []
            lora_keys = []
            for key in flat:
                if any(k in key for k in ('A_tilde', 'B_tilde', 'lora_a', 'lora_b')):
                    lora_keys.append(key)
                    pieces.append(flat[key].reshape(-1).astype(mx.float32))
            if not pieces:
                return grad

            g_vec = mx.concatenate(pieces)
            mx.eval(g_vec)

            # Project out: g_clean = g - (v^T g) v
            coeff = mx.sum(v_format * g_vec)
            g_clean = g_vec - coeff * v_format
            mx.eval(g_clean)

            # Unflatten back
            offset = 0
            for key in lora_keys:
                size = flat[key].size
                shape = flat[key].shape
                flat[key] = g_clean[offset:offset + size].reshape(shape)
                offset += size

            return tree_unflatten(flat)

        return hook

    def _derive_entropy_floor_or_fail(
        self,
        *,
        baseline_entropy: float | None,
        dataset_samples: int,
        scope: str,
    ) -> float:
        """Derive entropy floor or raise strict insufficiency failure."""
        if baseline_entropy is None or baseline_entropy <= 0.0:
            baseline_status = (
                "baseline_non_positive"
                if baseline_entropy is not None
                else "baseline_unavailable"
            )
            raise TrainingDerivationError(
                failure_class="insufficient_entropy_baseline",
                detail=(
                    f"Entropy regularization requested in {scope} but baseline entropy "
                    "could not be measured."
                ),
                diagnostics={
                    "baseline_entropy_status": baseline_status,
                    "baseline_entropy_value": baseline_entropy,
                    "dataset_samples": dataset_samples,
                    "checks": [
                        "tokenization",
                        "dataset_non_empty",
                        "logits_finite",
                    ],
                },
            )
        eps_f32 = float(mx.finfo(mx.float32).eps)
        return baseline_entropy * (1.0 - eps_f32 ** 0.5)

    def _derive_spectral_ceiling(
        self,
        *,
        sigma_k_min: float,
        sigma_max_global: float,
    ) -> float:
        """Derive static learning rate ceiling from adapter geometry (Weyl 1912).

        Delegates to :func:`mass_step_size.derive_spectral_ceiling` for the
        pure-math derivation. This method adds logging.
        """
        from modelcypher.core.domain.training.mass_step_size import (
            derive_spectral_ceiling,
        )

        ceiling = derive_spectral_ceiling(
            sigma_k_min=sigma_k_min,
            sigma_max_global=sigma_max_global,
        )
        logger.info(
            "Spectral ceiling (Weyl): eta_ceiling = sigma_k_min/sigma_max = %.4e/%.4e = %.4e",
            sigma_k_min, sigma_max_global, ceiling,
        )
        return ceiling
