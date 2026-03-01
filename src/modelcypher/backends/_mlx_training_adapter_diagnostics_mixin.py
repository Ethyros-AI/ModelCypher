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

"""Diagnostics and utility methods for :class:`MLXTrainingAdapter`."""

from __future__ import annotations

from typing import Any

from modelcypher.backends.mlx_training_adapter_core import *  # noqa: F403


class _MLXTrainingAdapterDiagnosticsMixin:
    def _compute_certificate_quantities(
        self,
        model,
        grad: Any,
        eval_dataset,
        batch_size: int,
        seq_length: int,
        n_batches: int,
        mean_token_entropy: float | None,
        repetition_rate: float | None,
        grad_norm_history: list[float] | None = None,
        seed: int = 0,
    ):
        """Compute all quantities for the geometric stopping certificate.

        Orchestrates: gradient norm, validation gradient,
        Hessian-vector product, bootstrap CI, per-batch worst-group bounds.

        Returns:
            ``StoppingCertificate`` from ``check_stopping_certificate()``.
        """
        from mlx.utils import tree_flatten as mlx_flatten

        from modelcypher.core.domain.training.geometric_early_stopping import (
            check_stopping_certificate,
        )
        from modelcypher.core.support.statistics import bootstrap_ci

        # 1. Gradient norm: ||g|| = sqrt(g^T g)
        # P removed (falsification 2026-02-23: P ≈ I, weight space Euclidean).
        grad_flat = dict(mlx_flatten(grad))
        dot = sum(
            float(mx.sum(grad_flat[k] * grad_flat[k]))
            for k in grad_flat
        )
        grad_norm = math.sqrt(max(dot, 0.0))

        # 2. Direction d_t = g_t (no preconditioner)
        d_t = grad_flat

        # 3. Validation gradient
        grad_val = self._compute_val_gradient(
            model, eval_dataset, batch_size, seq_length, n_batches,
        )

        alignment = 0.0
        curvature = 0.0
        per_batch_alignments: list[float] = []
        per_batch_curvatures: list[float] = []
        per_batch_ci_half_widths: list[float] = []

        if grad_val is not None:
            # 4. Alignment: a_t = grad_val^T @ d_t
            alignment = sum(
                float(mx.sum(grad_val[k] * d_t[k]))
                for k in grad_val if k in d_t
            )

            # 5. Val HVP: H_val @ d_t
            hvp = self._compute_val_hvp(
                model, eval_dataset, batch_size, seq_length, n_batches, d_t,
            )
            if hvp is not None:
                # 6. Curvature: b_t = d_t^T @ H_val @ d_t
                curvature = sum(
                    float(mx.sum(d_t[k] * hvp[k]))
                    for k in d_t if k in hvp
                )

            # 7. Per-batch worst-group: per-batch alignment with aggregate curvature
            from mlx_lm.tuner.trainer import default_loss, iterate_batches

            loss_vg = nn.value_and_grad(model, default_loss)
            batch_count = 0
            for batch, lengths in iterate_batches(
                eval_dataset, batch_size, seq_length, loop=False,
            ):
                if batch_count >= n_batches:
                    break
                try:
                    (loss_i, _), grads_i = loss_vg(model, batch, lengths)
                    mx.eval(loss_i)
                    flat_i = dict(mlx_flatten(grads_i))
                    a_i = sum(
                        float(mx.sum(flat_i[k] * d_t[k]))
                        for k in flat_i if k in d_t
                    )
                    per_batch_alignments.append(a_i)
                    # Use aggregate curvature (avoids per-batch HVP)
                    per_batch_curvatures.append(curvature)
                except Exception:
                    pass
                batch_count += 1

        # 8. Per-batch losses for bootstrap CI
        per_batch_losses = self._compute_per_batch_val_losses(
            model, eval_dataset, batch_size, seq_length, n_batches,
        )

        val_ci_half_width = 0.0
        if len(per_batch_losses) >= 2:
            # Data-derived bootstrap parameters (G1 compliance):
            # n_bootstrap = n^2 — quadratic in sample count ensures
            # bootstrap SE ∝ 1/n, matching the CLT convergence rate.
            # confidence = 1 - 1/n_bootstrap — tail probability scales
            # inversely with resample count (tighter CI with more data).
            n = len(per_batch_losses)
            _n_bootstrap = n * n
            _confidence = 1.0 - 1.0 / _n_bootstrap
            lower, upper = bootstrap_ci(
                per_batch_losses,
                confidence=_confidence,
                n_bootstrap=_n_bootstrap,
                seed=seed,
            )
            ci_half = (upper - lower) / 2.0
            # Keep CI distinguishable from numerical noise. A zero-width CI with
            # real validation data makes the improvement test vacuous.
            numeric_floor = math.sqrt(math.ldexp(1.0, -23))
            val_ci_half_width = max(ci_half, numeric_floor)

        # Per-batch CI half-widths (each batch gets the aggregate CI as proxy)
        per_batch_ci_half_widths = [val_ci_half_width] * len(per_batch_alignments)

        return check_stopping_certificate(
            grad_norm=grad_norm,
            grad_norm_history=grad_norm_history,
            alignment=alignment,
            curvature=curvature,
            val_ci_half_width=val_ci_half_width,
            per_batch_alignments=per_batch_alignments if per_batch_alignments else None,
            per_batch_curvatures=per_batch_curvatures if per_batch_curvatures else None,
            per_batch_ci_half_widths=per_batch_ci_half_widths if per_batch_ci_half_widths else None,
            mean_token_entropy=mean_token_entropy,
            repetition_rate=repetition_rate,
        )

    def _iter_nb_lora_modules(self, model):
        """Yield (layer_key, NBLoRALinear) pairs from model tree."""
        base = getattr(model, "model", model)
        if not hasattr(base, "layers"):
            return

        for layer_idx, layer in enumerate(base.layers):
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    proj = getattr(attn, proj_name, None)
                    if isinstance(proj, NBLoRALinear):
                        key = f"model.layers.{layer_idx}.self_attn.{proj_name}.weight"
                        yield key, proj

            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                for proj_name in ("up_proj", "down_proj", "gate_proj"):
                    proj = getattr(mlp, proj_name, None)
                    if isinstance(proj, NBLoRALinear):
                        key = f"model.layers.{layer_idx}.mlp.{proj_name}.weight"
                        yield key, proj

                experts = getattr(mlp, "experts", None)
                if experts is not None:
                    if isinstance(experts, dict):
                        expert_items = list(experts.items())
                    else:
                        expert_items = list(enumerate(experts))
                    for raw_expert_idx, expert in expert_items:
                        if expert is None:
                            continue
                        expert_idx = int(raw_expert_idx)
                        for proj_name in ("gate_proj", "up_proj", "down_proj"):
                            proj = getattr(expert, proj_name, None)
                            if isinstance(proj, NBLoRALinear):
                                key = (
                                    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}."
                                    f"{proj_name}.weight"
                                )
                                yield key, proj

                shared_expert = getattr(mlp, "shared_expert", None)
                if shared_expert is not None:
                    for proj_name in ("gate_proj", "up_proj", "down_proj"):
                        proj = getattr(shared_expert, proj_name, None)
                        if isinstance(proj, NBLoRALinear):
                            key = (
                                f"model.layers.{layer_idx}.mlp.shared_expert."
                                f"{proj_name}.weight"
                            )
                            yield key, proj

    def _compute_topological_metrics(
        self,
        model,
        tokenizer,
        probe_texts: list[str],
    ) -> dict[str, Any]:
        """Compute topological fingerprint and Ricci curvature from activation cloud.

        Experimental. Used for grokking phase detection: tracks Betti numbers,
        persistence entropy, and Ollivier-Ricci curvature per epoch.

        Args:
            model: Current model state.
            tokenizer: Tokenizer for the model.
            probe_texts: Short texts to collect activations from.

        Returns:
            Dict with topo_betti_0, topo_betti_1, topo_persistence_entropy,
            topo_mean_ricci_curvature, topo_ricci_curvature_std.
        """
        try:
            # 1. Collect activations for probe texts
            acts = self._backend.collect_hidden_activations(
                model, tokenizer, probe_texts,
            )
            if not acts:
                return {}

            # 2. Pool to [n_probes, hidden_dim] for middle layer
            mid_layer = max(acts.keys()) // 2
            layer_act = acts[mid_layer]  # [batch, seq, hidden]
            # Mean-pool over seq dimension → [batch, hidden]
            pooled = self._backend.mean(layer_act, axis=1)
            self._backend.eval(pooled)

            # Convert to list[list[float]] for TopologicalFingerprint
            points = pooled.tolist()

            # 3. Topological fingerprint (Betti numbers + persistence entropy)
            from modelcypher.core.domain.geometry.topological_fingerprint import (
                BackendTopologicalFingerprint,
            )

            topo = BackendTopologicalFingerprint(self._backend)
            fingerprint = topo.compute(points)

            # 4. Ollivier-Ricci curvature
            from modelcypher.core.domain.geometry.ollivier_ricci import (
                OllivierRicciCurvature,
            )

            orc = OllivierRicciCurvature(self._backend)
            ricci = orc.compute(pooled)

            return {
                "topo_betti_0": fingerprint.betti_numbers.get(0, 0),
                "topo_betti_1": fingerprint.betti_numbers.get(1, 0),
                "topo_persistence_entropy": fingerprint.summary.persistence_entropy,
                "topo_mean_ricci_curvature": ricci.mean_edge_curvature,
                "topo_ricci_curvature_std": ricci.std_edge_curvature,
            }
        except Exception:
            logger.debug("Topological metrics computation failed", exc_info=True)
            return {}

    def _compute_dimensional_snapshot(
        self,
        model,
        tokenizer,
        probe_texts: list[str],
        epoch: int,
    ):
        """Collect activations on probe texts, compute expansion ratio.

        Processes each text separately to handle variable sequence lengths,
        then merges token-level activations per layer before computing ID.

        Returns a DimensionalSnapshot or None on failure.
        """
        try:
            from modelcypher.core.domain.training.dimensional_monitor import (
                compute_expansion_from_activations,
            )

            # Process each text individually to avoid seq-length mismatch
            merged: dict[int, list] = {}
            for text in probe_texts:
                acts = self._backend.collect_hidden_activations(
                    model, tokenizer, [text],
                )
                if not acts:
                    continue
                for layer_idx, act in acts.items():
                    # act shape: [1, seq, hidden] → reshape to [seq, hidden]
                    shape = act.shape
                    if len(shape) == 3:
                        reshaped = self._backend.reshape(act, (shape[1], shape[2]))
                    else:
                        reshaped = act
                    if layer_idx not in merged:
                        merged[layer_idx] = []
                    merged[layer_idx].append(reshaped)

            if not merged:
                return None

            # Concatenate all tokens per layer: list of [seq_i, hidden] → [total_tokens, hidden]
            combined = {}
            for layer_idx, arrays in merged.items():
                combined[layer_idx] = self._backend.concatenate(arrays, axis=0)
                self._backend.eval(combined[layer_idx])

            return compute_expansion_from_activations(combined, self._backend, epoch)
        except Exception:
            logger.debug("Dimensional snapshot computation failed", exc_info=True)
            return None

    def _compute_rss_metrics(
        self,
        model,
        tokenizer,
        base_activations: dict[int, list],
        eval_samples: list | None,
        # Spearman rank-correlation significance is weak below n=10 (Zar 1972);
        # use 2x that minimum for more stable RSS drift estimates.
        n_probes: int = 20,
    ):
        """Compute outer similarity between base and adapted model representations.

        Uses anchor-relative representations at the middle layer.
        Returns an OuterSimilarityResult or None on failure.

        Reference: Kucukahmetler et al. (2026), TMLR.

        Note: RSS alignment != accuracy. These metrics track geometric drift
        of the adapted model relative to the base, not task performance.
        See online_eval for correctness measurement (Kucukahmetler et al. 2026).
        """
        try:
            from modelcypher.core.domain.geometry.relative_representation import (
                compute_anchor_embeddings,
                compute_outer_similarity,
                compute_relative_representation,
            )

            if not eval_samples or not base_activations:
                return None

            # Pick middle layer
            layer_indices = sorted(base_activations.keys())
            if not layer_indices:
                return None
            mid_layer = layer_indices[len(layer_indices) // 2]

            base_list = base_activations.get(mid_layer, [])
            if len(base_list) < 2:
                return None

            # Get anchor embeddings from the (frozen) embedding matrix
            embed_matrix = None
            base_model = model
            # Navigate model structure to find embedding weights
            for attr in ("base", "model"):
                if hasattr(base_model, attr):
                    base_model = getattr(base_model, attr)
            if hasattr(base_model, "embed_tokens"):
                embed_matrix = base_model.embed_tokens.weight
            if embed_matrix is None:
                return None

            anchors, anchor_ids = compute_anchor_embeddings(
                embed_matrix, tokenizer,
            )
            if len(anchor_ids) < 3:
                return None

            base_stack = mx.stack(base_list[:n_probes])
            mx.eval(base_stack)

            # Collect adapted model activations for same probes
            # 200 chars is ~50-67 tokens for English (3-4 chars/token), enough
            # for one sentence-scale activation fingerprint without long-context bias.
            probe_texts = [s["text"][:200] for s in eval_samples[:n_probes]]
            adapted_list = []
            for text in probe_texts[:len(base_list)]:
                acts = self._backend.collect_hidden_activations(
                    model, tokenizer, [text],
                )
                if mid_layer in acts:
                    act = acts[mid_layer]
                    shape = act.shape
                    if len(shape) == 3:
                        pooled = mx.mean(act, axis=1)
                        pooled = mx.reshape(pooled, (-1,))
                    else:
                        pooled = mx.reshape(act, (-1,))
                    mx.eval(pooled)
                    adapted_list.append(pooled)

            if len(adapted_list) != len(base_list[:n_probes]):
                return None

            adapted_stack = mx.stack(adapted_list)
            mx.eval(adapted_stack)

            # Compute relative representations and outer similarity
            rel_base = compute_relative_representation(base_stack, anchors)
            rel_adapted = compute_relative_representation(adapted_stack, anchors)
            return compute_outer_similarity(rel_base, rel_adapted)
        except Exception:
            logger.debug("RSS metric computation failed", exc_info=True)
            return None

    def _probe_entropy_and_repetition(
        self,
        model,
        tokenizer,
        n_sequences: int = 3,
        # For 4-gram repetition, >=16 generated tokens gives >=4 n-gram instances;
        # 64 keeps a 4x margin while staying lightweight.
        max_tokens: int = 64,
    ) -> tuple[float | None, float | None]:
        """Generate short sequences and measure entropy + repetition.

        Returns (mean_token_entropy, repetition_rate) or (None, None) on failure.
        Entropy: mean Shannon entropy per token across all generated sequences.
        Repetition: fraction of 4-grams that are repeated.
        """
        if tokenizer is None:
            return None, None

        prompts = ["The", "Once upon a time", "In the beginning"]

        all_entropies: list[float] = []
        all_tokens: list[int] = []

        try:
            for prompt in prompts[:n_sequences]:
                input_ids = mx.array(tokenizer.encode(prompt))[None]  # (1, seq_len)

                generated_tokens: list[int] = []
                for _ in range(max_tokens):
                    logits = model(input_ids)  # (1, seq_len, vocab_size)
                    next_logits = logits[:, -1, :]  # (1, vocab_size)

                    # Entropy from softmax distribution
                    probs = mx.softmax(next_logits, axis=-1)
                    log_probs = mx.log(probs + mx.finfo(probs.dtype).tiny)
                    entropy = -mx.sum(probs * log_probs, axis=-1)
                    all_entropies.append(float(entropy[0]))

                    # Greedy next token
                    next_token = int(mx.argmax(next_logits, axis=-1)[0])
                    generated_tokens.append(next_token)

                    # EOS check
                    if hasattr(tokenizer, "eos_token_id") and next_token == tokenizer.eos_token_id:
                        break

                    input_ids = mx.concatenate(
                        [input_ids, mx.array([[next_token]])], axis=1,
                    )

                all_tokens.extend(generated_tokens)

        except Exception:
            logger.debug("Entropy/repetition probe failed", exc_info=True)
            return None, None

        mean_entropy = sum(all_entropies) / len(all_entropies) if all_entropies else None

        # 4-gram repetition rate (Papineni et al. 2002 BLEU uses up to 4-grams).
        n = 4
        if len(all_tokens) >= n:
            ngrams = [tuple(all_tokens[i : i + n]) for i in range(len(all_tokens) - n + 1)]
            repetition_rate = 1.0 - len(set(ngrams)) / len(ngrams) if ngrams else 0.0
        else:
            repetition_rate = 0.0

        return mean_entropy, repetition_rate

    def derive_critical_batch_size(
        self,
        model,
        train_dataset,
        seq_length: int,
        n_samples: int | None = None,
    ) -> int:
        """Derive critical batch size from gradient noise scale.

        B_crit = Var(g) / ||E[g]||^2 = 1 / SNR

        Uses two streaming passes over per-sample gradients (batch_size=1):
        1) mean gradient + mean norm
        2) variance around the mean gradient

        This avoids storing all per-sample gradient trees in memory.
        Same math as event-buffer path in lora_memory_store.derive_critical_batch_size().

        n_samples: number of samples for variance estimation.  When None
        (default), derived from dataset size: min(len(dataset), max(20,
        len(dataset) // 10)).  Floor of 20 provides a statistical minimum
        for second-moment estimation; 10 % of dataset scales with data.
        """
        if n_samples is None:
            n_samples = min(len(train_dataset), max(20, len(train_dataset) // 10))
        from mlx.utils import tree_flatten as mlx_flatten
        from mlx_lm.tuner.trainer import default_loss, iterate_batches

        loss_vg = nn.value_and_grad(model, default_loss)

        def _norm_sq(flat_grads: dict[str, Any]) -> float:
            total = 0.0
            for grad in flat_grads.values():
                g = grad.astype(mx.float32)
                total += float(mx.sum(g * g))
            return total

        count = 0
        total_norm_sum = 0.0
        mean_grad: dict[str, Any] = {}

        # Pass 1: accumulate mean gradient and mean norm.
        for batch, lengths in iterate_batches(
            train_dataset, 1, seq_length, loop=False, seed=0,
        ):
            (loss, _), grads = loss_vg(model, batch, lengths)
            mx.eval(loss)

            flat_grads = dict(mlx_flatten(grads))
            if flat_grads:
                mx.eval(*flat_grads.values())

            norm_sq = _norm_sq(flat_grads)
            total_norm_sum += math.sqrt(norm_sq)

            if count == 0:
                mean_grad = {k: v.astype(mx.float32) for k, v in flat_grads.items()}
            else:
                for k, v in flat_grads.items():
                    if k in mean_grad:
                        mean_grad[k] = mean_grad[k] + v.astype(mx.float32)
                    else:
                        mean_grad[k] = v.astype(mx.float32)

            count += 1
            if count >= n_samples:
                break

        if count < 2:
            logger.info("Too few samples for B_crit estimation, defaulting to 1")
            return 1

        inv_count = 1.0 / float(count)
        for key, grad_sum in list(mean_grad.items()):
            mean_grad[key] = grad_sum * inv_count
        if mean_grad:
            mx.eval(*mean_grad.values())

        total_norm_sum / float(count)

        # Pass 2: accumulate variance around mean gradient.
        variance_sum = 0.0
        second_count = 0
        for batch, lengths in iterate_batches(
            train_dataset, 1, seq_length, loop=False, seed=0,
        ):
            (loss, _), grads = loss_vg(model, batch, lengths)
            mx.eval(loss)

            flat_grads = dict(mlx_flatten(grads))
            if flat_grads:
                mx.eval(*flat_grads.values())

            diff_sq = 0.0
            for key, grad in flat_grads.items():
                if key not in mean_grad:
                    continue
                diff = grad.astype(mx.float32) - mean_grad[key]
                diff_sq += float(mx.sum(diff * diff))
            variance_sum += diff_sq

            second_count += 1
            if second_count >= count:
                break

        variance = variance_sum / float(count - 1)
        mean_grad_norm_sq = _norm_sq(mean_grad)
        eps = math.ldexp(1.0, -23)  # IEEE 754 float32 epsilon
        snr = mean_grad_norm_sq / (variance + eps)

        if not math.isfinite(snr) or snr <= 0:
            logger.info("Could not estimate gradient noise, defaulting to 1")
            return 1

        b_crit = max(1, math.ceil(1.0 / snr))
        b_crit = min(b_crit, len(train_dataset))

        logger.info(
            "B_crit = %d (SNR=%.6f, variance=%.6f, %d samples)",
            b_crit, snr, variance, count,
        )
        return b_crit

    @staticmethod
    def _is_memory_pressure_exception(exc: Exception) -> bool:
        """Return True when an exception message indicates memory exhaustion."""
        msg = str(exc).lower()
        return any(
            token in msg
            for token in (
                "out of memory",
                "oom",
                "failed to allocate",
                "allocation failure",
                "memory",
                "metal",
                "mps",
            )
        )

    def derive_memory_safe_micro_batch_size(
        self,
        model,
        train_dataset,
        seq_length: int,
        logical_batch_size: int,
        seed: int = 0,
    ) -> int:
        """Find the largest micro-batch that fits device memory.

        Uses measured forward/backward probes on longest-token samples, then
        binary-searches the feasible interval [1, logical_batch_size].
        """
        if logical_batch_size <= 1:
            return 1
        if not train_dataset:
            return 1

        from mlx.utils import tree_flatten as mlx_flatten
        from mlx_lm.tuner.trainer import default_loss, iterate_batches

        max_candidate = min(int(logical_batch_size), len(train_dataset))
        if max_candidate <= 1:
            return 1

        loss_vg = nn.value_and_grad(model, default_loss)

        def _token_length(sample: Any) -> int:
            tokens = sample[0]
            shape = getattr(tokens, "shape", None)
            if shape is not None and len(shape) > 0:
                return int(shape[0])
            try:
                return int(len(tokens))
            except Exception:
                return 0

        longest_first = sorted(
            train_dataset,
            key=_token_length,
            reverse=True,
        )

        def _probe(candidate_bs: int) -> tuple[bool, float | None]:
            probe_size = min(candidate_bs, len(longest_first))
            if probe_size <= 0:
                return False, None

            probe_dataset = longest_first[:probe_size]
            peak_gb = None
            try:
                if hasattr(self._backend, "clear_cache"):
                    self._backend.clear_cache()
                if hasattr(self._backend, "reset_peak_memory"):
                    self._backend.reset_peak_memory()

                batch, lengths = next(
                    iterate_batches(
                        probe_dataset,
                        probe_size,
                        seq_length,
                        loop=False,
                        seed=seed,
                    )
                )
                (loss, _), grads = loss_vg(model, batch, lengths)
                flat = dict(mlx_flatten(grads))
                if flat:
                    mx.eval(loss, *flat.values())
                else:
                    mx.eval(loss)

                if hasattr(self._backend, "get_peak_memory_gb"):
                    peak_gb = float(self._backend.get_peak_memory_gb())
                return True, peak_gb
            except Exception as exc:
                if not self._is_memory_pressure_exception(exc):
                    raise
                logger.info(
                    "Memory probe rejected micro_batch=%d: %s",
                    candidate_bs,
                    exc,
                )
                return False, None
            finally:
                if hasattr(self._backend, "clear_cache"):
                    self._backend.clear_cache()

        ok1, peak1 = _probe(1)
        if not ok1:
            raise RuntimeError(
                "Training OOM at micro_batch=1; model/data do not fit device memory."
            )

        # Probe from the safe side first to avoid catastrophic max-size probes.
        low = 1
        high = max_candidate + 1
        low_peak = peak1
        while low + 1 < high:
            mid = (low + high) // 2
            mid_ok, _mid_peak = _probe(mid)
            if mid_ok:
                low = mid
                low_peak = _mid_peak
            else:
                high = mid

        if low_peak is not None:
            logger.info(
                "Memory-safe micro batch size=%d (peak=%.2f GB)",
                low,
                low_peak,
            )
        else:
            logger.info("Memory-safe micro batch size=%d", low)
        return low

    def _resolve_parent_and_attr(self, model, layer_key: str) -> tuple[Any, str]:
        path_parts = layer_key.replace(".weight", "").split(".")
        obj = model
        for part in path_parts[:-1]:
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        return obj, path_parts[-1]

    def _module_name_from_layer_key(self, layer_key: str) -> str:
        parts = layer_key.replace(".weight", "").split(".")
        if len(parts) >= 5 and parts[0] == "model" and parts[1] == "layers":
            return ".".join(parts[3:])
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return layer_key.replace(".weight", "")

    def _expert_key_from_layer_key(self, layer_key: str) -> str | None:
        """Return canonical expert identifier for expert projection keys."""
        parts = layer_key.replace(".weight", "").split(".")
        if (
            len(parts) >= 7
            and parts[0] == "model"
            and parts[1] == "layers"
            and parts[3] == "mlp"
            and parts[4] == "experts"
        ):
            return f"L{parts[2]}.E{parts[5]}"
        return None
