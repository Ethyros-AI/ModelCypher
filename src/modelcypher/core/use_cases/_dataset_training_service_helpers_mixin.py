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

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
import time
from pathlib import Path
from typing import Any

from modelcypher.core.domain.dataset_loading import load_jsonl_dataset
from modelcypher.core.domain.training.exceptions import TrainingDerivationError

logger = logging.getLogger(__name__)

_MOE_EXPERT_WEIGHT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$",
)


class _DatasetTrainingServiceHelperMixin:
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
            adapted_stacks: dict[int, Any] = {}
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
                adapted_stacks[layer_idx] = adapted_stack

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

            # Mode connectivity: barrier along linear interpolation path between
            # base and adapted activations.  Weight space is Euclidean (proven
            # cross-family 2026-02-23), so linear activation interpolation is the
            # correct geodesic.  n_steps=5 gives endpoints + 3 interior points —
            # minimum for barrier detection.
            try:
                from modelcypher.core.domain.geometry.mode_connectivity import (
                    analyze_mode_connectivity,
                )

                mc_barriers: dict[int, float] = {}
                mc_normalized: dict[int, float] = {}
                for layer_idx, b_stack in base_stacks.items():
                    a_stack = adapted_stacks.get(layer_idx)
                    if a_stack is None:
                        continue

                    # Closure: CKA divergence from base at interpolated activations
                    _b_ref = b_stack
                    _backend_ref = self._backend

                    def _mc_loss(interpolated: Any, _br: Any = _b_ref, _be: Any = _backend_ref) -> float:
                        return 1.0 - compute_linear_cka_from_activations(_br, interpolated, _be)

                    mc_result = analyze_mode_connectivity(
                        b_stack, a_stack, _mc_loss, n_steps=5, backend=self._backend,
                    )
                    mc_barriers[layer_idx] = mc_result.barrier_height
                    mc_normalized[layer_idx] = mc_result.normalized_barrier

                if mc_barriers:
                    result_dict["mode_connectivity_barrier"] = max(mc_barriers.values())
                    result_dict["mode_connectivity_normalized_barrier"] = max(mc_normalized.values())
                    result_dict["mode_connectivity_method"] = "linear"
                    logger.info(
                        "Mode connectivity: max_barrier=%.4f, max_normalized=%.4f (%d layers)",
                        result_dict["mode_connectivity_barrier"],
                        result_dict["mode_connectivity_normalized_barrier"],
                        len(mc_barriers),
                    )
            except Exception:
                logger.debug("Mode connectivity analysis failed", exc_info=True)

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

    def _parse_target_expert_keys(
        self,
        target_experts: list[str] | str | None,
    ) -> set[str]:
        """Parse `Lx.Ey` selectors into MoE projection weight keys."""
        if target_experts is None:
            return set()

        tokens: list[str] = []
        if isinstance(target_experts, str):
            tokens.extend(part.strip() for part in target_experts.split(","))
        else:
            for item in target_experts:
                tokens.extend(part.strip() for part in str(item).split(","))

        parsed: set[str] = set()
        for token in tokens:
            if not token:
                continue
            if token.startswith("model.layers.") and token.endswith(".weight"):
                parsed.add(token)
                continue

            match = re.fullmatch(r"[Ll](\d+)\.[Ee](\d+)", token)
            if match is None:
                continue

            layer_idx = int(match.group(1))
            expert_idx = int(match.group(2))
            prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
            parsed.update({
                f"{prefix}.gate_proj.weight",
                f"{prefix}.up_proj.weight",
                f"{prefix}.down_proj.weight",
            })
        return parsed

    def _load_moe_topology(self, model_path: Path):
        """Load MoE topology from model config, if available."""
        try:
            from modelcypher.ports.model_architecture_factory import load_config
            from modelcypher.core.domain.moe.topology import MoETopology

            config = load_config(model_path)
            return MoETopology.from_config(config)
        except Exception:
            return None

    def _build_moe_target_selection(
        self,
        *,
        target_modules: list[str],
        geometries: dict[str, Any],
        rank_overrides: dict[str, int],
        topology,
        num_layers: int,
    ):
        """Build ExpertTargetSelection for targeted MoE experts."""
        try:
            from modelcypher.core.domain.moe.expert_selection import (
                ExpertTarget,
                ExpertTargetSelection,
            )
            from modelcypher.core.domain.moe.topology import MoETopology
        except Exception:
            return None

        grouped: dict[tuple[int, int], dict[str, Any]] = {}
        for key in sorted(target_modules):
            match = _MOE_EXPERT_WEIGHT_RE.match(key)
            if match is None:
                continue
            layer_idx = int(match.group(1))
            expert_idx = int(match.group(2))
            proj_name = match.group(3)
            geom = geometries.get(key)
            if geom is None:
                continue
            group = grouped.setdefault(
                (layer_idx, expert_idx),
                {"geometries": {}, "ranks": []},
            )
            group["geometries"][proj_name] = geom
            group["ranks"].append(int(rank_overrides.get(key, geom.tail_dims)))

        if not grouped:
            return None

        if topology is None:
            max_expert_idx = max(expert_idx for _layer, expert_idx in grouped)
            moe_layers = sorted({layer for layer, _expert in grouped})
            topology = MoETopology(
                num_experts=max_expert_idx + 1,
                num_experts_per_tok=1,
                moe_intermediate_size=0,
                has_shared_expert=False,
                shared_expert_intermediate_size=None,
                moe_layer_indices=moe_layers,
                num_layers=num_layers,
            )

        targets = []
        estimated_params_total = 0
        for (layer_idx, expert_idx), payload in sorted(grouped.items()):
            proj_geometries = payload["geometries"]
            representative = proj_geometries.get("gate_proj")
            if representative is None:
                representative = next(iter(proj_geometries.values()))
            ranks = payload["ranks"] or [representative.tail_dims]
            rank = max(1, max(int(r) for r in ranks))
            target = ExpertTarget(
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                category="primary",
                routing_frequency=0.0,
                geometry=representative,
                rank=rank,
            )
            targets.append(target)
            for geom in proj_geometries.values():
                out_features, in_features = geom.shape
                estimated_params_total += rank * (int(in_features) + int(out_features) + 1)

        return ExpertTargetSelection(
            targets=targets,
            saturated=[],
            skipped=[],
            topology=topology,
            estimated_params_total=estimated_params_total,
        )

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



__all__ = ["_DatasetTrainingServiceHelperMixin"]
