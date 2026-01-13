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

"""Layer alignment helpers for probe stage."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.geometry.gram_aligner import GramAligner
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    sqrt_scalar,
)
from modelcypher.core.use_cases.merge.stages.probe_helpers import (
    _promote_precision,
    _proportional_layer_index,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    layer_mapping: dict[int, int]
    feature_transforms: dict[int, Any]
    scale_ratios: dict[int, float]
    attention_transforms: dict[int, Any]
    k_transforms: dict[int, Any]
    v_transforms: dict[int, Any]
    intermediate_transforms: dict[int, Any]
    gate_transforms: dict[int, Any]  # PRE-SiLU for gate_proj/up_proj stitching
    layer_cka_scores: dict[int, float]
    layer_cka_scores_raw: dict[int, float]
    cgls_iterations_by_layer: dict[int, int]
    rbf_consistency_hidden: dict[str, float] | None


def _activation_count(backend: "Backend", acts: Any) -> int:
    if hasattr(acts, "shape") and len(backend.shape(acts)) == 2:
        return int(backend.shape(acts)[0])
    return len(acts)


def _activation_dim(backend: "Backend", acts: Any) -> int:
    if hasattr(acts, "shape"):
        shape = backend.shape(acts)
        if len(shape) >= 2:
            return int(shape[1])
        if len(shape) == 1:
            return int(shape[0])
    if acts:
        sample = acts[0]
        if hasattr(sample, "shape"):
            return int(sample.shape[-1])
    return 0


def _stack_activations(
    backend: "Backend",
    acts: Any,
    n_samples: int,
) -> "Array":
    if hasattr(acts, "shape") and len(backend.shape(acts)) == 2:
        stacked = acts[:n_samples, :]
    else:
        stacked = backend.stack(acts[:n_samples], axis=0)
    return _promote_precision(stacked, backend)


def align_layers(
    *,
    source_layer_activations: dict[int, "Array"],
    target_layer_activations: dict[int, "Array"],
    source_intermediate_activations: dict[int, "Array"],
    target_intermediate_activations: dict[int, "Array"],
    source_gate_activations: dict[int, "Array"] | None = None,
    target_gate_activations: dict[int, "Array"] | None = None,
    backend: "Backend",
) -> AlignmentResult:
    layer_mapping: dict[int, int] = {}
    feature_transforms: dict[int, Any] = {}
    scale_ratios: dict[int, float] = {}
    attention_transforms: dict[int, Any] = {}
    k_transforms: dict[int, Any] = {}
    v_transforms: dict[int, Any] = {}
    intermediate_transforms: dict[int, Any] = {}
    gate_transforms: dict[int, Any] = {}  # PRE-SiLU for cross-arch gate/up stitching
    layer_cka_scores: dict[int, float] = {}
    layer_cka_scores_raw: dict[int, float] = {}
    cgls_iterations_by_layer: dict[int, int] = {}
    rbf_consistency_hidden: dict[str, float] | None = None

    if not (source_layer_activations and target_layer_activations):
        return AlignmentResult(
            layer_mapping=layer_mapping,
            feature_transforms=feature_transforms,
            scale_ratios=scale_ratios,
            attention_transforms=attention_transforms,
            k_transforms=k_transforms,
            v_transforms=v_transforms,
            intermediate_transforms=intermediate_transforms,
            gate_transforms=gate_transforms,
            layer_cka_scores=layer_cka_scores,
            layer_cka_scores_raw=layer_cka_scores_raw,
            cgls_iterations_by_layer=cgls_iterations_by_layer,
            rbf_consistency_hidden=rbf_consistency_hidden,
        )

    source_layers = sorted(source_layer_activations.keys())
    target_layers = sorted(target_layer_activations.keys())
    n_source = len(source_layers)
    n_target = len(target_layers)

    alignment_tasks: list[tuple[int, list[int]]] = []
    for tgt_idx in range(n_target):
        src_idx = _proportional_layer_index(tgt_idx, n_target, n_source)
        alignment_tasks.append((tgt_idx, [src_idx]))

    logger.info(
        "PROBE: Aligning %d target layers (proportional depth mapping)...",
        len(alignment_tasks),
    )

    rbf_consistency_checked = False

    def _align_target_group(
        tgt_idx: int,
        src_indices: list[int],
        F_init: "Array | None" = None,
    ) -> dict:
        nonlocal rbf_consistency_checked, rbf_consistency_hidden
        tgt_layer = target_layers[tgt_idx]
        src_layers_list = [source_layers[i] for i in src_indices]

        result: dict[str, Any] = {
            "tgt_layer": tgt_layer,
            "src_layers": src_layers_list,
            "raw_cka": 0.0,
            "achieved_cka": 0.0,
            "geodesic_cka": 0.0,
            "numerical_deviation": 0.0,
            "feature_transform": None,
            "attention_transform": None,
            "k_transform": None,
            "v_transform": None,
            "intermediate_transform": None,
            "gate_transform": None,  # PRE-SiLU for cross-arch gate/up stitching
            "linear_iterations": 0,
            "error": None,
        }

        src_act_lists = [source_layer_activations[s] for s in src_layers_list]
        tgt_list = target_layer_activations[tgt_layer]

        n_samples = _activation_count(backend, tgt_list)
        for s_list in src_act_lists:
            n_samples = min(n_samples, _activation_count(backend, s_list))

        if n_samples < 2:
            raise RuntimeError(
                f"Insufficient samples for {src_layers_list} -> {tgt_layer}: {n_samples}"
            )

        src_dim = _activation_dim(backend, src_act_lists[0]) if src_act_lists else 0
        tgt_dim = _activation_dim(backend, tgt_list)
        logger.info(
            "ALIGNMENT: Layer %d <- %s: n_samples=%d (need >=%d for full-rank src, >=%d for full-rank tgt)",
            tgt_layer,
            src_layers_list,
            n_samples,
            src_dim,
            tgt_dim,
        )

        local_aligner = GramAligner(backend=backend)

        try:
            src_stacks = []
            src_dims = []
            for s_list in src_act_lists:
                stack = _stack_activations(backend, s_list, n_samples)
                src_stacks.append(stack)
                src_dims.append(int(backend.shape(stack)[1]))

            tgt_stacked = _stack_activations(backend, tgt_list, n_samples)

            if len(src_stacks) == 1:
                src_combined = src_stacks[0]
            else:
                src_combined = backend.concatenate(src_stacks, axis=1)

            backend.eval(src_combined, tgt_stacked)

            from modelcypher.core.domain.geometry.cka import (
                compute_cka_backend,
                compute_linear_cka,
            )

            result["raw_cka"] = float(
                compute_cka_backend(src_combined, tgt_stacked, backend)
            )

            alignment_result = local_aligner.find_perfect_alignment(
                src_combined,
                tgt_stacked,
                F_init=F_init,
            )

            F_arr = alignment_result.feature_transform
            aligned = backend.matmul(src_combined, F_arr)
            backend.eval(aligned)
            linear_cka = float(
                compute_linear_cka(aligned, tgt_stacked, backend=backend)
            )
            linear_deviation = abs(1.0 - linear_cka)
            result["achieved_cka"] = linear_cka
            result["numerical_deviation"] = linear_deviation
            result["geodesic_cka"] = alignment_result.achieved_cka
            result["linear_iterations"] = alignment_result.linear_iterations
            cgls_iterations_by_layer[tgt_layer] = alignment_result.linear_iterations
            logger.info(
                "ALIGNMENT: Layer %d <- %s: solver iters=%d",
                tgt_layer,
                src_layers_list,
                alignment_result.linear_iterations,
            )

            if not rbf_consistency_checked:
                from modelcypher.core.domain.geometry.cka import compute_cka

                rbf_result = compute_cka(aligned, tgt_stacked, backend=backend)
                rbf_val = rbf_result.best if rbf_result.is_valid else float("nan")

                precision = sqrt_scalar(machine_epsilon(backend, aligned), backend)
                rbf_deviation = abs(1.0 - rbf_val) if rbf_val == rbf_val else float("inf")
                agreement_deviation = abs(rbf_val - linear_cka) if rbf_val == rbf_val else float("inf")

                rbf_consistency_hidden = {
                    "rbf_cka": float(rbf_val) if rbf_val == rbf_val else 0.0,
                    "rbf_deviation": float(rbf_deviation),
                    "linear_deviation": float(linear_deviation),
                    "agreement_deviation": float(agreement_deviation),
                    "precision_threshold": float(precision),
                    "layer": float(tgt_layer),
                }
                if linear_deviation > precision:
                    logger.error(
                        "PROBE: Linear CKA deviation %.2e > precision %.2e for layer %d.",
                        linear_deviation,
                        precision,
                        tgt_layer,
                    )
                if rbf_deviation > precision:
                    logger.info(
                        "PROBE: Geodesic CKA deviation %.2e > precision %.2e for layer %d.",
                        rbf_deviation,
                        precision,
                        tgt_layer,
                    )
                if agreement_deviation > precision:
                    logger.info(
                        "PROBE: Geodesic vs linear CKA deviation %.2e > precision %.2e for layer %d.",
                        agreement_deviation,
                        precision,
                        tgt_layer,
                    )
                rbf_consistency_checked = True

            result["F_arr_raw"] = F_arr

            split_transforms: dict[int, Any] = {}
            start_idx = 0
            for s_layer, s_dim in zip(src_layers_list, src_dims):
                F_slice = F_arr[start_idx : start_idx + s_dim, :]
                split_transforms[s_layer] = F_slice
                start_idx += s_dim

            result["feature_transform"] = split_transforms
            result["scale_ratio"] = alignment_result.scale_ratio

            layer_precision = sqrt_scalar(machine_epsilon(backend, aligned), backend)
            if linear_deviation > layer_precision:
                logger.warning(
                    "PROBE: Layer %s -> %d linear CKA deviation=%.2e > precision %.2e.",
                    src_layers_list,
                    tgt_layer,
                    linear_deviation,
                    layer_precision,
                )

            split_inter_transforms: dict[int, Any] = {}
            for s_layer in src_layers_list:
                src_inter_acts = source_intermediate_activations.get(s_layer)
                tgt_inter_acts = target_intermediate_activations.get(tgt_layer)

                if src_inter_acts is None or tgt_inter_acts is None:
                    logger.debug(
                        "PROBE INTER: No intermediate activations for %s -> %d",
                        s_layer,
                        tgt_layer,
                    )
                    continue

                inter_samples = min(
                    _activation_count(backend, src_inter_acts),
                    _activation_count(backend, tgt_inter_acts),
                    n_samples,
                )
                if inter_samples < 2:
                    logger.warning(
                        "PROBE INTER: Insufficient samples for %s -> %d: %d",
                        s_layer,
                        tgt_layer,
                        inter_samples,
                    )
                    continue

                src_inter_stacked = _stack_activations(
                    backend, src_inter_acts, inter_samples
                )
                tgt_inter_stacked = _stack_activations(
                    backend, tgt_inter_acts, inter_samples
                )
                backend.eval(src_inter_stacked, tgt_inter_stacked)

                try:
                    inter_result = local_aligner.find_perfect_alignment(
                        src_inter_stacked, tgt_inter_stacked
                    )
                    I_arr = inter_result.feature_transform
                    split_inter_transforms[s_layer] = I_arr

                    src_inter_dim = int(backend.shape(src_inter_stacked)[1])
                    tgt_inter_dim = int(backend.shape(tgt_inter_stacked)[1])
                    logger.info(
                        "PROBE INTER DIRECT: %s -> %d: I=[%d, %d] (src_inter=%d, tgt_inter=%d)",
                        s_layer,
                        tgt_layer,
                        int(backend.shape(I_arr)[0]),
                        int(backend.shape(I_arr)[1]),
                        src_inter_dim,
                        tgt_inter_dim,
                    )
                except Exception as inter_err:
                    logger.warning(
                        "PROBE INTER: Direct alignment failed for %s -> %d: %s",
                        s_layer,
                        tgt_layer,
                        inter_err,
                    )

            if split_inter_transforms:
                result["intermediate_transform"] = split_inter_transforms

            # GATE TRANSFORMS: PRE-SiLU activations for cross-architecture gate/up stitching
            # The intermediate transform is computed on POST-SiLU activations (SiLU(gate)*up),
            # but gate_proj/up_proj output to PRE-SiLU space. For cross-architecture merging,
            # compressions don't commute with SiLU, so we need separate alignment.
            split_gate_transforms: dict[int, Any] = {}
            if source_gate_activations and target_gate_activations:
                for s_layer in src_layers_list:
                    src_gate_acts = source_gate_activations.get(s_layer)
                    tgt_gate_acts = target_gate_activations.get(tgt_layer)

                    if src_gate_acts is None or tgt_gate_acts is None:
                        logger.debug(
                            "PROBE GATE: No gate activations for %s -> %d",
                            s_layer,
                            tgt_layer,
                        )
                        continue

                    gate_samples = min(
                        _activation_count(backend, src_gate_acts),
                        _activation_count(backend, tgt_gate_acts),
                        n_samples,
                    )
                    if gate_samples < 2:
                        logger.warning(
                            "PROBE GATE: Insufficient samples for %s -> %d: %d",
                            s_layer,
                            tgt_layer,
                            gate_samples,
                        )
                        continue

                    src_gate_stacked = _stack_activations(
                        backend, src_gate_acts, gate_samples
                    )
                    tgt_gate_stacked = _stack_activations(
                        backend, tgt_gate_acts, gate_samples
                    )
                    backend.eval(src_gate_stacked, tgt_gate_stacked)

                    try:
                        gate_result = local_aligner.find_perfect_alignment(
                            src_gate_stacked, tgt_gate_stacked
                        )
                        G_arr = gate_result.feature_transform
                        split_gate_transforms[s_layer] = G_arr

                        src_gate_dim = int(backend.shape(src_gate_stacked)[1])
                        tgt_gate_dim = int(backend.shape(tgt_gate_stacked)[1])
                        logger.info(
                            "PROBE GATE DIRECT: %s -> %d: G=[%d, %d] (src_gate=%d, tgt_gate=%d)",
                            s_layer,
                            tgt_layer,
                            int(backend.shape(G_arr)[0]),
                            int(backend.shape(G_arr)[1]),
                            src_gate_dim,
                            tgt_gate_dim,
                        )
                    except Exception as gate_err:
                        logger.warning(
                            "PROBE GATE: Direct alignment failed for %s -> %d: %s",
                            s_layer,
                            tgt_layer,
                            gate_err,
                        )

            if split_gate_transforms:
                result["gate_transform"] = split_gate_transforms

        except Exception as e:
            raise RuntimeError(
                f"GramAligner failed for {src_layers_list} -> {tgt_layer}: {e}"
            ) from e

        del src_combined, tgt_stacked, src_stacks, F_arr
        try:
            del alignment_result
        except NameError:
            pass
        backend.eval()

        return result

    successful_alignments: dict[int, dict] = {}

    completed = 0
    for tgt_idx, src_indices in alignment_tasks:
        tgt_layer = target_layers[tgt_idx]
        F_init = None

        if successful_alignments:
            aligned_layers = list(successful_alignments.keys())
            closest_layer = min(aligned_layers, key=lambda l: abs(l - tgt_layer))
            neighbor_data = successful_alignments[closest_layer]
            F_init = neighbor_data.get("F")
            logger.info(
                "ZIPPER: Layer %d warm-starting from layer %d",
                tgt_layer,
                closest_layer,
            )
        else:
            logger.debug("ZIPPER: Layer %d has no successful neighbors yet", tgt_layer)

        result = _align_target_group(tgt_idx, src_indices, F_init=F_init)

        tgt_layer = result["tgt_layer"]
        src_layers = result["src_layers"]

        layer_mapping[tgt_layer] = src_layers[0]

        if result["feature_transform"] is None:
            raise RuntimeError(
                f"GramAligner returned no transform for {src_layers} -> {tgt_layer}. "
                "This should never happen if the geometry is correct."
            )
        feature_transforms[tgt_layer] = result["feature_transform"]
        layer_cka_scores[tgt_layer] = result["achieved_cka"]
        layer_cka_scores_raw[tgt_layer] = result["raw_cka"]

        if "scale_ratio" in result:
            scale_ratios[tgt_layer] = result["scale_ratio"]

        if result.get("F_arr_raw") is not None:
            successful_alignments[tgt_layer] = {
                "F": result["F_arr_raw"],
                "R": result.get("R_raw", None),
            }

        completed += 1
        logger.info(
            "PROBE ALIGNMENT: Layer %d/%d complete (tgt=%d, linear_CKA=%.4f, raw_CKA=%.4f)",
            completed,
            len(alignment_tasks),
            tgt_layer,
            result["achieved_cka"],
            result["raw_cka"],
        )

        if result["attention_transform"] is not None:
            attention_transforms[tgt_layer] = result["attention_transform"]

        if result.get("k_transform") is not None:
            k_transforms[tgt_layer] = result["k_transform"]

        if result.get("v_transform") is not None:
            v_transforms[tgt_layer] = result["v_transform"]

        if result.get("intermediate_transform") is not None:
            intermediate_transforms[tgt_layer] = result["intermediate_transform"]

        if result.get("gate_transform") is not None:
            gate_transforms[tgt_layer] = result["gate_transform"]

        if completed % 5 == 0 or completed == len(alignment_tasks):
            logger.info(
                "PROBE: Aligned %d/%d target layers (zipper: %d warm-started)...",
                completed,
                len(alignment_tasks),
                len(successful_alignments),
            )

    logger.info(
        "PROBE: Cross-architecture layer alignment found %d mappings "
        "(source: %d layers, target: %d layers)",
        len(alignment_tasks),
        n_source,
        n_target,
    )

    return AlignmentResult(
        layer_mapping=layer_mapping,
        feature_transforms=feature_transforms,
        scale_ratios=scale_ratios,
        attention_transforms=attention_transforms,
        k_transforms=k_transforms,
        v_transforms=v_transforms,
        intermediate_transforms=intermediate_transforms,
        gate_transforms=gate_transforms,
        layer_cka_scores=layer_cka_scores,
        layer_cka_scores_raw=layer_cka_scores_raw,
        cgls_iterations_by_layer=cgls_iterations_by_layer,
        rbf_consistency_hidden=rbf_consistency_hidden,
    )
