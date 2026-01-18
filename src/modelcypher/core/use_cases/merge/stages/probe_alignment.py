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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.geometry.fisher_information import (
    fisher_compatibility_score,
)
from modelcypher.core.domain.geometry.gram_aligner import GramAligner
from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    sqrt_scalar,
)
from modelcypher.core.use_cases.merge.stages.probe_helpers import (
    _promote_precision,
    _proportional_layer_index,
    compute_numerical_rank,
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
    layer_cka_scores: dict[int, float]  # Geodesic CKA only
    cgls_iterations_by_layer: dict[int, int]
    gram_condition_numbers_by_layer: dict[int, float] = field(default_factory=dict)
    linear_residuals_by_layer: dict[int, float] = field(default_factory=dict)
    numerical_deviation_by_layer: dict[int, float] = field(default_factory=dict)
    precision_thresholds_by_layer: dict[int, float] = field(default_factory=dict)
    # Fisher compatibility scores for merge success prediction
    fisher_compatibility_by_layer: dict[int, float] = field(default_factory=dict)
    fisher_recommendations_by_layer: dict[int, str] = field(default_factory=dict)


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


def _log_manifold_diagnostic(
    activations: "Array",
    layer_idx: int,
    model_name: str,
    backend: "Backend",
) -> float | None:
    """Log intrinsic dimension as a diagnostic measurement.

    Returns the intrinsic dimension, or None if computation fails.
    No thresholds, no judgments - just the measurement.
    """
    try:
        estimator = IntrinsicDimension(backend)
        result = estimator.compute(activations)
        intrinsic_dim = result.intrinsic_dimension
        shape = backend.shape(activations)
        ambient_dim = int(shape[1])

        logger.debug(
            "MANIFOLD [%s Layer %d]: ID=%.1f, ambient=%d",
            model_name,
            layer_idx,
            intrinsic_dim,
            ambient_dim,
        )
        return intrinsic_dim
    except Exception as exc:
        logger.debug(
            "MANIFOLD [%s Layer %d]: Could not compute ID: %s",
            model_name,
            layer_idx,
            exc,
        )
        return None


def align_layers(
    *,
    source_layer_activations: dict[int, "Array"],
    target_layer_activations: dict[int, "Array"],
    source_intermediate_activations: dict[int, "Array"],
    target_intermediate_activations: dict[int, "Array"],
    source_gate_activations: dict[int, "Array"] | None = None,
    target_gate_activations: dict[int, "Array"] | None = None,
    backend: "Backend",
    require_full_rank: bool = False,
) -> AlignmentResult:
    """Align layers between source and target models.

    Args:
        source_layer_activations: Source model hidden state activations by layer.
        target_layer_activations: Target model hidden state activations by layer.
        source_intermediate_activations: Source intermediate (FFN) activations.
        target_intermediate_activations: Target intermediate (FFN) activations.
        source_gate_activations: Optional source gate activations.
        target_gate_activations: Optional target gate activations.
        backend: Backend for tensor operations.
        require_full_rank: If True, raise RuntimeError when activation rank < hidden_dim.
            This ensures alignment is only attempted with full-rank activation matrices.

    Returns:
        AlignmentResult with transforms and diagnostics.

    Raises:
        RuntimeError: If require_full_rank=True and any layer has rank < hidden_dim.
    """
    layer_mapping: dict[int, int] = {}
    feature_transforms: dict[int, Any] = {}
    scale_ratios: dict[int, float] = {}
    attention_transforms: dict[int, Any] = {}
    k_transforms: dict[int, Any] = {}
    v_transforms: dict[int, Any] = {}
    intermediate_transforms: dict[int, Any] = {}
    gate_transforms: dict[int, Any] = {}  # PRE-SiLU for cross-arch gate/up stitching
    layer_cka_scores: dict[int, float] = {}
    cgls_iterations_by_layer: dict[int, int] = {}
    gram_condition_numbers_by_layer: dict[int, float] = {}
    linear_residuals_by_layer: dict[int, float] = {}
    numerical_deviation_by_layer: dict[int, float] = {}
    precision_thresholds_by_layer: dict[int, float] = {}
    fisher_compatibility_by_layer: dict[int, float] = {}
    fisher_recommendations_by_layer: dict[int, str] = {}

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
            cgls_iterations_by_layer=cgls_iterations_by_layer,
            gram_condition_numbers_by_layer=gram_condition_numbers_by_layer,
            linear_residuals_by_layer=linear_residuals_by_layer,
            numerical_deviation_by_layer=numerical_deviation_by_layer,
            precision_thresholds_by_layer=precision_thresholds_by_layer,
            fisher_compatibility_by_layer=fisher_compatibility_by_layer,
            fisher_recommendations_by_layer=fisher_recommendations_by_layer,
        )

    # =========================================================================
    # RANK VALIDATION (when require_full_rank=True)
    # =========================================================================
    # Full-rank activation matrices ensure no information loss during alignment.
    # If rank < hidden_dim, some directions aren't mapped, which can cause
    # knowledge loss during merge transfer.
    if require_full_rank:
        rank_deficient_layers: list[tuple[int, int, int, str]] = []

        # Check source layers
        for layer_idx, acts in source_layer_activations.items():
            if hasattr(acts, "shape"):
                stacked = _promote_precision(acts, backend)
            else:
                stacked = _promote_precision(
                    backend.stack(acts, axis=0), backend
                )
            rank, hidden_dim = compute_numerical_rank(stacked, backend)
            if rank < hidden_dim:
                rank_deficient_layers.append((layer_idx, rank, hidden_dim, "source"))

        # Check target layers
        for layer_idx, acts in target_layer_activations.items():
            if hasattr(acts, "shape"):
                stacked = _promote_precision(acts, backend)
            else:
                stacked = _promote_precision(
                    backend.stack(acts, axis=0), backend
                )
            rank, hidden_dim = compute_numerical_rank(stacked, backend)
            if rank < hidden_dim:
                rank_deficient_layers.append((layer_idx, rank, hidden_dim, "target"))

        if rank_deficient_layers:
            # Build detailed diagnostic message
            diagnostics = []
            for layer_idx, rank, hidden_dim, model in rank_deficient_layers:
                coverage = 100.0 * rank / hidden_dim if hidden_dim > 0 else 0.0
                diagnostics.append(
                    f"  {model} layer {layer_idx}: rank={rank}/{hidden_dim} ({coverage:.1f}%)"
                )
            raise RuntimeError(
                f"ALIGNMENT FAILED: Rank-deficient activations detected.\n"
                f"Full-rank coverage is required but the following layers have deficits:\n"
                + "\n".join(diagnostics)
                + "\n\nTo fix this, either:\n"
                "1. Add more diverse probes to the atlas\n"
                "2. Use orthogonal probe generation to augment rank\n"
                "3. Set require_full_rank=False to proceed with partial coverage (not recommended)"
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

    def _align_target_group(
        tgt_idx: int,
        src_indices: list[int],
    ) -> dict:
        tgt_layer = target_layers[tgt_idx]
        src_layers_list = [source_layers[i] for i in src_indices]

        result: dict[str, Any] = {
            "tgt_layer": tgt_layer,
            "src_layers": src_layers_list,
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
            "fisher_compatibility": 0.0,
            "fisher_recommendation": "",
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

        local_aligner = GramAligner(backend=backend, use_geodesic_alignment=False)

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

            # Log intrinsic dimension as diagnostic (no thresholds - just measurement)
            _log_manifold_diagnostic(src_combined, tgt_layer, "source", backend)
            _log_manifold_diagnostic(tgt_stacked, tgt_layer, "target", backend)

            # Compute Fisher compatibility BEFORE alignment as an early predictor
            # This measures whether source and target have compatible loss curvature
            try:
                fisher_result = fisher_compatibility_score(
                    src_combined, tgt_stacked, backend=backend
                )
                result["fisher_compatibility"] = fisher_result.compatibility_score
                result["fisher_recommendation"] = fisher_result.recommendation
                fisher_compatibility_by_layer[tgt_layer] = fisher_result.compatibility_score
                fisher_recommendations_by_layer[tgt_layer] = fisher_result.recommendation

                logger.info(
                    "FISHER: Layer %d compatibility=%.6f (%s)",
                    tgt_layer,
                    fisher_result.compatibility_score,
                    fisher_result.recommendation,
                )
            except Exception as fisher_err:
                logger.debug(
                    "FISHER: Could not compute compatibility for layer %d: %s",
                    tgt_layer,
                    fisher_err,
                )

            alignment_result = local_aligner.find_perfect_alignment(
                src_combined,
                tgt_stacked,
            )

            F_arr = alignment_result.feature_transform
            aligned = backend.matmul(src_combined, F_arr)
            backend.eval(aligned)

            # Primary metric: geodesic RBF CKA (what the alignment optimizes)
            geodesic_cka = alignment_result.achieved_cka
            geodesic_deviation = abs(1.0 - geodesic_cka)
            result["achieved_cka"] = geodesic_cka
            result["numerical_deviation"] = geodesic_deviation
            result["geodesic_cka"] = geodesic_cka  # Same as achieved_cka (for clarity)
            result["linear_iterations"] = alignment_result.linear_iterations
            cgls_iterations_by_layer[tgt_layer] = alignment_result.linear_iterations
            gram_condition_numbers_by_layer[tgt_layer] = alignment_result.gram_condition_number
            linear_residuals_by_layer[tgt_layer] = alignment_result.linear_residual
            numerical_deviation_by_layer[tgt_layer] = alignment_result.numerical_deviation
            precision_thresholds_by_layer[tgt_layer] = alignment_result.precision_threshold
            split_transforms: dict[int, Any] = {}
            start_idx = 0
            for s_layer, s_dim in zip(src_layers_list, src_dims):
                F_slice = F_arr[start_idx : start_idx + s_dim, :]
                split_transforms[s_layer] = F_slice
                start_idx += s_dim

            result["feature_transform"] = split_transforms
            result["scale_ratio"] = alignment_result.scale_ratio

            layer_precision = sqrt_scalar(machine_epsilon(backend, aligned), backend)
            if geodesic_deviation > layer_precision:
                logger.debug(
                    "PROBE: Layer %s -> %d geodesic CKA deviation=%.2e > precision %.2e.",
                    src_layers_list,
                    tgt_layer,
                    geodesic_deviation,
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
                    logger.debug(
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
                    logger.debug(
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
                        logger.debug(
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
                        logger.debug(
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

    completed = 0
    for tgt_idx, src_indices in alignment_tasks:
        tgt_layer = target_layers[tgt_idx]
        result = _align_target_group(tgt_idx, src_indices)

        tgt_layer = result["tgt_layer"]
        src_layers = result["src_layers"]

        layer_mapping[tgt_layer] = src_layers[0]

        if result["feature_transform"] is None:
            raise RuntimeError(
                f"GramAligner returned no transform for {src_layers} -> {tgt_layer}. "
                "Unexpected empty transform."
            )
        feature_transforms[tgt_layer] = result["feature_transform"]
        layer_cka_scores[tgt_layer] = result["achieved_cka"]

        if "scale_ratio" in result:
            scale_ratios[tgt_layer] = result["scale_ratio"]

        completed += 1
        logger.info(
            "PROBE ALIGNMENT: Layer %d/%d complete (tgt=%d, geodesic_CKA=%.4f)",
            completed,
            len(alignment_tasks),
            tgt_layer,
            result["achieved_cka"],
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
                "PROBE: Aligned %d/%d target layers...",
                completed,
                len(alignment_tasks),
            )

    logger.info(
        "PROBE: Cross-architecture layer alignment found %d mappings "
        "(source: %d layers, target: %d layers)",
        len(alignment_tasks),
        n_source,
        n_target,
    )

    # Log summary of Fisher compatibility across layers
    if fisher_compatibility_by_layer:
        scores = list(fisher_compatibility_by_layer.values())
        mean_fisher = sum(scores) / len(scores)
        min_fisher = min(scores)
        max_fisher = max(scores)
        logger.info(
            "FISHER SUMMARY: Mean=%.6f, min=%.6f, max=%.6f",
            mean_fisher,
            min_fisher,
            max_fisher,
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
        cgls_iterations_by_layer=cgls_iterations_by_layer,
        gram_condition_numbers_by_layer=gram_condition_numbers_by_layer,
        linear_residuals_by_layer=linear_residuals_by_layer,
        numerical_deviation_by_layer=numerical_deviation_by_layer,
        precision_thresholds_by_layer=precision_thresholds_by_layer,
        fisher_compatibility_by_layer=fisher_compatibility_by_layer,
        fisher_recommendations_by_layer=fisher_recommendations_by_layer,
    )
