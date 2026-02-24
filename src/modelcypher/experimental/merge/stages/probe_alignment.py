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
from modelcypher.core.domain.geometry.hot_layer_matcher import (
    coupling_to_assignment,
    hot_layer_matching,
)

from .probe_helpers import (
    _promote_precision,
    compute_numerical_rank,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class MergeAlignmentResult:
    """Result of layer alignment computation.

    Contains only what's needed for downstream processing:
    - Transforms for each activation space (hidden, attention, intermediate, gate)
    - CKA scores (geodesic, diagnostic for overlap on probes)
    - Layer mapping (which source layer aligns to which target layer)
    - HOT coupling (for transfer strength weighting)
    """
    layer_mapping: dict[int, int]
    feature_transforms: dict[int, Any]
    scale_ratios: dict[int, float]
    attention_transforms: dict[int, Any]
    k_transforms: dict[int, Any]
    v_transforms: dict[int, Any]
    intermediate_transforms: dict[int, Any]
    gate_transforms: dict[int, Any]  # PRE-SiLU for gate_proj/up_proj stitching
    layer_cka_scores: dict[int, float]  # Geodesic CKA - validation only
    # HOT soft coupling matrix [n_source_layers, n_target_layers]
    # Each entry represents optimal mass transport between layer pairs.
    # Used to weight transfer strength: high coupling = strong alignment = transfer more.
    layer_coupling: list[list[float]] | None = None


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
    require_full_rank: bool = False,
    layer_filter: list[int] | None = None,
) -> MergeAlignmentResult:
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
        layer_filter: If provided, only align these target layer indices.
            This enables bottleneck-only alignment for massive speedup.

    Returns:
        MergeAlignmentResult with transforms and diagnostics.

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

    if not (source_layer_activations and target_layer_activations):
        return MergeAlignmentResult(
            layer_mapping=layer_mapping,
            feature_transforms=feature_transforms,
            scale_ratios=scale_ratios,
            attention_transforms=attention_transforms,
            k_transforms=k_transforms,
            v_transforms=v_transforms,
            intermediate_transforms=intermediate_transforms,
            gate_transforms=gate_transforms,
            layer_cka_scores=layer_cka_scores,
            layer_coupling=None,
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
                "ALIGNMENT FAILED: Rank-deficient activations detected.\n"
                "Full-rank coverage is required but the following layers have deficits:\n"
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

    # =========================================================================
    # HUNGARIAN LAYER MATCHING - Closed-form optimal assignment
    # =========================================================================
    # Use Hierarchical Optimal Transport (HOT) for layer matching.
    # HOT produces soft couplings (many-to-many) instead of rigid 1-to-1.
    # This handles depth mismatches naturally and provides a global alignment score.
    #
    # Reference: Shah & Khosla (2025) "Representational Alignment Across Model
    # Layers and Brain Regions with Hierarchical Optimal Transport" arXiv:2510.01706
    hot_result = hot_layer_matching(
        source_layer_activations=source_layer_activations,
        target_layer_activations=target_layer_activations,
        backend=backend,
    )

    # Convert soft coupling to hard assignment for backward compatibility
    layer_mapping = coupling_to_assignment(
        hot_result.layer_coupling,
        hot_result.source_layers,
        hot_result.target_layers,
        backend,
    )

    # Build alignment tasks from HOT matching
    # layer_mapping maps target_layer -> source_layer
    alignment_tasks: list[tuple[int, list[int]]] = []
    for tgt_idx, tgt_layer in enumerate(target_layers):
        # BOTTLENECK-ONLY OPTIMIZATION: If layer_filter provided, skip non-bottleneck layers
        if layer_filter is not None and tgt_layer not in layer_filter:
            continue

        if tgt_layer in layer_mapping:
            src_layer = layer_mapping[tgt_layer]
            # Find the index of src_layer in source_layers
            src_idx = source_layers.index(src_layer)
            alignment_tasks.append((tgt_idx, [src_idx]))
        else:
            # Unmatched target layer (shouldn't happen with proper marginals)
            logger.warning(
                "HOT: Target layer %d has no matching source layer", tgt_layer
            )

    if layer_filter is not None:
        logger.info(
            "PROBE: BOTTLENECK-ONLY MODE - aligning %d/%d layers (filter=%s)",
            len(alignment_tasks),
            n_target,
            layer_filter,
        )
    else:
        logger.info(
            "PROBE: Aligning %d target layers (HOT optimal matching, score=%.4f)...",
            len(alignment_tasks),
            hot_result.alignment_score,
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
            "numerical_deviation": 0.0,
            "feature_transform": None,
            "attention_transform": None,
            "k_transform": None,
            "v_transform": None,
            "intermediate_transform": None,
            "gate_transform": None,
            "linear_iterations": 0,
            "error": None,
        }

        src_act_lists = [source_layer_activations[s] for s in src_layers_list]
        tgt_list = target_layer_activations[tgt_layer]

        n_samples = _activation_count(backend, tgt_list)
        for s_list in src_act_lists:
            n_samples = min(n_samples, _activation_count(backend, s_list))

        # Information-theoretic minimum: need at least 2 points to compute any
        # relational structure (distances, covariance, alignment). Single point
        # has no relationships to align.
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

            alignment_result = local_aligner.find_perfect_alignment(
                src_combined,
                tgt_stacked,
            )

            F_arr = alignment_result.feature_transform
            achieved_cka = alignment_result.achieved_cka
            result["achieved_cka"] = achieved_cka
            split_transforms: dict[int, Any] = {}
            start_idx = 0
            for s_layer, s_dim in zip(src_layers_list, src_dims):
                F_slice = F_arr[start_idx : start_idx + s_dim, :]
                split_transforms[s_layer] = F_slice
                start_idx += s_dim

            result["feature_transform"] = split_transforms
            result["scale_ratio"] = alignment_result.scale_ratio

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
                # Information-theoretic minimum: 2 points required for relational alignment
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
                    # Information-theoretic minimum: 2 points required for relational alignment
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

            if split_gate_transforms:
                result["gate_transform"] = split_gate_transforms

        except Exception as e:
            raise RuntimeError(
                f"GramAligner failed for {src_layers_list} -> {tgt_layer}: {e}"
            ) from e

        del src_combined, tgt_stacked, src_stacks, F_arr
        if "alignment_result" in locals():
            del alignment_result
        backend.eval()

        return result

    completed = 0
    for tgt_idx, src_indices in alignment_tasks:
        tgt_layer = target_layers[tgt_idx]
        logger.info("ALIGNMENT: Starting layer %d (task %d/%d)...", tgt_layer, completed + 1, len(alignment_tasks))
        try:
            result = _align_target_group(tgt_idx, src_indices)
        except Exception as e:
            logger.error("ALIGNMENT FAILED: Layer %d crashed with %s: %s", tgt_layer, type(e).__name__, e)
            import traceback
            logger.error("TRACEBACK:\n%s", traceback.format_exc())
            raise

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

        # Log ~20 times during run, regardless of task count
        log_interval = max(1, len(alignment_tasks) // 20)
        if completed % log_interval == 0 or completed == len(alignment_tasks):
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

    # Convert HOT coupling matrix to list for storage
    coupling_list: list[list[float]] | None = None
    if hot_result.layer_coupling is not None:
        coupling_arr = hot_result.layer_coupling
        shape = backend.shape(coupling_arr)
        coupling_list = []
        for i in range(int(shape[0])):
            row_vals = []
            for j in range(int(shape[1])):
                val = backend.take(backend.take(coupling_arr, backend.array([i]), axis=0), backend.array([j]), axis=1)
                row_vals.append(float(backend.to_scalar(val)))
            coupling_list.append(row_vals)

    return MergeAlignmentResult(
        layer_mapping=layer_mapping,
        feature_transforms=feature_transforms,
        scale_ratios=scale_ratios,
        attention_transforms=attention_transforms,
        k_transforms=k_transforms,
        v_transforms=v_transforms,
        intermediate_transforms=intermediate_transforms,
        gate_transforms=gate_transforms,
        layer_cka_scores=layer_cka_scores,
        layer_coupling=coupling_list,
    )
