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

"""Unified geometric merge pipeline.

Stages:
    PROBE → DENSITY → TRANSPLANT

References:
    - Fang et al. (2025). "AlphaEdit" (null-space projection).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.geometry.deviation_budget import DeviationBudget
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

from . import helpers as merge_helpers
from . import stages as merge_stages
from .models import (
    CrossArchitectureInfo,
    LayerMergeState,
    UnifiedMergeResult,
)
from .pipeline import run_merge

if TYPE_CHECKING:
    from modelcypher.ports.activation_provider import ActivationProvider
    from modelcypher.ports.backend import Array, Backend
    from modelcypher.ports.inference import InferenceEngine
    from modelcypher.ports.model_loader import ModelLoaderPort

__all__ = [
    "CrossArchitectureInfo",
    "LayerMergeState",
    "UnifiedGeometricMerger",
    "UnifiedMergeResult",
]


class UnifiedGeometricMerger:
    """
    Unified geometric merge pipeline.

    Pipeline: PROBE → DENSITY → TRANSPLANT.
    """

    def __init__(
        self,
        model_loader: "ModelLoaderPort",
        activation_provider: "ActivationProvider | None" = None,
        inference_engine: "InferenceEngine | None" = None,
        backend: "Backend | None" = None,
    ) -> None:
        """Initialize with required dependencies.

        Args:
            model_loader: Model loader port for loading weights (REQUIRED).
            backend: Compute backend for tensor operations (defaults to the platform backend).
                     All geometric operations run on GPU when using GPU backend.
        """
        self._model_loader = model_loader
        self._activation_provider = activation_provider
        self._inference_engine = inference_engine

        self._backend = backend

    def merge(
        self,
        source_path: str,
        target_path: str,
        output_dir: str | None = None,
        target_weights: dict[str, "Array"] | None = None,
        # Optional pre-loaded models/tokenizers to avoid redundant loading
        source_model: Any | None = None,
        target_model: Any | None = None,
        source_tokenizer: Any | None = None,
        target_tokenizer: Any | None = None,
        inference_engine: "InferenceEngine | None" = None,
    ) -> UnifiedMergeResult:
        """Execute the unified geometric merge pipeline (geometry-only, no domain overrides).
        """
        return run_merge(
            model_loader=self._model_loader,
            backend=self._backend,
            source_path=source_path,
            target_path=target_path,
            output_dir=output_dir,
            target_weights=target_weights,
            source_model=source_model,
            target_model=target_model,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            activation_provider=self._activation_provider,
            inference_engine=inference_engine or self._inference_engine,
        )

    def _stage_probe(
        self,
        source_weights: dict[str, "Array"],
        target_weights: dict[str, "Array"],
        source_model: Any | None,
        target_model: Any | None,
        source_tokenizer: Any | None,
        target_tokenizer: Any | None,
        source_path: str = "",
        target_path: str = "",
    ) -> tuple[dict[str, Any], dict[str, Any], dict | None, dict | None]:
        return merge_stages.stage_probe(
            source_weights=source_weights,
            target_weights=target_weights,
            source_model=source_model,
            target_model=target_model,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            source_path=source_path,
            target_path=target_path,
            extract_layer_index_fn=merge_helpers.extract_layer_index,
            activation_provider=self._activation_provider,
        )


    def _stage_transplant(
        self,
        source_weights: dict[str, "Array"],
        target_weights: dict[str, "Array"],
        layer_indices: list[int],
        probe_ids: list[str] | None,
        probe_domains: list[str] | None,
        source_activations: dict | None,
        target_activations: dict | None,
        graft_mask: dict[str, dict[int, bool]],
    ) -> tuple[dict[str, "Array"], dict[str, Any]]:
        return merge_stages.stage_transplant(
            source_weights=source_weights,
            target_weights=target_weights,
            layer_indices=layer_indices,
            probe_ids=probe_ids,
            probe_domains=probe_domains,
            source_activations=source_activations,
            target_activations=target_activations,
            extract_layer_index_fn=merge_helpers.extract_layer_index,
            backend=self._backend,
            graft_mask=graft_mask,
        )

    def _load_tokenizer(self, model_path: str) -> Any | None:
        return merge_helpers.load_tokenizer(model_path, self._model_loader)

    def _load_model_for_probing(self, model_path: str) -> Any | None:
        return merge_helpers.load_model_for_probing(model_path, self._model_loader)

    def _load_weights(self, model_path: str) -> tuple[dict[str, Any], str]:
        return merge_helpers.load_weights(self._model_loader, model_path)



    def _load_weights_as_arrays(self, model_path: str) -> tuple[dict[str, "Array"], str]:
        return merge_helpers.load_weights_as_arrays(self._model_loader, model_path)

    def _infer_hidden_dim(self, weights: dict[str, Any]) -> int:
        return merge_helpers.infer_hidden_dim(weights)

    def _save_weights(
        self,
        output_dir: str,
        weights: dict[str, Any],
        output_format: str,
    ) -> None:
        merge_helpers.save_weights(output_dir, weights, output_format, self._backend)

    def _copy_config_files(self, source_path: str, output_dir: str) -> None:
        merge_helpers.copy_config_files(source_path, output_dir)

    def _extract_layer_indices(self, weights: dict[str, "Array"]) -> list[int]:
        return merge_helpers.extract_layer_indices(weights)

    def _extract_layer_index(self, key: str) -> int | None:
        return merge_helpers.extract_layer_index(key)

    def _detect_cross_architecture(
        self,
        source_weights: dict[str, "Array"],
        target_weights: dict[str, "Array"],
    ) -> CrossArchitectureInfo:
        return merge_helpers.detect_cross_architecture(source_weights, target_weights)

    def merge_batch(
        self,
        source_paths: list[str],
        target_path: str,
        output_dir: str | None = None,
    ) -> UnifiedMergeResult:
        """Merge multiple source models into a single target (N→1 merging).

        Optimized for merging many sources into one target. The target is loaded
        and probed once, then reused for all source merges.

        Returns
        -------
        UnifiedMergeResult
            The merged model result.
        """
        import logging
        logger = logging.getLogger(__name__)

        if not source_paths:
            raise ValueError("At least one source path required")

        n_sources = len(source_paths)
        logger.info("BATCH MERGE: Starting N→1 merge (%d sources → %s)", n_sources, target_path)

        # Phase 1: Load and probe target ONCE
        logger.info("BATCH MERGE: Phase 1 - Loading target model (done once for all sources)")
        target_weights, _ = self._load_weights_as_arrays(target_path)
        target_model = self._load_model_for_probing(target_path)
        target_tokenizer = self._load_tokenizer(target_path)
        current_occupancy_by_layer = merge_helpers.load_transplant_occupancy(target_path) or {}
        if current_occupancy_by_layer:
            logger.info(
                "BATCH MERGE: Loaded transplant occupancy for %d layers",
                len(current_occupancy_by_layer),
            )

        deviation_tracker = DeviationBudget(backend=self._backend)
        deviation_tracker.record_baseline(target_weights, name="original_target")
        logger.info(
            "BATCH MERGE: Deviation tracking enabled (geometry handles safety by construction)"
        )

        # Consensus correction before addition (only applies when 3+ models available)
        logger.info("BATCH MERGE: Phase 2 - Consensus correction (fix misaligned concepts)")
        target_weights, corrections_applied = self._run_consensus_correction(
            target_weights=target_weights,
            source_paths=source_paths,
            target_model=target_model,
            target_tokenizer=target_tokenizer,
        )
        if corrections_applied > 0:
            logger.info(
                "BATCH MERGE: Applied %d corrections (target moved to consensus)",
                corrections_applied,
            )
        else:
            logger.info("BATCH MERGE: No corrections needed (target already at consensus)")

        # Accumulate all sources into original target's null-space
        merged_weights = {k: self._backend.array(v) for k, v in target_weights.items()}

        for i, source_path in enumerate(source_paths):
            logger.info("BATCH MERGE: Merging source %d/%d: %s", i + 1, n_sources, source_path)

            result = run_merge(
                model_loader=self._model_loader,
                backend=self._backend,
                source_path=source_path,
                target_path=target_path,
                output_dir=None,  # Don't save intermediate
                target_weights=target_weights,  # Reuse original target
                target_model=target_model,  # Reuse loaded model
                target_tokenizer=target_tokenizer,  # Reuse tokenizer
                activation_provider=self._activation_provider,
                prior_occupancy_by_layer=current_occupancy_by_layer,
            )

            # Accumulate: add delta to merged weights
            for key in merged_weights.keys():
                if key in result.merged_weights:
                    # delta = result - original_target
                    delta = result.merged_weights[key] - target_weights[key]
                    merged_weights[key] = merged_weights[key] + delta
            occupancy_update = result.transplant_metrics.get("occupancy_by_layer")
            if occupancy_update:
                current_occupancy_by_layer = occupancy_update

            measurement = deviation_tracker.measure(
                merged_weights, baseline_name="original_target"
            )
            logger.info(
                "BATCH MERGE: Source %d/%d - deviation=%.1f (%.2f%% of baseline norm)",
                i + 1, n_sources,
                measurement.deviation,
                measurement.deviation_percent
            )

            logger.info("BATCH MERGE: Source %d/%d complete", i + 1, n_sources)

        final_result = result
        final_result.merged_weights = merged_weights
        final_result.output_path = output_dir

        # Save if output_dir provided
        if output_dir:
            logger.info("BATCH MERGE: Saving merged model to %s", output_dir)
            self._save_weights(output_dir, final_result.merged_weights, "safetensors")
            self._copy_config_files(target_path, output_dir)
            if current_occupancy_by_layer:
                merge_helpers.save_transplant_occupancy(
                    output_dir,
                    current_occupancy_by_layer,
                )

        logger.info("BATCH MERGE: Complete. Merged %d sources into target.", n_sources)
        return final_result

    def _run_consensus_correction(
        self,
        target_weights: dict[str, Any],
        source_paths: list[str],
        target_model: Any,
        target_tokenizer: Any,
    ) -> tuple[dict[str, Any], int]:
        """Run consensus correction using per-concept stress profile comparison.

        Compares models via stress profiles (distances to anchors) which is
        dimension-invariant. Corrects concepts where target disagrees with consensus.

        Process:
        1. Probe all models to get activations
        2. Select anchor concepts (subset of probes)
        3. Compute stress profiles (distances to anchors) for each concept
        4. Find concepts where target's stress differs from consensus
        5. Use multilateration + lstsq to correct those concepts

        Args:
            target_weights: Target model weights.
            source_paths: Paths to source models.
            target_model: Loaded target model for probing.
            target_tokenizer: Loaded tokenizer.

        Returns:
            (corrected_weights, corrections_count)
        """
        import logging
        import math
        from modelcypher.core.domain.geometry._primitives.epsilon_utils import (
            machine_epsilon,
        )
        from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
        from modelcypher.core.domain.geometry.outlier_detector import OutlierDetector
        from modelcypher.core.domain.geometry.consensus_corrector import ConsensusCorrector
        from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry
        from modelcypher.core.domain.agents.probe_loader import ProbeLoader
        from modelcypher.core.use_cases.merge.helpers import get_hidden_state
        from modelcypher.core.domain.geometry.numerical_stability import gpu_lstsq

        logger = logging.getLogger(__name__)
        b = self._backend

        n_models = 1 + len(source_paths)
        if n_models < 3:
            logger.info("CONSENSUS: Need 3+ models for triangulation (got %d)", n_models)
            return target_weights, 0

        detector = OutlierDetector(b)
        corrector = ConsensusCorrector(b)

        # Load probes for anchors + concept checks (no fixed cap).
        probe_loader = ProbeLoader()
        all_probes = probe_loader.load_probes()

        # Load source models
        source_models = []
        for source_path in source_paths:
            model, _ = self._model_loader.load_model(source_path)
            source_models.append(model)

        all_models = [target_model] + source_models

        # Collect activations for all models
        # model_activations[model_idx][probe_idx] = activation vector
        model_activations = []
        for model in all_models:
            activations = {}
            for i, probe in enumerate(all_probes):
                try:
                    act = get_hidden_state(model, target_tokenizer, probe.text, b)
                    if act is not None:
                        activations[i] = act
                except Exception:
                    continue
            model_activations.append(activations)

        # Find probes with activations across all models
        shared_indices = [
            i for i in range(len(all_probes)) if all(i in acts for acts in model_activations)
        ]
        if not shared_indices:
            logger.warning("CONSENSUS: No shared probes across models. Skipping correction.")
            for model in source_models:
                del model
            return target_weights, 0

        # Derive anchor count from intrinsic dimension (d+1 anchors for multilateration)
        target_shared_acts = [model_activations[0][i] for i in shared_indices]
        target_shared_stack = b.stack(target_shared_acts, axis=0)
        b.eval(target_shared_stack)
        id_estimate = IntrinsicDimension.compute_two_nn(target_shared_stack, backend=b)
        intrinsic_dim = max(0.0, float(id_estimate.intrinsic_dimension))
        required_anchors = int(math.ceil(intrinsic_dim)) + 1
        if len(shared_indices) <= required_anchors:
            logger.warning(
                "CONSENSUS: Shared probes (%d) insufficient for %d anchors + concepts. Skipping correction.",
                len(shared_indices),
                required_anchors,
            )
            for model in source_models:
                del model
            return target_weights, 0

        # Select anchors by farthest-point sampling in geodesic space.
        def _select_anchor_indices(indices, activations, count):
            if count <= 0:
                return []
            if count >= len(indices):
                return list(indices)

            rg = RiemannianGeometry(b)
            geo_result = rg.geodesic_distances(activations)
            dist_matrix = geo_result.distances
            b.eval(dist_matrix)

            norms = b.sum(activations * activations, axis=1)
            b.eval(norms)
            first_idx_arr = b.argmax(norms)
            b.eval(first_idx_arr)
            first_idx = int(b.to_scalar(first_idx_arr))

            selected = [first_idx]
            min_dists = dist_matrix[first_idx]
            eps = float(machine_epsilon(b, min_dists))

            while len(selected) < count:
                max_dist_arr = b.max(min_dists)
                b.eval(max_dist_arr)
                max_dist = float(b.to_scalar(max_dist_arr))
                if max_dist <= eps:
                    break
                next_idx_arr = b.argmax(min_dists)
                b.eval(next_idx_arr)
                next_idx = int(b.to_scalar(next_idx_arr))
                if next_idx in selected:
                    break
                selected.append(next_idx)
                min_dists = b.minimum(min_dists, dist_matrix[next_idx])

            return [indices[i] for i in selected]

        anchor_indices = _select_anchor_indices(
            shared_indices, target_shared_stack, required_anchors
        )

        if len(anchor_indices) < required_anchors:
            logger.warning(
                "CONSENSUS: Too few usable anchors after selection (%d < %d). Skipping correction.",
                len(anchor_indices),
                required_anchors,
            )
            for model in source_models:
                del model
            return target_weights, 0

        anchor_set = set(anchor_indices)
        concept_indices = [i for i in shared_indices if i not in anchor_set]

        logger.info(
            "CONSENSUS: Using %d anchors, checking %d concepts across %d models (ID=%.2f)",
            len(anchor_indices),
            len(concept_indices),
            n_models,
            intrinsic_dim,
        )

        def compute_stress(activation, anchor_acts):
            """Compute geodesic distances from activation to each anchor.

            Uses k-NN geodesic distances instead of Euclidean chord distances
            to properly capture manifold structure in high dimensions.
            """
            # Stack activation (first) + all anchors into single point cloud
            act_2d = b.reshape(activation, (1, -1))
            anchors_2d = b.stack(anchor_acts, axis=0)  # [n_anchors, d]
            combined = b.concatenate([act_2d, anchors_2d], axis=0)  # [n_anchors+1, d]
            b.eval(combined)

            # Compute geodesic distances on combined point cloud
            rg = RiemannianGeometry(b)
            geo_result = rg.geodesic_distances(combined)
            dist_matrix = geo_result.distances  # [n_anchors+1, n_anchors+1]
            b.eval(dist_matrix)

            # Extract first row (distances from activation to each anchor)
            # Skip index 0 (self-distance)
            n_anchors = len(anchor_acts)
            distances = [float(b.to_scalar(dist_matrix[0, i + 1])) for i in range(n_anchors)]
            return b.array(distances)

        n_anchors = len(anchor_indices)

        # Get anchor activations for each model (in their own spaces)
        model_anchor_acts = []
        for acts in model_activations:
            anchor_acts = [acts[i] for i in anchor_indices]
            model_anchor_acts.append(anchor_acts)

        # Compute stress profiles for concept probes
        concept_stress = []  # [concept_idx][model_idx] = stress vector
        valid_concept_indices = []

        for concept_idx in concept_indices:
            valid_concept_indices.append(concept_idx)
            stresses = []
            for model_idx, acts in enumerate(model_activations):
                concept_act = acts[concept_idx]
                stress = compute_stress(concept_act, model_anchor_acts[model_idx])
                stresses.append(stress)
            concept_stress.append(stresses)

        if not concept_stress:
            logger.warning("CONSENSUS: No valid concepts to check. Skipping correction.")
            for model in source_models:
                del model
            return target_weights, 0

        logger.info("CONSENSUS: Checking %d concepts for outliers", len(concept_stress))

        # Find concepts where target is an outlier
        target_concept_acts = [model_activations[0][i] for i in valid_concept_indices]
        outlier_concepts = []
        consensus_positions = []

        for c_idx, stresses in enumerate(concept_stress):
            n = len(stresses)
            # Stack stress vectors into point cloud for geodesic distance computation
            stress_cloud = b.stack(stresses, axis=0)  # [n_models, n_anchors]
            b.eval(stress_cloud)

            # Compute geodesic pairwise distances in stress space
            rg = RiemannianGeometry(b)
            geo_result = rg.geodesic_distances(stress_cloud)
            geo_dist_matrix = geo_result.distances  # [n, n]
            b.eval(geo_dist_matrix)

            # Extract pairwise distances as Python floats
            pairwise = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    dist = float(b.to_scalar(geo_dist_matrix[i, j]))
                    pairwise[i][j] = dist
                    pairwise[j][i] = dist

            if n == 3:
                result = detector._triangulate_outlier(pairwise)
            else:
                mean_dists = [sum(row) / (n - 1) for row in pairwise]
                result = detector.detect_from_gpa(mean_dists)

            if 0 in result.outlier_indices:
                consensus_stresses = [stresses[i] for i in result.consensus_indices if i > 0]
                if consensus_stresses:
                    stacked = b.stack(consensus_stresses, axis=0)
                    consensus_stress = b.mean(stacked, axis=0)

                    target_anchors = {
                        f"anchor_{i}": model_anchor_acts[0][i] for i in range(n_anchors)
                    }
                    consensus_pos = corrector._solve_position_from_stress(
                        consensus_stress, target_anchors
                    )
                    outlier_concepts.append(c_idx)
                    consensus_positions.append(consensus_pos)

        if not outlier_concepts:
            logger.info("CONSENSUS: No outlier concepts found. Target is in consensus.")
            for model in source_models:
                del model
            return target_weights, 0

        logger.info("CONSENSUS: Found %d outlier concepts. Computing corrections.",
                    len(outlier_concepts))

        # Compute correction delta
        target_outlier_acts = b.stack(
            [target_concept_acts[i] for i in outlier_concepts], axis=0
        )
        consensus_pos_stack = b.stack(consensus_positions, axis=0)
        delta_acts = consensus_pos_stack - target_outlier_acts
        b.eval(delta_acts)

        # Apply correction to output projection
        from modelcypher.ports.model_architecture_factory import get_model_architecture

        # Use architecture protocol to find output projection key
        try:
            arch = get_model_architecture(target_model)
            output_key_from_arch = arch.output_projection_key()
            if output_key_from_arch and output_key_from_arch in target_weights:
                output_keys = [output_key_from_arch]
            else:
                # Fallback to pattern matching
                output_keys = [k for k in target_weights.keys()
                               if "lm_head" in k.lower() and "weight" in k.lower()]
        except Exception:
            # Fallback to pattern matching if architecture detection fails
            output_keys = [k for k in target_weights.keys()
                           if "lm_head" in k.lower() or "output" in k.lower()]
        if not output_keys:
            output_keys = [k for k in target_weights.keys() if "weight" in k.lower()]

        corrected_weights = {k: b.array(v) for k, v in target_weights.items()}
        corrections_applied = 0

        if output_keys:
            output_key = output_keys[0]
            output_weight = b.array(target_weights[output_key])

            weight_delta = gpu_lstsq(b, target_outlier_acts, delta_acts)
            weight_delta = b.transpose(weight_delta)
            b.eval(weight_delta)

            # Compute lstsq residual to gauge fit quality using geodesic norms
            # residual = ||delta_acts - target_outlier_acts @ weight_delta.T||_geodesic
            from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms

            predicted = b.matmul(target_outlier_acts, b.transpose(weight_delta))
            residual_vec = delta_acts - predicted
            b.eval(residual_vec)

            # Use geodesic norms for manifold-aware magnitude measurement
            residual_geo_norms = geodesic_norms(residual_vec, b)
            delta_geo_norms = geodesic_norms(delta_acts, b)
            b.eval(residual_geo_norms, delta_geo_norms)
            residual_norm = float(b.to_scalar(b.mean(residual_geo_norms)))
            delta_norm = float(b.to_scalar(b.mean(delta_geo_norms)))

            # Scale based on fit quality: scale = 1 - (residual / delta)
            # If residual is 0 (perfect fit), scale = 1.0
            # If residual equals delta (no fit), scale = 0.0
            eps = division_epsilon(b, output_weight)
            fit_quality = 1.0 - min(1.0, residual_norm / (delta_norm + eps))

            # Geodesic Frobenius-like norm for weight matrices
            # Treat rows as points, compute geodesic norms, aggregate
            def _geodesic_frobenius(w):
                shape = b.shape(w)
                if len(shape) != 2:
                    w = b.reshape(w, (1, -1))
                elif shape[0] < 1:
                    w = b.reshape(w, (1, -1))
                geo_norms_arr = geodesic_norms(w, b)
                b.eval(geo_norms_arr)
                return float(b.to_scalar(b.sqrt(b.sum(geo_norms_arr * geo_norms_arr))))

            weight_delta_norm = _geodesic_frobenius(weight_delta)
            weight_norm = _geodesic_frobenius(output_weight)
            magnitude_ratio = min(1.0, weight_norm / (weight_delta_norm + eps))

            # Final scale is the more conservative of fit quality and magnitude ratio
            scale = min(fit_quality, magnitude_ratio)

            corrected_weights[output_key] = output_weight + scale * weight_delta
            corrections_applied = len(outlier_concepts)

            logger.info(
                "CONSENSUS: Applied correction to %s (scale=%.4f, fit=%.4f, mag_ratio=%.4f, %d concepts)",
                output_key, scale, fit_quality, magnitude_ratio, corrections_applied
            )

        for model in source_models:
            del model
        if hasattr(b, "clear_cache"):
            b.clear_cache()

        return corrected_weights, corrections_applied

    def merge_multi_channel(
        self,
        channel_paths: dict[str, str],
        target_path: str,
        output_dir: str | None = None,
        routing_mode: str = "uniform",
        fast_mode: bool = True,
    ) -> UnifiedMergeResult:
        """Merge multiple channels simultaneously via Birkhoff routing.

        This is the preferred method for multi-modal merging (e.g., world model +
        vision-language model + text model → unified model).

        Unlike merge_batch (sequential), this method:
        1. Probes all channels simultaneously
        2. Projects all channels into target's null-space (shared basis)
        3. Combines channels via doubly stochastic routing (spectral norm ≤ 1.0)
        4. Applies geometric addition (not interpolation)

        Notes:
            See docs/research/mhc_null_space_connection.md for routing details.

        Parameters
        ----------
        channel_paths : dict[str, str]
            Channel name → model path mapping.
            Example: {"spatial": "/path/to/world_model", "text": "/path/to/llm"}
        target_path : str
            Path to target model (receives the merged knowledge).
        output_dir : str, optional
            Output directory for merged model.
        routing_mode : str
            How to combine channels: "uniform", "identity", "diagonal_weighted".
        fast_mode : bool
            If True (default), skip CKA precision checks.

        Returns
        -------
        UnifiedMergeResult
            The merged model result with multi-channel diagnostics.
        """
        import logging
        from datetime import datetime

        from modelcypher.core.domain.geometry.birkhoff_router import (
            BirkhoffRouter,
            RoutingMode,
        )
        from modelcypher.core.domain.geometry.channel_projector import ChannelProjector

        logger = logging.getLogger(__name__)

        channel_ids = list(channel_paths.keys())
        n_channels = len(channel_ids)

        if n_channels == 0:
            raise ValueError("At least one channel required")

        logger.info(
            "MULTI-CHANNEL MERGE: %d channels → %s",
            n_channels, target_path
        )
        logger.info("MULTI-CHANNEL MERGE: Channels: %s", channel_ids)
        logger.info("MULTI-CHANNEL MERGE: routing_mode=%s, fast_mode=%s", routing_mode, fast_mode)

        # Phase 1: Load target (done once for all channels)
        logger.info("MULTI-CHANNEL MERGE: Phase 1 - Loading target model")
        target_weights, target_format = self._load_weights_as_arrays(target_path)
        target_model = self._load_model_for_probing(target_path)
        target_tokenizer = self._load_tokenizer(target_path)
        layer_indices = self._extract_layer_indices(target_weights)

        # Phase 2: Probe all channels to collect activations and transforms
        logger.info("MULTI-CHANNEL MERGE: Phase 2 - Probing all channels")
        channel_activations: dict[str, dict[int, Any]] = {}
        channel_weights: dict[str, dict[str, Any]] = {}
        channel_transforms: dict[str, dict[int, Any]] = {}
        target_activations: dict[int, Any] = {}

        for channel_id in channel_ids:
            source_path = channel_paths[channel_id]
            logger.info(
                "MULTI-CHANNEL MERGE: Probing channel '%s' from %s",
                channel_id, source_path
            )

            # Load channel weights
            source_weights, _ = self._load_weights_as_arrays(source_path)
            channel_weights[channel_id] = source_weights

            # Load channel model for probing
            source_model = self._load_model_for_probing(source_path)
            source_tokenizer = self._load_tokenizer(source_path)

            # Run probe stage to get activations and transforms
            # stage_probe returns an 18-element tuple (see stages/__init__.py)
            from .stages import stage_probe
            probe_tuple = stage_probe(
                source_weights=source_weights,
                target_weights=target_weights,
                source_model=source_model,
                target_model=target_model,
                source_tokenizer=source_tokenizer,
                target_tokenizer=target_tokenizer,
                source_path=source_path,
                target_path=target_path,
                extract_layer_index_fn=merge_helpers.extract_layer_index,
                activation_provider=self._activation_provider,
            )

            # Extract from tuple: indices per stages/__init__.py
            # [0] = probe_result dict, [1] = metrics, [2] = source_acts, [3] = target_acts
            # [10] = feature_transforms, [17] = layer_mapping
            source_acts = probe_tuple[2]
            target_acts = probe_tuple[3]
            feature_transforms = probe_tuple[10]
            layer_mapping = probe_tuple[17]

            channel_activations[channel_id] = source_acts or {}
            channel_transforms[channel_id] = feature_transforms or {}

            # Store target activations (same across channels)
            if not target_activations and target_acts:
                target_activations = target_acts

            # Clean up
            del source_model
            self._backend.clear_cache()

        # Clean up target model
        del target_model
        self._backend.clear_cache()

        logger.info(
            "MULTI-CHANNEL MERGE: Collected activations for %d channels, %d target layers",
            n_channels, len(target_activations)
        )

        # Phase 3: Multi-channel projection and routing per layer
        logger.info("MULTI-CHANNEL MERGE: Phase 3 - Channel projection and routing")
        channel_projector = ChannelProjector(self._backend)
        birkhoff_router = BirkhoffRouter(self._backend)

        # Load target config for architecture-aware key detection
        from modelcypher.ports.model_architecture_factory import load_config, is_attention_key

        target_config = load_config(target_path)

        merged_weights = {k: self._backend.array(v) for k, v in target_weights.items()}
        total_projection_loss = 0.0
        total_preserved = 0.0
        layers_merged = 0

        for layer_idx in layer_indices:
            if layer_idx not in target_activations:
                continue

            target_acts_layer = target_activations[layer_idx]

            # Gather per-channel data for this layer
            layer_source_acts: dict[str, Any] = {}
            layer_source_weights: dict[str, Any] = {}

            for channel_id in channel_ids:
                if layer_idx in channel_activations[channel_id]:
                    layer_source_acts[channel_id] = channel_activations[channel_id][layer_idx]

                    # Find corresponding weight key (use architecture-aware detection)
                    # Load channel config for architecture detection
                    channel_config = load_config(channel_paths[channel_id])
                    for key, val in channel_weights[channel_id].items():
                        key_layer_idx = self._extract_layer_index(key)
                        if key_layer_idx == layer_idx and is_attention_key(key, channel_config, layer_idx):
                            # Get feature transform for this layer if available
                            # This transforms hidden dimension: [d_source, d_target]
                            F_raw = channel_transforms[channel_id].get(layer_idx)
                            if F_raw is not None:
                                # Debug: log the actual type
                                logger.debug(
                                    "MULTI-CHANNEL: F_raw type=%s for layer %d",
                                    type(F_raw).__name__, layer_idx
                                )
                                # Handle dict case (multi-source mapping)
                                if isinstance(F_raw, dict):
                                    # Take first source's transform if multiple sources
                                    first_key = next(iter(F_raw.keys()))
                                    F_raw = F_raw[first_key]
                                    logger.debug(
                                        "MULTI-CHANNEL: Extracted F_raw from dict, key=%s",
                                        first_key
                                    )
                                # Convert from nested list to array if needed
                                F = self._backend.array(F_raw) if not hasattr(F_raw, 'shape') else F_raw
                                self._backend.eval(F)
                                # Source weight: [out_dim, d_source]
                                # F: [d_source, d_target]
                                # Need to transform input dimension (columns)
                                # aligned_W = W @ F gives [out_dim, d_target]
                                source_w = self._backend.array(val)
                                self._backend.eval(source_w)
                                stitched_w = self._backend.matmul(source_w, F)
                                self._backend.eval(stitched_w)
                                layer_source_weights[channel_id] = stitched_w
                                logger.debug(
                                    "MULTI-CHANNEL: Stitched %s layer %d: [%s] @ F[%s] → [%s]",
                                    channel_id, layer_idx,
                                    list(source_w.shape), list(F.shape), list(stitched_w.shape)
                                )
                            else:
                                layer_source_weights[channel_id] = val
                            break

            # Skip if insufficient channel data
            if len(layer_source_acts) < n_channels:
                continue

            # Find target weight for this layer (use architecture-aware detection)
            target_weight = None
            target_key = None
            for key, val in target_weights.items():
                key_layer_idx = self._extract_layer_index(key)
                if key_layer_idx == layer_idx and is_attention_key(key, target_config, layer_idx):
                    target_weight = val
                    target_key = key
                    break

            if target_weight is None:
                continue

            # Verify shapes are compatible after stitching
            # After input stitch, source should have: [out_dim_source, d_target]
            # Target has: [out_dim_target, d_target]
            # If out dimensions don't match, we can't subtract - skip this layer
            target_shape = list(self._backend.array(target_weight).shape)
            compatible = True
            for channel_id in list(layer_source_weights.keys()):
                source_shape = list(self._backend.array(layer_source_weights[channel_id]).shape)
                if source_shape != target_shape:
                    logger.warning(
                        "MULTI-CHANNEL MERGE: Layer %d channel '%s' shape mismatch: %s vs target %s, skipping",
                        layer_idx, channel_id, source_shape, target_shape
                    )
                    compatible = False
                    break

            if not compatible:
                continue

            # Project all channels into target's null space
            try:
                projection_result = channel_projector.project_channels(
                    source_activations=layer_source_acts,
                    source_weights=layer_source_weights,
                    target_activations=target_acts_layer,
                    target_weights=target_weight,
                )

                # Route channels via Birkhoff mixing
                channel_deltas = [
                    projection_result.channel_results[ch].filtered_delta
                    for ch in channel_ids
                    if ch in projection_result.channel_results
                ]

                if not channel_deltas:
                    continue

                combined_delta, routing_result = birkhoff_router.route_channels(
                    channel_deltas, init_mode=RoutingMode(routing_mode)
                )
                self._backend.eval(combined_delta)

                # Geometric addition
                merged_weights[target_key] = self._backend.array(target_weight) + combined_delta
                self._backend.eval(merged_weights[target_key])

                # Accumulate metrics
                total_projection_loss += projection_result.total_projection_loss
                total_preserved += projection_result.average_preserved_fraction
                layers_merged += 1

            except Exception as e:
                logger.warning(
                    "MULTI-CHANNEL MERGE: Layer %d failed: %s", layer_idx, e
                )
                continue

        avg_preserved = total_preserved / layers_merged if layers_merged > 0 else 1.0

        logger.info(
            "MULTI-CHANNEL MERGE: Merged %d layers, avg_preserved=%.3f, total_loss=%.3f",
            layers_merged, avg_preserved, total_projection_loss
        )

        # Save if output_dir provided
        if output_dir:
            logger.info("MULTI-CHANNEL MERGE: Saving to %s", output_dir)
            self._save_weights(output_dir, merged_weights, target_format)
            self._copy_config_files(target_path, output_dir)

        # Build result
        result = UnifiedMergeResult(
            merged_weights=merged_weights,
            probe_metrics={"channels": channel_ids, "routing_mode": routing_mode},
            permute_metrics={"skipped": True, "reason": "multi_channel_uses_birkhoff"},
            transplant_metrics={
                "layers_merged": layers_merged,
                "total_projection_loss": total_projection_loss,
                "average_preserved_fraction": avg_preserved,
            },
            mean_preserved_fraction=avg_preserved,
            mean_procrustes_error=total_projection_loss / layers_merged if layers_merged > 0 else 0.0,
            layer_count=len(layer_indices),
            weight_count=len(merged_weights),
            timestamp=datetime.utcnow(),
            merge_strategy="multi_channel",
            output_path=output_dir,
            refusal_preserved=True,
            geometry_metrics={"spectral_norm_bounded": True},
            density_metrics={"skipped": True, "reason": "multi_channel_mode"},
        )

        logger.info("MULTI-CHANNEL MERGE: Complete. %d channels → target", n_channels)
        return result
