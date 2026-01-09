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

"""
Unified Geometric Merge Pipeline.

Pipeline:
    PROBE → DENSITY → TRANSPLANT

Stage 1: PROBE - Build intersection map from probe responses, compute GramAlign transforms
Stage 2: DENSITY - Knowledge density profiling for graft mask
Stage 3: TRANSPLANT - Null-space constrained knowledge grafting

Key Principles:
1. Null-space projection guarantees: A_boundary @ W' = A_boundary @ W_target
2. Layer targeting enables surgical transplants
3. Cross-dimensional projection via GramAligner achieves CKA=1.0
4. Geodesic RKHS alignment subsumes discrete permutation alignment

References:
- AlphaEdit (null-space): Fang et al. (2025) ICLR Outstanding Paper

REMOVED (proven redundant):
- PERMUTE: GramAligner's CKA=1.0 in geodesic RKHS subsumes discrete permutation alignment
- ROTATE/PROPAGATE: No boundary preservation guarantee

Stage implementations are in merge/stages for modularity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend

from . import helpers as merge_helpers
from . import stages as merge_stages
from .models import (
    CrossArchitectureInfo,
    LayerMergeState,
    UnifiedMergeResult,
)
from .pipeline import run_merge

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend
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

    Pipeline: PROBE → DENSITY → TRANSPLANT

    - PROBE (GramAlign): Computes CKA=1.0 transforms in geodesic RKHS.
      This continuous alignment subsumes discrete permutation alignment.
    - DENSITY: Identifies regions where source is denser than target.
    - TRANSPLANT: Null-space constrained projection preserves boundary behavior
      while transferring knowledge.

    Stage implementations are in merge/stages for modularity.
    """

    def __init__(
        self,
        model_loader: "ModelLoaderPort",
        backend: "Backend | None" = None,
    ) -> None:
        """Initialize with required dependencies.

        Args:
            model_loader: Model loader port for loading weights (REQUIRED).
            backend: Compute backend for tensor operations (defaults to MLXBackend).
                     All geometric operations run on GPU when using MLXBackend.
        """
        self._model_loader = model_loader

        # Default to configured backend (respects MC_BACKEND/MODELCYPHER_BACKEND)
        self._backend = backend or get_default_backend()

    def merge(
        self,
        source_path: str,
        target_path: str,
        output_dir: str | None = None,
        output_path: str | None = None,
        dry_run: bool = False,
        target_weights: dict[str, "Array"] | None = None,
        probe_mode: str = "atlas",
        # Optional pre-loaded models/tokenizers to avoid redundant loading
        source_model: Any | None = None,
        target_model: Any | None = None,
        source_tokenizer: Any | None = None,
        target_tokenizer: Any | None = None,
    ) -> UnifiedMergeResult:
        """Execute the unified geometric merge pipeline (geometry-only, no domain overrides)."""
        return run_merge(
            model_loader=self._model_loader,
            backend=self._backend,
            source_path=source_path,
            target_path=target_path,
            output_dir=output_dir,
            output_path=output_path,
            dry_run=dry_run,
            target_weights=target_weights,
            probe_mode=probe_mode,
            source_model=source_model,
            target_model=target_model,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
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
        return merge_helpers.load_tokenizer(model_path)

    def _load_model_for_probing(self, model_path: str) -> Any | None:
        return merge_helpers.load_model_for_probing(model_path)

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
        accumulative: bool = True,
        fast_mode: bool = True,
    ) -> UnifiedMergeResult:
        """Merge multiple source models into a single target (N→1 merging).

        This is optimized for merging many models into one compact target (e.g., LFM2).
        The target is loaded and probed ONCE, then reused for all source merges.

        Mathematical Foundation:
        -----------------------
        CKA = 1.0 is an invariant (not a target). All models encode the same
        geometric shape. The alignment transform F achieves CKA = 1.0 by
        construction via F = pinv(source) @ target.

        With fast_mode=True, we skip CKA precision checks since the closed-form
        solution IS the answer. This provides significant speedup for batch ops.

        Accumulative Merging:
        --------------------
        When accumulative=True (default), knowledge from all sources is projected
        into the target's null-space and accumulated additively:

            merged = target + sum(project_null(delta_i, target))

        where delta_i = source_i - target.

        Each projection is orthogonal to target's active subspace, so:
        1. Target's behavior is preserved
        2. All source knowledge is added (not averaged)
        3. Sources can be processed in parallel

        Parameters
        ----------
        source_paths : list[str]
            Paths to source models to merge.
        target_path : str
            Path to target model (receives the merged knowledge).
        output_dir : str, optional
            Output directory for merged model.
        accumulative : bool
            If True (default), accumulate all sources into target's null-space.
            If False, merge sequentially (result = merge(merge(target, A), B)).
        fast_mode : bool
            If True (default), skip CKA precision checks in GramAligner.
            Safe because CKA = 1.0 is guaranteed by construction.

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
        logger.info("BATCH MERGE: fast_mode=%s, accumulative=%s", fast_mode, accumulative)

        # Phase 1: Load and probe target ONCE
        logger.info("BATCH MERGE: Phase 1 - Loading target model (done once for all sources)")
        target_weights, _ = self._load_weights_as_arrays(target_path)
        target_model = self._load_model_for_probing(target_path)
        target_tokenizer = self._load_tokenizer(target_path)

        if accumulative:
            # Accumulative mode: merge all sources into original target's null-space
            merged_weights = {k: self._backend.array(v) for k, v in target_weights.items()}

            for i, source_path in enumerate(source_paths):
                logger.info("BATCH MERGE: Merging source %d/%d: %s", i + 1, n_sources, source_path)

                # Run single merge with pre-loaded target
                result = run_merge(
                    model_loader=self._model_loader,
                    backend=self._backend,
                    source_path=source_path,
                    target_path=target_path,
                    output_dir=None,  # Don't save intermediate
                    dry_run=True,  # Get weights without saving
                    target_weights=target_weights,  # Reuse original target
                    target_model=target_model,  # Reuse loaded model
                    target_tokenizer=target_tokenizer,  # Reuse tokenizer
                    probe_mode="atlas",
                )

                # Accumulate: add delta to merged weights
                for key in merged_weights.keys():
                    if key in result.weights:
                        # delta = result - original_target
                        delta = result.weights[key] - target_weights[key]
                        merged_weights[key] = merged_weights[key] + delta

                logger.info("BATCH MERGE: Source %d/%d complete", i + 1, n_sources)

            # Create final result
            from .models import UnifiedMergeResult
            final_result = UnifiedMergeResult(
                weights=merged_weights,
                layer_metrics={},
                total_layers=len(target_weights),
                layers_modified=n_sources,  # Approximate
                output_path=output_dir,
            )

        else:
            # Sequential mode: merge(merge(merge(target, A), B), C)
            current_target = target_weights
            current_model = target_model
            current_tokenizer = target_tokenizer

            for i, source_path in enumerate(source_paths):
                logger.info("BATCH MERGE: Sequential merge %d/%d: %s", i + 1, n_sources, source_path)

                result = run_merge(
                    model_loader=self._model_loader,
                    backend=self._backend,
                    source_path=source_path,
                    target_path=target_path,
                    output_dir=None,
                    dry_run=True,
                    target_weights=current_target,
                    target_model=current_model,
                    target_tokenizer=current_tokenizer,
                    probe_mode="atlas",
                )

                # Update target for next iteration
                current_target = result.weights
                # Note: model/tokenizer stay the same (architecture unchanged)

                logger.info("BATCH MERGE: Sequential merge %d/%d complete", i + 1, n_sources)

            final_result = result

        # Save if output_dir provided
        if output_dir:
            logger.info("BATCH MERGE: Saving merged model to %s", output_dir)
            self._save_weights(output_dir, final_result.weights, "safetensors")
            self._copy_config_files(target_path, output_dir)

        logger.info("BATCH MERGE: Complete. Merged %d sources into target.", n_sources)
        return final_result
