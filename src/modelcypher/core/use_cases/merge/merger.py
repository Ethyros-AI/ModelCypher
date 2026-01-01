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
    VOCAB → PROBE → PERMUTE → TRANSPLANT → VALIDATE

Stage 0: VOCABULARY - Cross-vocabulary embedding alignment
Stage 1: PROBE - Build intersection map from probe responses
Stage 2: PERMUTE - Git Re-Basin permutation alignment (same-arch only)
Stage 3: TRANSPLANT - Null-space constrained knowledge grafting
Stage 4: VALIDATE - Safety checks (numerical + content)

Key Principles:
1. Null-space projection guarantees: A_boundary @ W' = A_boundary @ W_target
2. Layer targeting enables surgical transplants
3. Cross-dimensional projection via GRAM_TRANSPORT
4. Permutation alignment reduces delta magnitude before transplant (same-arch)

References:
- Git Re-Basin: Ainsworth et al. (2023) arXiv:2209.04836
- AlphaEdit (null-space): Fang et al. (2025) ICLR Outstanding Paper

REMOVED (proven broken):
- rotate_blend: Alpha-blending has no constraint, destroys coherence
- ROTATE/BLEND/PROPAGATE: Only served rotate_blend

Stage implementations are in merge_stages/ subpackage for modularity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend

from . import helpers as merge_helpers
from . import stages as merge_stages
from .models import (
    CrossArchitectureInfo,
    LayerMergeState,
    UnifiedMergeConfig,
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
    "UnifiedMergeConfig",
    "UnifiedMergeResult",
    "unified_merge",
]


class UnifiedGeometricMerger:
    """
    Unified geometric merge pipeline.

    Pipeline: VOCAB → PROBE → PERMUTE → TRANSPLANT → VALIDATE

    - PERMUTE (Git Re-Basin): Solves permutation symmetry for same-architecture models.
      Reduces delta magnitude before transplant by aligning neuron orderings.
    - TRANSPLANT: Null-space constrained projection preserves boundary behavior
      while transferring knowledge.

    Stage implementations are in merge_stages/ for modularity.
    """

    def __init__(
        self,
        model_loader: "ModelLoaderPort",
        config: UnifiedMergeConfig | None = None,
        backend: "Backend | None" = None,
    ) -> None:
        """Initialize with required dependencies.

        Args:
            model_loader: Model loader port for loading weights (REQUIRED).
            config: Merge configuration (optional, defaults to default config).
            backend: Compute backend for tensor operations (defaults to MLXBackend).
                     All geometric operations run on GPU when using MLXBackend.
        """
        self._model_loader = model_loader
        self.config = config or UnifiedMergeConfig()

        # Default to configured backend (respects MC_BACKEND/MODELCYPHER_BACKEND)
        self._backend = backend or get_default_backend()

    def merge(
        self,
        source_path: str,
        target_path: str,
        output_dir: str | None = None,
        output_path: str | None = None,
        dry_run: bool = False,
        use_full_geometry: bool = True,
        knowledge_delta_mask_path: str | None = None,
        transplant_domains: list[str] | None = None,
        target_weights: dict[str, "Array"] | None = None,
        config: "UnifiedMergeConfig | None" = None,
    ) -> UnifiedMergeResult:
        """Execute the unified geometric merge pipeline."""
        return run_merge(
            model_loader=self._model_loader,
            backend=self._backend,
            default_config=self.config,
            source_path=source_path,
            target_path=target_path,
            output_dir=output_dir,
            output_path=output_path,
            dry_run=dry_run,
            use_full_geometry=use_full_geometry,
            knowledge_delta_mask_path=knowledge_delta_mask_path,
            transplant_domains=transplant_domains,
            target_weights=target_weights,
            config=config,
        )

    # Convenience wrappers to preserve internal helper access in tests/callers.
    def _stage_vocabulary(
        self,
        source_weights: dict[str, "Array"],
        target_weights: dict[str, "Array"],
        source_tokenizer: Any | None,
        target_tokenizer: Any | None,
    ) -> tuple[dict[str, "Array"], dict[str, Any], bool, Any | None]:
        return merge_stages.stage_vocabulary(
            source_weights=source_weights,
            target_weights=target_weights,
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
        alignment_map: Any | None = None,
        config_override: UnifiedMergeConfig | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict | None, dict | None]:
        # ProbeConfig was REMOVED. Probe always uses precise mode with all probes.
        # config_override is kept for API compatibility but ignored.
        return merge_stages.stage_probe(
            source_weights=source_weights,
            target_weights=target_weights,
            source_model=source_model,
            target_model=target_model,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            alignment_map=alignment_map,
            extract_layer_index_fn=merge_helpers.extract_layer_index,
        )

    def _stage_permute(
        self,
        source_weights: dict[str, Any],
        target_weights: dict[str, Any],
        intersection_map_obj: Any | None,
        layer_confidences: dict[int, float],
        enable_permutation: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # PermuteConfig was REMOVED. Permutation always runs.
        # enable_permutation is kept for API compatibility but ignored.
        return merge_stages.stage_permute(
            source_weights=source_weights,
            target_weights=target_weights,
            intersection_map_obj=intersection_map_obj,
            layer_confidences=layer_confidences,
            backend=self._backend,
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
        config: UnifiedMergeConfig,
    ) -> tuple[dict[str, "Array"], dict[str, Any]]:
        return merge_stages.stage_transplant(
            source_weights=source_weights,
            target_weights=target_weights,
            layer_indices=layer_indices,
            probe_ids=probe_ids,
            probe_domains=probe_domains,
            source_activations=source_activations,
            target_activations=target_activations,
            config=config,
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

    def _load_weights_cpu(self, model_path: str) -> tuple[dict[str, Any], str]:
        return merge_helpers.load_weights_cpu(self._model_loader, model_path)

    def _load_weights_as_arrays(self, model_path: str) -> tuple[dict[str, "Array"], str]:
        return merge_helpers.load_weights_as_arrays(self._model_loader, model_path)

    def _load_knowledge_delta_mask(self, mask_path: str) -> dict[int, float]:
        return merge_helpers.load_knowledge_delta_mask(mask_path)

    def _infer_hidden_dim(self, weights: dict[str, Any]) -> int:
        return merge_helpers.infer_hidden_dim(weights)

    def _require_vocab_phase_lock(
        self, vocab_metrics: dict[str, Any], vocab_aligned: bool
    ) -> None:
        merge_helpers.require_vocab_phase_lock(vocab_metrics, vocab_aligned)

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


def unified_merge(
    source: str,
    target: str,
    output_dir: str,
    model_loader: "ModelLoaderPort",
    config: UnifiedMergeConfig | None = None,
    dry_run: bool = False,
) -> UnifiedMergeResult:
    """
    Execute unified geometric merge.

    Convenience function that creates the merger and runs the merge.
    """
    merger = UnifiedGeometricMerger(model_loader=model_loader, config=config)

    return merger.merge(
        source_path=source,
        target_path=target,
        output_dir=output_dir,
        dry_run=dry_run,
    )
