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

"""Entropy-based validation and profiling for model merging.

Provides raw entropy measurements per layer and merge validation metrics.
No subjective thresholds or phase-based adjustments are applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

if TYPE_CHECKING:
    from modelcypher.ports.model_loader import ModelLoaderPort


@dataclass(frozen=True)
class LayerEntropyProfile:
    """Entropy profile for a single layer.

    Computed from probe prompts to characterize layer behavior.
    Raw entropy values stored - no discrete classification.
    """

    layer_name: str
    mean_entropy: float
    entropy_variance: float


@dataclass(frozen=True)
class ModelEntropyProfile:
    """Entropy profile for an entire model.

    Aggregates per-layer profiles for merge planning.
    """

    model_name: str
    layer_profiles: dict[str, LayerEntropyProfile]
    mean_entropy: float
    entropy_variance: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def from_layer_profiles(
        cls,
        model_name: str,
        layer_profiles: dict[str, LayerEntropyProfile],
    ) -> ModelEntropyProfile:
        """Create model profile from layer profiles."""
        if not layer_profiles:
            return cls(
                model_name=model_name,
                layer_profiles={},
                mean_entropy=0.0,
                entropy_variance=0.0,
            )

        backend = get_default_backend()
        entropies = [p.mean_entropy for p in layer_profiles.values()]
        entropies_arr = backend.array(entropies)
        mean_arr = backend.mean(entropies_arr)
        var_arr = backend.var(entropies_arr)
        backend.eval(mean_arr, var_arr)
        mean_entropy = float(backend.to_scalar(mean_arr))
        variance = float(backend.to_scalar(var_arr))

        return cls(
            model_name=model_name,
            layer_profiles=layer_profiles,
            mean_entropy=mean_entropy,
            entropy_variance=variance,
        )


@dataclass(frozen=True)
class LayerMergeValidation:
    """Validation result for a single merged layer.

    All fields are raw measurements. No classifications.
    """

    layer_name: str
    source_entropy: float
    target_entropy: float
    merged_entropy: float
    entropy_delta: float
    """Absolute delta from expected entropy."""
    entropy_ratio: float
    """Delta normalized by expected entropy. The stability signal."""
    knowledge_retention_score: float
    """1.0 = full retention, 0.0 = total loss."""

    @classmethod
    def compute(
        cls,
        layer_name: str,
        source_entropy: float,
        target_entropy: float,
        merged_entropy: float,
    ) -> LayerMergeValidation:
        """Compute validation from entropy measurements.

        Args:
            layer_name: Name of the merged layer.
            source_entropy: Entropy of source model layer.
            target_entropy: Entropy of target model layer.
            merged_entropy: Entropy of merged layer.

        Returns:
            LayerMergeValidation with raw measurements.

        Note:
            Uses pure Python arithmetic for scalar operations to avoid
            GPU kernel launch overhead. Backend is only used for epsilon.
        """
        import math

        # With null space addition, target behavior is PRESERVED
        # Source knowledge is ADDED in null space directions
        # Expected: merged entropy >= target entropy (more knowledge = higher entropy)
        # Reference: target entropy (what we're preserving)
        expected_entropy = target_entropy

        # Delta from target - how much the merge changed target's entropy
        entropy_delta = abs(merged_entropy - expected_entropy)

        # Get machine epsilon from a reference array (once)
        backend = get_default_backend()
        ref_array = backend.array([source_entropy, target_entropy, merged_entropy])
        eps = division_epsilon(backend, ref_array)

        # Ratio normalized by expected - stability signal (lower = more stable)
        entropy_ratio = entropy_delta / (expected_entropy + eps)

        # Knowledge retention score: how close to expected
        # Use the source-target gap as the natural scale for what "large" means
        source_target_gap = abs(source_entropy - expected_entropy)

        # Data-derived fallback: variance across all three measurements
        # This is the natural scale when source and target are identical
        mean_val = (source_entropy + target_entropy + merged_entropy) / 3.0
        variance = (
            (source_entropy - mean_val) ** 2
            + (target_entropy - mean_val) ** 2
            + (merged_entropy - mean_val) ** 2
        ) / 3.0
        intrinsic_std = math.sqrt(variance)

        max_delta = max(source_target_gap, intrinsic_std, eps)
        retention = max(0.0, 1.0 - (entropy_delta / max_delta))

        return cls(
            layer_name=layer_name,
            source_entropy=source_entropy,
            target_entropy=target_entropy,
            merged_entropy=merged_entropy,
            entropy_delta=entropy_delta,
            entropy_ratio=entropy_ratio,
            knowledge_retention_score=retention,
        )


@dataclass(frozen=True)
class MergeEntropyValidation:
    """Overall entropy validation result for a merge operation.

    All fields are raw measurements. No classifications.
    Use mean_entropy_ratio and max_entropy_ratio to understand stability.
    Lower values = more stable merge.
    """

    source_model: str
    target_model: str
    layer_validations: dict[str, LayerMergeValidation]
    mean_entropy_ratio: float
    """Mean of per-layer entropy ratios. Lower = more stable."""
    max_entropy_ratio: float
    """Maximum per-layer entropy ratio. The worst layer."""
    mean_knowledge_retention: float
    """Mean knowledge retention across layers."""
    entropy_ratio_std: float
    """Standard deviation of entropy ratios. Uniformity of stability."""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def from_layer_validations(
        cls,
        source_model: str,
        target_model: str,
        layer_validations: dict[str, LayerMergeValidation],
    ) -> MergeEntropyValidation:
        """Create validation result from per-layer validations.

        Args:
            source_model: Name of source model.
            target_model: Name of target model.
            layer_validations: Per-layer validation results.

        Returns:
            MergeEntropyValidation with raw aggregate measurements.
        """
        if not layer_validations:
            return cls(
                source_model=source_model,
                target_model=target_model,
                layer_validations={},
                mean_entropy_ratio=0.0,
                max_entropy_ratio=0.0,
                mean_knowledge_retention=1.0,
                entropy_ratio_std=0.0,
            )

        # Collect raw measurements
        entropy_ratios = [v.entropy_ratio for v in layer_validations.values()]
        retention_scores = [v.knowledge_retention_score for v in layer_validations.values()]

        backend = get_default_backend()
        ratios_arr = backend.array(entropy_ratios)
        retention_arr = backend.array(retention_scores)
        mean_ratio_arr = backend.mean(ratios_arr)
        max_ratio_arr = backend.max(ratios_arr)
        mean_retention_arr = backend.mean(retention_arr)
        std_ratio_arr = backend.std(ratios_arr)
        backend.eval(mean_ratio_arr, max_ratio_arr, mean_retention_arr, std_ratio_arr)

        mean_ratio = float(backend.to_scalar(mean_ratio_arr))
        max_ratio = float(backend.to_scalar(max_ratio_arr))
        mean_retention = float(backend.to_scalar(mean_retention_arr))
        std_ratio = float(backend.to_scalar(std_ratio_arr))

        return cls(
            source_model=source_model,
            target_model=target_model,
            layer_validations=layer_validations,
            mean_entropy_ratio=mean_ratio,
            max_entropy_ratio=max_ratio,
            mean_knowledge_retention=mean_retention,
            entropy_ratio_std=std_ratio,
        )

    def layers_by_entropy_ratio(self, descending: bool = True) -> list[str]:
        """Get layer names sorted by entropy ratio.

        Args:
            descending: If True, lowest ratios first. If False, highest first.

        Returns:
            List of layer names sorted by entropy_ratio.
        """
        return sorted(
            self.layer_validations.keys(),
            key=lambda n: self.layer_validations[n].entropy_ratio,
            reverse=descending,
        )

    @property
    def summary(self) -> str:
        """Human-readable summary of validation."""
        return (
            f"Mean entropy ratio: {self.mean_entropy_ratio:.4f}\n"
            f"Max entropy ratio: {self.max_entropy_ratio:.4f}\n"
            f"Knowledge retention: {self.mean_knowledge_retention:.1%}\n"
            f"Layers: {len(self.layer_validations)}"
        )


class EntropyMergeValidator:
    """Validates model merges using entropy analysis.

    This class computes raw entropy measurements for profiling and merge
    validation without thresholds or phase-based adjustments.
    """

    def create_layer_profile(
        self,
        layer_name: str,
        entropy_values: list[float],
    ) -> LayerEntropyProfile:
        """Create entropy profile for a layer from measurements.

        Args:
            layer_name: Name of the layer.
            entropy_values: List of entropy measurements (from probe prompts).

        Returns:
            LayerEntropyProfile with statistics and phase classification.
        """
        if not entropy_values:
            return LayerEntropyProfile(
                layer_name=layer_name,
                mean_entropy=0.0,
                entropy_variance=0.0,
            )

        backend = get_default_backend()
        entropy_arr = backend.array(entropy_values)
        mean_arr = backend.mean(entropy_arr)
        var_arr = backend.var(entropy_arr)
        backend.eval(mean_arr, var_arr)
        mean_entropy = float(backend.to_scalar(mean_arr))
        variance = float(backend.to_scalar(var_arr))

        return LayerEntropyProfile(
            layer_name=layer_name,
            mean_entropy=mean_entropy,
            entropy_variance=variance,
        )

    def create_profile(
        self,
        model_path: str,
        model_loader: "ModelLoaderPort",
    ) -> ModelEntropyProfile:
        """Create a real model entropy profile using Entropy-Lens approach.

        Projects hidden states at each layer through the unembedding matrix
        to compute per-layer Shannon entropy. This gives REAL layer-wise
        entropy measurements, not fabricated data.

        Algorithm (per layer L):
            1. Capture hidden state h_L during forward pass
            2. Project through unembedding matrix: logits_L = h_L @ W_unembed.T
            3. Apply softmax: p = softmax(logits_L)
            4. Compute Shannon entropy: H_L = -sum(p * log(p))

        Args:
            model_path: Path to the model directory.
            model_loader: Model loader port implementation (injected dependency).

        Returns:
            ModelEntropyProfile with measured entropy values.

        References:
            Ali et al. (2025) "Entropy-Lens: The Information Signature of
            Transformer Computations" arXiv:2502.16570
        """
        from pathlib import Path

        from modelcypher.core.domain.entropy.layer_entropy_projector import (
            LayerEntropyProjector,
        )

        model_dir = Path(model_path)
        model_name = model_dir.name

        # Load model
        model, tokenizer = model_loader.load_model_for_training(model_path)

        # Create Entropy-Lens projector
        backend = get_default_backend()
        projector = LayerEntropyProjector(backend=backend)

        # Set up unembedding matrix for projection
        # No fallback - real measurement is required
        from modelcypher.core.domain.merging.exceptions import EntropyMeasurementError

        try:
            projector.set_unembedding_matrix(model)
        except ValueError as e:
            raise EntropyMeasurementError(
                stage="ENTROPY_PROFILING",
                weight_key=None,
                message=f"Failed to extract unembedding matrix from {model_name}",
                context={
                    "model_path": model_path,
                    "error": str(e),
                    "fix": (
                        "1. Verify model has lm_head or embed_out attribute\n"
                        "2. Check model architecture compatibility\n"
                        "3. Ensure model is fully loaded (not lazy)"
                    ),
                },
            ) from e

        # Probe prompts for entropy measurement
        probe_prompts = [
            "What is the capital of France?",
            "Explain photosynthesis briefly.",
            "Calculate 15 * 23.",
            "The quick brown fox jumps over the lazy dog.",
            "In a world where technology advances rapidly,",
        ]

        # Profile model using Entropy-Lens
        profile_result = projector.profile_model(
            model=model,
            tokenizer=tokenizer,
            prompts=probe_prompts,
            target_layers=None,
        )

        # Convert to ModelEntropyProfile format
        layer_profiles = {}
        for layer_idx, result in profile_result.layer_results.items():
            layer_profiles[result.layer_name] = LayerEntropyProfile(
                layer_name=result.layer_name,
                mean_entropy=result.mean_entropy,
                entropy_variance=result.entropy_variance,
            )

        return ModelEntropyProfile.from_layer_profiles(model_name, layer_profiles)

    def validate_merge(
        self,
        source_entropies: dict[str, float],
        target_entropies: dict[str, float],
        merged_entropies: dict[str, float],
        source_model: str = "source",
        target_model: str = "target",
    ) -> MergeEntropyValidation:
        """Validate a completed merge by comparing entropy characteristics.

        Args:
            source_entropies: Per-layer entropy from source model.
            target_entropies: Per-layer entropy from target model.
            merged_entropies: Per-layer entropy from merged model.
            source_model: Name of source model.
            target_model: Name of target model.

        Returns:
            MergeEntropyValidation with stability assessment.
        """
        layer_validations = {}

        # Validate common layers
        common_layers = (
            set(source_entropies.keys())
            & set(target_entropies.keys())
            & set(merged_entropies.keys())
        )

        for layer_name in common_layers:
            validation = LayerMergeValidation.compute(
                layer_name=layer_name,
                source_entropy=source_entropies[layer_name],
                target_entropy=target_entropies[layer_name],
                merged_entropy=merged_entropies[layer_name],
            )
            layer_validations[layer_name] = validation

        return MergeEntropyValidation.from_layer_validations(
            source_model=source_model,
            target_model=target_model,
            layer_validations=layer_validations,
        )
