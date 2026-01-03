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
Domain Signal Profile.

Per-layer domain signals used to guide model merging.
Combines sparse-activation evidence (where a domain is under-utilized)
with gradient smoothness evidence (where a domain has converged).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class LayerSignal:
    """Signal bundle for a single layer."""

    # Sparsity score (0 = fully occupied, 1 = fully sparse)
    sparsity: float | None = None

    # Gradient variance across prompts (higher = noisier)
    gradient_variance: float | None = None

    # Gradient signal-to-noise ratio (higher = smoother)
    gradient_snr: float | None = None

    # Mean gradient L2 norm across prompts
    mean_gradient_norm: float | None = None

    # Number of gradient samples used for this layer
    gradient_sample_count: int | None = None

    # Domain geometry coherence scores (0 = no coherence, 1 = full coherence)
    # Added for domain-aware merge waypoints
    spatial_coherence: float | None = None
    social_coherence: float | None = None
    temporal_coherence: float | None = None
    moral_coherence: float | None = None


@dataclass(frozen=True)
class DomainSignalProfile:
    """
    Per-layer domain signals used to guide model merging.

    Combines sparse-activation evidence (where a domain is under-utilized)
    with gradient smoothness evidence (where a domain has converged).
    """

    # Per-layer signals. Key is the layer index.
    layer_signals: dict[int, LayerSignal]

    # Model identifier that produced this profile
    model_id: str

    # Domain name (e.g., "code", "creative")
    domain: str

    # Baseline domain used for sparsity comparison (e.g., "baseline")
    baseline_domain: str

    # Total number of layers detected in the model
    total_layers: int

    # Number of prompts processed
    prompt_count: int

    # Maximum tokens per prompt used during probing
    max_tokens_per_prompt: int

    # When this profile was generated
    generated_at: datetime

    # Optional notes or provenance details
    notes: str | None = None

    @staticmethod
    def create(
        layer_signals: dict[int, LayerSignal],
        model_id: str,
        domain: str,
        baseline_domain: str,
        total_layers: int,
        prompt_count: int,
        max_tokens_per_prompt: int,
        notes: str | None = None,
    ) -> DomainSignalProfile:
        """Create a new DomainSignalProfile with current timestamp."""
        return DomainSignalProfile(
            layer_signals=layer_signals,
            model_id=model_id,
            domain=domain,
            baseline_domain=baseline_domain,
            total_layers=total_layers,
            prompt_count=prompt_count,
            max_tokens_per_prompt=max_tokens_per_prompt,
            generated_at=datetime.utcnow(),
            notes=notes,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "layerSignals": {
                str(k): {
                    "sparsity": v.sparsity,
                    "gradientVariance": v.gradient_variance,
                    "gradientSNR": v.gradient_snr,
                    "meanGradientNorm": v.mean_gradient_norm,
                    "gradientSampleCount": v.gradient_sample_count,
                    "spatialCoherence": v.spatial_coherence,
                    "socialCoherence": v.social_coherence,
                    "temporalCoherence": v.temporal_coherence,
                    "moralCoherence": v.moral_coherence,
                }
                for k, v in self.layer_signals.items()
            },
            "modelId": self.model_id,
            "domain": self.domain,
            "baselineDomain": self.baseline_domain,
            "totalLayers": self.total_layers,
            "promptCount": self.prompt_count,
            "maxTokensPerPrompt": self.max_tokens_per_prompt,
            "generatedAt": self.generated_at.isoformat(),
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(data: dict) -> DomainSignalProfile:
        """Create from dictionary."""
        layer_signals = {}
        for k, v in data.get("layerSignals", {}).items():
            layer_signals[int(k)] = LayerSignal(
                sparsity=v.get("sparsity"),
                gradient_variance=v.get("gradientVariance"),
                gradient_snr=v.get("gradientSNR"),
                mean_gradient_norm=v.get("meanGradientNorm"),
                gradient_sample_count=v.get("gradientSampleCount"),
                spatial_coherence=v.get("spatialCoherence"),
                social_coherence=v.get("socialCoherence"),
                temporal_coherence=v.get("temporalCoherence"),
                moral_coherence=v.get("moralCoherence"),
            )

        return DomainSignalProfile(
            layer_signals=layer_signals,
            model_id=data.get("modelId", ""),
            domain=data.get("domain", ""),
            baseline_domain=data.get("baselineDomain", ""),
            total_layers=data.get("totalLayers", 0),
            prompt_count=data.get("promptCount", 0),
            max_tokens_per_prompt=data.get("maxTokensPerPrompt", 0),
            generated_at=datetime.fromisoformat(data["generatedAt"])
            if "generatedAt" in data
            else datetime.utcnow(),
            notes=data.get("notes"),
        )
