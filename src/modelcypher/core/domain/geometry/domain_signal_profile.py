"""
Domain Signal Profile.

Per-layer domain signals used to guide model merging.
Combines sparse-activation evidence (where a domain is under-utilized)
with gradient smoothness evidence (where a domain has converged).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class LayerSignal:
    """Signal bundle for a single layer."""

    # Sparsity score (0 = fully occupied, 1 = fully sparse)
    sparsity: Optional[float] = None

    # Gradient variance across prompts (higher = noisier)
    gradient_variance: Optional[float] = None

    # Gradient signal-to-noise ratio (higher = smoother)
    gradient_snr: Optional[float] = None

    # Mean gradient L2 norm across prompts
    mean_gradient_norm: Optional[float] = None

    # Number of gradient samples used for this layer
    gradient_sample_count: Optional[int] = None


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
    notes: Optional[str] = None

    @staticmethod
    def create(
        layer_signals: dict[int, LayerSignal],
        model_id: str,
        domain: str,
        baseline_domain: str,
        total_layers: int,
        prompt_count: int,
        max_tokens_per_prompt: int,
        notes: Optional[str] = None,
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
            )

        return DomainSignalProfile(
            layer_signals=layer_signals,
            model_id=data.get("modelId", ""),
            domain=data.get("domain", ""),
            baseline_domain=data.get("baselineDomain", ""),
            total_layers=data.get("totalLayers", 0),
            prompt_count=data.get("promptCount", 0),
            max_tokens_per_prompt=data.get("maxTokensPerPrompt", 0),
            generated_at=datetime.fromisoformat(data["generatedAt"]) if "generatedAt" in data else datetime.utcnow(),
            notes=data.get("notes"),
        )
