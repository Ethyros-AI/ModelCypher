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


# =============================================================================
# Domain Signal Scoring (Phase 2 Parity)
# =============================================================================


@dataclass(frozen=True)
class DomainSignalScores:
    """
    Computed domain signal scores for alpha adjustment.

    Used to modulate per-layer alpha based on domain characteristics.
    Higher combined_score → prefer target (higher alpha).
    Lower combined_score → prefer source (lower alpha).
    """

    # Sparsity-based score [0, 1]
    # Higher = target is sparser than source in this domain
    sparsity_score: float

    # Smoothness-based score [0, 1]
    # Higher = source has better gradient signal-to-noise ratio
    smoothness_score: float

    # Combined weighted score [0, 1]
    combined_score: float

    # Reliability of the scores [0, 1]
    # Based on how much data was available (prompts, gradient samples)
    reliability: float


@dataclass
class DomainSignalConfig:
    """Configuration for domain signal scoring."""

    # Weight for sparsity signal in combined score
    sparsity_weight: float = 0.5

    # Weight for smoothness signal in combined score
    smoothness_weight: float = 0.5

    # Epsilon for numerical stability
    epsilon: float = 1e-6

    # Number of prompts needed to fully trust sparsity signal
    prompt_target: int = 8

    # Number of gradient samples needed to fully trust smoothness signal
    gradient_sample_target: int = 8

    # Minimum alpha when domain signals are applied
    min_alpha: float = 0.2

    # Maximum alpha when domain signals are applied
    max_alpha: float = 0.95


def compute_domain_scores(
    source_profile: DomainSignalProfile,
    target_profile: DomainSignalProfile,
    layer: int,
    config: Optional[DomainSignalConfig] = None,
) -> Optional[DomainSignalScores]:
    """
    Compute domain signal scores for a specific layer.

    Mathematical Basis:
    - smoothness_score = source_snr / (source_snr + target_snr + epsilon)
      High source SNR → trust source more → lower alpha
    - sparsity_score = (target_sparsity - source_sparsity + 1) * 0.5
      Target sparser than source → trust target for this domain → higher alpha
    - combined_score = weighted average with reliability tracking
    - reliability = based on available prompt/gradient sample counts

    Args:
        source_profile: Domain signal profile from source model
        target_profile: Domain signal profile from target model
        layer: Layer index to compute scores for
        config: Scoring configuration (uses defaults if None)

    Returns:
        DomainSignalScores if sufficient data is available, None otherwise
    """
    if config is None:
        config = DomainSignalConfig()

    source_signal = source_profile.layer_signals.get(layer)
    target_signal = target_profile.layer_signals.get(layer)

    if source_signal is None or target_signal is None:
        return None

    # Compute smoothness score from gradient SNR
    smoothness_score = 0.5  # Default to neutral
    smoothness_reliability = 0.0

    if source_signal.gradient_snr is not None and target_signal.gradient_snr is not None:
        source_snr = max(0.0, source_signal.gradient_snr)
        target_snr = max(0.0, target_signal.gradient_snr)

        # Higher source SNR means source is better for this domain → lower alpha
        # So we invert: smoothness_score high → prefer target → higher alpha
        total_snr = source_snr + target_snr + config.epsilon
        smoothness_score = target_snr / total_snr  # Higher target SNR → higher score

        # Reliability based on gradient sample counts
        source_samples = source_signal.gradient_sample_count or 0
        target_samples = target_signal.gradient_sample_count or 0
        avg_samples = (source_samples + target_samples) / 2.0
        smoothness_reliability = min(1.0, avg_samples / config.gradient_sample_target)

    # Compute sparsity score
    sparsity_score = 0.5  # Default to neutral
    sparsity_reliability = 0.0

    if source_signal.sparsity is not None and target_signal.sparsity is not None:
        source_sparsity = source_signal.sparsity
        target_sparsity = target_signal.sparsity

        # If target is sparser than source, target is under-utilizing this domain
        # → prefer source for this domain → lower alpha
        # Normalize to [0, 1] where 0.5 is neutral
        sparsity_score = (source_sparsity - target_sparsity + 1.0) * 0.5

        # Reliability based on prompt counts
        source_prompts = source_profile.prompt_count
        target_prompts = target_profile.prompt_count
        avg_prompts = (source_prompts + target_prompts) / 2.0
        sparsity_reliability = min(1.0, avg_prompts / config.prompt_target)

    # Compute combined score with reliability weighting
    weight_sum = 0.0
    weighted_score = 0.0

    if smoothness_reliability > 0:
        effective_smoothness_weight = config.smoothness_weight * smoothness_reliability
        weighted_score += smoothness_score * effective_smoothness_weight
        weight_sum += effective_smoothness_weight

    if sparsity_reliability > 0:
        effective_sparsity_weight = config.sparsity_weight * sparsity_reliability
        weighted_score += sparsity_score * effective_sparsity_weight
        weight_sum += effective_sparsity_weight

    if weight_sum <= 0:
        return None

    combined_score = weighted_score / weight_sum

    # Overall reliability is the max of available signals
    base_weight_sum = config.smoothness_weight + config.sparsity_weight
    reliability = weight_sum / base_weight_sum if base_weight_sum > 0 else 0.0

    return DomainSignalScores(
        sparsity_score=sparsity_score,
        smoothness_score=smoothness_score,
        combined_score=combined_score,
        reliability=reliability,
    )


def domain_adjusted_alpha(
    base_alpha: float,
    scores: DomainSignalScores,
    strength: float = 1.0,
    min_alpha: float = 0.2,
    max_alpha: float = 0.95,
) -> float:
    """
    Adjust alpha based on domain signal scores.

    Formula:
        desired_alpha = combined_score * (max_alpha - min_alpha) + min_alpha
        adjustment = (desired_alpha - base_alpha) * strength * reliability
        final_alpha = clamp(base_alpha + adjustment, min_alpha, max_alpha)

    Args:
        base_alpha: Starting alpha value before domain adjustment
        scores: Computed domain signal scores
        strength: How much to apply domain adjustment [0, 1]
        min_alpha: Minimum allowed alpha
        max_alpha: Maximum allowed alpha

    Returns:
        Adjusted alpha value
    """
    if strength <= 0 or scores.reliability <= 0:
        return base_alpha

    # Map combined_score to desired alpha
    desired_alpha = scores.combined_score * (max_alpha - min_alpha) + min_alpha

    # Apply adjustment scaled by strength and reliability
    adjustment = (desired_alpha - base_alpha) * strength * scores.reliability
    adjusted_alpha = base_alpha + adjustment

    return max(min_alpha, min(max_alpha, adjusted_alpha))


@dataclass
class DomainSignalDecision:
    """Complete decision record for domain signal-based alpha adjustment."""

    layer: int
    base_alpha: float
    adjusted_alpha: float
    applied: bool
    reason: str
    scores: Optional[DomainSignalScores] = None

    @staticmethod
    def skipped(layer: int, base_alpha: float, reason: str) -> "DomainSignalDecision":
        """Create a decision record for skipped adjustment."""
        return DomainSignalDecision(
            layer=layer,
            base_alpha=base_alpha,
            adjusted_alpha=base_alpha,
            applied=False,
            reason=reason,
        )

    @staticmethod
    def applied(
        layer: int,
        base_alpha: float,
        adjusted_alpha: float,
        scores: DomainSignalScores,
    ) -> "DomainSignalDecision":
        """Create a decision record for applied adjustment."""
        return DomainSignalDecision(
            layer=layer,
            base_alpha=base_alpha,
            adjusted_alpha=adjusted_alpha,
            applied=True,
            reason=f"Domain signal applied (reliability={scores.reliability:.2f})",
            scores=scores,
        )


def compute_domain_adjusted_alphas(
    source_profile: DomainSignalProfile,
    target_profile: DomainSignalProfile,
    base_alphas: dict[int, float],
    strength: float = 1.0,
    config: Optional[DomainSignalConfig] = None,
) -> tuple[dict[int, float], list[DomainSignalDecision]]:
    """
    Compute domain-adjusted alphas for all layers.

    Args:
        source_profile: Domain signal profile from source model
        target_profile: Domain signal profile from target model
        base_alphas: Base alpha values by layer
        strength: How much to apply domain adjustment [0, 1]
        config: Scoring configuration

    Returns:
        Tuple of (adjusted_alphas, decisions)
    """
    if config is None:
        config = DomainSignalConfig()

    adjusted_alphas: dict[int, float] = {}
    decisions: list[DomainSignalDecision] = []

    for layer, base_alpha in base_alphas.items():
        scores = compute_domain_scores(
            source_profile=source_profile,
            target_profile=target_profile,
            layer=layer,
            config=config,
        )

        if scores is None:
            adjusted_alphas[layer] = base_alpha
            decisions.append(
                DomainSignalDecision.skipped(
                    layer=layer,
                    base_alpha=base_alpha,
                    reason="Insufficient domain signal data",
                )
            )
            continue

        adjusted_alpha = domain_adjusted_alpha(
            base_alpha=base_alpha,
            scores=scores,
            strength=strength,
            min_alpha=config.min_alpha,
            max_alpha=config.max_alpha,
        )

        adjusted_alphas[layer] = adjusted_alpha
        decisions.append(
            DomainSignalDecision.applied(
                layer=layer,
                base_alpha=base_alpha,
                adjusted_alpha=adjusted_alpha,
                scores=scores,
            )
        )

    return adjusted_alphas, decisions
