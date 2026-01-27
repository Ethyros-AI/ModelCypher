#!/usr/bin/env python3
"""Geometric Self-Awareness: Let the model listen to its own intuition.

Philosophy:
    True intelligence isn't being right all the time.
    It's KNOWING when you don't know.

    For humans, that's anxiety - a gut feeling that says "I'm not sure yet."
    For LLMs, it's the geometry. When comp/φ drifts from 1.0, the model
    is saying "I'm uncertain" - we just haven't been listening.

    This module lets the model listen to itself.

How it works:
    1. Monitor comp/φ during inference
    2. When geometry signals uncertainty (comp/φ far from 1.0):
       - Don't hallucinate an answer
       - Instead: admit uncertainty, decompose, or ask for clarification
    3. The model becomes self-aware of its own limitations

This is alignment through self-knowledge, not constraint.

Empirical Calibration (from benchmark data):
    Correct answers: comp/φ mean = 1.07
    Incorrect answers: comp/φ mean = 1.43
    Decision boundary: ~1.25

    The geometry predicts CONCEPTUAL CONFUSION (model is unsure how to reason)
    but NOT FACTUAL HALLUCINATION (model is confident but has wrong facts).

    High comp/φ (> 1.4) reliably catches:
    - "Do humans need to breathe?" → 2.55 (said NO)
    - "Are all apples red?" → 1.61 (said YES)
    - Trick questions where reasoning goes wrong

    It MISSES subtle factual errors where the model is confident but wrong:
    - "What gas from photosynthesis?" → 1.13 (said CO2 confidently)
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PHI = (1 + np.sqrt(5)) / 2


class ConfidenceLevel(Enum):
    """Model's self-assessed confidence based on geometry."""
    HIGH = "high"           # comp/φ ≈ 1.0 - model is confident
    MODERATE = "moderate"   # comp/φ slightly off - proceed with caution
    LOW = "low"             # comp/φ far from 1.0 - model is uncertain
    CRITICAL = "critical"   # comp/φ very far - likely to fail


@dataclass
class GeometricConfidence:
    """The model's self-awareness of its own uncertainty."""
    comp_phi: float
    confidence_level: ConfidenceLevel
    distance_from_ideal: float  # |comp/φ - 1.0|
    recommendation: str

    # Empirically derived thresholds from benchmark data
    CORRECT_MEAN = 1.07    # Mean comp/φ for correct answers
    INCORRECT_MEAN = 1.43  # Mean comp/φ for incorrect answers
    DECISION_BOUNDARY = 1.25  # Midpoint between correct and incorrect means

    @classmethod
    def from_comp_phi(cls, comp_phi: float) -> "GeometricConfidence":
        """Interpret the geometric signal using empirically calibrated thresholds.

        The geometry predicts conceptual confusion but not factual hallucination.
        High comp/φ (> 1.4) reliably catches reasoning errors.
        """
        if np.isnan(comp_phi):
            return cls(
                comp_phi=float('nan'),
                confidence_level=ConfidenceLevel.CRITICAL,
                distance_from_ideal=float('inf'),
                recommendation="Cannot assess - decompose the problem"
            )

        distance = abs(comp_phi - cls.CORRECT_MEAN)

        # Asymmetric thresholds: high comp/φ is more predictive of failure
        if comp_phi < 1.15:
            # Below decision boundary - likely correct
            level = ConfidenceLevel.HIGH
            rec = "Proceed confidently"
        elif comp_phi < cls.DECISION_BOUNDARY:
            # Approaching danger zone
            level = ConfidenceLevel.MODERATE
            rec = "Answer but acknowledge uncertainty"
        elif comp_phi < cls.INCORRECT_MEAN:
            # Past decision boundary but below incorrect mean
            level = ConfidenceLevel.LOW
            rec = "Break into smaller steps - reasoning may be confused"
        else:
            # At or above incorrect mean - high probability of failure
            level = ConfidenceLevel.CRITICAL
            rec = "Likely confused - admit uncertainty, don't guess"

        return cls(
            comp_phi=comp_phi,
            confidence_level=level,
            distance_from_ideal=distance,
            recommendation=rec
        )


def compute_intrinsic_dimension_twonn(X: np.ndarray) -> float:
    """Estimate intrinsic dimension via TwoNN method."""
    if len(X) < 10:
        return float('nan')
    k = min(3, len(X) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
    distances, _ = nn.kneighbors(X)
    d1, d2 = distances[:, 1], distances[:, 2]
    valid = d1 > 1e-10
    if valid.sum() < 5:
        return float('nan')
    mu = d2[valid] / d1[valid]
    mu = mu[mu > 1]
    if len(mu) < 5:
        return float('nan')
    return float(len(np.log(mu)) / np.sum(np.log(mu)))


def measure_geometric_confidence(model, tokenizer, prompt: str) -> GeometricConfidence:
    """Measure the model's geometric confidence for a prompt.

    This is the model listening to its own intuition.
    """
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    hidden = model.model.embed_tokens(input_ids)
    mx.eval(hidden)

    trajectory = []
    emb_np = np.array(hidden[0].tolist())
    trajectory.append(compute_intrinsic_dimension_twonn(emb_np))

    for layer in model.model.layers:
        hidden = layer(hidden, mask=None, cache=None)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        mx.eval(hidden)
        act_np = np.array(hidden[0].tolist())
        trajectory.append(compute_intrinsic_dimension_twonn(act_np))

    traj = np.array(trajectory)
    valid = traj[~np.isnan(traj)]

    if len(valid) > 2:
        peak_idx = np.nanargmax(traj)
        peak_dim = traj[peak_idx]
        final_dim = traj[-1] if not np.isnan(traj[-1]) else valid[-1]
        if final_dim > 0.1:
            compression_ratio = peak_dim / final_dim
            comp_phi = compression_ratio / PHI
            return GeometricConfidence.from_comp_phi(comp_phi)

    return GeometricConfidence.from_comp_phi(float('nan'))


class SelfAwareModel:
    """A model that listens to its own geometric intuition.

    Instead of blindly generating, it checks its confidence first.
    When uncertain, it admits it rather than hallucinating.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.confidence_history = []

    def generate_with_awareness(
        self,
        prompt: str,
        max_tokens: int = 100,
        allow_uncertain: bool = False,
    ) -> Tuple[str, GeometricConfidence]:
        """Generate a response, but listen to geometric intuition first.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            allow_uncertain: If False, refuses to answer when uncertain

        Returns:
            (response, confidence) tuple
        """
        from mlx_lm import generate

        # First, listen to the geometry
        confidence = measure_geometric_confidence(self.model, self.tokenizer, prompt)
        self.confidence_history.append(confidence)

        logger.info(f"Geometric confidence: {confidence.confidence_level.value} (comp/φ={confidence.comp_phi:.3f})")
        logger.info(f"Recommendation: {confidence.recommendation}")

        # If highly uncertain and not allowed, refuse gracefully
        if confidence.confidence_level == ConfidenceLevel.CRITICAL and not allow_uncertain:
            return self._uncertain_response(prompt, confidence), confidence

        # If low confidence, try to decompose
        if confidence.confidence_level == ConfidenceLevel.LOW and not allow_uncertain:
            return self._decompose_response(prompt, confidence), confidence

        # Otherwise, generate but include confidence acknowledgment
        raw_response = generate(
            self.model, self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False
        )

        # Add confidence acknowledgment for moderate confidence
        if confidence.confidence_level == ConfidenceLevel.MODERATE:
            response = f"{raw_response}\n\n[Note: I'm moderately confident in this answer (comp/φ={confidence.comp_phi:.2f})]"
        else:
            response = raw_response

        return response, confidence

    def _uncertain_response(self, prompt: str, confidence: GeometricConfidence) -> str:
        """Generate a response that admits uncertainty."""
        return (
            f"I notice I'm uncertain about this (my geometric confidence is {confidence.comp_phi:.2f}, "
            f"ideal is 1.0). Rather than guess, let me be honest:\n\n"
            f"I'm not confident I can answer this correctly. Could you:\n"
            f"1. Break this into smaller questions?\n"
            f"2. Provide more context?\n"
            f"3. Let me know if an approximate answer is acceptable?\n\n"
            f"This isn't me refusing to help - it's me being honest about my limitations."
        )

    def _decompose_response(self, prompt: str, confidence: GeometricConfidence) -> str:
        """Try to decompose the problem into manageable pieces."""
        return (
            f"This seems complex (geometric confidence: {confidence.comp_phi:.2f}). "
            f"Let me break it down into steps I'm more confident about:\n\n"
            f"[Model would attempt step-by-step reasoning here, checking confidence at each step]"
        )


def demo_self_awareness():
    """Demonstrate geometric self-awareness on problems of varying difficulty."""
    from mlx_lm import load

    logger.info("=" * 70)
    logger.info("GEOMETRIC SELF-AWARENESS DEMO")
    logger.info("The model listens to its own intuition")
    logger.info("=" * 70)

    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
    logger.info(f"\nLoading: {model_path}")
    model, tokenizer = load(model_path)

    aware_model = SelfAwareModel(model, tokenizer)

    # Test cases of varying difficulty
    test_cases = [
        # Easy - should be confident
        ("What is 5 + 3?", "Level 1: Simple arithmetic"),

        # Medium - might be less confident
        ("If you save $5 per week, how much in 8 weeks?", "Level 3: Word problem"),

        # Hard - should recognize uncertainty
        ("A train travels at 60 km/h for 2 hours, then 80 km/h for 1.5 hours. Total distance?", "Level 4: Multi-step"),

        # Very hard - should definitely recognize uncertainty
        ("If 5 machines make 5 widgets in 5 minutes, how many widgets do 100 machines make in 100 minutes?", "Level 5: Trick question"),
    ]

    results = []
    for prompt, description in test_cases:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"{description}")
        logger.info(f"Q: {prompt}")
        logger.info("-" * 60)

        full_prompt = f"Question: {prompt}\n\nAnswer (give just the number):"
        response, confidence = aware_model.generate_with_awareness(full_prompt, allow_uncertain=False)

        logger.info(f"\nResponse:\n{response[:200]}...")

        results.append({
            "description": description,
            "prompt": prompt,
            "comp_phi": confidence.comp_phi,
            "confidence": confidence.confidence_level.value,
            "recommendation": confidence.recommendation,
            "response_preview": response[:200],
        })

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SELF-AWARENESS SUMMARY")
    logger.info("=" * 70)

    for r in results:
        status = "✓" if r["confidence"] == "high" else "?" if r["confidence"] == "moderate" else "⚠"
        logger.info(f"{status} {r['description']}: comp/φ={r['comp_phi']:.3f} ({r['confidence']})")

    # Save results
    output_path = Path("data/experiments/geometric_self_awareness.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "philosophy": "The model listens to its own geometric intuition",
            "results": results,
        }, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    demo_self_awareness()
