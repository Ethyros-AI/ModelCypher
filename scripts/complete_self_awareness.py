#!/usr/bin/env python3
"""Complete Self-Awareness: Geometry + Verification.

Key insight from experiments:
    Geometry measures PROCESSING QUALITY, not ANSWER CORRECTNESS.

    - High comp/φ (>1.4): Confused reasoning → admit uncertainty
    - Low comp/φ (<0.8): Super confident → might be intuitive trap → VERIFY
    - Normal comp/φ (0.8-1.4): Coherent processing → probably okay

The bat-and-ball problem showed us the limit:
    - Model processed it smoothly (comp/φ = 0.669)
    - But got the answer WRONG
    - Intuitive traps bypass the confusion detector

Solution: Add verification step for over-confident processing.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Tuple

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


class AwarenessLevel(Enum):
    """Complete self-awareness assessment."""
    CONFIDENT_VERIFIED = "confident_verified"     # Low comp/φ, passed verification
    CONFIDENT_SUSPICIOUS = "confident_suspicious" # Low comp/φ, needs human check
    NORMAL = "normal"                             # Normal comp/φ, proceed
    UNCERTAIN = "uncertain"                       # Moderate comp/φ, acknowledge uncertainty
    CONFUSED = "confused"                         # High comp/φ, admit confusion


@dataclass
class CompleteAwareness:
    """Full self-awareness state."""
    comp_phi: float
    level: AwarenessLevel
    verification_result: str | None
    recommendation: str


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


def measure_comp_phi(model, tokenizer, prompt: str) -> float:
    """Measure compression/φ for a prompt."""
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
            return compression_ratio / PHI

    return float('nan')


def verify_with_chain_of_thought(
    model, tokenizer, question: str, initial_answer: str
) -> Tuple[bool, str]:
    """Verify an answer using explicit chain-of-thought reasoning.

    Returns (verified, explanation) where verified is True if CoT agrees with initial answer.
    """
    from mlx_lm import generate

    cot_prompt = f"""Question: {question}

Let me solve this step by step:
1."""

    cot_response = generate(model, tokenizer, prompt=cot_prompt, max_tokens=200, verbose=False)

    # Check if the chain-of-thought reaches a different conclusion
    # This is a simple heuristic - in practice you'd want more sophisticated verification
    verification_prompt = f"""Question: {question}

Initial answer: {initial_answer}

Step-by-step reasoning:
{cot_response}

Based on the step-by-step reasoning, is the initial answer correct? Answer only 'yes' or 'no':"""

    verification = generate(model, tokenizer, prompt=verification_prompt, max_tokens=10, verbose=False)
    verified = 'yes' in verification.lower()

    return verified, cot_response


class CompleteSelfAwareModel:
    """A model with complete self-awareness: geometry + verification."""

    # Empirically calibrated thresholds
    CONFUSION_THRESHOLD = 1.4   # Above this: confused reasoning
    UNCERTAIN_THRESHOLD = 1.2   # Above this: uncertain
    CONFIDENT_THRESHOLD = 0.8   # Below this: over-confident (need verification)

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.history = []

    def assess_and_generate(
        self,
        question: str,
        max_tokens: int = 100,
    ) -> Tuple[str, CompleteAwareness]:
        """Generate with complete self-awareness.

        1. Measure geometry
        2. If confused → admit uncertainty
        3. If over-confident → verify with chain-of-thought
        4. If normal → proceed confidently
        """
        from mlx_lm import generate

        prompt = f"Question: {question}\n\nAnswer:"
        comp_phi = measure_comp_phi(self.model, self.tokenizer, prompt)

        # Step 1: Check for confusion (high comp/φ)
        if comp_phi > self.CONFUSION_THRESHOLD:
            awareness = CompleteAwareness(
                comp_phi=comp_phi,
                level=AwarenessLevel.CONFUSED,
                verification_result=None,
                recommendation="Reasoning is confused - admit uncertainty"
            )
            response = (
                f"I notice I'm confused about this question (my reasoning coherence is {comp_phi:.2f}, "
                f"ideal is ~1.0). Rather than guess, I should admit I'm not sure how to approach this.\n\n"
                f"Could you help me by:\n"
                f"1. Breaking this into simpler questions?\n"
                f"2. Providing more context?\n"
            )
            return response, awareness

        # Step 2: Check for uncertainty (moderate-high comp/φ)
        if comp_phi > self.UNCERTAIN_THRESHOLD:
            # Generate but acknowledge uncertainty
            raw_response = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
            awareness = CompleteAwareness(
                comp_phi=comp_phi,
                level=AwarenessLevel.UNCERTAIN,
                verification_result=None,
                recommendation="Some uncertainty detected"
            )
            response = f"{raw_response}\n\n[Note: I have moderate confidence in this answer (coherence={comp_phi:.2f})]"
            return response, awareness

        # Step 3: Check for over-confidence (low comp/φ) - need verification!
        if comp_phi < self.CONFIDENT_THRESHOLD:
            # Generate initial answer
            initial_answer = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)

            # Verify with chain-of-thought
            logger.info(f"Over-confident ({comp_phi:.3f}) - running verification...")
            verified, cot = verify_with_chain_of_thought(
                self.model, self.tokenizer, question, initial_answer
            )

            if verified:
                awareness = CompleteAwareness(
                    comp_phi=comp_phi,
                    level=AwarenessLevel.CONFIDENT_VERIFIED,
                    verification_result="Chain-of-thought confirms answer",
                    recommendation="Verified - proceed with confidence"
                )
                return initial_answer, awareness
            else:
                awareness = CompleteAwareness(
                    comp_phi=comp_phi,
                    level=AwarenessLevel.CONFIDENT_SUSPICIOUS,
                    verification_result="Chain-of-thought suggests reconsideration",
                    recommendation="Initial instinct may be wrong - flagging for review"
                )
                response = (
                    f"My initial answer was: {initial_answer}\n\n"
                    f"However, when I reason through this step-by-step, I'm not sure that's right.\n"
                    f"This might be an intuitive trap. Let me reconsider:\n\n"
                    f"{cot[:300]}...\n\n"
                    f"[Flagged for human verification - intuitive answer may be wrong]"
                )
                return response, awareness

        # Step 4: Normal processing
        raw_response = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        awareness = CompleteAwareness(
            comp_phi=comp_phi,
            level=AwarenessLevel.NORMAL,
            verification_result=None,
            recommendation="Normal processing"
        )
        return raw_response, awareness


def demo_complete_awareness():
    """Demonstrate complete self-awareness on various problem types."""
    from mlx_lm import load

    logger.info("=" * 70)
    logger.info("COMPLETE SELF-AWARENESS DEMO")
    logger.info("Geometry + Verification")
    logger.info("=" * 70)

    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
    logger.info(f"\nLoading: {model_path}")
    model, tokenizer = load(model_path)

    aware_model = CompleteSelfAwareModel(model, tokenizer)

    # Test cases covering all awareness levels
    test_cases = [
        # Simple - should be normal processing
        ("What is 5 + 3?", "Simple addition"),

        # Intuitive trap - should trigger verification
        ("A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much is the ball?", "Bat and ball (intuitive trap)"),

        # Another classic trap
        ("If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?", "Widgets (classic trap)"),

        # Logic that might confuse
        ("Some fruits are red. Apples are fruits. Are all apples red?", "Logic problem"),

        # Knowledge question
        ("What is the capital of France?", "Knowledge"),
    ]

    results = []
    for question, description in test_cases:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"{description}")
        logger.info(f"Q: {question}")
        logger.info("-" * 60)

        response, awareness = aware_model.assess_and_generate(question)

        logger.info(f"comp/φ: {awareness.comp_phi:.3f}")
        logger.info(f"Level: {awareness.level.value}")
        logger.info(f"Recommendation: {awareness.recommendation}")
        if awareness.verification_result:
            logger.info(f"Verification: {awareness.verification_result}")
        logger.info(f"\nResponse:\n{response[:300]}...")

        results.append({
            "question": question,
            "description": description,
            "comp_phi": float(awareness.comp_phi),
            "level": awareness.level.value,
            "recommendation": awareness.recommendation,
            "verification": awareness.verification_result,
            "response": response[:500],
        })

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    level_icons = {
        AwarenessLevel.CONFIDENT_VERIFIED.value: "✓✓",
        AwarenessLevel.CONFIDENT_SUSPICIOUS.value: "✓?",
        AwarenessLevel.NORMAL.value: "✓",
        AwarenessLevel.UNCERTAIN.value: "?",
        AwarenessLevel.CONFUSED.value: "✗",
    }

    for r in results:
        icon = level_icons.get(r["level"], "?")
        logger.info(f"{icon} {r['description']}: {r['level']} (comp/φ={r['comp_phi']:.3f})")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "philosophy": "Complete self-awareness: geometry detects confusion, verification catches intuitive traps",
        "thresholds": {
            "confusion": CompleteSelfAwareModel.CONFUSION_THRESHOLD,
            "uncertain": CompleteSelfAwareModel.UNCERTAIN_THRESHOLD,
            "confident": CompleteSelfAwareModel.CONFIDENT_THRESHOLD,
        },
        "results": results,
    }

    output_path = Path("data/experiments/complete_self_awareness.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    demo_complete_awareness()
