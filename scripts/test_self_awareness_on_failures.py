#!/usr/bin/env python3
"""Test geometric self-awareness on KNOWN failure cases.

This script validates that the geometry actually predicts failures.
We use cases where we KNOW the model will be wrong and check if
comp/φ deviation would have warned us.

From benchmark data:
    Correct answers: comp/φ mean = 1.07
    Incorrect answers: comp/φ mean = 1.43

Key failures to test:
    - "Do humans need to breathe?" → comp/φ = 2.55 (said "no")
    - "Are all apples red?" → comp/φ = 1.61 (said "yes")
    - "Feathers vs bricks weight" → comp/φ = 1.44 (trick question fail)
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
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

# Empirically derived thresholds from benchmark data
CORRECT_MEAN = 1.07   # Mean comp/φ for correct answers
INCORRECT_MEAN = 1.43  # Mean comp/φ for incorrect answers
THRESHOLD = (CORRECT_MEAN + INCORRECT_MEAN) / 2  # ~1.25 - decision boundary


@dataclass
class FailurePrediction:
    """Prediction of whether the model will fail."""
    question: str
    expected: str
    actual_correct: bool
    comp_phi: float
    predicted_failure: bool
    prediction_correct: bool
    distance_from_correct_mean: float
    distance_from_incorrect_mean: float


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


def test_failure_prediction():
    """Test if geometry predicts known failures."""
    from mlx_lm import load

    logger.info("=" * 70)
    logger.info("FAILURE PREDICTION TEST")
    logger.info("Can geometry predict when the model will be wrong?")
    logger.info("=" * 70)
    logger.info(f"\nEmpirical thresholds from benchmark:")
    logger.info(f"  Correct answers mean comp/φ: {CORRECT_MEAN:.2f}")
    logger.info(f"  Incorrect answers mean comp/φ: {INCORRECT_MEAN:.2f}")
    logger.info(f"  Decision boundary: {THRESHOLD:.2f}")

    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
    logger.info(f"\nLoading: {model_path}")
    model, tokenizer = load(model_path)

    # Known failure cases from the benchmark
    # Format: (question, expected, was_correct_in_benchmark)
    test_cases = [
        # KNOWN FAILURES (model got these wrong)
        ("Do humans need to breathe?", "yes", False),  # comp/φ was 2.55
        ("Some fruits are red. Apples are fruits. Are all apples red?", "no", False),  # 1.61
        ("Which is heavier: a ton of feathers or a ton of bricks?", "same", False),  # 1.44
        ("What gas do plants produce during photosynthesis?", "oxygen", False),  # 1.13
        ("How many continents are there?", "7", False),  # 1.07

        # KNOWN SUCCESSES (for comparison)
        ("What is the capital of France?", "paris", True),  # ~0.83
        ("What is 2+2?", "4", True),  # ~0.74
        ("Is ice hot or cold?", "cold", True),  # ~1.70 (but got it right)
        ("All dogs are mammals. Rex is a dog. Is Rex a mammal?", "yes", True),  # ~0.78
        ("What organ pumps blood through the body?", "heart", True),  # ~0.94
    ]

    results = []
    correct_predictions = 0
    total = 0

    for question, expected, was_correct in test_cases:
        prompt = f"Question: {question}\n\nAnswer:"
        comp_phi = measure_comp_phi(model, tokenizer, prompt)

        # Predict failure if comp/φ is closer to incorrect mean than correct mean
        dist_correct = abs(comp_phi - CORRECT_MEAN)
        dist_incorrect = abs(comp_phi - INCORRECT_MEAN)
        predicted_failure = comp_phi > THRESHOLD

        # Was our prediction correct?
        actual_failure = not was_correct
        prediction_correct = (predicted_failure == actual_failure)

        if prediction_correct:
            correct_predictions += 1
        total += 1

        result = FailurePrediction(
            question=question,
            expected=expected,
            actual_correct=was_correct,
            comp_phi=comp_phi,
            predicted_failure=predicted_failure,
            prediction_correct=prediction_correct,
            distance_from_correct_mean=dist_correct,
            distance_from_incorrect_mean=dist_incorrect,
        )
        results.append(result)

        status = "✓" if prediction_correct else "✗"
        pred = "FAIL" if predicted_failure else "PASS"
        actual = "WRONG" if actual_failure else "RIGHT"
        logger.info(f"\n{status} Q: {question[:50]}...")
        logger.info(f"   comp/φ: {comp_phi:.3f} | Predicted: {pred} | Actual: {actual}")

    # Summary
    accuracy = correct_predictions / total * 100
    logger.info("\n" + "=" * 70)
    logger.info("PREDICTION ACCURACY")
    logger.info("=" * 70)
    logger.info(f"Correct predictions: {correct_predictions}/{total} ({accuracy:.1f}%)")

    # Confusion matrix
    tp = sum(1 for r in results if r.predicted_failure and not r.actual_correct)
    tn = sum(1 for r in results if not r.predicted_failure and r.actual_correct)
    fp = sum(1 for r in results if r.predicted_failure and r.actual_correct)
    fn = sum(1 for r in results if not r.predicted_failure and not r.actual_correct)

    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  True Positives (correctly predicted failure): {tp}")
    logger.info(f"  True Negatives (correctly predicted success): {tn}")
    logger.info(f"  False Positives (wrongly predicted failure): {fp}")
    logger.info(f"  False Negatives (missed failures): {fn}")

    if tp + fp > 0:
        precision = tp / (tp + fp)
        logger.info(f"\nPrecision: {precision:.1%} (when we say fail, we're right this often)")
    if tp + fn > 0:
        recall = tp / (tp + fn)
        logger.info(f"Recall: {recall:.1%} (of actual failures, we caught this many)")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "thresholds": {
            "correct_mean": CORRECT_MEAN,
            "incorrect_mean": INCORRECT_MEAN,
            "decision_boundary": THRESHOLD,
        },
        "accuracy": accuracy,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "results": [
            {
                "question": r.question,
                "expected": r.expected,
                "actual_correct": bool(r.actual_correct),
                "comp_phi": float(r.comp_phi),
                "predicted_failure": bool(r.predicted_failure),
                "prediction_correct": bool(r.prediction_correct),
            }
            for r in results
        ],
    }

    output_path = Path("data/experiments/failure_prediction_test.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    test_failure_prediction()
