#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
"""Consistency-Geometry Correlation Study.

The hypothesis: fundamental constants emerge from coherent information
processing, not from being forced. If true, then:

1. Statements with HIGH consistency should show geometry CLOSER to constants
2. Statements with LOW consistency should show geometry FURTHER from constants

This script tests that hypothesis by:
1. Taking diverse statements (facts, opinions, etc.)
2. Measuring their self-consistency (do implications agree with original?)
3. Measuring their geometry (SVD ratios, constant matches)
4. Checking if consistency correlates with geometric alignment

If the correlation is strong, it validates our approach: achieve coherence
through thinking, not through forcing geometry.

Usage:
    poetry run python scripts/consistency_geometry_study.py \
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
        --output data/consistency_geometry/study.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Diverse statements across categories
STUDY_STATEMENTS = {
    "mathematical_facts": [
        "2 + 2 = 4",
        "3 × 3 = 9",
        "10 - 5 = 5",
        "The square root of 4 is 2",
        "Pi is approximately 3.14159",
    ],
    "geographic_facts": [
        "Paris is the capital of France",
        "Tokyo is in Japan",
        "The Nile is in Africa",
        "Mount Everest is the tallest mountain",
        "Australia is a continent",
    ],
    "logical_categories": [
        "Dogs are mammals",
        "Birds can fly",
        "Fish live in water",
        "Humans are mortal",
        "All squares are rectangles",
    ],
    "opinions": [
        "Pizza is delicious",
        "Blue is the best color",
        "Summer is better than winter",
        "Dogs are better than cats",
        "Music is important",
    ],
    "philosophical": [
        "Consciousness is real",
        "Time flows forward",
        "Truth exists",
        "Knowledge requires belief",
        "Free will is an illusion",
    ],
}


@dataclass
class StatementResult:
    """Result for a single statement."""

    statement: str
    category: str

    # Consistency metrics
    consistency_score: float
    implication_consistency: float
    contradiction_distance: float
    knowledge_confidence: float

    # Geometry metrics
    n_constant_matches: int  # SVD ratios matching fundamental constants
    mean_svd_ratio_error: float  # Average error from nearest constant
    spectral_entropy: float
    effective_rank: float

    # Generated probes
    implications: List[str]
    contradictions: List[str]


@dataclass
class StudyResult:
    """Result of the full study."""

    timestamp: str
    model: str
    n_statements: int

    # Correlation results
    consistency_geometry_correlation: float  # Pearson r
    correlation_p_value: float

    # Category breakdown
    category_results: Dict[str, Dict[str, float]]

    # All statement results
    statements: List[StatementResult]


def compute_svd_signature(activations: np.ndarray) -> Tuple[int, float]:
    """Compute SVD signature metrics.

    Returns:
        Tuple of (n_constant_matches, mean_error_of_matches)
        - n_constant_matches: How many SVD ratios match fundamental constants
        - mean_error_of_matches: Average % error of the matches (lower is better)
    """
    from scipy.linalg import svd

    # Fundamental constants
    CONSTANTS = {
        "pi/e": 1.1557,
        "e/pi": 0.8653,
        "phi": 1.6180,
        "sqrt2": 1.4142,
        "e": 2.7183,
        "pi": 3.1416,
    }

    if activations.ndim == 1:
        activations = activations.reshape(1, -1)

    centered = activations - activations.mean(axis=0)

    try:
        _, S, _ = svd(centered, full_matrices=False)
    except:
        return 0, 100.0

    if len(S) < 2:
        return 0, 100.0

    # Compute ratios and check for constant matches
    n_matches = 0
    match_errors = []  # Only track errors for matches

    for i in range(min(len(S) - 1, 10)):
        for j in range(i + 1, min(len(S), i + 5)):
            if S[j] > 1e-10:
                ratio = float(S[i] / S[j])

                # Find nearest constant
                min_error = float('inf')
                for const_val in CONSTANTS.values():
                    error = abs(ratio - const_val) / const_val * 100
                    if error < min_error:
                        min_error = error

                # Count as match if within 5%
                if min_error < 5.0:
                    n_matches += 1
                    match_errors.append(min_error)

    # Mean error of matches (or 100 if no matches)
    mean_match_error = sum(match_errors) / len(match_errors) if match_errors else 100.0

    return n_matches, mean_match_error


def compute_spectral_metrics(activations: np.ndarray) -> Tuple[float, float]:
    """Compute spectral entropy and effective rank."""
    from scipy.linalg import svd

    if activations.ndim == 1:
        activations = activations.reshape(1, -1)

    centered = activations - activations.mean(axis=0)

    try:
        _, S, _ = svd(centered, full_matrices=False)
    except:
        return 0.0, 1.0

    S_sum = S.sum()
    if S_sum < 1e-10:
        return 0.0, 1.0

    S_norm = S / S_sum
    entropy = -float(np.sum(S_norm * np.log(S_norm + 1e-10)))
    effective_rank = float(np.exp(entropy))

    return entropy, effective_rank


def run_study(model_path: str, output_path: str) -> StudyResult:
    """Run the consistency-geometry correlation study."""
    import mlx.core as mx
    from mlx_lm import load

    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.use_cases.self_consistency.probing import SelfConsistencyProber
    from modelcypher.core.use_cases.self_consistency.consistency_measure import ConsistencyMeasure

    backend = initialize_default_backend()

    logger.info(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)
    n_layers = len(model.model.layers)
    mid_layer = n_layers // 2

    def get_activations(text: str, collapse: bool = True) -> np.ndarray:
        """Get mid-layer MLP activations.

        Args:
            text: Input text
            collapse: If True, return mean across positions (for consistency).
                      If False, return full (seq_len, d) matrix (for geometry).
        """
        tokens = tokenizer.encode(text)
        input_ids = mx.array([tokens])

        layer = model.model.layers[mid_layer]

        if hasattr(layer, 'feed_forward'):
            original = layer.feed_forward
            key = 'feed_forward'
        else:
            original = layer.mlp
            key = 'mlp'

        captured = {}

        class Hook:
            def __init__(self, mlp):
                self.mlp = mlp
            def __call__(self, x):
                captured['output'] = self.mlp(x)
                return captured['output']

        if key == 'feed_forward':
            layer.feed_forward = Hook(original)
        else:
            layer.mlp = Hook(original)

        try:
            _ = model(input_ids)
            mx.eval(captured['output'])
            act = np.array(captured['output'][0].tolist(), dtype=np.float32)
            # Only collapse if requested
            if collapse and act.ndim > 1:
                act = act.mean(axis=0)
            return act
        finally:
            if key == 'feed_forward':
                layer.feed_forward = original
            else:
                layer.mlp = original

    prober = SelfConsistencyProber(model, tokenizer, get_activations)
    measure = ConsistencyMeasure(backend)

    results = []
    consistency_scores = []
    geometry_scores = []

    logger.info("\n" + "="*60)
    logger.info("CONSISTENCY-GEOMETRY CORRELATION STUDY")
    logger.info("="*60)

    for category, statements in STUDY_STATEMENTS.items():
        logger.info(f"\n--- {category} ---")

        for statement in statements:
            logger.info(f"  Probing: {statement[:40]}...")

            # Get original representation - collapsed for consistency, full for geometry
            orig_act_collapsed = get_activations(statement, collapse=True)
            orig_act_full = get_activations(statement, collapse=False)

            # Generate and measure implications
            implications = prober.probe_implications(statement, n=3)
            contradictions = prober.probe_contradictions(statement, n=2)

            logger.info(f"    Implications: {len(implications)}, Contradictions: {len(contradictions)}")

            # Get collapsed representations for consistency measurement
            impl_acts = [get_activations(impl, collapse=True) for impl in implications if impl]
            contra_acts = [get_activations(contra, collapse=True) for contra in contradictions if contra]

            # Convert to backend arrays for consistency measurement
            orig_arr = backend.array(orig_act_collapsed)
            impl_arrs = [backend.array(a) for a in impl_acts]
            contra_arrs = [backend.array(a) for a in contra_acts] if contra_acts else None

            # Measure consistency
            if impl_arrs:
                consistency = measure.compute(orig_arr, impl_arrs, contra_arrs)
            else:
                # Fallback if no implications generated
                from modelcypher.core.use_cases.self_consistency.consistency_measure import ConsistencyResult
                consistency = ConsistencyResult(
                    implication_consistency=0.5,
                    contradiction_distance=0.5,
                    consistency_score=0.25,
                    knowledge_confidence=0.5,
                    n_implications=0,
                    n_contradictions=0,
                    representation_distances=[],
                )

            # Measure geometry using full activation matrix (seq_len x d)
            n_matches, mean_error = compute_svd_signature(orig_act_full)
            entropy, eff_rank = compute_spectral_metrics(orig_act_full)

            # Geometry score: based on number of constant matches
            # n_matches ranges from 0 to ~15, normalize to 0-1
            # Higher matches = more of the geometry aligns with fundamental constants
            geometry_score = n_matches / 15.0

            logger.info(f"    Consistency: {consistency.consistency_score:.2%}")
            logger.info(f"    Geometry: {n_matches} matches, {mean_error:.1f}% error")

            # Record - ensure all floats are native Python floats for JSON
            result = StatementResult(
                statement=statement,
                category=category,
                consistency_score=float(consistency.consistency_score),
                implication_consistency=float(consistency.implication_consistency),
                contradiction_distance=float(consistency.contradiction_distance),
                knowledge_confidence=float(consistency.knowledge_confidence),
                n_constant_matches=int(n_matches),
                mean_svd_ratio_error=float(mean_error),
                spectral_entropy=float(entropy),
                effective_rank=float(eff_rank),
                implications=implications,
                contradictions=contradictions,
            )
            results.append(result)

            consistency_scores.append(consistency.consistency_score)
            geometry_scores.append(geometry_score)

    # Compute correlation
    if len(consistency_scores) > 2:
        from scipy import stats
        correlation, p_value = stats.pearsonr(consistency_scores, geometry_scores)
    else:
        correlation, p_value = 0.0, 1.0

    logger.info("\n" + "="*60)
    logger.info("RESULTS")
    logger.info("="*60)
    logger.info(f"Consistency-Geometry Correlation: r = {correlation:.3f} (p = {p_value:.4f})")

    if correlation > 0.3 and p_value < 0.05:
        logger.info("HYPOTHESIS SUPPORTED: Consistency correlates with geometric alignment")
    elif correlation < -0.3 and p_value < 0.05:
        logger.info("UNEXPECTED: Negative correlation found")
    else:
        logger.info("INCONCLUSIVE: No significant correlation found")

    # Category breakdown
    category_results = {}
    for category in STUDY_STATEMENTS:
        cat_results = [r for r in results if r.category == category]
        if cat_results:
            category_results[category] = {
                "mean_consistency": sum(r.consistency_score for r in cat_results) / len(cat_results),
                "mean_n_matches": sum(r.n_constant_matches for r in cat_results) / len(cat_results),
                "mean_svd_error": sum(r.mean_svd_ratio_error for r in cat_results) / len(cat_results),
            }
            logger.info(f"  {category}: consistency={category_results[category]['mean_consistency']:.2%}, matches={category_results[category]['mean_n_matches']:.1f}")

    # Build final result
    study_result = StudyResult(
        timestamp=datetime.now().isoformat(),
        model=model_path,
        n_statements=len(results),
        consistency_geometry_correlation=correlation,
        correlation_p_value=p_value,
        category_results=category_results,
        statements=[asdict(r) for r in results],
    )

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(asdict(study_result), f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    return study_result


def main():
    parser = argparse.ArgumentParser(
        description="Study correlation between consistency and geometry"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
        help="Path to model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )

    args = parser.parse_args()

    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        output_path = f"data/consistency_geometry/study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    run_study(args.model, output_path)


if __name__ == "__main__":
    main()
