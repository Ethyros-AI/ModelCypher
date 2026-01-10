#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
# AGPL-3.0-or-later
"""Generalization test for the Geometric Knowledge Thesis.

Tests whether alignment learned on training probes generalizes to held-out concepts.
This is the critical test for validating the thesis.

Key parameters:
- n_train >> d (full rank) for valid test
- Test on held-out words not seen during alignment fitting
- Random baseline to validate test configuration
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class GeneralizationResult:
    """Results from the generalization test."""

    model_a: str
    model_b: str
    d_source: int
    d_target: int
    n_train: int
    n_test: int
    rank_bound: int
    is_full_rank: bool

    # CKA on training set (should be ~1.0 due to Procrustes)
    train_cka_linear: float
    train_cka_geodesic: float

    # CKA on held-out test set (validates generalization)
    test_cka_linear: float
    test_cka_geodesic: float

    # Random baseline (sanity check)
    random_train_cka: float
    random_test_cka: float
    random_baseline_valid: bool

    elapsed_seconds: float


def run_generalization_test(
    model_a_path: Path,
    model_b_path: Path,
    n_train: int = 0,  # 0 = auto (2*max(d_a, d_b))
    n_test: int = 500,
    seed: int = 42,
) -> GeneralizationResult:
    """Run the generalization test between two models."""
    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.backends import get_backend
    from modelcypher.cli.commands.geometry.helpers import (
        extract_anchor_activations,
        resolve_model_backbone,
    )
    from modelcypher.core.domain._backend import set_default_backend
    from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
    from modelcypher.core.domain.geometry.cka import compute_cka, compute_linear_cka
    from modelcypher.core.domain.geometry.gram_aligner import find_alignment

    start_time = time.perf_counter()

    # Initialize backend
    backend = get_backend("mlx")
    set_default_backend(backend)

    # Load models
    logger.info(f"Loading model A: {model_a_path}")
    model_a, tok_a = load_model_for_training(str(model_a_path))

    logger.info(f"Loading model B: {model_b_path}")
    model_b, tok_b = load_model_for_training(str(model_b_path))

    # Resolve backbones
    embed_a, layers_a, norm_a = resolve_model_backbone(model_a, getattr(model_a, "model_type", None))
    embed_b, layers_b, norm_b = resolve_model_backbone(model_b, getattr(model_b, "model_type", None))

    if embed_a is None or embed_b is None:
        raise ValueError("Could not resolve model backbones")

    d_source = int(embed_a.weight.shape[-1])
    d_target = int(embed_b.weight.shape[-1])
    d_max = max(d_source, d_target)
    d_min = min(d_source, d_target)
    layer_a = len(layers_a) // 2
    layer_b = len(layers_b) // 2

    # Auto-derive n_train for full-rank alignment
    if n_train == 0:
        n_train = 2 * d_max

    total_needed = n_train + n_test

    logger.info(f"Geometry: d_source={d_source}, d_target={d_target}, d_max={d_max}")
    logger.info(f"Samples: n_train={n_train}, n_test={n_test}, total={total_needed}")

    # Get probe words from unified atlas
    all_probes = UnifiedAtlasInventory.all_probes()
    probe_words = [p.name for p in all_probes]

    if len(probe_words) < total_needed:
        logger.warning(f"Only {len(probe_words)} probes available, need {total_needed}")
        # Reduce proportionally
        ratio = len(probe_words) / total_needed
        n_train = int(n_train * ratio * 0.9)  # 90% to train
        n_test = len(probe_words) - n_train
        logger.warning(f"Reduced to n_train={n_train}, n_test={n_test}")

    # Collect activations from both models
    logger.info(f"Collecting activations for {len(probe_words)} probes...")
    acts_a = extract_anchor_activations(probe_words, tok_a, embed_a, layers_a, norm_a, layer_a, backend)
    acts_b = extract_anchor_activations(probe_words, tok_b, embed_b, layers_b, norm_b, layer_b, backend)

    # Find common words
    common = [w for w in probe_words if w in acts_a and w in acts_b]
    logger.info(f"Common words: {len(common)}")

    # Shuffle and split into train/test
    rng = random.Random(seed)
    rng.shuffle(common)

    train_words = common[:n_train]
    test_words = common[n_train:n_train + n_test]

    logger.info(f"Train words: {len(train_words)}, Test words: {len(test_words)}")

    # Stack activations
    source_train = backend.stack([acts_a[w] for w in train_words], axis=0)
    target_train = backend.stack([acts_b[w] for w in train_words], axis=0)
    source_train = backend.astype(source_train, "float32")
    target_train = backend.astype(target_train, "float32")
    backend.eval(source_train, target_train)

    source_test = backend.stack([acts_a[w] for w in test_words], axis=0)
    target_test = backend.stack([acts_b[w] for w in test_words], axis=0)
    source_test = backend.astype(source_test, "float32")
    target_test = backend.astype(target_test, "float32")
    backend.eval(source_test, target_test)

    n_train_actual = int(source_train.shape[0])
    n_test_actual = int(source_test.shape[0])
    rank_bound = min(n_train_actual, d_source, d_target)
    is_full_rank = rank_bound >= d_min

    logger.info(f"Rank bound: {rank_bound}, Full rank: {is_full_rank}")

    # Find alignment on training data
    logger.info("Computing alignment on training data...")
    alignment = find_alignment(source_train, target_train, backend)
    aligned_train = backend.matmul(source_train, alignment.feature_transform)
    aligned_test = backend.matmul(source_test, alignment.feature_transform)
    backend.eval(aligned_train, aligned_test)

    # Compute CKA on train and test
    logger.info("Computing CKA metrics...")
    train_cka_linear = compute_linear_cka(aligned_train, target_train, backend)
    train_cka_geo = compute_cka(aligned_train, target_train, backend)

    test_cka_linear = compute_linear_cka(aligned_test, target_test, backend)
    test_cka_geo = compute_cka(aligned_test, target_test, backend)

    logger.info(f"Train CKA: linear={train_cka_linear:.4f}, geodesic={train_cka_geo.cka:.4f}")
    logger.info(f"Test CKA: linear={test_cka_linear:.4f}, geodesic={test_cka_geo.cka:.4f}")

    # Random baseline
    logger.info("Computing random baseline...")
    rng2 = random.Random(seed + 42)
    random_train_data = [[rng2.gauss(0, 1) for _ in range(d_target)] for _ in range(n_train_actual)]
    random_test_data = [[rng2.gauss(0, 1) for _ in range(d_target)] for _ in range(n_test_actual)]

    random_train = backend.array(random_train_data)
    random_test = backend.array(random_test_data)
    random_train = backend.astype(random_train, "float32")
    random_test = backend.astype(random_test, "float32")
    backend.eval(random_train, random_test)

    random_alignment = find_alignment(source_train, random_train, backend)
    aligned_train_random = backend.matmul(source_train, random_alignment.feature_transform)
    aligned_test_random = backend.matmul(source_test, random_alignment.feature_transform)
    backend.eval(aligned_train_random, aligned_test_random)

    random_train_cka_linear = compute_linear_cka(aligned_train_random, random_train, backend)
    random_test_cka_linear = compute_linear_cka(aligned_test_random, random_test, backend)

    # Use geodesic CKA for random baseline (more reliable for high-d manifolds)
    random_train_cka_geo = compute_cka(aligned_train_random, random_train, backend)
    random_test_cka_geo = compute_cka(aligned_test_random, random_test, backend)

    # Random baseline is valid if geodesic test CKA is low (< 0.3)
    # This confirms the test isn't trivially achieving high CKA
    random_test_cka = random_test_cka_geo.cka if random_test_cka_geo.is_valid else random_test_cka_linear
    random_train_cka = random_train_cka_geo.cka if random_train_cka_geo.is_valid else random_train_cka_linear
    random_baseline_valid = random_test_cka < 0.3

    logger.info(f"Random baseline: train={random_train_cka:.4f}, test={random_test_cka:.4f}")
    logger.info(f"Random baseline valid: {random_baseline_valid}")

    elapsed = time.perf_counter() - start_time

    return GeneralizationResult(
        model_a=model_a_path.name,
        model_b=model_b_path.name,
        d_source=d_source,
        d_target=d_target,
        n_train=n_train_actual,
        n_test=n_test_actual,
        rank_bound=rank_bound,
        is_full_rank=is_full_rank,
        train_cka_linear=train_cka_linear,
        train_cka_geodesic=train_cka_geo.cka if train_cka_geo.is_valid else 0.0,
        test_cka_linear=test_cka_linear,
        test_cka_geodesic=test_cka_geo.cka if test_cka_geo.is_valid else 0.0,
        random_train_cka=random_train_cka,
        random_test_cka=random_test_cka,
        random_baseline_valid=random_baseline_valid,
        elapsed_seconds=elapsed,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generalization test for Geometric Knowledge Thesis")
    parser.add_argument(
        "--model-a",
        type=str,
        required=True,
        help="Path to model A (source)",
    )
    parser.add_argument(
        "--model-b",
        type=str,
        required=True,
        help="Path to model B (target)",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=0,
        help="Training samples (0=auto: 2*max(d_a, d_b))",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=500,
        help="Test samples",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/generalization_test.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    model_a_path = Path(args.model_a)
    model_b_path = Path(args.model_b)

    if not model_a_path.exists():
        # Try as model name
        model_a_path = Path("/Volumes/CodeCypher/models/mlx-community") / args.model_a
    if not model_b_path.exists():
        model_b_path = Path("/Volumes/CodeCypher/models/mlx-community") / args.model_b

    result = run_generalization_test(
        model_a_path,
        model_b_path,
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
    )

    # Save results
    output_path = Path(__file__).parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "experiment": "generalization_test",
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "Alignment learned on training probes generalizes to held-out concepts",
        "result": asdict(result),
    }

    with output_path.open("w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("GENERALIZATION TEST RESULTS")
    print("=" * 70)
    print(f"\nModels:")
    print(f"  A (source): {result.model_a}")
    print(f"  B (target): {result.model_b}")
    print(f"\nGeometry:")
    print(f"  d_source = {result.d_source}")
    print(f"  d_target = {result.d_target}")
    print(f"  n_train = {result.n_train}")
    print(f"  n_test = {result.n_test}")
    print(f"  rank(F) = {result.rank_bound}")
    print(f"  Full rank: {'YES' if result.is_full_rank else 'NO'}")
    print(f"\nCKA Results:")
    print(f"  Train CKA (linear):   {result.train_cka_linear:.4f}")
    print(f"  Train CKA (geodesic): {result.train_cka_geodesic:.4f}")
    print(f"  Test CKA (linear):    {result.test_cka_linear:.4f}")
    print(f"  Test CKA (geodesic):  {result.test_cka_geodesic:.4f}")
    print(f"\nRandom Baseline:")
    print(f"  Random train CKA: {result.random_train_cka:.4f}")
    print(f"  Random test CKA:  {result.random_test_cka:.4f}")
    print(f"  Valid control: {'YES' if result.random_baseline_valid else 'NO (test may be misconfigured)'}")

    # Interpretation
    print(f"\n" + "-" * 70)
    print("INTERPRETATION:")

    if not result.is_full_rank:
        print(f"  WARNING: rank(F)={result.rank_bound} < d_target={result.d_target}")
        print(f"  Results may not be valid - alignment is rank-deficient")

    if not result.random_baseline_valid:
        print(f"  WARNING: Random baseline also achieves high CKA")
        print(f"  Test is likely misconfigured (n too small)")
    elif result.test_cka_geodesic > 0.9:
        print(f"  THESIS VALIDATED: Test CKA > 0.9")
        print(f"  Alignment generalizes to held-out concepts!")
    elif result.test_cka_geodesic > 0.7:
        print(f"  PARTIAL: Test CKA = {result.test_cka_geodesic:.2f}")
        print(f"  Good generalization - probe set captures shared structure")
    else:
        print(f"  NEEDS INVESTIGATION: Test CKA = {result.test_cka_geodesic:.2f}")
        print(f"  Either probe coverage is limited or these models differ")

    print("=" * 70)


if __name__ == "__main__":
    main()
