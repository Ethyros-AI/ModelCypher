#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
# AGPL-3.0-or-later
"""Generalization test for the Geometric Knowledge Thesis.

Tests whether alignment learned on training probes generalizes to held-out concepts.

Mathematical foundation:
- F = pinv(X_source) @ X_target is the closed-form Procrustes solution
- With n < d (rank-deficient), CKA = 1.0 on training data is guaranteed
- With n >= d (full rank), CKA < 1.0 indicates structural mismatch
- Generalization is measured by applying F to held-out samples

This script outputs raw measurements only. No interpretation thresholds.
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class GeneralizationResult:
    """Raw measurements from the generalization test.

    All values are direct measurements - no derived interpretations.
    """

    model_a: str
    model_b: str
    d_source: int
    d_target: int
    n_train: int
    n_test: int

    # Rank analysis (derived from linear algebra)
    rank_bound: int  # min(n_train, d_source, d_target)
    is_full_rank: bool  # rank_bound >= min(d_source, d_target)

    # CKA measurements on training set
    train_cka_linear: float
    train_cka_geodesic: float

    # CKA measurements on held-out test set
    test_cka_linear: float
    test_cka_geodesic: float

    # Random baseline measurements (control)
    random_train_cka: float
    random_test_cka: float

    # Timing
    elapsed_seconds: float


def run_generalization_test(
    model_a_path: Path,
    model_b_path: Path,
    n_train: int = 0,  # 0 = auto: 2 * max(d_source, d_target) for full rank
    n_test: int = 500,
    seed: int = 0,
) -> GeneralizationResult:
    """Run the generalization test between two models.

    Args:
        model_a_path: Path to source model
        model_b_path: Path to target model
        n_train: Training samples. 0 = auto (2 * max(d) for full rank guarantee)
        n_test: Held-out test samples
        seed: Random seed for reproducibility

    Returns:
        GeneralizationResult with raw measurements only
    """
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

    backend = get_backend("mlx")
    set_default_backend(backend)

    logger.info(f"Loading model A: {model_a_path}")
    model_a, tok_a = load_model_for_training(str(model_a_path))

    logger.info(f"Loading model B: {model_b_path}")
    model_b, tok_b = load_model_for_training(str(model_b_path))

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

    # n_train = 2 * d_max ensures full rank (overdetermined system)
    # This is derived from linear algebra: rank(F) = min(n, d_source, d_target)
    if n_train == 0:
        n_train = 2 * d_max

    total_needed = n_train + n_test

    logger.info(f"d_source={d_source}, d_target={d_target}")
    logger.info(f"n_train={n_train}, n_test={n_test}")

    all_probes = UnifiedAtlasInventory.all_probes()
    probe_words = [p.name for p in all_probes]

    if len(probe_words) < total_needed:
        logger.warning(f"Available probes ({len(probe_words)}) < needed ({total_needed})")
        # Maintain train/test ratio from requested values
        train_ratio = n_train / total_needed
        n_train = int(len(probe_words) * train_ratio)
        n_test = len(probe_words) - n_train
        logger.warning(f"Adjusted: n_train={n_train}, n_test={n_test}")

    logger.info(f"Collecting activations for {len(probe_words)} probes...")
    acts_a = extract_anchor_activations(probe_words, tok_a, embed_a, layers_a, norm_a, layer_a, backend)
    acts_b = extract_anchor_activations(probe_words, tok_b, embed_b, layers_b, norm_b, layer_b, backend)

    common = [w for w in probe_words if w in acts_a and w in acts_b]
    logger.info(f"Common words: {len(common)}")

    rng = random.Random(seed)
    rng.shuffle(common)

    train_words = common[:n_train]
    test_words = common[n_train:n_train + n_test]

    logger.info(f"Train: {len(train_words)}, Test: {len(test_words)}")

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

    # Rank bound from linear algebra: rank(F) <= min(n, d_source, d_target)
    rank_bound = min(n_train_actual, d_source, d_target)
    is_full_rank = rank_bound >= d_min

    logger.info(f"rank_bound={rank_bound}, is_full_rank={is_full_rank}")

    logger.info("Computing alignment...")
    alignment = find_alignment(source_train, target_train, backend)
    aligned_train = backend.matmul(source_train, alignment.feature_transform)
    aligned_test = backend.matmul(source_test, alignment.feature_transform)
    backend.eval(aligned_train, aligned_test)

    logger.info("Computing CKA...")
    train_cka_linear = compute_linear_cka(aligned_train, target_train, backend)
    train_cka_geo = compute_cka(aligned_train, target_train, backend)
    test_cka_linear = compute_linear_cka(aligned_test, target_test, backend)
    test_cka_geo = compute_cka(aligned_test, target_test, backend)

    logger.info(f"Train CKA: linear={train_cka_linear:.6f}, geodesic={train_cka_geo.cka:.6f}")
    logger.info(f"Test CKA: linear={test_cka_linear:.6f}, geodesic={test_cka_geo.cka:.6f}")

    # Random baseline: use different seed for independence
    # Offset by n_train to ensure non-overlapping random sequences
    logger.info("Computing random baseline...")
    rng_baseline = random.Random(seed + n_train_actual)
    random_train_data = [[rng_baseline.gauss(0, 1) for _ in range(d_target)] for _ in range(n_train_actual)]
    random_test_data = [[rng_baseline.gauss(0, 1) for _ in range(d_target)] for _ in range(n_test_actual)]

    random_train = backend.array(random_train_data)
    random_test = backend.array(random_test_data)
    random_train = backend.astype(random_train, "float32")
    random_test = backend.astype(random_test, "float32")
    backend.eval(random_train, random_test)

    random_alignment = find_alignment(source_train, random_train, backend)
    aligned_train_random = backend.matmul(source_train, random_alignment.feature_transform)
    aligned_test_random = backend.matmul(source_test, random_alignment.feature_transform)
    backend.eval(aligned_train_random, aligned_test_random)

    # Use geodesic CKA for random baseline (accounts for manifold structure)
    random_train_cka_geo = compute_cka(aligned_train_random, random_train, backend)
    random_test_cka_geo = compute_cka(aligned_test_random, random_test, backend)

    random_train_cka = random_train_cka_geo.cka if random_train_cka_geo.is_valid else 0.0
    random_test_cka = random_test_cka_geo.cka if random_test_cka_geo.is_valid else 0.0

    logger.info(f"Random baseline: train={random_train_cka:.6f}, test={random_test_cka:.6f}")

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
        elapsed_seconds=elapsed,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generalization test")
    parser.add_argument("--model-a", type=str, required=True, help="Source model path")
    parser.add_argument("--model-b", type=str, required=True, help="Target model path")
    parser.add_argument("--n-train", type=int, default=0, help="Training samples (0=auto: 2*max(d))")
    parser.add_argument("--n-test", type=int, default=500, help="Test samples")
    parser.add_argument("--output", type=str, default="results/generalization_test.json", help="Output file")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    import os
    model_base = os.environ.get(
        "MODELCYPHER_MODEL_PATH",
        str(Path.home() / ".cache/huggingface/hub")
    )

    model_a_path = Path(args.model_a)
    model_b_path = Path(args.model_b)

    if not model_a_path.exists():
        model_a_path = Path(model_base) / args.model_a
    if not model_b_path.exists():
        model_b_path = Path(model_base) / args.model_b

    result = run_generalization_test(
        model_a_path,
        model_b_path,
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
    )

    output_path = Path(__file__).parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "_schema": "mc.geometry.generalization_test.v1",
        "timestamp": datetime.now().isoformat(),
        "result": asdict(result),
    }

    with output_path.open("w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Results: {output_path}")

    # Raw measurements output
    logger.info("=" * 60)
    logger.info("GENERALIZATION TEST")
    logger.info("=" * 60)
    logger.info(f"model_a: {result.model_a}")
    logger.info(f"model_b: {result.model_b}")
    logger.info(f"d_source: {result.d_source}")
    logger.info(f"d_target: {result.d_target}")
    logger.info(f"n_train: {result.n_train}")
    logger.info(f"n_test: {result.n_test}")
    logger.info(f"rank_bound: {result.rank_bound}")
    logger.info(f"is_full_rank: {result.is_full_rank}")
    logger.info(f"train_cka_linear: {result.train_cka_linear:.6f}")
    logger.info(f"train_cka_geodesic: {result.train_cka_geodesic:.6f}")
    logger.info(f"test_cka_linear: {result.test_cka_linear:.6f}")
    logger.info(f"test_cka_geodesic: {result.test_cka_geodesic:.6f}")
    logger.info(f"random_train_cka: {result.random_train_cka:.6f}")
    logger.info(f"random_test_cka: {result.random_test_cka:.6f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
