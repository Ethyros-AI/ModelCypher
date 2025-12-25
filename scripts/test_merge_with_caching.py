#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Test script for caching performance with real model weights.
# Run with: poetry run python scripts/test_merge_with_caching.py

"""Test caching performance with real model weights."""

from __future__ import annotations

import time
from pathlib import Path

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.cache import ComputationCache
from modelcypher.core.domain.geometry.cka import compute_cka, compute_cka_backend


def load_model_weights(model_path: Path, backend):
    """Load model weights directly from safetensor files."""
    import mlx.core as mx

    # Find safetensor files
    safetensor_files = list(model_path.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    # Load all weights
    all_weights = {}
    for sf_path in safetensor_files:
        file_weights = mx.load(str(sf_path))
        all_weights.update(file_weights)

    # Extract attention weight layers for testing
    weights = {}
    for name, param in all_weights.items():
        if "self_attn" in name and "weight" in name:
            weights[name] = param
            if len(weights) >= 10:  # Just take 10 layers for testing
                break

    return weights


def test_cka_with_real_weights():
    """Test CKA caching with real model weights."""
    print("=" * 70)
    print("REAL MODEL WEIGHT CACHING TEST")
    print("=" * 70)

    backend = get_default_backend()
    cache = ComputationCache.shared()
    cache.clear_all()

    print(f"\nBackend: {backend.__class__.__name__}")

    # Check for available models
    models_dir = Path("/Volumes/CodeCypher/models/mlx-community")
    source_path = models_dir / "Qwen2.5-0.5B-Instruct-bf16"
    target_path = models_dir / "Qwen2.5-3B-Instruct-bf16"

    if not source_path.exists() or not target_path.exists():
        print("Required models not found, using synthetic data instead")
        return test_synthetic_caching()

    print(f"\nSource: {source_path.name}")
    print(f"Target: {target_path.name}")

    # Load models
    print("\nLoading weights...")
    start = time.perf_counter()
    source_weights = load_model_weights(source_path, backend)
    target_weights = load_model_weights(target_path, backend)
    load_time = time.perf_counter() - start
    print(f"Models loaded in {load_time:.2f}s")
    print(f"Source layers: {len(source_weights)}")
    print(f"Target layers: {len(target_weights)}")

    # Get matching layer names (by position)
    source_names = list(source_weights.keys())
    target_names = list(target_weights.keys())
    n_pairs = min(len(source_names), len(target_names), 5)

    print(f"\nTesting CKA on {n_pairs} layer pairs...")

    # Clear cache
    cache.clear_all()
    stats_before = cache.get_stats()

    # First pass - cold cache
    print("\n" + "-" * 70)
    print("FIRST PASS (Cold Cache)")
    print("-" * 70)

    cka_results = []
    start = time.perf_counter()

    for i in range(n_pairs):
        src_w = source_weights[source_names[i]]
        tgt_w = target_weights[target_names[i]]

        # Reshape to 2D if needed (flatten extra dims)
        src_shape = src_w.shape
        tgt_shape = tgt_w.shape

        if len(src_shape) > 2:
            src_w = backend.reshape(src_w, (src_shape[0], -1))
        if len(tgt_shape) > 2:
            tgt_w = backend.reshape(tgt_w, (tgt_shape[0], -1))

        # Ensure same number of samples (rows)
        n_samples = min(src_w.shape[0], tgt_w.shape[0], 100)
        src_w = src_w[:n_samples]
        tgt_w = tgt_w[:n_samples]

        result = compute_cka(src_w, tgt_w, backend)
        cka_results.append(result.cka)
        print(f"  Layer {i}: CKA = {result.cka:.4f}")

    first_pass_time = time.perf_counter() - start
    stats_after_first = cache.get_stats()

    print(f"\nFirst pass time: {first_pass_time:.3f}s")
    print(f"Cache stats: hits={stats_after_first.hits}, misses={stats_after_first.misses}")

    # Second pass - warm cache (same weights)
    print("\n" + "-" * 70)
    print("SECOND PASS (Warm Cache - Same Weights)")
    print("-" * 70)

    start = time.perf_counter()

    for i in range(n_pairs):
        src_w = source_weights[source_names[i]]
        tgt_w = target_weights[target_names[i]]

        src_shape = src_w.shape
        tgt_shape = tgt_w.shape

        if len(src_shape) > 2:
            src_w = backend.reshape(src_w, (src_shape[0], -1))
        if len(tgt_shape) > 2:
            tgt_w = backend.reshape(tgt_w, (tgt_shape[0], -1))

        n_samples = min(src_w.shape[0], tgt_w.shape[0], 100)
        src_w = src_w[:n_samples]
        tgt_w = tgt_w[:n_samples]

        result = compute_cka(src_w, tgt_w, backend)
        print(f"  Layer {i}: CKA = {result.cka:.4f}")

    second_pass_time = time.perf_counter() - start
    stats_after_second = cache.get_stats()

    print(f"\nSecond pass time: {second_pass_time:.3f}s")
    print(f"Speedup: {first_pass_time/second_pass_time:.2f}x")

    new_hits = stats_after_second.hits - stats_after_first.hits
    new_misses = stats_after_second.misses - stats_after_first.misses
    print(f"New cache hits: {new_hits}")
    print(f"New cache misses: {new_misses}")

    # Third pass - test compute_cka_backend
    print("\n" + "-" * 70)
    print("THIRD PASS (compute_cka_backend)")
    print("-" * 70)

    start = time.perf_counter()

    for i in range(n_pairs):
        src_w = source_weights[source_names[i]]
        tgt_w = target_weights[target_names[i]]

        src_shape = src_w.shape
        tgt_shape = tgt_w.shape

        if len(src_shape) > 2:
            src_w = backend.reshape(src_w, (src_shape[0], -1))
        if len(tgt_shape) > 2:
            tgt_w = backend.reshape(tgt_w, (tgt_shape[0], -1))

        n_samples = min(src_w.shape[0], tgt_w.shape[0], 100)
        src_w = src_w[:n_samples]
        tgt_w = tgt_w[:n_samples]

        cka = compute_cka_backend(src_w, tgt_w, backend)
        print(f"  Layer {i}: CKA = {cka:.4f}")

    third_pass_time = time.perf_counter() - start
    stats_after_third = cache.get_stats()

    print(f"\nThird pass time: {third_pass_time:.3f}s")

    # Final stats
    print("\n" + "-" * 70)
    print("FINAL CACHE STATISTICS")
    print("-" * 70)

    final_stats = cache.get_stats()
    sizes = cache.get_cache_sizes()

    print(f"\nCache sizes:")
    for name, size in sizes.items():
        print(f"  {name}: {size} entries")

    print(f"\nStatistics:")
    print(f"  Total operations: {final_stats.hits + final_stats.misses}")
    print(f"  Total hits: {final_stats.hits}")
    print(f"  Total misses: {final_stats.misses}")
    print(f"  Hit rate: {final_stats.hit_rate:.1%}")
    print(f"  Total compute time saved: {final_stats.total_compute_time_saved_ms:.1f}ms")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


def test_synthetic_caching():
    """Fallback test with synthetic data."""
    print("\nRunning synthetic data test...")

    backend = get_default_backend()
    cache = ComputationCache.shared()
    cache.clear_all()

    backend.random_seed(42)

    # Create synthetic "layer weights"
    n_layers = 10
    n_samples = 100
    n_features = 256

    source_weights = [backend.random_normal((n_samples, n_features)) for _ in range(n_layers)]
    target_weights = [backend.random_normal((n_samples, n_features)) for _ in range(n_layers)]

    print(f"Testing with {n_layers} synthetic layer pairs ({n_samples}x{n_features})")

    # First pass
    print("\nFirst pass (cold cache)...")
    start = time.perf_counter()
    for i in range(n_layers):
        compute_cka(source_weights[i], target_weights[i], backend)
    first_time = time.perf_counter() - start
    stats_first = cache.get_stats()
    print(f"  Time: {first_time:.3f}s, Misses: {stats_first.misses}")

    # Second pass
    print("\nSecond pass (warm cache)...")
    start = time.perf_counter()
    for i in range(n_layers):
        compute_cka(source_weights[i], target_weights[i], backend)
    second_time = time.perf_counter() - start
    stats_second = cache.get_stats()
    new_hits = stats_second.hits - stats_first.hits
    print(f"  Time: {second_time:.3f}s, Hits: {new_hits}")
    print(f"  Speedup: {first_time/second_time:.2f}x")

    final_stats = cache.get_stats()
    print(f"\nFinal hit rate: {final_stats.hit_rate:.1%}")
    print(f"Compute time saved: {final_stats.total_compute_time_saved_ms:.1f}ms")


if __name__ == "__main__":
    test_cka_with_real_weights()
