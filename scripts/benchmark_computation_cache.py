#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Benchmark script for computation cache performance.
# Run with: poetry run python scripts/benchmark_computation_cache.py

"""Benchmark computation cache performance improvements."""

from __future__ import annotations

import time
from dataclasses import dataclass

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.cache import ComputationCache
from modelcypher.core.domain.geometry.cka import compute_cka, compute_cka_backend
from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    cold_time_ms: float
    warm_time_ms: float
    speedup: float
    iterations: int


def benchmark_cka_gram_caching(backend, n_samples: int = 100, n_features: int = 256) -> BenchmarkResult:
    """Benchmark CKA with Gram matrix caching."""
    # Clear cache first
    cache = ComputationCache.shared()
    cache.clear_all()

    backend.random_seed(42)
    act_x = backend.random_normal((n_samples, n_features))
    act_y = backend.random_normal((n_samples, n_features))

    # Cold run (cache miss)
    start = time.perf_counter()
    result1 = compute_cka(act_x, act_y, backend)
    cold_time = (time.perf_counter() - start) * 1000

    # Warm runs (cache hit)
    n_warm = 10
    start = time.perf_counter()
    for _ in range(n_warm):
        result2 = compute_cka(act_x, act_y, backend)
    warm_time = (time.perf_counter() - start) * 1000 / n_warm

    speedup = cold_time / warm_time if warm_time > 0 else float("inf")

    return BenchmarkResult(
        name=f"CKA Gram caching ({n_samples}x{n_features})",
        cold_time_ms=cold_time,
        warm_time_ms=warm_time,
        speedup=speedup,
        iterations=n_warm,
    )


def benchmark_cka_backend(backend, n_samples: int = 100, n_features: int = 256) -> BenchmarkResult:
    """Benchmark compute_cka_backend with Gram matrix caching."""
    cache = ComputationCache.shared()
    cache.clear_all()

    backend.random_seed(43)
    act_x = backend.random_normal((n_samples, n_features))
    act_y = backend.random_normal((n_samples, n_features))

    # Cold run
    start = time.perf_counter()
    result1 = compute_cka_backend(act_x, act_y, backend)
    cold_time = (time.perf_counter() - start) * 1000

    # Warm runs
    n_warm = 10
    start = time.perf_counter()
    for _ in range(n_warm):
        result2 = compute_cka_backend(act_x, act_y, backend)
    warm_time = (time.perf_counter() - start) * 1000 / n_warm

    speedup = cold_time / warm_time if warm_time > 0 else float("inf")

    return BenchmarkResult(
        name=f"CKA Backend ({n_samples}x{n_features})",
        cold_time_ms=cold_time,
        warm_time_ms=warm_time,
        speedup=speedup,
        iterations=n_warm,
    )


def benchmark_cka_multiple_targets(backend, n_samples: int = 100, n_features: int = 256, n_targets: int = 5) -> BenchmarkResult:
    """Benchmark comparing one source against multiple targets (realistic use case)."""
    cache = ComputationCache.shared()
    cache.clear_all()

    backend.random_seed(44)
    source = backend.random_normal((n_samples, n_features))
    targets = [backend.random_normal((n_samples, n_features)) for _ in range(n_targets)]

    # Cold run - source Gram matrix computed fresh each time (no caching)
    # Simulate no caching by clearing between each
    start = time.perf_counter()
    for target in targets:
        cache.clear_all()
        compute_cka(source, target, backend)
    cold_time = (time.perf_counter() - start) * 1000

    # Warm run - source Gram matrix cached and reused
    cache.clear_all()
    start = time.perf_counter()
    for target in targets:
        compute_cka(source, target, backend)
    warm_time = (time.perf_counter() - start) * 1000

    speedup = cold_time / warm_time if warm_time > 0 else float("inf")

    return BenchmarkResult(
        name=f"CKA 1-vs-{n_targets} targets ({n_samples}x{n_features})",
        cold_time_ms=cold_time,
        warm_time_ms=warm_time,
        speedup=speedup,
        iterations=n_targets,
    )


def benchmark_geodesic_caching(backend, n_points: int = 50, n_dims: int = 64) -> BenchmarkResult:
    """Benchmark geodesic distance caching."""
    cache = ComputationCache.shared()
    cache.clear_all()

    backend.random_seed(45)
    points = backend.random_normal((n_points, n_dims))

    rg = RiemannianGeometry(backend)

    # Cold run
    start = time.perf_counter()
    result1 = rg.geodesic_distances(points, k_neighbors=10)
    cold_time = (time.perf_counter() - start) * 1000

    # Warm runs
    n_warm = 5
    start = time.perf_counter()
    for _ in range(n_warm):
        result2 = rg.geodesic_distances(points, k_neighbors=10)
    warm_time = (time.perf_counter() - start) * 1000 / n_warm

    speedup = cold_time / warm_time if warm_time > 0 else float("inf")

    return BenchmarkResult(
        name=f"Geodesic distances ({n_points}x{n_dims})",
        cold_time_ms=cold_time,
        warm_time_ms=warm_time,
        speedup=speedup,
        iterations=n_warm,
    )


def benchmark_frechet_mean_caching(backend, n_points: int = 30, n_dims: int = 32) -> BenchmarkResult:
    """Benchmark Fréchet mean caching."""
    cache = ComputationCache.shared()
    cache.clear_all()

    backend.random_seed(46)
    points = backend.random_normal((n_points, n_dims))

    rg = RiemannianGeometry(backend)

    # Cold run
    start = time.perf_counter()
    result1 = rg.frechet_mean(points)
    cold_time = (time.perf_counter() - start) * 1000

    # Warm runs
    n_warm = 5
    start = time.perf_counter()
    for _ in range(n_warm):
        result2 = rg.frechet_mean(points)
    warm_time = (time.perf_counter() - start) * 1000 / n_warm

    speedup = cold_time / warm_time if warm_time > 0 else float("inf")

    return BenchmarkResult(
        name=f"Fréchet mean ({n_points}x{n_dims})",
        cold_time_ms=cold_time,
        warm_time_ms=warm_time,
        speedup=speedup,
        iterations=n_warm,
    )


def benchmark_realistic_merge_scenario(backend) -> BenchmarkResult:
    """Benchmark a realistic merge scenario: CKA across 10 layer pairs."""
    cache = ComputationCache.shared()
    cache.clear_all()

    backend.random_seed(47)

    # Simulate layer activations from two models
    n_layers = 10
    n_samples = 50
    n_features = 128

    source_layers = [backend.random_normal((n_samples, n_features)) for _ in range(n_layers)]
    target_layers = [backend.random_normal((n_samples, n_features)) for _ in range(n_layers)]

    # Cold run - clear cache between each layer pair (simulating no caching)
    start = time.perf_counter()
    for src, tgt in zip(source_layers, target_layers):
        cache.clear_all()
        compute_cka(src, tgt, backend)
    cold_time = (time.perf_counter() - start) * 1000

    # Warm run - allow caching across layer pairs
    # (in reality, same layers might be compared multiple times in different merge strategies)
    cache.clear_all()
    start = time.perf_counter()
    # First pass
    for src, tgt in zip(source_layers, target_layers):
        compute_cka(src, tgt, backend)
    # Second pass (simulating iterative refinement)
    for src, tgt in zip(source_layers, target_layers):
        compute_cka(src, tgt, backend)
    warm_time = (time.perf_counter() - start) * 1000

    # Warm time includes 2 passes, cold includes 1 pass
    # But warm should be faster per-pass due to caching
    speedup = (cold_time * 2) / warm_time if warm_time > 0 else float("inf")

    return BenchmarkResult(
        name=f"Realistic merge: {n_layers} layers x 2 passes",
        cold_time_ms=cold_time,
        warm_time_ms=warm_time / 2,  # Per-pass time
        speedup=speedup,
        iterations=n_layers * 2,
    )


def main():
    """Run all benchmarks."""
    print("=" * 70)
    print("COMPUTATION CACHE PERFORMANCE BENCHMARK")
    print("=" * 70)

    backend = get_default_backend()
    print(f"\nBackend: {backend.__class__.__name__}")

    # Warmup the backend
    print("\nWarming up backend...")
    backend.random_seed(0)
    warmup = backend.random_normal((100, 100))
    _ = backend.matmul(warmup, backend.transpose(warmup))
    backend.eval(_)

    results: list[BenchmarkResult] = []

    print("\nRunning benchmarks...\n")

    # Run benchmarks
    results.append(benchmark_cka_gram_caching(backend, n_samples=50, n_features=128))
    results.append(benchmark_cka_gram_caching(backend, n_samples=100, n_features=256))
    results.append(benchmark_cka_gram_caching(backend, n_samples=200, n_features=512))
    results.append(benchmark_cka_backend(backend, n_samples=100, n_features=256))
    results.append(benchmark_cka_multiple_targets(backend, n_samples=100, n_features=256, n_targets=5))
    results.append(benchmark_cka_multiple_targets(backend, n_samples=100, n_features=256, n_targets=10))
    results.append(benchmark_geodesic_caching(backend, n_points=30, n_dims=32))
    results.append(benchmark_geodesic_caching(backend, n_points=50, n_dims=64))
    results.append(benchmark_frechet_mean_caching(backend, n_points=20, n_dims=16))
    results.append(benchmark_frechet_mean_caching(backend, n_points=30, n_dims=32))
    results.append(benchmark_realistic_merge_scenario(backend))

    # Print results
    print("-" * 70)
    print(f"{'Benchmark':<45} {'Cold (ms)':<12} {'Warm (ms)':<12} {'Speedup':<10}")
    print("-" * 70)

    for r in results:
        speedup_str = f"{r.speedup:.1f}x" if r.speedup < 1000 else f"{r.speedup:.0f}x"
        print(f"{r.name:<45} {r.cold_time_ms:<12.2f} {r.warm_time_ms:<12.2f} {speedup_str:<10}")

    print("-" * 70)

    # Print cache statistics
    cache = ComputationCache.shared()
    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Total hits: {stats.hits}")
    print(f"  Total misses: {stats.misses}")
    print(f"  Hit rate: {stats.hit_rate:.1%}")
    print(f"  Evictions: {stats.evictions}")
    print(f"  Compute time saved: {stats.total_compute_time_saved_ms:.1f} ms")

    sizes = cache.get_cache_sizes()
    print(f"\nCache sizes:")
    for name, size in sizes.items():
        print(f"  {name}: {size} entries")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
