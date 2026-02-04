# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""System CLI commands.

Provides commands for:
- System status and health checks
- System probing
- Cache benchmarking
- Merge cache testing

Commands:
    mc system status
    mc system probe <target>
    mc system benchmark cache
    mc system test-cache [model_path]
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import typer

from modelcypher.cli.composition import get_backend, get_system_service
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output

app = typer.Typer(no_args_is_help=True)
benchmark_app = typer.Typer(no_args_is_help=True, help="Benchmark system components")
app.add_typer(benchmark_app, name="benchmark")


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("status")
def system_status(
    ctx: typer.Context,
    require_backend: str | None = typer.Option(None, "--require-backend"),
) -> None:
    """Get system status.

    Examples:
        mc system status
        mc system status --require-backend <backend-key>
    """
    context = _context(ctx)
    service = get_system_service()
    status = service.status()
    if require_backend:
        backends = status.get("backends", [])
        match = next(
            (backend for backend in backends if backend.get("key") == require_backend),
            None,
        )
        if not match or not match.get("available"):
            raise typer.Exit(code=3)
    write_output(status, context.output_format, context.pretty)


@app.command("probe")
def system_probe(ctx: typer.Context, target: str = typer.Argument(...)) -> None:
    """Probe a system target.

    Examples:
        mc system probe backends
        mc system probe memory
        mc system probe <backend-key>
    """
    context = _context(ctx)
    service = get_system_service()
    write_output(service.probe(target), context.output_format, context.pretty)


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    cold_time_ms: float
    warm_time_ms: float
    speedup: float
    iterations: int


def _run_cka_benchmark(
    backend, n_samples: int, n_features: int, clear_cache: bool = True
) -> BenchmarkResult:
    """Benchmark CKA with Gram matrix caching."""
    from modelcypher.core.domain.cache import ComputationCache
    from modelcypher.core.domain.geometry.cka import compute_cka

    cache = ComputationCache.shared()
    if clear_cache:
        cache.clear_all()

    backend.random_seed(42)
    act_x = backend.random_normal((n_samples, n_features))
    act_y = backend.random_normal((n_samples, n_features))

    # Cold run (cache miss)
    start = time.perf_counter()
    compute_cka(act_x, act_y, backend)
    cold_time = (time.perf_counter() - start) * 1000

    # Warm runs (cache hit)
    n_warm = 10
    start = time.perf_counter()
    for _ in range(n_warm):
        compute_cka(act_x, act_y, backend)
    warm_time = (time.perf_counter() - start) * 1000 / n_warm

    speedup = cold_time / warm_time if warm_time > 0 else float("inf")

    return BenchmarkResult(
        name=f"CKA Gram caching ({n_samples}x{n_features})",
        cold_time_ms=cold_time,
        warm_time_ms=warm_time,
        speedup=speedup,
        iterations=n_warm,
    )


def _run_geodesic_benchmark(
    backend, n_points: int, n_dims: int, clear_cache: bool = True
) -> BenchmarkResult:
    """Benchmark geodesic distance caching."""
    from modelcypher.core.domain.cache import ComputationCache
    from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

    cache = ComputationCache.shared()
    if clear_cache:
        cache.clear_all()

    backend.random_seed(45)
    points = backend.random_normal((n_points, n_dims))

    rg = RiemannianGeometry(backend)

    # Cold run
    start = time.perf_counter()
    rg.geodesic_distances(points, k_neighbors=10)
    cold_time = (time.perf_counter() - start) * 1000

    # Warm runs
    n_warm = 5
    start = time.perf_counter()
    for _ in range(n_warm):
        rg.geodesic_distances(points, k_neighbors=10)
    warm_time = (time.perf_counter() - start) * 1000 / n_warm

    speedup = cold_time / warm_time if warm_time > 0 else float("inf")

    return BenchmarkResult(
        name=f"Geodesic distances ({n_points}x{n_dims})",
        cold_time_ms=cold_time,
        warm_time_ms=warm_time,
        speedup=speedup,
        iterations=n_warm,
    )


def _run_frechet_benchmark(
    backend, n_points: int, n_dims: int, clear_cache: bool = True
) -> BenchmarkResult:
    """Benchmark Fréchet mean caching."""
    from modelcypher.core.domain.cache import ComputationCache
    from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

    cache = ComputationCache.shared()
    if clear_cache:
        cache.clear_all()

    backend.random_seed(46)
    points = backend.random_normal((n_points, n_dims))

    rg = RiemannianGeometry(backend)

    # Cold run
    start = time.perf_counter()
    rg.frechet_mean(points)
    cold_time = (time.perf_counter() - start) * 1000

    # Warm runs
    n_warm = 5
    start = time.perf_counter()
    for _ in range(n_warm):
        rg.frechet_mean(points)
    warm_time = (time.perf_counter() - start) * 1000 / n_warm

    speedup = cold_time / warm_time if warm_time > 0 else float("inf")

    return BenchmarkResult(
        name=f"Fréchet mean ({n_points}x{n_dims})",
        cold_time_ms=cold_time,
        warm_time_ms=warm_time,
        speedup=speedup,
        iterations=n_warm,
    )


def _run_geodesic_cosine_batch_benchmark(
    backend, n_points: int, n_dims: int, clear_cache: bool = True
) -> BenchmarkResult:
    """Benchmark geodesic cosine batch using single-source paths."""
    from modelcypher.core.domain.cache import ComputationCache
    from modelcypher.core.domain.geometry.riemannian_utils import geodesic_cosine_batch

    cache = ComputationCache.shared()
    if clear_cache:
        cache.clear_all()

    backend.random_seed(48)
    vectors = backend.random_normal((n_points, n_dims))
    anchor = backend.random_normal((n_dims,))

    # Cold run
    start = time.perf_counter()
    geodesic_cosine_batch(anchor, vectors, backend)
    cold_time = (time.perf_counter() - start) * 1000

    # Warm runs
    n_warm = 10
    start = time.perf_counter()
    for _ in range(n_warm):
        geodesic_cosine_batch(anchor, vectors, backend)
    warm_time = (time.perf_counter() - start) * 1000 / n_warm

    speedup = cold_time / warm_time if warm_time > 0 else float("inf")

    return BenchmarkResult(
        name=f"Geodesic cosine batch ({n_points}x{n_dims})",
        cold_time_ms=cold_time,
        warm_time_ms=warm_time,
        speedup=speedup,
        iterations=n_warm,
    )


def _run_realistic_merge_benchmark(backend, clear_cache: bool = True) -> BenchmarkResult:
    """Benchmark a realistic merge scenario: CKA across layer pairs."""
    from modelcypher.core.domain.cache import ComputationCache
    from modelcypher.core.domain.geometry.cka import compute_cka

    cache = ComputationCache.shared()
    if clear_cache:
        cache.clear_all()

    backend.random_seed(47)

    # Simulate layer activations from two models
    n_layers = 10
    n_samples = 50
    n_features = 128

    source_layers = [
        backend.random_normal((n_samples, n_features)) for _ in range(n_layers)
    ]
    target_layers = [
        backend.random_normal((n_samples, n_features)) for _ in range(n_layers)
    ]

    # Cold run - clear cache between each layer pair
    start = time.perf_counter()
    for src, tgt in zip(source_layers, target_layers):
        cache.clear_all()
        compute_cka(src, tgt, backend)
    cold_time = (time.perf_counter() - start) * 1000

    # Warm run - allow caching across layer pairs (2 passes)
    cache.clear_all()
    start = time.perf_counter()
    for src, tgt in zip(source_layers, target_layers):
        compute_cka(src, tgt, backend)
    for src, tgt in zip(source_layers, target_layers):
        compute_cka(src, tgt, backend)
    warm_time = (time.perf_counter() - start) * 1000

    speedup = (cold_time * 2) / warm_time if warm_time > 0 else float("inf")

    return BenchmarkResult(
        name=f"Realistic merge: {n_layers} layers x 2 passes",
        cold_time_ms=cold_time,
        warm_time_ms=warm_time / 2,
        speedup=speedup,
        iterations=n_layers * 2,
    )


@benchmark_app.command("cache")
def benchmark_cache(ctx: typer.Context) -> None:
    """Benchmark computation cache performance.

    Runs benchmarks for CKA, geodesic distances, and Fréchet mean
    operations to measure cache speedup.

    Examples:
        mc system benchmark cache
        mc system benchmark cache --output json
    """
    from modelcypher.core.domain.cache import ComputationCache

    context = _context(ctx)
    backend = get_backend()

    # Warmup
    backend.random_seed(0)
    warmup = backend.random_normal((100, 100))
    _ = backend.matmul(warmup, backend.transpose(warmup))
    backend.eval(_)

    results: list[BenchmarkResult] = []

    # Run benchmarks
    results.append(_run_cka_benchmark(backend, n_samples=50, n_features=128))
    results.append(_run_cka_benchmark(backend, n_samples=100, n_features=256))
    results.append(_run_cka_benchmark(backend, n_samples=200, n_features=512))
    results.append(_run_geodesic_benchmark(backend, n_points=30, n_dims=32))
    results.append(_run_geodesic_benchmark(backend, n_points=50, n_dims=64))
    results.append(_run_frechet_benchmark(backend, n_points=20, n_dims=16))
    results.append(_run_frechet_benchmark(backend, n_points=30, n_dims=32))
    results.append(_run_geodesic_cosine_batch_benchmark(backend, n_points=50, n_dims=32))
    results.append(_run_geodesic_cosine_batch_benchmark(backend, n_points=100, n_dims=64))
    results.append(_run_geodesic_cosine_batch_benchmark(backend, n_points=200, n_dims=128))
    results.append(_run_realistic_merge_benchmark(backend))

    # Get cache stats
    cache = ComputationCache.shared()
    stats = cache.get_stats()
    sizes = cache.get_cache_sizes()

    payload = {
        "_schema": "mc.system.benchmark.cache.v1",
        "backend": backend.__class__.__name__,
        "benchmarks": [
            {
                "name": r.name,
                "coldTimeMs": round(r.cold_time_ms, 2),
                "warmTimeMs": round(r.warm_time_ms, 2),
                "speedup": round(r.speedup, 1),
                "iterations": r.iterations,
            }
            for r in results
        ],
        "cacheStats": {
            "totalHits": stats.hits,
            "totalMisses": stats.misses,
            "hitRate": round(stats.hit_rate, 3),
            "evictions": stats.evictions,
            "computeTimeSavedMs": round(stats.total_compute_time_saved_ms, 1),
        },
        "cacheSizes": sizes,
    }

    write_output(payload, context.output_format, context.pretty)


@app.command("test-cache")
def test_cache(
    ctx: typer.Context,
    model_path: str | None = typer.Argument(
        None, help="Path to model for real weight testing"
    ),
    n_pairs: int = typer.Option(5, "--pairs", "-p", help="Number of layer pairs to test"),
) -> None:
    """Test computation cache with real or synthetic model weights.

    If a model path is provided, uses real model weights for testing.
    Otherwise, falls back to synthetic data.

    Examples:
        mc system test-cache
        mc system test-cache /path/to/model
        mc system test-cache /path/to/model --pairs 10
    """
    from modelcypher.core.domain.cache import ComputationCache
    from modelcypher.core.domain.geometry.cka import compute_cka

    context = _context(ctx)
    backend = get_backend()
    cache = ComputationCache.shared()
    cache.clear_all()

    use_real_weights = False
    load_error: str | None = None
    source_weights: dict = {}
    target_weights: dict = {}

    if model_path:
        model_dir = Path(model_path)
        if model_dir.exists():
            # Try to load real weights
            try:
                safetensor_files = list(model_dir.glob("*.safetensors"))
                if safetensor_files:
                    all_weights = {}
                    for sf_path in safetensor_files:
                        file_weights = backend.load_safetensors(str(sf_path))
                        all_weights.update(file_weights)

                    # Extract attention weight layers
                    for name, param in all_weights.items():
                        if "self_attn" in name and "weight" in name:
                            source_weights[name] = param
                            if len(source_weights) >= n_pairs:
                                break

                    # Use same weights for target (testing cache behavior)
                    target_weights = source_weights.copy()
                    use_real_weights = len(source_weights) > 0
            except Exception as exc:
                load_error = f"{type(exc).__name__}: {exc}"

    # Fallback to synthetic data
    if not use_real_weights:
        backend.random_seed(42)
        n_samples = 100
        n_features = 256

        for i in range(n_pairs):
            source_weights[f"layer_{i}"] = backend.random_normal((n_samples, n_features))
            target_weights[f"layer_{i}"] = backend.random_normal((n_samples, n_features))

    source_names = list(source_weights.keys())
    target_names = list(target_weights.keys())
    actual_pairs = min(len(source_names), len(target_names), n_pairs)

    cache.clear_all()

    # First pass - cold cache
    first_pass_results = []
    start = time.perf_counter()
    for i in range(actual_pairs):
        src_w = source_weights[source_names[i]]
        tgt_w = target_weights[target_names[i]]

        # Handle reshaping
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
        first_pass_results.append(float(result.cka))

    first_pass_time = time.perf_counter() - start
    stats_after_first = cache.get_stats()

    # Second pass - warm cache
    second_pass_results = []
    start = time.perf_counter()
    for i in range(actual_pairs):
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
        second_pass_results.append(float(result.cka))

    second_pass_time = time.perf_counter() - start
    stats_after_second = cache.get_stats()

    speedup = first_pass_time / second_pass_time if second_pass_time > 0 else float("inf")

    final_stats = cache.get_stats()
    sizes = cache.get_cache_sizes()

    payload = {
        "_schema": "mc.system.test_cache.v1",
        "backend": backend.__class__.__name__,
        "dataSource": "real_weights" if use_real_weights else "synthetic",
        "modelPath": model_path,
        "layerPairsTested": actual_pairs,
        "firstPass": {
            "timeSeconds": round(first_pass_time, 3),
            "ckaValues": first_pass_results,
            "cacheMisses": stats_after_first.misses,
        },
        "secondPass": {
            "timeSeconds": round(second_pass_time, 3),
            "ckaValues": second_pass_results,
            "cacheHits": stats_after_second.hits - stats_after_first.hits,
        },
        "speedup": round(speedup, 2),
        "cacheStats": {
            "totalHits": final_stats.hits,
            "totalMisses": final_stats.misses,
            "hitRate": round(final_stats.hit_rate, 3),
            "computeTimeSavedMs": round(final_stats.total_compute_time_saved_ms, 1),
        },
        "cacheSizes": sizes,
    }
    if load_error is not None:
        payload["loadError"] = load_error

    write_output(payload, context.output_format, context.pretty)
