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

"""Service layer for system cache benchmarks and diagnostics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


@dataclass(frozen=True)
class CacheBenchmarkMetric:
    """Timing metric for a benchmarked computation."""

    name: str
    cold_time_ms: float
    warm_time_ms: float
    speedup: float
    iterations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "coldTimeMs": round(self.cold_time_ms, 2),
            "warmTimeMs": round(self.warm_time_ms, 2),
            "speedup": round(self.speedup, 1),
            "iterations": self.iterations,
        }


@dataclass(frozen=True)
class CacheBenchmarkResponse:
    """Response payload for cache benchmark command."""

    backend_name: str
    benchmarks: list[CacheBenchmarkMetric]
    cache_stats: dict[str, Any]
    cache_sizes: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "_schema": "mc.system.benchmark.cache.v1",
            "backend": self.backend_name,
            "benchmarks": [metric.to_dict() for metric in self.benchmarks],
            "cacheStats": self.cache_stats,
            "cacheSizes": self.cache_sizes,
        }


@dataclass(frozen=True)
class CacheProbeRequest:
    """Request for cache probe command."""

    model_path: str | None = None
    n_pairs: int = 5


@dataclass(frozen=True)
class CacheProbeResponse:
    """Response payload for cache probe command."""

    backend_name: str
    data_source: str
    model_path: str | None
    layer_pairs_tested: int
    first_pass_time_seconds: float
    first_pass_cka_values: list[float]
    first_pass_cache_misses: int
    second_pass_time_seconds: float
    second_pass_cka_values: list[float]
    second_pass_cache_hits: int
    speedup: float
    cache_stats: dict[str, Any]
    cache_sizes: dict[str, Any]
    load_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "_schema": "mc.system.test_cache.v1",
            "backend": self.backend_name,
            "dataSource": self.data_source,
            "modelPath": self.model_path,
            "layerPairsTested": self.layer_pairs_tested,
            "firstPass": {
                "timeSeconds": round(self.first_pass_time_seconds, 3),
                "ckaValues": self.first_pass_cka_values,
                "cacheMisses": self.first_pass_cache_misses,
            },
            "secondPass": {
                "timeSeconds": round(self.second_pass_time_seconds, 3),
                "ckaValues": self.second_pass_cka_values,
                "cacheHits": self.second_pass_cache_hits,
            },
            "speedup": round(self.speedup, 2),
            "cacheStats": self.cache_stats,
            "cacheSizes": self.cache_sizes,
        }
        if self.load_error is not None:
            payload["loadError"] = self.load_error
        return payload


class SystemCacheService:
    """Benchmark and validate cache behavior through injected backend."""

    def __init__(self, backend: "Backend") -> None:
        self._backend = backend

    def benchmark_cache(self) -> CacheBenchmarkResponse:
        """Run system cache benchmarks and return raw timing metrics."""
        from modelcypher.core.domain.cache import ComputationCache

        # Warmup to exclude one-time kernel initialization effects.
        self._backend.random_seed(0)
        warmup = self._backend.random_normal((100, 100))
        warmup_mm = self._backend.matmul(warmup, self._backend.transpose(warmup))
        self._backend.eval(warmup_mm)

        benchmarks = [
            self._run_cka_benchmark(n_samples=50, n_features=128),
            self._run_cka_benchmark(n_samples=100, n_features=256),
            self._run_cka_benchmark(n_samples=200, n_features=512),
            self._run_geodesic_benchmark(n_points=30, n_dims=32),
            self._run_geodesic_benchmark(n_points=50, n_dims=64),
            self._run_frechet_benchmark(n_points=20, n_dims=16),
            self._run_frechet_benchmark(n_points=30, n_dims=32),
            self._run_geodesic_cosine_batch_benchmark(n_points=50, n_dims=32),
            self._run_geodesic_cosine_batch_benchmark(n_points=100, n_dims=64),
            self._run_geodesic_cosine_batch_benchmark(n_points=200, n_dims=128),
            self._run_realistic_merge_benchmark(),
        ]

        cache = ComputationCache.shared()
        stats = cache.get_stats()
        sizes = cache.get_cache_sizes()

        return CacheBenchmarkResponse(
            backend_name=self._backend.__class__.__name__,
            benchmarks=benchmarks,
            cache_stats={
                "totalHits": stats.hits,
                "totalMisses": stats.misses,
                "hitRate": round(stats.hit_rate, 3),
                "evictions": stats.evictions,
                "computeTimeSavedMs": round(stats.total_compute_time_saved_ms, 1),
            },
            cache_sizes=sizes,
        )

    def test_cache(self, request: CacheProbeRequest) -> CacheProbeResponse:
        """Measure cache behavior on real model weights or synthetic data."""
        from modelcypher.core.domain.cache import ComputationCache

        cache = ComputationCache.shared()
        cache.clear_all()

        use_real_weights = False
        load_error: str | None = None
        source_weights: dict[str, Any] = {}
        target_weights: dict[str, Any] = {}

        if request.model_path:
            model_dir = Path(request.model_path)
            if model_dir.exists():
                try:
                    safetensor_files = list(model_dir.glob("*.safetensors"))
                    if safetensor_files:
                        all_weights: dict[str, Any] = {}
                        for sf_path in safetensor_files:
                            file_weights = self._backend.load_safetensors(str(sf_path))
                            all_weights.update(file_weights)

                        for name, param in all_weights.items():
                            if "self_attn" in name and "weight" in name:
                                source_weights[name] = param
                                if len(source_weights) >= request.n_pairs:
                                    break

                        target_weights = source_weights.copy()
                        use_real_weights = len(source_weights) > 0
                except Exception as exc:
                    load_error = f"{type(exc).__name__}: {exc}"

        if not use_real_weights:
            self._backend.random_seed(42)
            n_samples = 100
            n_features = 256
            for idx in range(request.n_pairs):
                source_weights[f"layer_{idx}"] = self._backend.random_normal(
                    (n_samples, n_features)
                )
                target_weights[f"layer_{idx}"] = self._backend.random_normal(
                    (n_samples, n_features)
                )

        source_names = list(source_weights.keys())
        target_names = list(target_weights.keys())
        actual_pairs = min(len(source_names), len(target_names), request.n_pairs)

        cache.clear_all()
        first_pass_seconds, first_pass_values = self._run_cka_pairs(
            source_weights=source_weights,
            target_weights=target_weights,
            source_names=source_names,
            target_names=target_names,
            n_pairs=actual_pairs,
        )
        stats_after_first = cache.get_stats()

        second_pass_seconds, second_pass_values = self._run_cka_pairs(
            source_weights=source_weights,
            target_weights=target_weights,
            source_names=source_names,
            target_names=target_names,
            n_pairs=actual_pairs,
        )
        stats_after_second = cache.get_stats()

        speedup = (
            first_pass_seconds / second_pass_seconds
            if second_pass_seconds > 0
            else float("inf")
        )

        final_stats = cache.get_stats()
        sizes = cache.get_cache_sizes()

        return CacheProbeResponse(
            backend_name=self._backend.__class__.__name__,
            data_source="real_weights" if use_real_weights else "synthetic",
            model_path=request.model_path,
            layer_pairs_tested=actual_pairs,
            first_pass_time_seconds=first_pass_seconds,
            first_pass_cka_values=first_pass_values,
            first_pass_cache_misses=stats_after_first.misses,
            second_pass_time_seconds=second_pass_seconds,
            second_pass_cka_values=second_pass_values,
            second_pass_cache_hits=stats_after_second.hits - stats_after_first.hits,
            speedup=speedup,
            cache_stats={
                "totalHits": final_stats.hits,
                "totalMisses": final_stats.misses,
                "hitRate": round(final_stats.hit_rate, 3),
                "computeTimeSavedMs": round(final_stats.total_compute_time_saved_ms, 1),
            },
            cache_sizes=sizes,
            load_error=load_error,
        )

    def _run_cka_benchmark(self, n_samples: int, n_features: int) -> CacheBenchmarkMetric:
        """Benchmark CKA with Gram matrix caching."""
        from modelcypher.core.domain.cache import ComputationCache
        from modelcypher.core.domain.geometry.cka import compute_cka

        cache = ComputationCache.shared()
        cache.clear_all()

        self._backend.random_seed(42)
        act_x = self._backend.random_normal((n_samples, n_features))
        act_y = self._backend.random_normal((n_samples, n_features))

        start = time.perf_counter()
        compute_cka(act_x, act_y, self._backend)
        cold_time = (time.perf_counter() - start) * 1000

        n_warm = 10
        start = time.perf_counter()
        for _ in range(n_warm):
            compute_cka(act_x, act_y, self._backend)
        warm_time = (time.perf_counter() - start) * 1000 / n_warm
        speedup = cold_time / warm_time if warm_time > 0 else float("inf")

        return CacheBenchmarkMetric(
            name=f"CKA Gram caching ({n_samples}x{n_features})",
            cold_time_ms=cold_time,
            warm_time_ms=warm_time,
            speedup=speedup,
            iterations=n_warm,
        )

    def _run_geodesic_benchmark(self, n_points: int, n_dims: int) -> CacheBenchmarkMetric:
        """Benchmark geodesic distance caching."""
        from modelcypher.core.domain.cache import ComputationCache
        from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

        cache = ComputationCache.shared()
        cache.clear_all()

        self._backend.random_seed(45)
        points = self._backend.random_normal((n_points, n_dims))
        rg = RiemannianGeometry(self._backend)

        start = time.perf_counter()
        rg.geodesic_distances(points, k_neighbors=10)
        cold_time = (time.perf_counter() - start) * 1000

        n_warm = 5
        start = time.perf_counter()
        for _ in range(n_warm):
            rg.geodesic_distances(points, k_neighbors=10)
        warm_time = (time.perf_counter() - start) * 1000 / n_warm
        speedup = cold_time / warm_time if warm_time > 0 else float("inf")

        return CacheBenchmarkMetric(
            name=f"Geodesic distances ({n_points}x{n_dims})",
            cold_time_ms=cold_time,
            warm_time_ms=warm_time,
            speedup=speedup,
            iterations=n_warm,
        )

    def _run_frechet_benchmark(self, n_points: int, n_dims: int) -> CacheBenchmarkMetric:
        """Benchmark Fréchet mean caching."""
        from modelcypher.core.domain.cache import ComputationCache
        from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

        cache = ComputationCache.shared()
        cache.clear_all()

        self._backend.random_seed(46)
        points = self._backend.random_normal((n_points, n_dims))
        rg = RiemannianGeometry(self._backend)

        start = time.perf_counter()
        rg.frechet_mean(points)
        cold_time = (time.perf_counter() - start) * 1000

        n_warm = 5
        start = time.perf_counter()
        for _ in range(n_warm):
            rg.frechet_mean(points)
        warm_time = (time.perf_counter() - start) * 1000 / n_warm
        speedup = cold_time / warm_time if warm_time > 0 else float("inf")

        return CacheBenchmarkMetric(
            name=f"Fréchet mean ({n_points}x{n_dims})",
            cold_time_ms=cold_time,
            warm_time_ms=warm_time,
            speedup=speedup,
            iterations=n_warm,
        )

    def _run_geodesic_cosine_batch_benchmark(
        self,
        n_points: int,
        n_dims: int,
    ) -> CacheBenchmarkMetric:
        """Benchmark geodesic cosine batch using single-source paths."""
        from modelcypher.core.domain.cache import ComputationCache
        from modelcypher.core.domain.geometry.riemannian_utils import geodesic_cosine_batch

        cache = ComputationCache.shared()
        cache.clear_all()

        self._backend.random_seed(48)
        vectors = self._backend.random_normal((n_points, n_dims))
        anchor = self._backend.random_normal((n_dims,))

        start = time.perf_counter()
        geodesic_cosine_batch(anchor, vectors, self._backend)
        cold_time = (time.perf_counter() - start) * 1000

        n_warm = 10
        start = time.perf_counter()
        for _ in range(n_warm):
            geodesic_cosine_batch(anchor, vectors, self._backend)
        warm_time = (time.perf_counter() - start) * 1000 / n_warm
        speedup = cold_time / warm_time if warm_time > 0 else float("inf")

        return CacheBenchmarkMetric(
            name=f"Geodesic cosine batch ({n_points}x{n_dims})",
            cold_time_ms=cold_time,
            warm_time_ms=warm_time,
            speedup=speedup,
            iterations=n_warm,
        )

    def _run_realistic_merge_benchmark(self) -> CacheBenchmarkMetric:
        """Benchmark realistic CKA merge scenario across layer pairs."""
        from modelcypher.core.domain.cache import ComputationCache
        from modelcypher.core.domain.geometry.cka import compute_cka

        cache = ComputationCache.shared()
        cache.clear_all()

        self._backend.random_seed(47)
        n_layers = 10
        n_samples = 50
        n_features = 128

        source_layers = [
            self._backend.random_normal((n_samples, n_features))
            for _ in range(n_layers)
        ]
        target_layers = [
            self._backend.random_normal((n_samples, n_features))
            for _ in range(n_layers)
        ]

        start = time.perf_counter()
        for src, tgt in zip(source_layers, target_layers):
            cache.clear_all()
            compute_cka(src, tgt, self._backend)
        cold_time = (time.perf_counter() - start) * 1000

        cache.clear_all()
        start = time.perf_counter()
        for src, tgt in zip(source_layers, target_layers):
            compute_cka(src, tgt, self._backend)
        for src, tgt in zip(source_layers, target_layers):
            compute_cka(src, tgt, self._backend)
        warm_time = (time.perf_counter() - start) * 1000

        speedup = (cold_time * 2) / warm_time if warm_time > 0 else float("inf")
        return CacheBenchmarkMetric(
            name=f"Realistic merge: {n_layers} layers x 2 passes",
            cold_time_ms=cold_time,
            warm_time_ms=warm_time / 2,
            speedup=speedup,
            iterations=n_layers * 2,
        )

    def _run_cka_pairs(
        self,
        source_weights: dict[str, Any],
        target_weights: dict[str, Any],
        source_names: list[str],
        target_names: list[str],
        n_pairs: int,
    ) -> tuple[float, list[float]]:
        """Compute CKA for layer pairs and return elapsed time and values."""
        from modelcypher.core.domain.geometry.cka import compute_cka

        values: list[float] = []
        start = time.perf_counter()
        for idx in range(n_pairs):
            src_w = source_weights[source_names[idx]]
            tgt_w = target_weights[target_names[idx]]

            src_shape = src_w.shape
            tgt_shape = tgt_w.shape

            if len(src_shape) > 2:
                src_w = self._backend.reshape(src_w, (src_shape[0], -1))
            if len(tgt_shape) > 2:
                tgt_w = self._backend.reshape(tgt_w, (tgt_shape[0], -1))

            n_samples = min(src_w.shape[0], tgt_w.shape[0], 100)
            src_w = src_w[:n_samples]
            tgt_w = tgt_w[:n_samples]

            cka_result = compute_cka(src_w, tgt_w, self._backend)
            values.append(float(cka_result.cka))

        elapsed = time.perf_counter() - start
        return elapsed, values


__all__ = [
    "CacheBenchmarkMetric",
    "CacheBenchmarkResponse",
    "CacheProbeRequest",
    "CacheProbeResponse",
    "SystemCacheService",
]
