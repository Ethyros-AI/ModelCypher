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

"""Geodesic trajectory analysis for Chain-of-Thought reasoning.

Experimental. Measures whether token-by-token activation trajectories
follow geodesics on the activation manifold or cut through voids.

Pipeline: generate text -> collect per-token trajectories -> compute
geodesic vs euclidean distances between consecutive tokens -> report
deviation.
"""

from __future__ import annotations

import math
import logging
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.domain.geometry.riemannian_core import RiemannianGeometry

if TYPE_CHECKING:
    from modelcypher.ports.activation_provider import ActivationProvider
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class GeodesicTrajectoryResult:
    """Geodesic trajectory analysis for a generation."""

    token_count: int
    layer_analyzed: int

    # Per-consecutive-step geodesic deviation: 0.0 = straight line, >0 = curved
    step_deviations: list[float]

    # Summary statistics
    mean_deviation: float
    max_deviation: float

    # Intrinsic dimension of the trajectory point cloud
    intrinsic_dimension: float

    # Total geodesic path length vs straight-line (first->last) distance
    # 1.0 = straight line, >1.0 = curved path
    path_length_ratio: float

    # Token annotation (when annotate_tokens=True)
    token_labels: list[str] | None = None
    annotated_steps: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "token_count": self.token_count,
            "layer_analyzed": self.layer_analyzed,
            "step_deviations": self.step_deviations,
            "mean_deviation": self.mean_deviation,
            "max_deviation": self.max_deviation,
            "intrinsic_dimension": self.intrinsic_dimension,
            "path_length_ratio": self.path_length_ratio,
        }
        if self.token_labels is not None:
            d["token_labels"] = self.token_labels
        if self.annotated_steps is not None:
            d["annotated_steps"] = self.annotated_steps
        return d


@dataclass
class CategoryGeodesicSummary:
    """Aggregated geodesic metrics for a prompt category."""

    category: str
    prompt_count: int
    mean_deviation: float
    mean_path_length_ratio: float
    mean_intrinsic_dimension: float
    results: list[GeodesicTrajectoryResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "prompt_count": self.prompt_count,
            "mean_deviation": self.mean_deviation,
            "mean_path_length_ratio": self.mean_path_length_ratio,
            "mean_intrinsic_dimension": self.mean_intrinsic_dimension,
            "results": [r.to_dict() for r in self.results],
        }


@dataclass
class GeodesicComparisonResult:
    """Comparative geodesic analysis across prompt categories."""

    categories: list[CategoryGeodesicSummary]
    model_path: str
    layer_analyzed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "layer_analyzed": self.layer_analyzed,
            "categories": [c.to_dict() for c in self.categories],
        }


@dataclass
class LayerGeodesicProfile:
    """Per-layer geodesic metrics for a single text."""

    layer: int
    mean_deviation: float
    max_deviation: float
    path_length_ratio: float
    intrinsic_dimension: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "mean_deviation": self.mean_deviation,
            "max_deviation": self.max_deviation,
            "path_length_ratio": self.path_length_ratio,
            "intrinsic_dimension": self.intrinsic_dimension,
        }


@dataclass
class GeodesicLayerProfileResult:
    """Geodesic deviation profile across all layers for a single text."""

    text: str
    token_count: int
    num_layers: int
    layer_profiles: list[LayerGeodesicProfile]
    peak_deviation_layer: int
    inflection_layer: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text[:80],
            "token_count": self.token_count,
            "num_layers": self.num_layers,
            "layer_profiles": [lp.to_dict() for lp in self.layer_profiles],
            "peak_deviation_layer": self.peak_deviation_layer,
            "inflection_layer": self.inflection_layer,
        }


@dataclass
class CategoryLayerProfile:
    """Aggregated per-layer geodesic metrics for a prompt category."""

    category: str
    prompt_count: int
    layer_profiles: list[LayerGeodesicProfile]
    peak_deviation_layer: int
    inflection_layer: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "prompt_count": self.prompt_count,
            "layer_profiles": [lp.to_dict() for lp in self.layer_profiles],
            "peak_deviation_layer": self.peak_deviation_layer,
            "inflection_layer": self.inflection_layer,
        }


@dataclass
class GeodesicLayerProfileBatchResult:
    """Geodesic deviation profiles across all layers for multiple categories."""

    model_path: str
    num_layers: int
    category_profiles: list[CategoryLayerProfile]
    divergence_onset_layer: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "num_layers": self.num_layers,
            "category_profiles": [cp.to_dict() for cp in self.category_profiles],
            "divergence_onset_layer": self.divergence_onset_layer,
        }


@dataclass
class GeodesicLayerComparisonResult:
    """Per-layer geodesic deviation comparison between baseline and adapted model."""

    model_path: str
    adapter_path: str
    num_layers: int
    token_count: int
    baseline_profiles: list[LayerGeodesicProfile]
    adapted_profiles: list[LayerGeodesicProfile]
    delta_deviations: list[float]  # adapted - baseline per layer
    max_divergence_layer: int
    max_divergence_magnitude: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "adapter_path": self.adapter_path,
            "num_layers": self.num_layers,
            "token_count": self.token_count,
            "baseline_profiles": [lp.to_dict() for lp in self.baseline_profiles],
            "adapted_profiles": [lp.to_dict() for lp in self.adapted_profiles],
            "delta_deviations": self.delta_deviations,
            "max_divergence_layer": self.max_divergence_layer,
            "max_divergence_magnitude": self.max_divergence_magnitude,
        }


@dataclass
class GeodesicLayerComparisonBatchResult:
    """Per-layer geodesic deviation comparison across categorized prompts."""

    model_path: str
    adapter_path: str
    num_layers: int
    baseline_batch: GeodesicLayerProfileBatchResult
    adapted_batch: GeodesicLayerProfileBatchResult
    mean_delta_by_layer: list[tuple[int, float]]  # (layer_idx, mean_delta)
    max_divergence_layer: int
    max_divergence_magnitude: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "adapter_path": self.adapter_path,
            "num_layers": self.num_layers,
            "baseline_batch": self.baseline_batch.to_dict(),
            "adapted_batch": self.adapted_batch.to_dict(),
            "mean_delta_by_layer": [
                {"layer": layer, "delta": delta}
                for layer, delta in self.mean_delta_by_layer
            ],
            "max_divergence_layer": self.max_divergence_layer,
            "max_divergence_magnitude": self.max_divergence_magnitude,
        }


class GeodesicTrajectoryService:
    """Measure geodesic properties of activation trajectories.

    Experimental. Does Chain-of-Thought trace geodesics on the activation
    manifold, or cut through topological voids?
    """

    def __init__(
        self,
        backend: "Backend",
        activation_provider: "ActivationProvider",
    ) -> None:
        self._backend = backend
        self._activation_provider = activation_provider

    @staticmethod
    def _median(values: list[float]) -> float:
        """Compute median of a non-empty list."""
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mid = n // 2
        if n % 2 == 0:
            return 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])
        return sorted_vals[mid]

    def _derive_inflection_threshold(self, deviations: list[float]) -> float:
        """Derive inflection threshold from noise floor and layer statistics.

        Uses Donoho-Johnstone universal threshold:
            mean + sqrt(2 log n_layers) * std
        and falls back to peak * sqrt(eps) when std is numerically negligible.
        """
        if not deviations:
            return 0.0

        n_layers = len(deviations)
        mean_dev = sum(deviations) / n_layers
        variance = sum((d - mean_dev) ** 2 for d in deviations) / n_layers
        std_dev = variance**0.5
        peak_dev = max(deviations)

        eps = machine_epsilon(self._backend, self._backend.array([peak_dev if peak_dev > 0 else 1.0]))
        sqrt_eps = math.sqrt(eps)
        std_floor = sqrt_eps * max(1.0, abs(mean_dev))

        if std_dev <= std_floor:
            return peak_dev * sqrt_eps

        return mean_dev + math.sqrt(2.0 * math.log(max(2, n_layers))) * std_dev

    def _find_inflection_layer(
        self,
        layer_profiles: list[LayerGeodesicProfile],
    ) -> int | None:
        """Find first layer where deviation significantly exceeds background noise."""
        if not layer_profiles:
            return None

        deviations = [lp.mean_deviation for lp in layer_profiles]
        threshold = self._derive_inflection_threshold(deviations)
        for lp in layer_profiles:
            if lp.mean_deviation > threshold:
                return lp.layer
        return None

    def measure(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        target_layer: int | None = None,
        annotate_tokens: bool = False,
    ) -> GeodesicTrajectoryResult:
        """Measure geodesic properties of a text's activation trajectory.

        Args:
            model: Loaded model.
            tokenizer: Tokenizer for the model.
            text: Full text to analyze (prompt + response).
            target_layer: Layer to analyze. Defaults to last layer.
            annotate_tokens: If True, include per-token labels and
                annotated step dicts in the result.

        Returns:
            GeodesicTrajectoryResult with per-step deviations and summary.
        """
        # Collect trajectory via activation provider
        trajectory = self._activation_provider.collect_trajectory_batch(
            model, tokenizer, [text]
        )

        # Validate layer selection
        available_layers = sorted(trajectory.positions.keys())
        if not available_layers:
            raise ValueError("No trajectory positions collected")

        if target_layer is None:
            target_layer = available_layers[-1]
        elif target_layer not in trajectory.positions:
            raise ValueError(
                f"Layer {target_layer} not in trajectory. "
                f"Available: {available_layers}"
            )

        return self._measure_from_trajectory(
            trajectory, target_layer, annotate_tokens, tokenizer, text,
        )

    def _measure_from_trajectory(
        self,
        trajectory: Any,
        target_layer: int,
        annotate_tokens: bool = False,
        tokenizer: Any = None,
        text: str = "",
    ) -> GeodesicTrajectoryResult:
        """Analyze geodesic properties at a single layer from a pre-collected trajectory.

        This is the core analysis logic, factored out of measure() so that
        measure_layer_profile() can collect once and analyze N layers.
        """
        b = self._backend

        positions = trajectory.positions[target_layer]  # [n_tokens, hidden_dim]
        n_tokens = trajectory.total_tokens

        if n_tokens < 3:
            raise ValueError(
                f"Need >= 3 tokens for geodesic analysis, got {n_tokens}"
            )

        # Compute geodesic distance matrix
        rg = RiemannianGeometry(backend=b)
        geo_result = rg.geodesic_distances(positions)
        geo_dist = geo_result.distances  # [n, n]

        # Compute euclidean distances between consecutive tokens
        step_deviations: list[float] = []
        geodesic_steps: list[float] = []

        for t in range(n_tokens - 1):
            diff = b.subtract(positions[t + 1 : t + 2], positions[t : t + 1])
            eucl = b.to_scalar(b.sqrt(b.sum(b.multiply(diff, diff))))
            geo = b.to_scalar(geo_dist[t][t + 1])
            geodesic_steps.append(geo)

            if eucl > 0:
                deviation = (geo / eucl) - 1.0
            else:
                deviation = 0.0
            step_deviations.append(deviation)

        # Path length ratio: sum of geodesic steps / geodesic(first, last)
        total_path = sum(geodesic_steps)
        endpoint_geo = b.to_scalar(geo_dist[0][n_tokens - 1])

        if endpoint_geo > 0:
            path_length_ratio = total_path / endpoint_geo
        else:
            path_length_ratio = 1.0

        # Intrinsic dimension of the trajectory point cloud
        dim_est = IntrinsicDimension(backend=b)
        two_nn = dim_est.compute(positions)
        intrinsic_dim = two_nn.intrinsic_dimension

        # Summary
        mean_dev = sum(step_deviations) / len(step_deviations) if step_deviations else 0.0
        max_dev = max(step_deviations) if step_deviations else 0.0

        # Token annotation (optional)
        token_labels: list[str] | None = None
        annotated_steps: list[dict[str, Any]] | None = None

        if annotate_tokens and tokenizer is not None:
            try:
                token_ids = tokenizer.encode(text)
                if not isinstance(token_ids, list):
                    token_ids = list(token_ids)
                token_labels = [
                    tokenizer.decode([tid]) for tid in token_ids[:n_tokens]
                ]
                annotated_steps = []
                for t in range(len(step_deviations)):
                    annotated_steps.append({
                        "from": token_labels[t] if t < len(token_labels) else "?",
                        "to": token_labels[t + 1] if t + 1 < len(token_labels) else "?",
                        "deviation": step_deviations[t],
                    })
            except Exception:
                logger.debug("Token annotation failed", exc_info=True)

        return GeodesicTrajectoryResult(
            token_count=n_tokens,
            layer_analyzed=target_layer,
            step_deviations=step_deviations,
            mean_deviation=mean_dev,
            max_deviation=max_dev,
            intrinsic_dimension=intrinsic_dim,
            path_length_ratio=path_length_ratio,
            token_labels=token_labels,
            annotated_steps=annotated_steps,
        )

    def measure_batch(
        self,
        model: Any,
        tokenizer: Any,
        categorized_prompts: dict[str, list[str]],
        model_path: str = "",
        target_layer: int | None = None,
        annotate_tokens: bool = False,
    ) -> GeodesicComparisonResult:
        """Run geodesic analysis across categorized prompt sets.

        Args:
            model: Loaded model.
            tokenizer: Tokenizer for the model.
            categorized_prompts: Dict mapping category name to list of prompts.
            model_path: Model path string for the result metadata.
            target_layer: Layer to analyze. Defaults to last layer.
            annotate_tokens: If True, annotate tokens in per-prompt results.

        Returns:
            GeodesicComparisonResult with per-category summaries.
        """
        categories: list[CategoryGeodesicSummary] = []
        resolved_layer = target_layer

        for cat_name, prompts in categorized_prompts.items():
            results: list[GeodesicTrajectoryResult] = []
            for prompt in prompts:
                try:
                    r = self.measure(
                        model, tokenizer, prompt, target_layer, annotate_tokens,
                    )
                    results.append(r)
                    if resolved_layer is None:
                        resolved_layer = r.layer_analyzed
                except Exception:
                    logger.warning(
                        "Skipping prompt in %s: %s", cat_name, prompt[:40],
                        exc_info=True,
                    )

            if results:
                categories.append(CategoryGeodesicSummary(
                    category=cat_name,
                    prompt_count=len(results),
                    mean_deviation=sum(r.mean_deviation for r in results) / len(results),
                    mean_path_length_ratio=sum(r.path_length_ratio for r in results) / len(results),
                    mean_intrinsic_dimension=sum(r.intrinsic_dimension for r in results) / len(results),
                    results=results,
                ))

        return GeodesicComparisonResult(
            categories=categories,
            model_path=model_path,
            layer_analyzed=resolved_layer or 0,
        )

    def measure_layer_profile(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
    ) -> GeodesicLayerProfileResult:
        """Measure geodesic deviation at every layer for a single text.

        Collects trajectory ONCE, then analyzes each layer from the cached
        trajectory.

        Args:
            model: Loaded model.
            tokenizer: Tokenizer for the model.
            text: Full text to analyze.

        Returns:
            GeodesicLayerProfileResult with per-layer deviation profiles.
        """
        trajectory = self._activation_provider.collect_trajectory_batch(
            model, tokenizer, [text]
        )
        available_layers = sorted(trajectory.positions.keys())
        if not available_layers:
            raise ValueError("No trajectory positions collected")

        layer_profiles: list[LayerGeodesicProfile] = []
        for layer_idx in available_layers:
            try:
                result = self._measure_from_trajectory(
                    trajectory, layer_idx, annotate_tokens=False,
                )
                layer_profiles.append(LayerGeodesicProfile(
                    layer=layer_idx,
                    mean_deviation=result.mean_deviation,
                    max_deviation=result.max_deviation,
                    path_length_ratio=result.path_length_ratio,
                    intrinsic_dimension=result.intrinsic_dimension,
                ))
            except Exception:
                logger.warning("Skipping layer %d", layer_idx, exc_info=True)

        if layer_profiles:
            peak_lp = max(layer_profiles, key=lambda lp: lp.mean_deviation)
            peak_layer = peak_lp.layer
            inflection = self._find_inflection_layer(layer_profiles)
        else:
            peak_layer = 0
            inflection = None

        return GeodesicLayerProfileResult(
            text=text[:80],
            token_count=trajectory.total_tokens,
            num_layers=len(available_layers),
            layer_profiles=layer_profiles,
            peak_deviation_layer=peak_layer,
            inflection_layer=inflection,
        )

    def measure_layer_profile_batch(
        self,
        model: Any,
        tokenizer: Any,
        categorized_prompts: dict[str, list[str]],
        model_path: str = "",
        divergence_mad_multiplier: float | None = None,
    ) -> GeodesicLayerProfileBatchResult:
        """Measure geodesic deviation profiles across layers for categorized prompts.

        For each category, computes per-prompt layer profiles and averages
        per-layer metrics across prompts.

        Args:
            model: Loaded model.
            tokenizer: Tokenizer for the model.
            categorized_prompts: Dict mapping category name to list of prompts.
            model_path: Model path string for result metadata.
            divergence_mad_multiplier: Optional MAD multiplier for cross-category
                divergence onset detection. If None, derives multiplier from
                Donoho-Johnstone universal threshold with MAD→σ scaling.

        Returns:
            GeodesicLayerProfileBatchResult with per-category per-layer profiles.
        """
        category_profiles: list[CategoryLayerProfile] = []
        num_layers = 0

        for cat_name, prompts in categorized_prompts.items():
            prompt_profiles: list[GeodesicLayerProfileResult] = []
            for prompt_text in prompts:
                try:
                    profile = self.measure_layer_profile(
                        model, tokenizer, prompt_text,
                    )
                    prompt_profiles.append(profile)
                    if profile.num_layers > num_layers:
                        num_layers = profile.num_layers
                except Exception:
                    logger.warning(
                        "Skipping prompt in %s: %s", cat_name, prompt_text[:40],
                        exc_info=True,
                    )

            if not prompt_profiles:
                continue

            # Average per-layer metrics across prompts in this category
            layer_data: dict[int, list[LayerGeodesicProfile]] = {}
            for pp in prompt_profiles:
                for lp in pp.layer_profiles:
                    layer_data.setdefault(lp.layer, []).append(lp)

            averaged: list[LayerGeodesicProfile] = []
            for layer_idx in sorted(layer_data.keys()):
                profiles = layer_data[layer_idx]
                n = len(profiles)
                averaged.append(LayerGeodesicProfile(
                    layer=layer_idx,
                    mean_deviation=sum(p.mean_deviation for p in profiles) / n,
                    max_deviation=sum(p.max_deviation for p in profiles) / n,
                    path_length_ratio=sum(p.path_length_ratio for p in profiles) / n,
                    intrinsic_dimension=sum(p.intrinsic_dimension for p in profiles) / n,
                ))

            if averaged:
                peak_lp = max(averaged, key=lambda lp: lp.mean_deviation)
                peak_layer = peak_lp.layer
                inflection = self._find_inflection_layer(averaged)
            else:
                peak_layer = 0
                inflection = None

            category_profiles.append(CategoryLayerProfile(
                category=cat_name,
                prompt_count=len(prompt_profiles),
                layer_profiles=averaged,
                peak_deviation_layer=peak_layer,
                inflection_layer=inflection,
            ))

        divergence_onset = self._find_divergence_onset(
            category_profiles,
            threshold_mad_multiplier=divergence_mad_multiplier,
        )

        return GeodesicLayerProfileBatchResult(
            model_path=model_path,
            num_layers=num_layers,
            category_profiles=category_profiles,
            divergence_onset_layer=divergence_onset,
        )

    @staticmethod
    def _find_divergence_onset(
        category_profiles: list[CategoryLayerProfile],
        threshold_mad_multiplier: float | None = None,
    ) -> int | None:
        """Find first layer where inter-category spread exceeds a MAD-relative threshold."""
        if len(category_profiles) < 2:
            return None

        all_deviations: list[float] = []
        for cp in category_profiles:
            all_deviations.extend(lp.mean_deviation for lp in cp.layer_profiles)
        if not all_deviations:
            return None

        median_dev = GeodesicTrajectoryService._median(all_deviations)
        abs_deviations = [abs(dev - median_dev) for dev in all_deviations]
        mad = GeodesicTrajectoryService._median(abs_deviations)

        backend = get_default_backend()
        deviations_arr = backend.array(all_deviations)
        sqrt_eps = math.sqrt(float(machine_epsilon(backend, deviations_arr)))
        max_dev = max(abs(dev) for dev in all_deviations)
        spread_floor = sqrt_eps * max(1.0, max_dev)

        all_layers: set[int] = set()
        for cp in category_profiles:
            for lp in cp.layer_profiles:
                all_layers.add(lp.layer)
        n_layers = max(2, len(all_layers))

        resolved_multiplier = threshold_mad_multiplier
        if resolved_multiplier is None:
            # For Gaussian noise: sigma ≈ MAD / Phi^{-1}(0.75),
            # then universal threshold sigma*sqrt(2 log n_layers).
            mad_to_sigma = 1.0 / statistics.NormalDist().inv_cdf(0.75)
            resolved_multiplier = mad_to_sigma * math.sqrt(
                2.0 * math.log(float(n_layers))
            )

        spread_threshold = max(resolved_multiplier * mad, spread_floor)

        for layer_idx in sorted(all_layers):
            devs = []
            for cp in category_profiles:
                for lp in cp.layer_profiles:
                    if lp.layer == layer_idx:
                        devs.append(lp.mean_deviation)
                        break
            if len(devs) >= 2 and max(devs) - min(devs) > spread_threshold:
                return layer_idx
        return None

    def compare_layer_profiles(
        self,
        baseline_model: Any,
        adapted_model: Any,
        tokenizer: Any,
        text: str,
        model_path: str = "",
        adapter_path: str = "",
    ) -> GeodesicLayerComparisonResult:
        """Compare per-layer geodesic deviation between baseline and adapted model.

        Runs measure_layer_profile on both models with the same text,
        then computes per-layer delta (adapted - baseline).

        Args:
            baseline_model: Base model (no adapter).
            adapted_model: Model with adapter fused.
            tokenizer: Shared tokenizer.
            text: Text to profile.
            model_path: For result metadata.
            adapter_path: For result metadata.

        Returns:
            GeodesicLayerComparisonResult with per-layer deltas.
        """
        baseline_result = self.measure_layer_profile(
            model=baseline_model, tokenizer=tokenizer, text=text,
        )
        adapted_result = self.measure_layer_profile(
            model=adapted_model, tokenizer=tokenizer, text=text,
        )

        return self._build_comparison(
            baseline_result, adapted_result, model_path, adapter_path,
        )

    def compare_layer_profiles_batch(
        self,
        baseline_model: Any,
        adapted_model: Any,
        tokenizer: Any,
        categorized_prompts: dict[str, list[str]],
        model_path: str = "",
        adapter_path: str = "",
    ) -> GeodesicLayerComparisonBatchResult:
        """Compare per-layer geodesic deviation across categorized prompts.

        Args:
            baseline_model: Base model (no adapter).
            adapted_model: Model with adapter fused.
            tokenizer: Shared tokenizer.
            categorized_prompts: Dict mapping category name to list of prompts.
            model_path: For result metadata.
            adapter_path: For result metadata.

        Returns:
            GeodesicLayerComparisonBatchResult with per-layer deltas.
        """
        baseline_batch = self.measure_layer_profile_batch(
            model=baseline_model, tokenizer=tokenizer,
            categorized_prompts=categorized_prompts, model_path=model_path,
        )
        adapted_batch = self.measure_layer_profile_batch(
            model=adapted_model, tokenizer=tokenizer,
            categorized_prompts=categorized_prompts, model_path=model_path,
        )

        # Compute mean delta across all categories per layer
        baseline_by_layer: dict[int, list[float]] = {}
        for cp in baseline_batch.category_profiles:
            for lp in cp.layer_profiles:
                baseline_by_layer.setdefault(lp.layer, []).append(lp.mean_deviation)

        adapted_by_layer: dict[int, list[float]] = {}
        for cp in adapted_batch.category_profiles:
            for lp in cp.layer_profiles:
                adapted_by_layer.setdefault(lp.layer, []).append(lp.mean_deviation)

        all_layers = sorted(set(baseline_by_layer) | set(adapted_by_layer))
        mean_deltas: list[tuple[int, float]] = []
        max_div_layer = 0
        max_div_mag = 0.0

        for layer_idx in all_layers:
            base_devs = baseline_by_layer.get(layer_idx, [0.0])
            adapt_devs = adapted_by_layer.get(layer_idx, [0.0])
            base_mean = sum(base_devs) / len(base_devs)
            adapt_mean = sum(adapt_devs) / len(adapt_devs)
            delta = adapt_mean - base_mean
            mean_deltas.append((layer_idx, delta))
            if abs(delta) > max_div_mag:
                max_div_mag = abs(delta)
                max_div_layer = layer_idx

        num_layers = max(
            baseline_batch.num_layers, adapted_batch.num_layers,
        )

        return GeodesicLayerComparisonBatchResult(
            model_path=model_path,
            adapter_path=adapter_path,
            num_layers=num_layers,
            baseline_batch=baseline_batch,
            adapted_batch=adapted_batch,
            mean_delta_by_layer=mean_deltas,
            max_divergence_layer=max_div_layer,
            max_divergence_magnitude=max_div_mag,
        )

    @staticmethod
    def _build_comparison(
        baseline: GeodesicLayerProfileResult,
        adapted: GeodesicLayerProfileResult,
        model_path: str,
        adapter_path: str,
    ) -> GeodesicLayerComparisonResult:
        """Build comparison result from two layer profiles."""
        # Index adapted profiles by layer for lookup
        adapted_by_layer = {lp.layer: lp for lp in adapted.layer_profiles}

        deltas: list[float] = []
        for lp in baseline.layer_profiles:
            adapted_lp = adapted_by_layer.get(lp.layer)
            if adapted_lp is not None:
                deltas.append(adapted_lp.mean_deviation - lp.mean_deviation)
            else:
                deltas.append(0.0)

        if deltas:
            max_idx = max(range(len(deltas)), key=lambda i: abs(deltas[i]))
            max_layer = baseline.layer_profiles[max_idx].layer
            max_mag = abs(deltas[max_idx])
        else:
            max_layer = 0
            max_mag = 0.0

        return GeodesicLayerComparisonResult(
            model_path=model_path,
            adapter_path=adapter_path,
            num_layers=baseline.num_layers,
            token_count=baseline.token_count,
            baseline_profiles=baseline.layer_profiles,
            adapted_profiles=adapted.layer_profiles,
            delta_deviations=deltas,
            max_divergence_layer=max_layer,
            max_divergence_magnitude=max_mag,
        )
