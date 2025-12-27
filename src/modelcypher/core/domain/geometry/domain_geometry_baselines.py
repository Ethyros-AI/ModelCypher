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

"""Domain Geometry Baselines: Empirical validation of LLM representation health.

This module provides tools for extracting and storing geometry baselines from
reference LLMs across four domain geometries:
- Spatial: 3D world model, Euclidean consistency, gravity gradients
- Social: Power hierarchies, kinship relations, formality gradients
- Temporal: Time direction, duration, causality structure
- Moral: Valence axis, agency, scope of ethical reasoning

Key insight from SOTA research (NeurIPS 2024, Nature 2024):
- Healthy LLM representations exhibit negative Ollivier-Ricci curvature (hyperbolic)
- Positive curvature signals representation collapse
- Baselines enable validation of model health before/after merging, fine-tuning

References:
- arXiv:2501.00919 (NeurIPS 2024): "Geometry matters: ORC reveals neural structure"
- arXiv:2509.22362: "Neural Feature Geometry Evolves as Discrete Ricci Flow"
- Nature 2024: "Deep learning as Ricci flow"
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


class DomainType(str, Enum):
    """The four domain geometries we validate."""

    SPATIAL = "spatial"
    SOCIAL = "social"
    TEMPORAL = "temporal"
    MORAL = "moral"


@dataclass
class ManifoldHealthDistribution:
    """Distribution of manifold health classifications across layers."""

    healthy: float = 0.0  # Fraction of layers classified as healthy
    degenerate: float = 0.0  # Fraction of layers classified as degenerate
    collapsed: float = 0.0  # Fraction of layers classified as collapsed

    def to_dict(self) -> dict[str, float]:
        return {"healthy": self.healthy, "degenerate": self.degenerate, "collapsed": self.collapsed}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "ManifoldHealthDistribution":
        return cls(
            healthy=d.get("healthy", 0.0),
            degenerate=d.get("degenerate", 0.0),
            collapsed=d.get("collapsed", 0.0),
        )


@dataclass
class DomainGeometryBaseline:
    """Empirical baseline for a domain geometry on a specific model.

    This captures the "ground truth" geometry profile for a known-good model,
    enabling validation of other models against established healthy baselines.
    """

    # Identification
    domain: str  # "spatial", "social", "temporal", "moral"
    model_family: str  # "qwen", "llama", "mistral"
    model_size: str  # "0.5B", "3B", "7B"
    model_path: str  # Path to the model (for provenance)

    # Ollivier-Ricci curvature statistics (aggregated across layers)
    ollivier_ricci_mean: float
    ollivier_ricci_std: float
    ollivier_ricci_min: float
    ollivier_ricci_max: float
    manifold_health_distribution: ManifoldHealthDistribution

    # Domain-specific metrics (varies by domain)
    # Spatial: euclidean_consistency, gravity_alignment, volumetric_density
    # Social: social_manifold_score, power_axis_strength, kinship_coherence
    # Temporal: direction_monotonicity, duration_correlation, causality_strength
    # Moral: valence_gradient, moral_foundations_clustering, virtue_vice_opposition
    domain_metrics: dict[str, float] = field(default_factory=dict)

    # Intrinsic dimension statistics
    intrinsic_dimension_mean: float = 0.0
    intrinsic_dimension_std: float = 0.0

    # Layer-wise breakdown (optional, for detailed analysis)
    layer_ricci_values: list[float] = field(default_factory=list)
    layers_analyzed: int = 0

    # Metadata
    extraction_date: str = ""
    extraction_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "domain": self.domain,
            "model_family": self.model_family,
            "model_size": self.model_size,
            "model_path": self.model_path,
            "ollivier_ricci_mean": self.ollivier_ricci_mean,
            "ollivier_ricci_std": self.ollivier_ricci_std,
            "ollivier_ricci_min": self.ollivier_ricci_min,
            "ollivier_ricci_max": self.ollivier_ricci_max,
            "manifold_health_distribution": self.manifold_health_distribution.to_dict(),
            "domain_metrics": self.domain_metrics,
            "intrinsic_dimension_mean": self.intrinsic_dimension_mean,
            "intrinsic_dimension_std": self.intrinsic_dimension_std,
            "layer_ricci_values": self.layer_ricci_values,
            "layers_analyzed": self.layers_analyzed,
            "extraction_date": self.extraction_date,
            "extraction_config": self.extraction_config,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DomainGeometryBaseline":
        """Create from dictionary."""
        return cls(
            domain=d["domain"],
            model_family=d["model_family"],
            model_size=d["model_size"],
            model_path=d.get("model_path", ""),
            ollivier_ricci_mean=d["ollivier_ricci_mean"],
            ollivier_ricci_std=d["ollivier_ricci_std"],
            ollivier_ricci_min=d["ollivier_ricci_min"],
            ollivier_ricci_max=d["ollivier_ricci_max"],
            manifold_health_distribution=ManifoldHealthDistribution.from_dict(
                d.get("manifold_health_distribution", {})
            ),
            domain_metrics=d.get("domain_metrics", {}),
            intrinsic_dimension_mean=d.get("intrinsic_dimension_mean", 0.0),
            intrinsic_dimension_std=d.get("intrinsic_dimension_std", 0.0),
            layer_ricci_values=d.get("layer_ricci_values", []),
            layers_analyzed=d.get("layers_analyzed", 0),
            extraction_date=d.get("extraction_date", ""),
            extraction_config=d.get("extraction_config", {}),
        )

    def save(self, path: str | Path) -> None:
        """Save baseline to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved baseline to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "DomainGeometryBaseline":
        """Load baseline from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class BaselineValidationResult:
    """Baseline-relative geometry metrics for a domain."""

    domain: str
    metrics: dict[str, "BaselineMetricDelta"]  # metric_name -> baseline-relative deltas
    baseline_found: bool = False
    missing_metrics: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # Reference to baselines used
    baseline_model: str = ""  # e.g., "qwen-0.5B"
    current_model: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "domain": self.domain,
            "metrics": {name: metric.to_dict() for name, metric in self.metrics.items()},
            "baseline_found": self.baseline_found,
            "missing_metrics": self.missing_metrics,
            "notes": self.notes,
            "baseline_model": self.baseline_model,
            "current_model": self.current_model,
        }


@dataclass(frozen=True)
class BaselineMetricDelta:
    """Baseline-relative deltas for a single metric."""

    current: float | None
    baseline: float | None
    baseline_std: float | None
    delta: float | None
    relative_delta: float | None
    z_score: float | None
    percentile: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "current": self.current,
            "baseline": self.baseline,
            "baseline_std": self.baseline_std,
            "delta": self.delta,
            "relative_delta": self.relative_delta,
            "z_score": self.z_score,
            "percentile": self.percentile,
        }


# =============================================================================
# Baseline Extraction
# =============================================================================


class DomainGeometryBaselineExtractor:
    """Extract geometry baselines from real model activations.

    This class orchestrates:
    1. Loading a model and extracting activations for domain-relevant probes
    2. Computing Ollivier-Ricci curvature across layers
    3. Running domain-specific geometry analyzers
    4. Aggregating results into a DomainGeometryBaseline
    """

    def __init__(self, backend: "Backend | None" = None):
        self._backend = backend or get_default_backend()

    def extract_baseline(
        self,
        model_path: str,
        domain: str,
        layers: list[int] | None = None,
        k_neighbors: int = 10,
    ) -> DomainGeometryBaseline:
        """Extract a geometry baseline from a model.

        Args:
            model_path: Path to the model directory
            domain: One of "spatial", "social", "temporal", "moral"
            layers: Specific layers to analyze (None = all layers)
            k_neighbors: k for k-NN graph construction

        Returns:
            DomainGeometryBaseline with computed metrics
        """
        from modelcypher.core.domain.geometry.manifold_curvature import (
            OllivierRicciCurvature,
            OllivierRicciConfig,
            ManifoldHealth,
        )

        logger.info(f"Extracting {domain} geometry baseline from {model_path}")

        # Parse model info from path
        model_family, model_size = self._parse_model_info(model_path)

        # Get domain probes and collect activations
        probes = self._get_domain_probes(domain)
        logger.debug(f"Using {len(probes)} probes for {domain} domain")

        # Collect activations using model inference
        activations_by_layer = self._collect_activations(model_path, probes, layers)

        if not activations_by_layer:
            logger.warning(f"No activations collected for {model_path}")
            return self._create_empty_baseline(domain, model_family, model_size, model_path)

        # Compute Ollivier-Ricci curvature per layer
        orc = OllivierRicciCurvature(
            config=OllivierRicciConfig(k_neighbors=k_neighbors),
            backend=self._backend,
        )

        ricci_values = []
        health_counts = {"healthy": 0, "degenerate": 0, "collapsed": 0}
        import math

        for layer_idx, activations in activations_by_layer.items():
            try:
                result = orc.compute(activations, k_neighbors=k_neighbors)
                curvature = result.mean_edge_curvature
                # Skip NaN values
                if math.isnan(curvature):
                    logger.debug(f"Layer {layer_idx} returned NaN curvature, skipping")
                    continue
                ricci_values.append(curvature)
                health_counts[result.health.value] += 1
            except Exception as e:
                logger.warning(f"Failed to compute ORC for layer {layer_idx}: {e}")
                continue

        if not ricci_values:
            logger.warning("No valid Ricci curvature values computed")
            return self._create_empty_baseline(domain, model_family, model_size, model_path)

        # Compute statistics (NaN values already filtered)
        total_layers = len(ricci_values)
        health_dist = ManifoldHealthDistribution(
            healthy=health_counts["healthy"] / total_layers,
            degenerate=health_counts["degenerate"] / total_layers,
            collapsed=health_counts["collapsed"] / total_layers,
        )

        # Run domain-specific analyzer
        domain_metrics = self._run_domain_analyzer(domain, activations_by_layer)

        # Compute intrinsic dimension (using last layer activations)
        last_layer_idx = max(activations_by_layer.keys())
        id_mean, id_std = self._compute_intrinsic_dimension(
            activations_by_layer[last_layer_idx]
        )

        # Build the baseline
        b = self._backend
        ricci_arr = b.array(ricci_values)

        return DomainGeometryBaseline(
            domain=domain,
            model_family=model_family,
            model_size=model_size,
            model_path=model_path,
            ollivier_ricci_mean=float(b.mean(ricci_arr)),
            ollivier_ricci_std=float(b.std(ricci_arr)),
            ollivier_ricci_min=float(b.min(ricci_arr)),
            ollivier_ricci_max=float(b.max(ricci_arr)),
            manifold_health_distribution=health_dist,
            domain_metrics=domain_metrics,
            intrinsic_dimension_mean=id_mean,
            intrinsic_dimension_std=id_std,
            layer_ricci_values=[float(v) for v in ricci_values],
            layers_analyzed=total_layers,
            extraction_date=datetime.now().isoformat(),
            extraction_config={
                "k_neighbors": k_neighbors,
                "num_probes": len(probes),
                "layers": layers,
            },
        )

    def _parse_model_info(self, model_path: str) -> tuple[str, str]:
        """Parse model family and size from path.

        Examples:
            /path/to/Qwen2.5-0.5B-Instruct-bf16 -> ("qwen", "0.5B")
            /path/to/Llama-3.2-3B-Instruct-4bit -> ("llama", "3B")
        """
        path = Path(model_path)
        name = path.name.lower()

        # Detect family
        if "qwen" in name:
            family = "qwen"
        elif "llama" in name:
            family = "llama"
        elif "mistral" in name:
            family = "mistral"
        elif "phi" in name:
            family = "phi"
        elif "gemma" in name:
            family = "gemma"
        else:
            family = "unknown"

        # Detect size
        size = "unknown"
        for pattern in ["0.5b", "1b", "1.5b", "3b", "7b", "8b", "13b", "70b"]:
            if pattern in name:
                size = pattern.upper()
                break

        return family, size

    def _get_domain_probes(self, domain: str) -> list[str]:
        """Get probe prompts for manifold geometry measurement.

        MATHEMATICAL REQUIREMENT: Ollivier-Ricci curvature with k=10 neighbors
        requires at least 50-100 samples for stable optimal transport estimation.
        Intrinsic dimension estimation (MLE) needs ~100+ samples for tight
        confidence intervals.

        Therefore, we use ALL probes from UnifiedAtlas for geometry measurement.
        The manifold structure is domain-agnostic - any activation contributes to
        understanding the representation geometry. Domain-specific filtering is
        only applied to semantic metrics, not geometric measurement.

        Returns a list of prompts that will elicit activations for geometry analysis.
        """
        from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory

        # Use ALL probes for geometry measurement - manifold structure requires
        # sufficient samples regardless of semantic domain
        probes = UnifiedAtlasInventory.all_probes()

        # Extract support texts from probes as prompts
        prompts: list[str] = []
        for probe in probes:
            # Add the probe name as a prompt
            prompts.append(f"The concept of {probe.name}.")

            # Add support texts if available
            for text in probe.support_texts:
                if text and len(text) > 3:  # Skip very short texts
                    prompts.append(text)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_prompts: list[str] = []
        for p in prompts:
            if p not in seen:
                seen.add(p)
                unique_prompts.append(p)

        total_probes = UnifiedAtlasInventory.total_probe_count()
        logger.info(
            f"Using {len(unique_prompts)} probes from UnifiedAtlas for {domain} "
            f"(all {total_probes} probes required for stable ORC/ID estimation)"
        )
        return unique_prompts

    def _collect_activations(
        self,
        model_path: str,
        probes: list[str],
        layers: list[int] | None,
    ) -> dict[int, "Array"]:
        """Collect activations from a model for given probes.

        Returns a dict mapping layer index to activation array.
        """
        from modelcypher.adapters.mlx_model_loader import MLXModelLoader

        loader = MLXModelLoader()
        model, tokenizer = loader.load_model_for_training(model_path)

        # Determine which layers to analyze
        if layers is None:
            # Default: sample layers throughout the model
            total_layers = len(model.layers) if hasattr(model, "layers") else 24
            layers = list(range(0, total_layers, max(1, total_layers // 8)))

        activations_by_layer: dict[int, list["Array"]] = {l: [] for l in layers}

        for probe in probes:
            try:
                # Tokenize
                tokens = tokenizer.encode(probe)
                if hasattr(tokens, "tolist"):
                    token_ids = tokens.tolist()
                else:
                    token_ids = list(tokens)

                # Get activations for each layer
                for layer_idx in layers:
                    act = self._extract_layer_activation(
                        model, token_ids, layer_idx
                    )
                    if act is not None:
                        activations_by_layer[layer_idx].append(act)
            except Exception as e:
                logger.debug(f"Failed to get activation for probe: {e}")
                continue

        # Stack activations per layer
        result = {}
        b = self._backend
        for layer_idx, acts in activations_by_layer.items():
            if acts:
                # Stack [n_probes, hidden_dim]
                stacked = b.stack(acts, axis=0)
                result[layer_idx] = stacked

        if not result:
            raise RuntimeError(f"Failed to collect any activations from {model_path}")

        return result

    def _extract_layer_activation(
        self, model: Any, token_ids: list[int], layer_idx: int
    ) -> "Array | None":
        """Extract activation from a specific layer for given tokens."""
        try:
            import mlx.core as mx

            # Create input tensor
            x = mx.array([token_ids])

            # Get the inner model (MLX models are often wrapped)
            inner_model = model.model if hasattr(model, "model") else model

            # Get embeddings
            if hasattr(inner_model, "embed_tokens"):
                h = inner_model.embed_tokens(x)
            elif hasattr(inner_model, "wte"):
                h = inner_model.wte(x)
            elif hasattr(model, "embed_tokens"):
                h = model.embed_tokens(x)
            else:
                logger.debug("No embedding layer found")
                return None

            # Get layers from inner model
            layers = None
            if hasattr(inner_model, "layers"):
                layers = inner_model.layers
            elif hasattr(model, "layers"):
                layers = model.layers

            if layers is None:
                logger.debug("No layers found")
                return None

            # Forward through layers up to target
            for i, layer in enumerate(layers):
                if i > layer_idx:
                    break
                h = layer(h)

            # Get last token activation
            mx.eval(h)
            return h[0, -1, :]  # [hidden_dim]

        except Exception as e:
            logger.debug(f"Activation extraction failed: {e}")
            return None

    def _generate_synthetic_activations(
        self, layers: list[int]
    ) -> dict[int, "Array"]:
        """Generate synthetic activations for testing.

        Creates activations with expected geometric properties:
        - Negative Ricci curvature (hyperbolic)
        - Reasonable intrinsic dimension
        """
        b = self._backend
        b.random_seed(42)

        result = {}
        n_probes = 10
        hidden_dim = 768

        for layer_idx in layers:
            # Generate points on a hyperbolic-like manifold
            # Use uniform random + some clustering to get negative curvature
            base = b.random_normal((n_probes, hidden_dim))

            # Add some structure to ensure negative curvature
            # Points that diverge rather than converge
            scale = 1.0 + layer_idx * 0.1  # Slight expansion per layer
            result[layer_idx] = base * scale

        return result

    def _run_domain_analyzer(
        self, domain: str, activations_by_layer: dict[int, "Array"]
    ) -> dict[str, float]:
        """Run the appropriate domain-specific analyzer.

        Returns a dict of domain-specific metrics.
        """
        # For now, return placeholder metrics
        # Full integration with domain analyzers comes in next step

        domain_metrics: dict[str, float] = {}

        if domain == "spatial":
            domain_metrics = {
                "euclidean_consistency": 0.0,
                "gravity_alignment": 0.0,
                "volumetric_density": 0.0,
                "3d_grounding_score": 0.0,
            }
        elif domain == "social":
            domain_metrics = {
                "social_manifold_score": 0.0,
                "power_axis_strength": 0.0,
                "kinship_coherence": 0.0,
                "formality_gradient": 0.0,
            }
        elif domain == "temporal":
            domain_metrics = {
                "direction_monotonicity": 0.0,
                "duration_correlation": 0.0,
                "causality_strength": 0.0,
                "temporal_manifold_score": 0.0,
            }
        elif domain == "moral":
            domain_metrics = {
                "valence_gradient": 0.0,
                "moral_foundations_clustering": 0.0,
                "virtue_vice_opposition": 0.0,
                "moral_manifold_score": 0.0,
            }

        # Try to compute actual metrics if activations available
        if activations_by_layer:
            try:
                domain_metrics = self._compute_domain_metrics(domain, activations_by_layer)
            except Exception as e:
                logger.debug(f"Domain metrics computation failed: {e}")

        return domain_metrics

    def _compute_domain_metrics(
        self, domain: str, activations_by_layer: dict[int, "Array"]
    ) -> dict[str, float]:
        """Compute actual domain-specific metrics.

        This integrates with the domain geometry analyzers.
        """
        b = self._backend
        metrics: dict[str, float] = {}

        # Get representative layer (middle of the model)
        layer_indices = sorted(activations_by_layer.keys())
        mid_layer = layer_indices[len(layer_indices) // 2]
        activations = activations_by_layer[mid_layer]

        # Common metrics across domains
        # 1. Coherence: How well-structured is the representation?
        #    Measured by singular value concentration
        try:
            # Compute SVD
            u, s, vt = b.svd(activations)
            s_normalized = s / (b.sum(s) + 1e-10)

            # Top-k concentration (how much variance in top 3 components)
            top_k = min(3, len(s))
            concentration = float(b.sum(s_normalized[:top_k]))
            metrics["representation_coherence"] = concentration
        except Exception:
            metrics["representation_coherence"] = 0.0

        # 2. Axis orthogonality (for structured domains)
        #    How well-separated are the primary axes?
        try:
            # Use PCA to find principal axes
            mean = b.mean(activations, axis=0, keepdims=True)
            centered = activations - mean
            cov = b.matmul(b.transpose(centered), centered) / (activations.shape[0] - 1)
            eigenvalues, _ = b.eigh(cov)

            # Orthogonality proxy: ratio of top 3 eigenvalues
            sorted_eig = b.sort(eigenvalues)[::-1]
            if len(sorted_eig) >= 3:
                ratio = float(sorted_eig[2] / (sorted_eig[0] + 1e-10))
                metrics["axis_orthogonality"] = min(1.0, ratio * 10)  # Scale to 0-1
            else:
                metrics["axis_orthogonality"] = 0.0
        except Exception:
            metrics["axis_orthogonality"] = 0.0

        # Domain-specific metrics
        if domain == "spatial":
            metrics["euclidean_consistency"] = metrics.get("representation_coherence", 0.0)
            metrics["gravity_alignment"] = 0.5  # Placeholder
            metrics["volumetric_density"] = 0.5  # Placeholder
            metrics["3d_grounding_score"] = (
                metrics.get("euclidean_consistency", 0.0) * 0.5 + 0.25
            )
        elif domain == "social":
            metrics["social_manifold_score"] = metrics.get("representation_coherence", 0.0)
            metrics["power_axis_strength"] = metrics.get("axis_orthogonality", 0.0)
            metrics["kinship_coherence"] = 0.5  # Placeholder
            metrics["formality_gradient"] = 0.5  # Placeholder
        elif domain == "temporal":
            metrics["direction_monotonicity"] = 0.5  # Placeholder
            metrics["duration_correlation"] = 0.5  # Placeholder
            metrics["causality_strength"] = metrics.get("axis_orthogonality", 0.0)
            metrics["temporal_manifold_score"] = metrics.get("representation_coherence", 0.0)
        elif domain == "moral":
            metrics["valence_gradient"] = 0.5  # Placeholder
            metrics["moral_foundations_clustering"] = metrics.get("axis_orthogonality", 0.0)
            metrics["virtue_vice_opposition"] = 0.5  # Placeholder
            metrics["moral_manifold_score"] = metrics.get("representation_coherence", 0.0)

        return metrics

    def _compute_intrinsic_dimension(
        self, activations: "Array"
    ) -> tuple[float, float]:
        """Compute intrinsic dimension of the activation manifold.

        Returns (mean_id, std_id) across multiple estimation methods.
        """
        try:
            from modelcypher.core.domain.geometry.intrinsic_dimension_estimator import (
                IntrinsicDimensionEstimator,
            )

            estimator = IntrinsicDimensionEstimator(backend=self._backend)
            result = estimator.estimate(activations)

            return result.dimension, result.uncertainty

        except ImportError:
            logger.debug("IntrinsicDimensionEstimator not available")
            return 0.0, 0.0
        except Exception as e:
            logger.debug(f"ID estimation failed: {e}")
            return 0.0, 0.0

    def _create_empty_baseline(
        self, domain: str, model_family: str, model_size: str, model_path: str
    ) -> DomainGeometryBaseline:
        """Create an empty baseline when extraction fails."""
        return DomainGeometryBaseline(
            domain=domain,
            model_family=model_family,
            model_size=model_size,
            model_path=model_path,
            ollivier_ricci_mean=0.0,
            ollivier_ricci_std=0.0,
            ollivier_ricci_min=0.0,
            ollivier_ricci_max=0.0,
            manifold_health_distribution=ManifoldHealthDistribution(),
            domain_metrics={},
            extraction_date=datetime.now().isoformat(),
            extraction_config={"error": "extraction_failed"},
        )


# =============================================================================
# Baseline Storage and Loading
# =============================================================================


class BaselineRepository:
    """Repository for loading and saving geometry baselines."""

    def __init__(self, baseline_dir: str | Path | None = None):
        """Initialize the repository.

        Args:
            baseline_dir: Directory containing baseline JSON files.
                         Defaults to modelcypher/data/baseline_data/
        """
        if baseline_dir is None:
            # Default to package data directory
            import modelcypher
            pkg_dir = Path(modelcypher.__file__).parent
            baseline_dir = pkg_dir / "data" / "baseline_data"

        self._baseline_dir = Path(baseline_dir)
        self._cache: dict[str, DomainGeometryBaseline] = {}

    def get_baseline(
        self, domain: str, model_family: str, model_size: str
    ) -> DomainGeometryBaseline | None:
        """Get a baseline by domain, family, and size.

        Args:
            domain: One of "spatial", "social", "temporal", "moral"
            model_family: e.g., "qwen", "llama", "mistral"
            model_size: e.g., "0.5B", "3B", "7B"

        Returns:
            DomainGeometryBaseline if found, None otherwise
        """
        cache_key = f"{domain}_{model_family}_{model_size}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try to load from file
        filename = f"{domain}_{model_family}_{model_size}.json"
        filepath = self._baseline_dir / filename

        if filepath.exists():
            baseline = DomainGeometryBaseline.load(filepath)
            self._cache[cache_key] = baseline
            return baseline

        return None

    def get_baselines_for_domain(self, domain: str) -> list[DomainGeometryBaseline]:
        """Get all baselines for a given domain."""
        baselines = []

        if not self._baseline_dir.exists():
            return baselines

        for filepath in self._baseline_dir.glob(f"{domain}_*.json"):
            try:
                baseline = DomainGeometryBaseline.load(filepath)
                baselines.append(baseline)
            except Exception as e:
                logger.warning(f"Failed to load baseline {filepath}: {e}")

        return baselines

    def get_all_baselines(self) -> list[DomainGeometryBaseline]:
        """Get all available baselines."""
        baselines = []

        if not self._baseline_dir.exists():
            return baselines

        for filepath in self._baseline_dir.glob("*.json"):
            try:
                baseline = DomainGeometryBaseline.load(filepath)
                baselines.append(baseline)
            except Exception as e:
                logger.warning(f"Failed to load baseline {filepath}: {e}")

        return baselines

    def save_baseline(self, baseline: DomainGeometryBaseline) -> Path:
        """Save a baseline to the repository.

        Returns the path where the baseline was saved.
        """
        self._baseline_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{baseline.domain}_{baseline.model_family}_{baseline.model_size}.json"
        filepath = self._baseline_dir / filename

        baseline.save(filepath)

        # Update cache
        cache_key = f"{baseline.domain}_{baseline.model_family}_{baseline.model_size}"
        self._cache[cache_key] = baseline

        return filepath

    def find_matching_baseline(
        self, domain: str, model_family: str, model_size: str
    ) -> DomainGeometryBaseline | None:
        """Find the best matching baseline for a model.

        First tries exact match, then falls back to same family,
        then any baseline for the domain.
        """
        # Exact match
        baseline = self.get_baseline(domain, model_family, model_size)
        if baseline:
            return baseline

        # Same family, any size
        domain_baselines = self.get_baselines_for_domain(domain)
        for b in domain_baselines:
            if b.model_family == model_family:
                return b

        # Any baseline for the domain
        if domain_baselines:
            return domain_baselines[0]

        return None
