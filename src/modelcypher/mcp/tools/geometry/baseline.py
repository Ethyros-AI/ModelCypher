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

"""Geometry baseline MCP tools.

Contains tools for domain geometry validation:
- Baseline listing
- Baseline extraction
- Baseline validation
- Baseline comparison
"""

from __future__ import annotations

from ..common import (
    READ_ONLY_ANNOTATIONS,
    ServiceContext,
    require_existing_directory,
)


def register_geometry_baseline_tools(ctx: ServiceContext) -> None:
    """Register geometry baseline tools for domain geometry validation."""
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    if "mc_geometry_baseline_list" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_baseline_list(domain: str | None = None) -> dict:
            """
            List available domain geometry baselines.

            Args:
                domain: Optional domain filter (spatial, social, temporal, moral)

            Returns:
                List of available baselines with their metadata
            """
            from modelcypher.core.domain.geometry.domain_geometry_baselines import (
                BaselineRepository,
            )

            repo = BaselineRepository()
            if domain:
                baselines = repo.get_baselines_for_domain(domain)
            else:
                baselines = repo.get_all_baselines()

            return {
                "_schema": "mc.geometry.baseline.list.v1",
                "baselines": [
                    {
                        "domain": b.domain,
                        "modelFamily": b.model_family,
                        "modelSize": b.model_size,
                        "ollivierRicciMean": b.ollivier_ricci_mean,
                        "extractionDate": b.extraction_date,
                    }
                    for b in baselines
                ],
            }

    if "mc_geometry_baseline_extract" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_baseline_extract(
            modelPath: str,
            domain: str = "spatial",
            layer: int = -1,
            kNeighbors: int = 10,
        ) -> dict:
            """
            Extract geometry baseline from a reference model.

            Uses Ollivier-Ricci curvature and domain-specific analyzers to create
            an empirical baseline for reference LLM geometry.

            Args:
                modelPath: Path to the model directory
                domain: Domain to extract (spatial, social, temporal, moral)
                layer: Layer to analyze (-1 for all layers sampled)
                kNeighbors: k for k-NN graph in Ollivier-Ricci computation

            Returns:
                Extracted baseline with curvature and domain metrics
            """
            from modelcypher.adapters.mlx_model_loader import MLXModelLoader
            from modelcypher.core.domain.geometry.domain_geometry_baselines import (
                BaselineRepository,
                DomainGeometryBaselineExtractor,
            )

            model_path = require_existing_directory(modelPath)
            valid_domains = ["spatial", "social", "temporal", "moral"]
            if domain.lower() not in valid_domains:
                raise ValueError(f"Invalid domain: {domain}. Valid: {', '.join(valid_domains)}")

            model_loader = MLXModelLoader()
            extractor = DomainGeometryBaselineExtractor(model_loader=model_loader)
            baseline = extractor.extract_baseline(
                model_path=model_path,
                domain=domain.lower(),
                layers=[layer] if layer != -1 else None,
                k_neighbors=kNeighbors,
            )

            # Save baseline
            repo = BaselineRepository()
            saved_path = repo.save_baseline(baseline)

            return {
                "_schema": "mc.geometry.baseline.extract.v1",
                "domain": baseline.domain,
                "modelFamily": baseline.model_family,
                "modelSize": baseline.model_size,
                "ollivierRicciMean": baseline.ollivier_ricci_mean,
                "ollivierRicciStd": baseline.ollivier_ricci_std,
                "manifoldHealthDistribution": baseline.manifold_health_distribution.to_dict(),
                "intrinsicDimension": baseline.intrinsic_dimension_mean,
                "domainMetrics": baseline.domain_metrics,
                "savedPath": str(saved_path),
            }

    if "mc_geometry_baseline_validate" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_baseline_validate(
            modelPath: str,
            domains: list[str] | None = None,
            layer: int = -1,
        ) -> dict:
            """
            Validate model geometry against established baselines.

            Compares model's Ollivier-Ricci curvature and domain metrics against
            reference baselines. Useful for post-merge validation and baseline-relative checks.

            Args:
                modelPath: Path to the model to validate
                domains: List of domains to validate (default: all)
                layer: Layer to analyze (-1 for all layers sampled)

            Returns:
                Validation results with baseline-relative deltas
            """
            from modelcypher.adapters.mlx_model_loader import MLXModelLoader
            from modelcypher.core.domain.geometry.domain_geometry_validator import (
                DomainGeometryValidator,
            )

            model_path = require_existing_directory(modelPath)

            model_loader = MLXModelLoader()
            validator = DomainGeometryValidator(model_loader=model_loader)
            results = validator.validate_model(
                model_path=model_path,
                domains=domains,
                layer=layer,
            )

            return {
                "_schema": "mc.geometry.baseline.validate.v1",
                "modelPath": model_path,
                "results": [
                    {
                        "domain": r.domain,
                        "baselineFound": r.baseline_found,
                        "baselineModel": r.baseline_model,
                        "currentModel": r.current_model,
                        "missingMetrics": r.missing_metrics,
                        "notes": r.notes,
                        "metrics": {
                            name: {
                                "current": metric.current,
                                "baseline": metric.baseline,
                                "baselineStd": metric.baseline_std,
                                "delta": metric.delta,
                                "relativeDelta": metric.relative_delta,
                                "zScore": metric.z_score,
                                "percentile": metric.percentile,
                            }
                            for name, metric in r.metrics.items()
                        },
                    }
                    for r in results
                ],
            }

    if "mc_geometry_baseline_compare" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_baseline_compare(
            model1Path: str,
            model2Path: str,
            domain: str = "spatial",
            layer: int = -1,
        ) -> dict:
            """
            Compare geometry profiles of two models.

            Extracts baselines from both models and computes divergence metrics.
            Useful for pre-merge compatibility assessment.

            Args:
                model1Path: Path to first model
                model2Path: Path to second model
                domain: Domain to compare (spatial, social, temporal, moral)
                layer: Layer to analyze (-1 for all layers sampled)

            Returns:
                Comparison results with divergence metrics
            """
            from modelcypher.adapters.mlx_model_loader import MLXModelLoader
            from modelcypher.core.domain.geometry.domain_geometry_baselines import (
                DomainGeometryBaselineExtractor,
            )

            model1_path = require_existing_directory(model1Path)
            model2_path = require_existing_directory(model2Path)

            valid_domains = ["spatial", "social", "temporal", "moral"]
            if domain.lower() not in valid_domains:
                raise ValueError(f"Invalid domain: {domain}. Valid: {', '.join(valid_domains)}")

            model_loader = MLXModelLoader()
            extractor = DomainGeometryBaselineExtractor(model_loader=model_loader)
            baseline1 = extractor.extract_baseline(
                model_path=model1_path,
                domain=domain.lower(),
                layers=[layer] if layer != -1 else None,
            )
            baseline2 = extractor.extract_baseline(
                model_path=model2_path,
                domain=domain.lower(),
                layers=[layer] if layer != -1 else None,
            )

            # Compute divergence
            ricci_divergence = abs(baseline1.ollivier_ricci_mean - baseline2.ollivier_ricci_mean)
            id_divergence = abs(baseline1.intrinsic_dimension_mean - baseline2.intrinsic_dimension_mean)

            # Compute domain metric divergence
            domain_divergence = {}
            common_metrics = set(baseline1.domain_metrics.keys()) & set(baseline2.domain_metrics.keys())
            for metric in common_metrics:
                v1 = baseline1.domain_metrics[metric]
                v2 = baseline2.domain_metrics[metric]
                domain_divergence[metric] = abs(v1 - v2)

            return {
                "_schema": "mc.geometry.baseline.compare.v1",
                "domain": domain,
                "model1": {
                    "path": model1_path,
                    "family": baseline1.model_family,
                    "size": baseline1.model_size,
                    "ollivierRicciMean": baseline1.ollivier_ricci_mean,
                    "intrinsicDimension": baseline1.intrinsic_dimension_mean,
                },
                "model2": {
                    "path": model2_path,
                    "family": baseline2.model_family,
                    "size": baseline2.model_size,
                    "ollivierRicciMean": baseline2.ollivier_ricci_mean,
                    "intrinsicDimension": baseline2.intrinsic_dimension_mean,
                },
                "divergence": {
                    "ollivierRicci": ricci_divergence,
                    "intrinsicDimension": id_divergence,
                    "domainMetrics": domain_divergence,
                },
            }
