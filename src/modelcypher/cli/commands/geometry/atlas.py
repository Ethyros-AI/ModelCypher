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

"""Unified atlas CLI commands."""

from __future__ import annotations

import logging

import typer

from modelcypher.cli.commands.geometry.helpers import (
    forward_through_backbone,
    resolve_model_backbone,
)
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output
from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
from modelcypher.core.domain.geometry.concept_dimensionality import (
    ConceptDimensionalityAnalyzer,
    ConceptDimensionalityReport,
    ConceptDimensionalityStudy,
)
from modelcypher.core.domain.geometry.riemannian_utils import frechet_mean
from modelcypher.core.support.array_utils import array_to_list

app = typer.Typer(no_args_is_help=True)
logger = logging.getLogger(__name__)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


class BackboneActivationProvider:
    def __init__(
        self,
        tokenizer,
        embed_tokens,
        layers,
        norm,
        backend,
        frechet_k_neighbors: int | None = None,
        frechet_max_k_neighbors: int | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._embed_tokens = embed_tokens
        self._layers = layers
        self._norm = norm
        self._backend = backend
        self._frechet_k_neighbors = frechet_k_neighbors
        self._frechet_max_k_neighbors = frechet_max_k_neighbors

    def get_activations(self, texts: list[str], layer: int) -> list[list[float]]:
        activations = []
        pending = []

        for text in texts:
            if not text:
                continue
            try:
                tokens = self._tokenizer.encode(text)
                if not tokens:
                    continue
                input_ids = self._backend.array([tokens])
                hidden = forward_through_backbone(
                    input_ids,
                    self._embed_tokens,
                    self._layers,
                    self._norm,
                    target_layer=layer,
                    backend=self._backend,
                )
                mean = frechet_mean(
                    hidden[0],
                    backend=self._backend,
                    k_neighbors=self._frechet_k_neighbors,
                    max_k_neighbors=self._frechet_max_k_neighbors,
                )
                self._backend.async_eval(mean)
                pending.append(mean)
                activations.append(mean)
            except Exception as exc:
                logger.debug("Activation failed for text '%s': %s", text, exc)
                continue

        if pending:
            self._backend.eval(*pending)

        return [array_to_list(self._backend, vec) for vec in activations]


def _report_payload(report: ConceptDimensionalityReport) -> dict:
    return {
        "layer": report.layer,
        "totalProbes": report.total_probes,
        "analyzedCount": report.analyzed_count,
        "skippedCount": report.skipped_count,
        "meanDimension": report.mean_dimension,
        "weightedMeanDimension": report.weighted_mean_dimension,
        "dimensionHistogram": report.dimension_histogram,
        "domainSummaries": [
            {
                "domain": summary.domain,
                "probeCount": summary.probe_count,
                "meanDimension": summary.mean_dimension,
                "dimensionHistogram": summary.dimension_histogram,
            }
            for summary in report.domain_summaries
        ],
        "results": [
            {
                "probeID": result.probe_id,
                "name": result.name,
                "source": result.source,
                "domain": result.domain,
                "category": result.category,
                "layer": result.layer,
                "supportTextCount": result.support_text_count,
                "sampleCount": result.sample_count,
                "usableCount": result.usable_count,
                "intrinsicDimension": result.intrinsic_dimension,
                "calibrationWeight": result.calibration_weight,
                "confidenceLower": result.ci_lower,
                "confidenceUpper": result.ci_upper,
            }
            for result in report.results
        ],
        "skipped": [
            {
                "probeID": item.probe_id,
                "name": item.name,
                "reason": item.reason,
                "supportTextCount": item.support_text_count,
                "calibrationWeight": item.calibration_weight,
                "activationCount": item.activation_count,
                "invalidCounts": item.invalid_counts,
            }
            for item in report.skipped
        ],
    }


@app.command("dimensionality")
def atlas_dimensionality(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to the model directory"),
) -> None:
    """Measure intrinsic dimension for UnifiedAtlas probes at a model layer."""
    context = _context(ctx)

    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.core.domain._backend import get_default_backend

    model, tokenizer = load_model_for_training(model_path)
    model_type = getattr(model, "model_type", "unknown")
    resolved = resolve_model_backbone(model, model_type)
    if not resolved:
        raise typer.BadParameter("Could not resolve model architecture.")

    embed_tokens, layers, norm = resolved
    num_layers = len(layers)
    target_layer = num_layers - 1

    probes = UnifiedAtlasInventory.all_probes()
    calibration_weights = {}

    backend = get_default_backend()
    provider = BackboneActivationProvider(
        tokenizer,
        embed_tokens,
        layers,
        norm,
        backend,
        frechet_k_neighbors=None,
        frechet_max_k_neighbors=None,
    )
    analyzer = ConceptDimensionalityAnalyzer(backend=backend)
    report = analyzer.analyze(
        probes=probes,
        activation_provider=provider,
        layer=target_layer,
        calibration_weights=calibration_weights,
    )

    payload = {
        "_schema": "mc.geometry.atlas.dimensionality.v1",
        "modelPath": model_path,
        **_report_payload(report),
    }

    if context.output_format == "text":
        histogram = report.dimension_histogram
        lines = [
            "CONCEPT DIMENSIONALITY (UNIFIED ATLAS)",
            f"Model: {model_path}",
            f"Layer: {target_layer}",
            f"Analyzed: {report.analyzed_count}/{report.total_probes} "
            f"(skipped {report.skipped_count})",
        ]
        if report.mean_dimension is not None:
            lines.append(f"Mean Dimension: {report.mean_dimension:.2f}")
        if report.weighted_mean_dimension is not None:
            lines.append(f"Weighted Mean: {report.weighted_mean_dimension:.2f}")
        lines.extend(
            [
                "Histogram: "
                f"1D {histogram.get('1D', 0)} | "
                f"2D {histogram.get('2D', 0)} | "
                f"3D {histogram.get('3D', 0)} | "
                f"4D+ {histogram.get('4D+', 0)}",
                "",
                "Domain Summaries:",
            ]
        )
        for summary in report.domain_summaries:
            mean_dim = summary.mean_dimension
            mean_text = f"{mean_dim:.2f}" if mean_dim is not None else "n/a"
            lines.append(
                f"  {summary.domain}: mean {mean_text}, "
                f"1D {summary.dimension_histogram.get('1D', 0)}, "
                f"2D {summary.dimension_histogram.get('2D', 0)}, "
                f"3D {summary.dimension_histogram.get('3D', 0)}, "
                f"4D+ {summary.dimension_histogram.get('4D+', 0)}"
            )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("dimensionality-study")
def atlas_dimensionality_study(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to the model directory"),
    include_results: bool = typer.Option(
        False,
        "--include-results/--summary-only",
        help="Include per-probe results for each layer",
    ),
) -> None:
    """Run atlas dimensionality across all layers and summarize structure."""
    context = _context(ctx)

    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.core.domain._backend import get_default_backend

    model, tokenizer = load_model_for_training(model_path)
    model_type = getattr(model, "model_type", "unknown")
    resolved = resolve_model_backbone(model, model_type)
    if not resolved:
        raise typer.BadParameter("Could not resolve model architecture.")

    embed_tokens, layers_module, norm = resolved
    num_layers = len(layers_module)
    resolved_layers = list(range(num_layers))

    probes = UnifiedAtlasInventory.all_probes()
    calibration_weights = {}

    backend = get_default_backend()
    provider = BackboneActivationProvider(
        tokenizer,
        embed_tokens,
        layers_module,
        norm,
        backend,
        frechet_k_neighbors=None,
        frechet_max_k_neighbors=None,
    )
    analyzer = ConceptDimensionalityAnalyzer(backend=backend)

    reports: list[ConceptDimensionalityReport] = []
    for layer_idx in resolved_layers:
        report = analyzer.analyze(
            probes=probes,
            activation_provider=provider,
            layer=layer_idx,
            calibration_weights=calibration_weights,
        )
        reports.append(report)

    study = ConceptDimensionalityStudy.summarize(reports)

    payload = {
        "_schema": "mc.geometry.atlas.dimensionality_study.v1",
        "modelPath": model_path,
        "layers": study.layers,
        "bottleneckLayer": study.bottleneck_layer,
        "bottleneckMeanDimension": study.bottleneck_mean_dimension,
        "endpointMeanDimension": study.endpoint_mean_dimension,
        "collapseRatio": study.collapse_ratio,
        "meanDomainRankCorrelation": study.mean_domain_rank_correlation,
        "domainRankCorrelations": [
            {
                "layerA": item.layer_a,
                "layerB": item.layer_b,
                "domainCount": item.domain_count,
                "spearman": item.spearman,
            }
            for item in study.domain_rank_correlations
        ],
        "layerSummaries": [
            {
                "layer": summary.layer,
                "meanDimension": summary.mean_dimension,
                "dimensionHistogram": summary.dimension_histogram,
                "domainMeanDimensions": summary.domain_mean_dimensions,
                "domainRank": summary.domain_rank,
            }
            for summary in study.layer_summaries
        ],
        "layerReports": [_report_payload(report) for report in reports]
        if include_results
        else None,
    }

    if context.output_format == "text":
        lines = [
            "ATLAS DIMENSIONALITY STUDY",
            f"Model: {model_path}",
            f"Layers: {', '.join(str(layer) for layer in study.layers)}",
        ]
        if study.bottleneck_layer is not None:
            lines.append(f"Bottleneck Layer: {study.bottleneck_layer}")
        if study.bottleneck_mean_dimension is not None:
            lines.append(f"Bottleneck Mean Dimension: {study.bottleneck_mean_dimension:.3f}")
        if study.collapse_ratio is not None:
            lines.append(f"Collapse Ratio: {study.collapse_ratio:.3f}")
        if study.mean_domain_rank_correlation is not None:
            lines.append(
                f"Mean Domain Rank Correlation: {study.mean_domain_rank_correlation:.3f}"
            )
        lines.append("")
        lines.append("Layer Summaries:")
        for summary in study.layer_summaries:
            mean_dim = summary.mean_dimension
            mean_text = f"{mean_dim:.3f}" if mean_dim is not None else "n/a"
            hist = summary.dimension_histogram
            lines.append(
                f"  L{summary.layer}: mean {mean_text} | "
                f"1D {hist.get('1D', 0)}, 2D {hist.get('2D', 0)}, "
                f"3D {hist.get('3D', 0)}, 4D+ {hist.get('4D+', 0)}"
            )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
