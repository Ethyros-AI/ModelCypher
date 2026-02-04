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

from __future__ import annotations

import logging
from pathlib import Path

import typer

from modelcypher.cli.output import write_error, write_output
from modelcypher.core.domain.atlas.unified_atlas import UnifiedAtlasInventory
from modelcypher.core.domain.geometry.atlas_protocols import enum_key
from modelcypher.core.domain.geometry.shared_manifold import (
    compute_alignment_transform,
    compute_diff_basis,
    compute_residual_matrix,
    compute_shared_manifold_report,
    derive_alignment_indices,
)

from .common import cleanup_memory, get_context, load_model_and_provider

logger = logging.getLogger(__name__)


def _select_probes(probe_count: int):
    probes = UnifiedAtlasInventory.all_probes()
    if not probes:
        raise ValueError("No atlas probes available.")
    if probe_count <= 0:
        raise ValueError("probe-count must be positive.")
    if probe_count >= len(probes):
        return probes
    step = max(1, len(probes) // probe_count)
    return probes[::step][:probe_count]


def _collect_activations(model_path: str, prompts: list[str], layer: int | None):
    _model, _tokenizer, backend, provider, num_layers = load_model_and_provider(model_path)

    if layer is None:
        layer_idx = max(0, num_layers // 2)
    else:
        if layer < 0 or layer >= num_layers:
            raise ValueError(f"Layer {layer} out of range for model with {num_layers} layers.")
        layer_idx = layer

    activations = provider.get_activations(prompts, layer_idx)
    if len(activations) != len(prompts):
        raise ValueError(
            "Activation collection returned mismatched sample counts. "
            "Reduce probe count or inspect probes."
        )

    arr = backend.array(activations)
    backend.eval(arr)
    return arr, backend, layer_idx


def _domain_summary(residuals):
    stats: dict[str, dict[str, float]] = {}
    for item in residuals:
        domain = item.domain or "unknown"
        bucket = stats.setdefault(domain, {"count": 0.0, "sum": 0.0, "sum_rel": 0.0})
        bucket["count"] += 1.0
        bucket["sum"] += float(item.residual_norm)
        bucket["sum_rel"] += float(item.residual_relative)
    summary = []
    for domain, bucket in stats.items():
        count = int(bucket["count"])
        mean = bucket["sum"] / count if count else 0.0
        mean_rel = bucket["sum_rel"] / count if count else 0.0
        summary.append(
            {
                "domain": domain,
                "count": count,
                "meanResidual": mean,
                "meanRelativeResidual": mean_rel,
            }
        )
    summary.sort(key=lambda item: item["meanResidual"], reverse=True)
    return summary


def _residual_payload(residuals, probe_lookup: dict[str, object], limit: int, reverse: bool):
    if limit <= 0:
        return []
    ordered = sorted(residuals, key=lambda item: item.residual_norm, reverse=reverse)[:limit]
    payload = []
    for item in ordered:
        probe = probe_lookup.get(item.probe_id)
        payload.append(
            {
                "probeID": item.probe_id,
                "name": getattr(probe, "name", item.probe_id),
                "domain": item.domain,
                "residualNorm": item.residual_norm,
                "residualRelative": item.residual_relative,
            }
        )
    return payload


def register(app: typer.Typer) -> None:
    @app.command("shared-manifold")
    def shared_manifold(
        ctx: typer.Context,
        source_path: str = typer.Argument(..., help="Path to source model directory"),
        target_path: str = typer.Argument(..., help="Path to target model directory"),
        probe_count: int = typer.Option(
            64, "--probe-count", help="Max number of probes to evaluate"
        ),
        layer: int | None = typer.Option(
            None, "--layer", help="Layer index (defaults to middle layer)"
        ),
        top_k: int = typer.Option(
            10, "--top-k", help="Number of probes to list for top/bottom residuals"
        ),
        output: Path | None = typer.Option(
            None, "--output-file", help="Path to save shared-manifold report JSON"
        ),
    ) -> None:
        """Measure shared-manifold coverage and model-specific residuals."""
        context = get_context(ctx)

        try:
            probes = _select_probes(probe_count)
            prompts = [
                (probe.support_texts[0] if probe.support_texts else probe.name)
                for probe in probes
            ]
            probe_ids = [probe.probe_id for probe in probes]
            probe_domains = [enum_key(probe.domain) for probe in probes]
            probe_lookup = {probe.probe_id: probe for probe in probes}

            logger.info("Collecting activations for source model...")
            source_arr, backend_a, layer_a = _collect_activations(source_path, prompts, layer)
            cleanup_memory()

            logger.info("Collecting activations for target model...")
            target_arr, backend_b, layer_b = _collect_activations(target_path, prompts, layer)
            cleanup_memory()

            if type(backend_a) is not type(backend_b):
                raise ValueError("Source and target backends must match.")

            train_idx, holdout_idx = derive_alignment_indices(source_arr, target_arr, backend_a)
            report = compute_shared_manifold_report(
                source_arr,
                target_arr,
                probe_ids,
                probe_domains=probe_domains,
                train_indices=train_idx,
                holdout_indices=holdout_idx,
                backend=backend_a,
            )

            transform = compute_alignment_transform(source_arr, target_arr, train_idx, backend_a)
            residuals = compute_residual_matrix(source_arr, target_arr, transform, backend_a)
            diff_basis = compute_diff_basis(residuals, backend_a)
            singular_values = backend_a.tolist(diff_basis.singular_values)
            if not isinstance(singular_values, list):
                singular_values = [singular_values]

            payload = {
                "_schema": "mc.geometry.research.shared_manifold.v1",
                "sourcePath": source_path,
                "targetPath": target_path,
                "layerA": layer_a,
                "layerB": layer_b,
                "probeCount": len(probes),
                "alignment": {
                    "trainSamples": report.alignment.train_samples,
                    "holdoutSamples": report.alignment.holdout_samples,
                    "trainCKA": report.alignment.train_cka,
                    "holdoutCKA": report.alignment.holdout_cka,
                    "rawHoldoutCKA": report.alignment.raw_holdout_cka,
                    "alignmentGain": report.alignment.alignment_gain,
                    "coverageRatio": report.alignment.coverage_ratio,
                },
                "residualSummary": {
                    "mean": report.residual_mean,
                    "max": report.residual_max,
                    "std": report.residual_std,
                },
                "domainSummary": _domain_summary(report.residuals),
                "topResiduals": _residual_payload(
                    report.residuals, probe_lookup, top_k, reverse=True
                ),
                "lowestResiduals": _residual_payload(
                    report.residuals, probe_lookup, top_k, reverse=False
                ),
                "diffBasis": {
                    "rank": diff_basis.rank,
                    "explainedVarianceRatio": diff_basis.explained_variance_ratio,
                    "singularValues": [float(x) for x in singular_values],
                },
            }

            if output:
                from modelcypher.utils.json import dump_json

                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(dump_json(payload, pretty=True))

            if context.output_format == "text":
                lines = [
                    "SHARED MANIFOLD REPORT",
                    f"Source: {source_path}",
                    f"Target: {target_path}",
                    f"Layer A: {layer_a}",
                    f"Layer B: {layer_b}",
                    f"Probes: {len(probes)}",
                    "",
                    "Alignment:",
                    f"  train CKA: {report.alignment.train_cka:.6f}",
                    f"  holdout CKA: {report.alignment.holdout_cka:.6f}",
                    f"  raw holdout CKA: {report.alignment.raw_holdout_cka:.6f}",
                    f"  coverage ratio: {report.alignment.coverage_ratio:.6f}",
                    "",
                    "Residuals:",
                    f"  mean: {report.residual_mean:.6f}",
                    f"  max: {report.residual_max:.6f}",
                    f"  std: {report.residual_std:.6f}",
                    "",
                    "Diff basis:",
                    f"  rank: {diff_basis.rank}",
                    f"  explained variance ratio: {diff_basis.explained_variance_ratio:.4f}",
                ]
                if top_k > 0 and report.residuals:
                    lines.append("")
                    lines.append("Top residual probes:")
                    for item in payload["topResiduals"]:
                        lines.append(
                            f"  [{item['domain']}] {item['name']}: "
                            f"{item['residualNorm']:.6f}"
                        )
                write_output("\n".join(lines), context.output_format, context.pretty)
                return

            write_output(payload, context.output_format, context.pretty)

        except Exception as exc:
            write_error(f"Shared-manifold report failed: {exc}", context.output_format)
            raise typer.Exit(1) from exc
