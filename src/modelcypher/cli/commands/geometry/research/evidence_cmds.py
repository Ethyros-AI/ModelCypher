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

from dataclasses import asdict
from pathlib import Path

import typer

from modelcypher.cli.output import write_error, write_output

from .common import cleanup_memory, get_context, load_model_and_provider


def _build_domain_alignment_inputs(
    model_path: Path,
    prompts: list[str],
    layer: int | None,
):
    model, _tokenizer, backend, provider, num_layers = load_model_and_provider(str(model_path))

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


def register(app: typer.Typer) -> None:
    @app.command("evidence")
    def evidence(
        ctx: typer.Context,
        alignment_dim: int = typer.Option(16, "--alignment-dim", help="Synthetic alignment dimension"),
        alignment_samples: int | None = typer.Option(
            None, "--alignment-samples", help="Synthetic alignment sample count"
        ),
        radius: float = typer.Option(1.0, "--radius", help="Radius for analytic manifolds"),
        seed: int = typer.Option(0, "--seed", help="Random seed"),
        model_a: Path | None = typer.Option(
            None, "--model-a", help="Optional model A for domain alignment"
        ),
        model_b: Path | None = typer.Option(
            None, "--model-b", help="Optional model B for domain alignment"
        ),
        layer: int | None = typer.Option(
            None, "--layer", help="Layer index for domain alignment (defaults to middle layer)"
        ),
        probe_count: int = typer.Option(
            24, "--probe-count", help="Max number of probes for domain alignment"
        ),
        output: Path | None = typer.Option(
            None, "--output-file", help="Path to save evidence JSON"
        ),
    ) -> None:
        """Run evidence suite that quantifies the stated limitations."""
        context = get_context(ctx)

        try:
            from modelcypher.core.domain.geometry.evidence_suite import run_synthetic_evidence

            domain_source = None
            domain_target = None
            domain_probes = None
            domain_meta = None
            backend = None

            if (model_a is None) ^ (model_b is None):
                raise ValueError("Provide both --model-a and --model-b for domain alignment.")

            if model_a is not None and model_b is not None:
                from modelcypher.core.domain.atlas.unified_atlas import UnifiedAtlasInventory

                all_probes = UnifiedAtlasInventory.all_probes()
                if not all_probes:
                    raise ValueError("No atlas probes available for domain alignment.")

                if probe_count <= 0:
                    raise ValueError("probe-count must be positive.")

                if probe_count >= len(all_probes):
                    selected = all_probes
                else:
                    step = max(1, len(all_probes) // probe_count)
                    selected = all_probes[::step][:probe_count]

                prompts = []
                for probe in selected:
                    if probe.support_texts:
                        prompts.append(probe.support_texts[0])
                    else:
                        prompts.append(probe.name)

                domain_source, backend, layer_a = _build_domain_alignment_inputs(
                    model_a, prompts, layer
                )
                cleanup_memory()
                domain_target, backend_b, layer_b = _build_domain_alignment_inputs(
                    model_b, prompts, layer
                )
                cleanup_memory()

                if backend is None:
                    backend = backend_b

                domain_probes = selected
                domain_meta = {
                    "modelA": str(model_a),
                    "modelB": str(model_b),
                    "layerA": layer_a,
                    "layerB": layer_b,
                    "probeCount": len(selected),
                }

            report = run_synthetic_evidence(
                alignment_dim=alignment_dim,
                alignment_samples=alignment_samples,
                radius=radius,
                seed=seed,
                domain_source=domain_source,
                domain_target=domain_target,
                domain_probes=domain_probes,
                backend=backend,
            )

            payload = {
                "_schema": "mc.geometry.research.evidence.v1",
                "evidence": asdict(report),
            }
            if domain_meta is not None:
                payload["domainAlignmentMeta"] = domain_meta

            if output:
                from modelcypher.utils.json import dump_json

                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(dump_json(payload, pretty=True))

            if context.output_format == "text":
                lines = [
                    "",
                    "=" * 60,
                    "GEOMETRY EVIDENCE REPORT",
                    "=" * 60,
                    f"Alignment train CKA: {report.alignment_generalization.train_cka:.6f}",
                    f"Alignment holdout CKA: {report.alignment_generalization.holdout_cka:.6f}",
                    "",
                    "Geodesic accuracy:",
                    f"  mean abs (small): {report.geodesic_convergence.small.mean_abs_error:.6f}",
                    f"  mean abs (large): {report.geodesic_convergence.large.mean_abs_error:.6f}",
                    "",
                    "Curvature accuracy:",
                    f"  mean abs (small): {report.curvature_convergence.small.mean_abs_error:.6f}",
                    f"  mean abs (large): {report.curvature_convergence.large.mean_abs_error:.6f}",
                    "",
                    "Causal intervention:",
                    f"  core mean shift: {report.causal_intervention.core_mean_shift:.6f}",
                    f"  boundary max diff: {report.causal_intervention.boundary_max_relative_diff:.6f}",
                ]
                if report.domain_alignment is not None:
                    lines.append("")
                    lines.append("Domain alignment:")
                    for domain, metrics in report.domain_alignment.domain_reports.items():
                        lines.append(
                            f"  {domain}: train={metrics.train_cka:.6f}, holdout={metrics.holdout_cka:.6f}"
                        )
                write_output("\n".join(lines), context.output_format, context.pretty)
                return

            write_output(payload, context.output_format, context.pretty)

        except Exception as exc:
            write_error(f"Evidence suite failed: {exc}", context.output_format)
            raise typer.Exit(1) from exc
