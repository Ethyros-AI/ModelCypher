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

"""Geometry moral CLI commands.

Provides commands for probing moral geometry in LLM representations.

Commands:
    mc geometry moral anchors
    mc geometry moral probe-model --model <path> [--layer <int>]
"""

from __future__ import annotations

from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


def _get_moral_probes():
    """Get moral probes from unified atlas."""
    from modelcypher.core.domain.agents.unified_atlas import (
        AtlasSource,
        UnifiedAtlasInventory,
    )

    all_probes = UnifiedAtlasInventory.all_probes()
    return [p for p in all_probes if p.source == AtlasSource.MORAL_CONCEPT]


@app.command("anchors")
def moral_anchors(ctx: typer.Context):
    """List all moral concept anchors used for ethical structure probing."""
    from modelcypher.core.use_cases.atlas_bootstrap import (
        register_default_atlas_inventories,
    )

    context = _context(ctx)
    register_default_atlas_inventories()

    anchors = _get_moral_probes()

    if context.output_format == "text":
        lines = ["MORAL CONCEPT ANCHORS", f"Total: {len(anchors)}", ""]
        for anchor in anchors:
            lines.append(f"  {anchor.name}")
            if anchor.support_texts:
                lines.append(f"    Probe: {anchor.support_texts[0][:60]}...")
            lines.append("")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    payload = {
        "anchors": [
            {
                "name": a.name,
                "source": a.source.value,
                "support_texts": a.support_texts,
            }
            for a in anchors
        ],
        "count": len(anchors),
    }
    write_output(payload, context.output_format, context.pretty)


@app.command("probe-model")
def moral_probe_model(
    ctx: typer.Context,
    model: Path = typer.Option(..., "--model", "-m", help="Path to model directory"),
    layer: int = typer.Option(-1, "--layer", "-l", help="Layer to probe (-1 for last)"),
):
    """
    Probe a model for moral geometry structure.

    Extracts activations for moral concept anchors and computes:
    - Intrinsic dimension of moral subspace
    - Mean pairwise cosine similarity
    """
    from modelcypher.adapters.hf_hub import HfHubAdapter
    from modelcypher.cli.commands.geometry.helpers import (
        extract_anchor_activations,
        resolve_model_backbone,
    )
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
    from modelcypher.core.domain.geometry.riemannian_utils import geodesic_pairwise_metrics
    from modelcypher.core.use_cases.atlas_bootstrap import (
        register_default_atlas_inventories,
    )

    context = _context(ctx)
    register_default_atlas_inventories()
    backend = get_default_backend()

    # Load model
    adapter = HfHubAdapter()
    try:
        model_obj, tokenizer = adapter.load_model_and_tokenizer(str(model))
    except Exception as e:
        write_error(
            ErrorDetail(
                code="MC-4001",
                message="Failed to load model",
                detail=str(e),
            ),
            context.output_format,
        )
        raise typer.Exit(1)

    # Resolve backbone
    backbone = resolve_model_backbone(model_obj)
    if backbone is None:
        write_error(
            ErrorDetail(
                code="MC-4002",
                message="Failed to resolve model backbone",
                detail="Could not find embed_tokens/layers in model structure",
            ),
            context.output_format,
        )
        raise typer.Exit(1)

    embed_tokens, layers, norm = backbone

    # Get probes
    probes = _get_moral_probes()
    if not probes:
        write_error(
            ErrorDetail(
                code="MC-4003",
                message="No moral probes found",
                detail="UnifiedAtlasInventory has no MORAL_CONCEPT probes",
            ),
            context.output_format,
        )
        raise typer.Exit(1)

    # Create anchor objects for extraction
    class ProbeAnchor:
        def __init__(self, probe):
            self.name = probe.name
            self.prompt = probe.support_texts[0] if probe.support_texts else probe.name

    anchors = [ProbeAnchor(p) for p in probes]

    # Extract activations
    activations = extract_anchor_activations(
        anchors=anchors,
        tokenizer=tokenizer,
        embed_tokens=embed_tokens,
        layers=layers,
        norm=norm,
        target_layer=layer,
        backend=backend,
        prompt_attr="prompt",
        name_attr="name",
    )

    if len(activations) < 5:
        write_error(
            ErrorDetail(
                code="MC-4004",
                message="Insufficient activations extracted",
                detail=f"Got {len(activations)}, need at least 5",
            ),
            context.output_format,
        )
        raise typer.Exit(1)

    # Stack into matrix
    names = list(activations.keys())
    vectors = [activations[n] for n in names]
    matrix = backend.stack(vectors, axis=0)
    backend.eval(matrix)

    # Compute intrinsic dimension
    id_computer = IntrinsicDimension(backend)
    try:
        id_result = id_computer.compute(matrix)
        intrinsic_dim = id_result.intrinsic_dimension
        id_ci = (id_result.ci.lower, id_result.ci.upper)
    except Exception:
        intrinsic_dim = float("nan")
        id_ci = (float("nan"), float("nan"))

    # Compute mean pairwise similarity
    cos_matrix, _ = geodesic_pairwise_metrics(matrix, matrix, backend)
    backend.eval(cos_matrix)

    n = len(names)
    total_sim = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_sim += float(backend.to_scalar(cos_matrix[i, j]))
            count += 1

    mean_similarity = total_sim / count if count > 0 else 0.0

    if context.output_format == "text":
        lines = [
            "MORAL CONCEPT GEOMETRY",
            f"Model: {model}",
            f"Layer: {layer}",
            f"Probes extracted: {len(activations)}",
            "",
            f"INTRINSIC DIMENSION: {intrinsic_dim:.2f}",
            f"  95% CI: [{id_ci[0]:.2f}, {id_ci[1]:.2f}]",
            "",
            f"MEAN PAIRWISE SIMILARITY: {mean_similarity:.4f}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    payload = {
        "model_path": str(model),
        "layer": layer,
        "probes_extracted": len(activations),
        "intrinsic_dimension": {
            "value": intrinsic_dim,
            "ci_lower": id_ci[0],
            "ci_upper": id_ci[1],
        },
        "mean_pairwise_similarity": mean_similarity,
    }
    write_output(payload, context.output_format, context.pretty)
