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

"""Geometry semantic primes CLI commands.

Provides commands for probing semantic prime structure in LLM representations.

Semantic primes are the irreducible concepts from which all meaning is built
(Wierzbicka, 1972). They include: I, YOU, SOMEONE, SOMETHING, PEOPLE, BODY,
WORDS, THIS, THE SAME, OTHER, ONE, TWO, MUCH/MANY, LITTLE/FEW, etc.

Commands:
    mc geometry primes list
    mc geometry primes probe-model --model <path> [--layer <int>]
    mc geometry primes compare --model-a <path> --model-b <path>
"""

from __future__ import annotations

from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.cli.warnings import warn_network
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("list")
def primes_list(ctx: typer.Context):
    """List all semantic primes from the NSM inventory (English 2014)."""
    from modelcypher.core.domain.agents.semantic_primes import SemanticPrimeInventory

    context = _context(ctx)
    primes = SemanticPrimeInventory.english2014()

    if context.output_format == "text":
        lines = ["SEMANTIC PRIMES (NSM English 2014)", f"Total: {len(primes)}", ""]
        by_category: dict[str, list] = {}
        for prime in primes:
            cat = prime.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(prime)

        for category, cat_primes in sorted(by_category.items()):
            lines.append(f"  {category}:")
            for prime in cat_primes:
                exponents = ", ".join(prime.english_exponents[:3])
                lines.append(f"    {prime.id}: {exponents}")
            lines.append("")

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    payload = {
        "primes": [
            {
                "id": p.id,
                "category": p.category.value,
                "english_exponents": p.english_exponents,
                "canonical": p.canonical_english,
            }
            for p in primes
        ],
        "count": len(primes),
    }
    write_output(payload, context.output_format, context.pretty)


@app.command("probe-model")
def primes_probe_model(
    ctx: typer.Context,
    model: Path = typer.Option(..., "--model", "-m", help="Path to model directory"),
    layer: int = typer.Option(-1, "--layer", "-l", help="Layer to probe (-1 for last)"),
):
    """
    Probe a model for semantic prime geometry.

    Extracts activations for all semantic primes and computes:
    - Intrinsic dimension of prime subspace
    - Category clustering quality
    - Prime-to-prime similarity matrix
    """
    from modelcypher.adapters.hf_hub import HfHubAdapter
    from modelcypher.cli.commands.geometry.helpers import (
        extract_anchor_activations,
        resolve_model_backbone,
    )
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.agents.semantic_primes import SemanticPrimeInventory
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
    from modelcypher.core.support.array_utils import array_to_list

    context = _context(ctx)
    backend = get_default_backend()

    # Load model
    warn_network(context, "Loading models from Hugging Face Hub if not cached.")
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

    # Get primes as anchor objects
    primes = SemanticPrimeInventory.english2014()

    # Create anchor-like objects for extraction
    class PrimeAnchor:
        def __init__(self, prime):
            self.name = prime.id
            self.prompt = prime.canonical_english

    anchors = [PrimeAnchor(p) for p in primes]

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

    if len(activations) < 10:
        write_error(
            ErrorDetail(
                code="MC-4004",
                message="Insufficient activations extracted",
                detail=f"Got {len(activations)}, need at least 10",
            ),
            context.output_format,
        )
        raise typer.Exit(1)

    # Stack activations into matrix
    prime_ids = list(activations.keys())
    vectors = [activations[pid] for pid in prime_ids]
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

    # Compute mean pairwise similarity by category
    prime_lookup = {p.id: p for p in primes}
    category_sims: dict[str, list[float]] = {}

    from modelcypher.core.domain.geometry.riemannian_utils import geodesic_pairwise_metrics

    cos_matrix, _ = geodesic_pairwise_metrics(matrix, matrix, backend)
    backend.eval(cos_matrix)
    cos_list = array_to_list(backend, cos_matrix)

    for i, pid_i in enumerate(prime_ids):
        for j, pid_j in enumerate(prime_ids):
            if i >= j:
                continue
            prime_i = prime_lookup.get(pid_i)
            prime_j = prime_lookup.get(pid_j)
            if prime_i and prime_j and prime_i.category == prime_j.category:
                cat = prime_i.category.value
                if cat not in category_sims:
                    category_sims[cat] = []
                # cos_list is flattened, need to index correctly
                idx = i * len(prime_ids) + j
                if idx < len(cos_list):
                    category_sims[cat].append(cos_list[idx])

    # Compute mean within-category similarity
    mean_by_category = {
        cat: sum(sims) / len(sims) if sims else 0.0
        for cat, sims in category_sims.items()
    }
    overall_within = (
        sum(sum(sims) for sims in category_sims.values())
        / sum(len(sims) for sims in category_sims.values())
        if any(category_sims.values())
        else 0.0
    )

    if context.output_format == "text":
        lines = [
            "SEMANTIC PRIME GEOMETRY",
            f"Model: {model}",
            f"Layer: {layer}",
            f"Primes extracted: {len(activations)}",
            "",
            f"INTRINSIC DIMENSION: {intrinsic_dim:.2f} (95% CI: {id_ci[0]:.2f}-{id_ci[1]:.2f})",
            "",
            "WITHIN-CATEGORY SIMILARITY:",
            f"  Overall: {overall_within:.4f}",
        ]
        for cat, sim in sorted(mean_by_category.items()):
            lines.append(f"  {cat}: {sim:.4f}")

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    payload = {
        "model_path": str(model),
        "layer": layer,
        "primes_extracted": len(activations),
        "intrinsic_dimension": {
            "value": intrinsic_dim,
            "ci_lower": id_ci[0],
            "ci_upper": id_ci[1],
        },
        "within_category_similarity": {
            "overall": overall_within,
            "by_category": mean_by_category,
        },
    }
    write_output(payload, context.output_format, context.pretty)


@app.command("compare")
def primes_compare(
    ctx: typer.Context,
    model_a: Path = typer.Option(..., "--model-a", "-a", help="Path to first model"),
    model_b: Path = typer.Option(..., "--model-b", "-b", help="Path to second model"),
    layer: int = typer.Option(-1, "--layer", "-l", help="Layer to probe (-1 for last)"),
):
    """
    Compare semantic prime representations between two models.

    Computes CKA (Centered Kernel Alignment) between the prime subspaces
    of two models to measure geometric similarity.
    """
    from modelcypher.adapters.hf_hub import HfHubAdapter
    from modelcypher.cli.commands.geometry.helpers import (
        extract_anchor_activations,
        resolve_model_backbone,
    )
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.agents.semantic_primes import SemanticPrimeInventory
    from modelcypher.core.domain.geometry.cka import compute_cka

    context = _context(ctx)
    backend = get_default_backend()

    # Get primes
    primes = SemanticPrimeInventory.english2014()

    class PrimeAnchor:
        def __init__(self, prime):
            self.name = prime.id
            self.prompt = prime.canonical_english

    anchors = [PrimeAnchor(p) for p in primes]

    # Load and extract from model A
    warn_network(context, "Loading models from Hugging Face Hub if not cached.")
    adapter = HfHubAdapter()
    try:
        model_a_obj, tokenizer_a = adapter.load_model_and_tokenizer(str(model_a))
    except Exception as e:
        write_error(
            ErrorDetail(code="MC-4001", message="Failed to load model A", detail=str(e)),
            context.output_format,
        )
        raise typer.Exit(1)

    backbone_a = resolve_model_backbone(model_a_obj)
    if backbone_a is None:
        write_error(
            ErrorDetail(code="MC-4002", message="Failed to resolve model A backbone", detail=""),
            context.output_format,
        )
        raise typer.Exit(1)

    activations_a = extract_anchor_activations(
        anchors=anchors,
        tokenizer=tokenizer_a,
        embed_tokens=backbone_a[0],
        layers=backbone_a[1],
        norm=backbone_a[2],
        target_layer=layer,
        backend=backend,
        prompt_attr="prompt",
        name_attr="name",
    )

    # Load and extract from model B
    try:
        model_b_obj, tokenizer_b = adapter.load_model_and_tokenizer(str(model_b))
    except Exception as e:
        write_error(
            ErrorDetail(code="MC-4001", message="Failed to load model B", detail=str(e)),
            context.output_format,
        )
        raise typer.Exit(1)

    backbone_b = resolve_model_backbone(model_b_obj)
    if backbone_b is None:
        write_error(
            ErrorDetail(code="MC-4002", message="Failed to resolve model B backbone", detail=""),
            context.output_format,
        )
        raise typer.Exit(1)

    activations_b = extract_anchor_activations(
        anchors=anchors,
        tokenizer=tokenizer_b,
        embed_tokens=backbone_b[0],
        layers=backbone_b[1],
        norm=backbone_b[2],
        target_layer=layer,
        backend=backend,
        prompt_attr="prompt",
        name_attr="name",
    )

    # Find common primes
    common_primes = sorted(set(activations_a.keys()) & set(activations_b.keys()))
    if len(common_primes) < 10:
        write_error(
            ErrorDetail(
                code="MC-4004",
                message="Insufficient common primes",
                detail=f"Got {len(common_primes)}, need at least 10",
            ),
            context.output_format,
        )
        raise typer.Exit(1)

    # Stack into matrices
    matrix_a = backend.stack([activations_a[p] for p in common_primes], axis=0)
    matrix_b = backend.stack([activations_b[p] for p in common_primes], axis=0)
    backend.eval(matrix_a, matrix_b)

    # Compute CKA
    cka_result = compute_cka(matrix_a, matrix_b, backend)

    if context.output_format == "text":
        lines = [
            "SEMANTIC PRIME COMPARISON",
            f"Model A: {model_a}",
            f"Model B: {model_b}",
            f"Layer: {layer}",
            f"Common primes: {len(common_primes)}",
            "",
            f"CKA (Centered Kernel Alignment): {cka_result.cka:.4f}",
            f"HSIC(A,B): {cka_result.hsic_xy:.4f}",
            f"HSIC(A,A): {cka_result.hsic_xx:.4f}",
            f"HSIC(B,B): {cka_result.hsic_yy:.4f}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    payload = {
        "model_a": str(model_a),
        "model_b": str(model_b),
        "layer": layer,
        "common_primes": len(common_primes),
        "cka": cka_result.cka,
        "hsic_xy": cka_result.hsic_xy,
        "hsic_xx": cka_result.hsic_xx,
        "hsic_yy": cka_result.hsic_yy,
    }
    write_output(payload, context.output_format, context.pretty)
