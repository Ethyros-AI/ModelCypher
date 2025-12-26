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

"""Semantic Primes CLI commands.

Probes how language models represent Wierzbicka's Natural Semantic Metalanguage
(NSM) primes - proposed universal concepts found across all languages.

Commands:
    mc geometry primes list
    mc geometry primes probe-model <model_path>
    mc geometry primes compare <activations_a> <activations_b>
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from modelcypher.cli.commands.geometry.helpers import (
    forward_through_backbone,
    resolve_model_backbone,
    save_activations_json,
)
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output
from modelcypher.core.domain._backend import get_default_backend

app = typer.Typer(no_args_is_help=True)
logger = logging.getLogger(__name__)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("list")
def primes_list(
    ctx: typer.Context,
    category: str = typer.Option(None, help="Filter by category"),
) -> None:
    """List all NSM semantic primes (Goddard & Wierzbicka 2014).

    These 65 primes are proposed universal concepts found across all languages.
    They serve as stable anchors for cross-model comparison.

    Examples:
        mc geometry primes list
        mc geometry primes list --category mentalPredicates
    """
    context = _context(ctx)

    from modelcypher.core.domain.agents.semantic_prime_atlas import (
        SemanticPrimeInventory,
    )

    primes = SemanticPrimeInventory.english_2014()

    if category:
        primes = [p for p in primes if p.category.value == category]

    categories = sorted(set(p.category.value for p in primes))

    payload = {
        "_schema": "mc.geometry.primes.list.v1",
        "primes": [
            {
                "id": p.id,
                "category": p.category.value,
                "exponents": p.english_exponents,
            }
            for p in primes
        ],
        "count": len(primes),
        "categories": categories,
    }

    if context.output_format == "text":
        lines = [
            "SEMANTIC PRIMES (NSM - Goddard & Wierzbicka 2014)",
            f"Total: {len(primes)} primes across {len(categories)} categories",
            "",
            f"{'ID':<20} {'Category':<30} Exponents",
            "-" * 80,
        ]
        for p in primes:
            exponents = ", ".join(p.english_exponents[:3])
            if len(p.english_exponents) > 3:
                exponents += "..."
            lines.append(f"{p.id:<20} {p.category.value:<30} {exponents}")

        if not category:
            lines.extend(
                [
                    "",
                    "Categories:",
                    *[f"  - {c}" for c in categories],
                ]
            )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("probe-model")
def primes_probe_model(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to the model directory"),
    layer: int = typer.Option(-1, help="Layer to analyze (default is last)"),
    output_file: str = typer.Option(None, "--output-file", "-o", help="File to save activations"),
) -> None:
    """Probe a model for semantic prime representations.

    Extracts hidden state activations for each NSM prime and computes
    pairwise similarities to measure the model's semantic structure.

    Examples:
        mc geometry primes probe-model ./model
        mc geometry primes probe-model ./model --output-file primes.json
    """
    context = _context(ctx)

    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.backends.mlx_backend import MLXBackend
    from modelcypher.core.domain.agents.semantic_prime_atlas import (
        SemanticPrimeInventory,
    )
    from modelcypher.core.domain.geometry.cka import compute_cka

    typer.echo(f"Loading model from {model_path}...")
    model, tokenizer = load_model_for_training(model_path)

    model_type = getattr(model, "model_type", "unknown")
    resolved = resolve_model_backbone(model, model_type)

    if not resolved:
        typer.echo("Error: Could not resolve architecture.", err=True)
        raise typer.Exit(1)

    embed_tokens, layers, norm = resolved
    num_layers = len(layers)
    target_layer = layer if layer >= 0 else num_layers - 1
    typer.echo(f"Architecture resolved: {num_layers} layers, probing layer {target_layer}")

    backend = MLXBackend()
    primes = SemanticPrimeInventory.english_2014()
    prime_activations = {}

    typer.echo(f"Probing {len(primes)} semantic primes...")

    for prime in primes:
        try:
            # Use first exponent as probe text
            probe_text = prime.english_exponents[0] if prime.english_exponents else prime.id
            tokens = tokenizer.encode(probe_text)
            input_ids = backend.array([tokens])

            hidden = forward_through_backbone(
                input_ids,
                embed_tokens,
                layers,
                norm,
                target_layer=target_layer,
                backend=backend,
            )

            activation = backend.mean(hidden[0], axis=0)
            backend.eval(activation)
            prime_activations[prime.id] = activation

        except Exception as e:
            typer.echo(f"  Warning: Failed prime {prime.id}: {e}", err=True)

    if not prime_activations:
        typer.echo("Error: No activations extracted.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Extracted {len(prime_activations)} prime activations.")

    # Save activations if requested
    if output_file:
        save_activations_json(prime_activations, output_file, backend)
        typer.echo(f"Saved activations to {output_file}")

    # Compute category coherence (CKA within categories)
    category_primes: dict[str, list] = {}
    for prime in primes:
        cat = prime.category.value
        if cat not in category_primes:
            category_primes[cat] = []
        if prime.id in prime_activations:
            category_primes[cat].append(backend.to_numpy(prime_activations[prime.id]))

    category_coherence = {}
    for cat, acts in category_primes.items():
        if len(acts) >= 2:
            # Stack into matrix using backend and compute self-CKA
            X = backend.to_numpy(backend.stack([backend.array(a) for a in acts]))
            result = compute_cka(X, X)
            category_coherence[cat] = result.cka
        else:
            category_coherence[cat] = None

    # Compute overall structure score
    all_acts = [backend.to_numpy(a) for a in prime_activations.values()]
    X_all = backend.to_numpy(backend.stack([backend.array(a) for a in all_acts]))
    overall_result = compute_cka(X_all, X_all)

    payload = {
        "_schema": "mc.geometry.primes.probe.v1",
        "model_path": model_path,
        "layer": target_layer,
        "primes_probed": len(prime_activations),
        "total_primes": len(primes),
        "overall_coherence": overall_result.cka,
        "category_coherence": {k: v for k, v in category_coherence.items() if v is not None},
        "interpretation": (
            "Strong semantic structure - primes form coherent clusters."
            if overall_result.cka > 0.7
            else "Moderate semantic structure - some prime clustering detected."
            if overall_result.cka > 0.4
            else "Weak semantic structure - primes are diffusely represented."
        ),
    }

    if context.output_format == "text":
        lines = [
            "=" * 60,
            f"SEMANTIC PRIME ANALYSIS: {Path(model_path).name}",
            "=" * 60,
            "",
            f"Primes Probed: {len(prime_activations)}/{len(primes)}",
            f"Layer Analyzed: {target_layer}",
            f"Overall Coherence (CKA): {overall_result.cka:.3f}",
            "",
            "-" * 40,
            "Category Coherence:",
        ]
        for cat, score in sorted(category_coherence.items()):
            if score is not None:
                lines.append(f"  {cat}: {score:.3f}")
        lines.extend(
            [
                "",
                "=" * 60,
                payload["interpretation"],
                "=" * 60,
            ]
        )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("compare")
def primes_compare(
    ctx: typer.Context,
    activations_a: str = typer.Argument(..., help="JSON file with model A prime activations"),
    activations_b: str = typer.Argument(..., help="JSON file with model B prime activations"),
) -> None:
    """Compare semantic prime representations between two models.

    Computes CKA similarity to measure how similarly two models
    represent the NSM semantic primes.

    Examples:
        mc geometry primes compare model_a_primes.json model_b_primes.json
    """
    context = _context(ctx)

    from modelcypher.core.domain.geometry.cka import compute_cka

    # Load activations
    acts_a = json.loads(Path(activations_a).read_text())
    acts_b = json.loads(Path(activations_b).read_text())

    # Find common primes
    common_primes = sorted(set(acts_a.keys()) & set(acts_b.keys()))

    if len(common_primes) < 2:
        typer.echo("Error: Need at least 2 common primes to compare.", err=True)
        raise typer.Exit(1)

    # Build matrices using backend
    backend = get_default_backend()
    X = backend.to_numpy(backend.stack([backend.array(acts_a[p]) for p in common_primes]))
    Y = backend.to_numpy(backend.stack([backend.array(acts_b[p]) for p in common_primes]))

    # Compute CKA
    result = compute_cka(X, Y)

    # Find most similar and divergent primes
    # Handle dimension mismatch by projecting to common space
    dim_a = len(acts_a[common_primes[0]])
    dim_b = len(acts_b[common_primes[0]])

    prime_similarities = []
    if dim_a == dim_b:
        # Same dimensions - use direct cosine similarity
        from modelcypher.core.domain.geometry.vector_math import VectorMath

        for prime in common_primes:
            vec_a = acts_a[prime]
            vec_b = acts_b[prime]
            sim = VectorMath.cosine_similarity(vec_a, vec_b)
            prime_similarities.append((prime, sim if sim is not None else 0.0))
    else:
        # Different dimensions - use per-prime CKA via row correlation
        # Project to common space by computing normalized correlation
        # between the prime's position relative to other primes
        for i, prime in enumerate(common_primes):
            # Each prime's similarity is how correlated its relationships are
            # to other primes in both spaces
            row_a = X[i]  # This prime's embedding in model A
            row_b = Y[i]  # This prime's embedding in model B

            # Compute correlation with centroid (how typical is this prime)
            centroid_a = X.mean(axis=0)
            centroid_b = Y.mean(axis=0)

            # Use normalized dot product with centroid as proxy
            norm_a = float((row_a @ centroid_a) / (backend.to_numpy(backend.norm(backend.array(row_a))) * backend.to_numpy(backend.norm(backend.array(centroid_a))) + 1e-8))
            norm_b = float((row_b @ centroid_b) / (backend.to_numpy(backend.norm(backend.array(row_b))) * backend.to_numpy(backend.norm(backend.array(centroid_b))) + 1e-8))

            # Similarity = how similarly positioned this prime is in both spaces
            sim = 1.0 - abs(norm_a - norm_b)
            prime_similarities.append((prime, sim))

    prime_similarities.sort(key=lambda x: x[1], reverse=True)
    most_similar = [p for p, _ in prime_similarities[:5]]
    most_divergent = [p for p, _ in prime_similarities[-5:]]

    payload = {
        "_schema": "mc.geometry.primes.compare.v1",
        "model_a": activations_a,
        "model_b": activations_b,
        "common_primes": len(common_primes),
        "cka_similarity": result.cka,
        "most_similar_primes": most_similar,
        "most_divergent_primes": most_divergent,
        "interpretation": (
            "Models have highly similar semantic prime structure."
            if result.cka > 0.8
            else "Models have moderately similar semantic structure."
            if result.cka > 0.5
            else "Models have divergent semantic prime representations."
        ),
    }

    if context.output_format == "text":
        lines = [
            "=" * 60,
            "SEMANTIC PRIME COMPARISON",
            "=" * 60,
            "",
            f"Model A: {Path(activations_a).name}",
            f"Model B: {Path(activations_b).name}",
            f"Common Primes: {len(common_primes)}",
            "",
            f"CKA Similarity: {result.cka:.3f}",
            "",
            "-" * 40,
            "Most Similar Primes:",
            *[f"  - {p}" for p in most_similar],
            "",
            "Most Divergent Primes:",
            *[f"  - {p}" for p in most_divergent],
            "",
            "=" * 60,
            payload["interpretation"],
            "=" * 60,
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
