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

"""Cross-cultural geometry CLI commands."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output
from modelcypher.core.domain.geometry.cross_cultural_geometry import (
    CrossCulturalGeometry,
)
from modelcypher.utils.json import dump_json

app = typer.Typer(no_args_is_help=True)
logger = logging.getLogger(__name__)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


def _flatten_gram(matrix: list) -> list[float]:
    if not matrix:
        return []
    if isinstance(matrix[0], list):
        return [float(value) for row in matrix for value in row]
    return [float(value) for value in matrix]


@app.command("analyze")
def cross_cultural_analyze(
    ctx: typer.Context,
    input_file: str = typer.Argument(..., help="JSON file with grams/primes"),
    output_file: str | None = typer.Option(None, "--output-file", "-o"),
) -> None:
    """Analyze cross-cultural geometry from two Gram matrices.

    Input JSON format:
    {
      "gramA": [[...], ...] or [...],
      "gramB": [[...], ...] or [...],
      "primeIds": ["prime_a", ...],
      "primeCategories": {"prime_a": "category", ...}
    }
    """
    context = _context(ctx)
    data = json.loads(Path(input_file).read_text(encoding="utf-8"))

    gram_a = _flatten_gram(data.get("gramA", []))
    gram_b = _flatten_gram(data.get("gramB", []))
    prime_ids = data.get("primeIds", [])
    prime_categories = data.get("primeCategories", {})

    if not prime_ids:
        raise typer.BadParameter("primeIds is required and must be non-empty")

    n = len(prime_ids)
    if len(gram_a) != n * n or len(gram_b) != n * n:
        raise typer.BadParameter(
            f"Gram sizes must match primeIds length (expected {n*n}, got {len(gram_a)} and {len(gram_b)})"
        )

    result = CrossCulturalGeometry.analyze(gram_a, gram_b, prime_ids, prime_categories)
    if result is None:
        raise typer.BadParameter("Cross-cultural analysis failed; check gram sizes and inputs.")

    alignment = CrossCulturalGeometry.analyze_alignment(gram_a, gram_b, n)

    payload = {
        "_schema": "mc.geometry.cross_cultural.analyze.v2",
        "primeIds": list(prime_ids),
        "gramRoughnessA": result.gram_roughness_a,
        "gramRoughnessB": result.gram_roughness_b,
        "mergedGramRoughness": result.merged_gram_roughness,
        "roughnessReduction": result.roughness_reduction,
        "rowCorrelations": result.row_correlations,
        "rowSharpnessA": result.row_sharpness_a,
        "rowSharpnessB": result.row_sharpness_b,
        "rowSharpnessRatio": result.row_sharpness_ratio,
        "categoryDivergence": result.category_divergence,
        "alignment": {
            "cka": alignment.cka,
            "rawPearson": alignment.raw_pearson,
            "alignmentGap": alignment.alignment_gap,
        }
        if alignment
        else None,
    }

    if output_file:
        Path(output_file).write_text(dump_json(payload, pretty=context.pretty), encoding="utf-8")

    if context.output_format == "text":
        lines = [
            "CROSS-CULTURAL GEOMETRY",
            "",
            f"Gram Roughness A: {result.gram_roughness_a:.6f}",
            f"Gram Roughness B: {result.gram_roughness_b:.6f}",
            f"Merged Gram Roughness: {result.merged_gram_roughness:.6f}",
            f"Roughness Reduction: {result.roughness_reduction:.6f}",
        ]
        if alignment:
            lines.extend(
                [
                    "",
                    "Alignment:",
                    f"  CKA: {alignment.cka:.3f}",
                    f"  Raw Pearson: {alignment.raw_pearson:.3f}",
                    f"  Alignment Gap: {alignment.alignment_gap:.3f}",
                ]
            )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
