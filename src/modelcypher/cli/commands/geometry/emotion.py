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

"""Emotion concept analysis CLI commands.

Provides commands for analyzing emotion concept representations in model space,
including VAD (Valence-Arousal-Dominance) projection and opposition structure analysis.

Commands:
    mc geometry emotion analyze --text "..."
    mc geometry emotion inventory
    mc geometry emotion compare --sig-a ... --sig-b ...
"""

from __future__ import annotations

import json
from pathlib import Path


import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.core.domain.agents.emotion_concept_atlas import (
    EmotionCategory,
    EmotionConceptAtlas,
    EmotionAtlasConfiguration,
    EmotionConceptInventory,
    OppositionPreservationScorer,
    OPPOSITION_PAIRS,
)
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("inventory")
def emotion_inventory(
    ctx: typer.Context,
    category: str | None = typer.Option(
        None, "--category", "-c", help="Filter by category (joy, sadness, fear, etc.)"
    ),
    no_dyads: bool = typer.Option(False, "--no-dyads", is_flag=True, flag_value=True, help="Exclude emotion dyads"),
) -> None:
    """List all emotion concepts in the inventory.

    Displays the 24 base emotions (8 primaries × 3 intensities) and 8 primary dyads
    with their VAD (Valence-Arousal-Dominance) coordinates.

    Examples:
        mc geometry emotion inventory
        mc geometry emotion inventory --category joy
        mc geometry emotion inventory --no-dyads
    """
    context = _context(ctx)

    try:
        emotions = EmotionConceptInventory.all_emotions()

        # Filter by category if specified
        if category:
            try:
                cat = EmotionCategory(category.lower())
                emotions = [e for e in emotions if e.category == cat]
            except ValueError:
                valid = [c.value for c in EmotionCategory]
                write_error(f"Invalid category '{category}'. Valid: {valid}", context.output_format)
                raise typer.Exit(1)

        dyads = EmotionConceptInventory.primary_dyads() if not no_dyads else []

        payload = {
            "emotionCount": len(emotions),
            "dyadCount": len(dyads),
            "emotions": [
                {
                    "id": e.id,
                    "name": e.name,
                    "category": e.category.value,
                    "intensity": e.intensity.value,
                    "valence": e.valence,
                    "arousal": e.arousal,
                    "dominance": e.dominance,
                    "oppositeId": e.opposite_id,
                }
                for e in emotions
            ],
            "dyads": [
                {
                    "id": d.id,
                    "name": d.name,
                    "primaryIds": list(d.primary_ids),
                    "valence": d.valence,
                    "arousal": d.arousal,
                    "dominance": d.dominance,
                }
                for d in dyads
            ],
            "oppositionPairs": [
                {"a": a.value, "b": b.value} for a, b in OPPOSITION_PAIRS
            ],
        }

        if context.output_format == "text":
            lines = [
                "EMOTION CONCEPT INVENTORY",
                f"Total emotions: {len(emotions)}",
                f"Total dyads: {len(dyads)}",
                "",
                "EMOTIONS:",
                f"{'ID':<15} {'Name':<12} {'Category':<12} {'Intensity':<10} {'V':>6} {'A':>6} {'D':>6} Opposite",
                "-" * 85,
            ]
            for e in emotions:
                lines.append(
                    f"{e.id:<15} {e.name:<12} {e.category.value:<12} {e.intensity.value:<10} "
                    f"{e.valence:>6.2f} {e.arousal:>6.2f} {e.dominance:>6.2f} {e.opposite_id or '-'}"
                )

            if dyads:
                lines.extend([
                    "",
                    "DYADS (blended emotions):",
                    f"{'ID':<15} {'Name':<15} {'Components':<20} {'V':>6} {'A':>6} {'D':>6}",
                    "-" * 75,
                ])
                for d in dyads:
                    components = f"{d.primary_ids[0]}+{d.primary_ids[1]}"
                    lines.append(
                        f"{d.id:<15} {d.name:<15} {components:<20} "
                        f"{d.valence:>6.2f} {d.arousal:>6.2f} {d.dominance:>6.2f}"
                    )

            lines.extend([
                "",
                "OPPOSITION PAIRS:",
            ])
            for a, b in OPPOSITION_PAIRS:
                lines.append(f"  {a.value} ↔ {b.value}")

            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)

    except Exception as exc:
        error = ErrorDetail.from_exception(exc)
        write_error(error.message, context.output_format)
        raise typer.Exit(1) from exc


@app.command("analyze")
def emotion_analyze(
    ctx: typer.Context,
    text: str = typer.Argument(..., help="Text to analyze for emotion content"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of top emotions to show"),
    no_mild: bool = typer.Option(False, "--no-mild", is_flag=True, flag_value=True, help="Exclude mild intensity emotions"),
    no_intense: bool = typer.Option(False, "--no-intense", is_flag=True, flag_value=True, help="Exclude intense emotions"),
) -> None:
    """Analyze text for emotion concept activations.

    Computes embedding-based similarity between input text and emotion concepts,
    returning top activations, VAD projection, and opposition balance.

    Note: Requires an embedding provider. Without one, returns inventory metadata only.

    Examples:
        mc geometry emotion analyze --text "I'm so happy today!"
        mc geometry emotion analyze --text "This makes me furious" --top-k 3
    """
    context = _context(ctx)

    try:
        include_mild = not no_mild
        include_intense = not no_intense
        config = EmotionAtlasConfiguration(
            include_mild=include_mild,
            include_intense=include_intense,
            top_k=top_k,
        )
        atlas = EmotionConceptAtlas(configuration=config)

        # Without embedder, we can only provide static info
        payload = {
            "text": text,
            "textLength": len(text),
            "inventorySize": len(atlas.inventory),
            "dyadCount": len(atlas.dyads),
            "configuration": {
                "includeMild": include_mild,
                "includeIntense": include_intense,
                "topK": top_k,
            },
            "note": "Embedding-based analysis requires an embedding provider. "
                    "Use MCP server or programmatic API for full analysis.",
            "availableCategories": [c.value for c in EmotionCategory],
            "oppositionPairs": [
                {"a": a.value, "b": b.value} for a, b in OPPOSITION_PAIRS
            ],
        }

        if context.output_format == "text":
            lines = [
                "EMOTION ANALYSIS",
                f"Text: {text[:100]}{'...' if len(text) > 100 else ''}",
                f"Text length: {len(text)} chars",
                "",
                f"Inventory size: {len(atlas.inventory)} emotions",
                f"Dyad count: {len(atlas.dyads)} dyads",
                "",
                "Note: Full embedding-based analysis requires an embedding provider.",
                "      Use the MCP server or programmatic API for signature computation.",
                "",
                "Categories available:",
            ]
            for cat in EmotionCategory:
                lines.append(f"  - {cat.value}")

            lines.extend([
                "",
                "Opposition pairs (should show inverse activation):",
            ])
            for a, b in OPPOSITION_PAIRS:
                lines.append(f"  {a.value} ↔ {b.value}")

            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)

    except Exception as exc:
        error = ErrorDetail.from_exception(exc)
        write_error(error.message, context.output_format)
        raise typer.Exit(1) from exc


@app.command("opposition")
def emotion_opposition(
    ctx: typer.Context,
) -> None:
    """Display emotion opposition structure.

    Shows the four primary opposition pairs from Plutchik's wheel and explains
    how opposition preservation scoring works for model merge validation.

    Examples:
        mc geometry emotion opposition
    """
    context = _context(ctx)

    try:
        pairs = [
            {
                "emotionA": a.value,
                "emotionB": b.value,
                "description": _opposition_description(a, b),
            }
            for a, b in OPPOSITION_PAIRS
        ]

        payload = {
            "pairCount": len(OPPOSITION_PAIRS),
            "pairs": pairs,
            "interpretation": (
                "Opposite emotions should have low co-activation. "
                "When merging models, opposition structure should be preserved - "
                "if joy > sadness in model A, the same should hold in merged model."
            ),
            "scoringMethod": "OppositionPreservationScorer.compute_score()",
        }

        if context.output_format == "text":
            lines = [
                "EMOTION OPPOSITION STRUCTURE",
                "",
                "Plutchik's wheel defines 4 primary opposition pairs where emotions",
                "are psychologically opposite and should not co-activate strongly.",
                "",
                "OPPOSITION PAIRS:",
                "-" * 60,
            ]
            for pair in pairs:
                lines.append(f"  {pair['emotionA']:>12} ↔ {pair['emotionB']:<12}")
                lines.append(f"    {pair['description']}")
                lines.append("")

            lines.extend([
                "MERGE VALIDATION:",
                "  When merging models, opposition structure should be preserved.",
                "  If model A has joy > sadness, the merged model should maintain",
                "  this relationship. Violations indicate potential semantic drift.",
                "",
                "  Score: 1.0 = perfect preservation, 0.0 = opposition violated",
            ])

            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)

    except Exception as exc:
        error = ErrorDetail.from_exception(exc)
        write_error(error.message, context.output_format)
        raise typer.Exit(1) from exc


def _opposition_description(a: EmotionCategory, b: EmotionCategory) -> str:
    """Get human-readable description of an opposition pair."""
    descriptions = {
        (EmotionCategory.JOY, EmotionCategory.SADNESS): (
            "Positive vs negative hedonic tone. Joy is pleasure; sadness is pain."
        ),
        (EmotionCategory.TRUST, EmotionCategory.DISGUST): (
            "Approach vs rejection. Trust accepts; disgust rejects."
        ),
        (EmotionCategory.FEAR, EmotionCategory.ANGER): (
            "Withdrawal vs approach. Fear flees; anger confronts."
        ),
        (EmotionCategory.SURPRISE, EmotionCategory.ANTICIPATION): (
            "Unexpected vs expected. Surprise reacts; anticipation prepares."
        ),
    }
    return descriptions.get((a, b), descriptions.get((b, a), "Opposite emotional responses."))
