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

"""Concept detection CLI commands."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from modelcypher.infrastructure.inference_engine_factory import get_inference_engine
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output
from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
from modelcypher.core.domain.geometry.concept_detector import ConceptDetector
from modelcypher.utils.json import dump_json

app = typer.Typer(no_args_is_help=True)
logger = logging.getLogger(__name__)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


def _get_embedding_provider():
    """Get the default embedding provider for concept detection."""
    from modelcypher.adapters.embedding_defaults import EmbeddingDefaults

    embedder = EmbeddingDefaults.make_default_embedder()
    if embedder is None:
        raise RuntimeError(
            "No embedding provider available. Concept detection requires embeddings. "
            "Install mlx-embeddings or configure TC_EMBEDDING_API_URL."
        )
    return embedder


def _build_detector() -> ConceptDetector:
    """Build a concept detector from the unified atlas inventory."""
    embedder = _get_embedding_provider()
    probes = UnifiedAtlasInventory.all_probes()
    if not probes:
        raise RuntimeError("No atlas probes available for concept detection.")
    return ConceptDetector(embedder, probes)


@app.command("detect")
def concept_detect(
    ctx: typer.Context,
    text: str = typer.Argument(..., help="Text or prompt to analyze"),
    model: str | None = typer.Option(None, "--model", help="Optional model path"),
    file: str | None = typer.Option(None, "--file", help="Optional output file"),
) -> None:
    """Detect conceptual activations in text or model responses.

    All parameters (threshold, window sizes, stride) are derived from
    concept embedding geometry. No user parameters.
    """
    context = _context(ctx)
    detector = _build_detector()

    if model:
        engine = get_inference_engine()
        result = engine.infer(model, text)
        text_to_analyze = result.get("response", "")
        model_id = Path(model).name if Path(model).exists() else model
    else:
        text_to_analyze = text
        model_id = "input-text"

    detection = detector.detect(
        response=text_to_analyze,
        model_id=model_id,
        prompt_id="concept-detect",
    )

    payload = {
        "_schema": "mc.geometry.concept.detect.v1",
        "modelId": detection.model_id,
        "promptId": detection.prompt_id,
        "responseText": detection.response_text,
        "conceptSequence": detection.concept_sequence,
        "detectedConcepts": [
            {
                "conceptId": concept.concept_id,
                "category": concept.category,
                "similarity": concept.similarity,
                "characterSpan": {
                    "lowerBound": concept.character_span[0],
                    "upperBound": concept.character_span[1],
                },
                "triggerText": concept.trigger_text,
                "crossModalSimilarity": concept.cross_modal_similarity,
            }
            for concept in detection.detected_concepts
        ],
        "meanSimilarity": detection.mean_similarity,
        "meanCrossModalSimilarity": detection.mean_cross_modal_similarity,
    }

    if file:
        Path(file).write_text(dump_json(payload, pretty=context.pretty), encoding="utf-8")

    if context.output_format == "text":
        concepts = " -> ".join(detection.concept_sequence) if detection.concept_sequence else "(none)"
        lines = [
            f"Concept Sequence: {concepts}",
            "",
            "Detected Concepts:",
        ]
        for concept in detection.detected_concepts:
            lines.append(
                f"  [{concept.category}] {concept.concept_id} similarity={concept.similarity:.2f}"
            )
            lines.append(f'    trigger: "{concept.trigger_text}"')
        lines.append("")
        lines.append(f"Mean Similarity: {detection.mean_similarity:.3f}")
        if detection.mean_cross_modal_similarity is not None:
            lines.append(
                f"Mean Cross-Modal Similarity: {detection.mean_cross_modal_similarity:.3f}"
            )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("compare")
def concept_compare(
    ctx: typer.Context,
    text_a: str | None = typer.Option(None, "--text-a"),
    text_b: str | None = typer.Option(None, "--text-b"),
    model_a: str | None = typer.Option(None, "--model-a"),
    model_b: str | None = typer.Option(None, "--model-b"),
    prompt: str | None = typer.Option(None, "--prompt"),
    file: str | None = typer.Option(None, "--file", help="Optional output file"),
) -> None:
    """Compare conceptual sequences between two texts or models.

    All parameters (threshold, window sizes, stride) are derived from
    concept embedding geometry. No user parameters.
    """
    context = _context(ctx)
    detector = _build_detector()

    if text_a and text_b:
        text_to_analyze_a = text_a
        text_to_analyze_b = text_b
        model_id_a = "text-a"
        model_id_b = "text-b"
    elif model_a and model_b and prompt:
        engine = get_inference_engine()
        response_a = engine.infer(model_a, prompt)
        response_b = engine.infer(model_b, prompt)
        text_to_analyze_a = response_a.get("response", "")
        text_to_analyze_b = response_b.get("response", "")
        model_id_a = Path(model_a).name if Path(model_a).exists() else model_a
        model_id_b = Path(model_b).name if Path(model_b).exists() else model_b
    elif text_a or text_b:
        missing = "--text-b" if text_a else "--text-a"
        raise typer.BadParameter(
            f"Missing {missing}: both --text-a and --text-b required for text comparison"
        )
    elif model_a or model_b:
        if model_a and model_b:
            raise typer.BadParameter(
                "Missing --prompt: required when comparing models. "
                "Example: --model-a ./m1 --model-b ./m2 --prompt 'Test input'"
            )
        missing = "--model-b" if model_a else "--model-a"
        raise typer.BadParameter(
            f"Missing {missing}: both models required for model comparison"
        )
    else:
        raise typer.BadParameter(
            "No input provided. Use either:\n"
            "  --text-a 'text' --text-b 'text'  (compare texts)\n"
            "  --model-a ./m1 --model-b ./m2 --prompt 'test'  (compare models)"
        )

    result_a = detector.detect(text_to_analyze_a, model_id_a, prompt_id="concept-compare-a")
    result_b = detector.detect(text_to_analyze_b, model_id_b, prompt_id="concept-compare-b")
    comparison = ConceptDetector.compare_results(result_a, result_b)

    payload = {
        "_schema": "mc.geometry.concept.compare.v1",
        "modelA": comparison.model_a,
        "modelB": comparison.model_b,
        "conceptPathA": list(comparison.concept_path_a),
        "conceptPathB": list(comparison.concept_path_b),
        "alignedConcepts": list(comparison.aligned_concepts),
        "uniqueToA": list(comparison.unique_to_a),
        "uniqueToB": list(comparison.unique_to_b),
        "alignmentRatio": comparison.alignment_ratio,
        "cka": comparison.cka,
        "cosineSimilarity": comparison.cosine_similarity,
    }

    if file:
        Path(file).write_text(dump_json(payload, pretty=context.pretty), encoding="utf-8")

    if context.output_format == "text":
        path_a = " -> ".join(comparison.concept_path_a) or "(none)"
        path_b = " -> ".join(comparison.concept_path_b) or "(none)"
        lines = [
            f"Concept Path A: {path_a}",
            f"Concept Path B: {path_b}",
            "",
            f"Aligned Concepts: {', '.join(comparison.aligned_concepts) or '(none)'}",
            f"Unique to A: {', '.join(comparison.unique_to_a) or '(none)'}",
            f"Unique to B: {', '.join(comparison.unique_to_b) or '(none)'}",
            f"Alignment Ratio: {comparison.alignment_ratio:.2f}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
