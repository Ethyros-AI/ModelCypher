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

from modelcypher.adapters.local_inference import LocalInferenceEngine
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output
from modelcypher.core.domain.geometry.concept_detector import (
    ConceptDetector,
    Configuration,
    create_default_detector,
)
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


def _parse_window_sizes(value: str | None) -> tuple[int, ...] | None:
    """Parse comma-separated window sizes string into tuple."""
    if not value:
        return None
    sizes: list[int] = []
    for part in value.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        sizes.append(int(stripped))
    return tuple(sizes) if sizes else None


def _build_detector(
    window_sizes: tuple[int, ...] | None = None,
    stride: int = 5,
    collapse_consecutive: bool = True,
    max_concepts: int = 30,
):
    """Build a concept detector with default embeddings.

    Note: threshold is now derived from embedding separability, not user-provided.
    """
    config = Configuration(
        # threshold derived from embeddings - not user parameter
        detection_threshold=None,
        window_sizes=window_sizes or Configuration().window_sizes,
        stride=stride,
        collapse_consecutive=collapse_consecutive,
        max_concepts_per_response=max_concepts,
    )
    embedder = _get_embedding_provider()
    return create_default_detector(embedder, config)


@app.command("detect")
def concept_detect(
    ctx: typer.Context,
    text: str = typer.Argument(..., help="Text or prompt to analyze"),
    model: str | None = typer.Option(None, "--model", help="Optional model path"),
    window_sizes: str | None = typer.Option(
        None, "--window-sizes", help="Comma-separated word window sizes (e.g., 10,20,30)"
    ),
    stride: int = typer.Option(5, "--stride", help="Word stride between windows"),
    max_concepts: int = typer.Option(30, "--max-concepts", help="Max concepts per response"),
    collapse: bool = typer.Option(
        True,
        "--collapse/--no-collapse",
        help="Collapse consecutive identical concepts",
    ),
    file: str | None = typer.Option(None, "--file", help="Optional output file"),
) -> None:
    """Detect conceptual activations in text or model responses.

    Detection threshold is derived from concept embedding geometry,
    not user-provided. This ensures reliable, reproducible detection.
    """
    context = _context(ctx)
    detector = _build_detector(
        window_sizes=_parse_window_sizes(window_sizes),
        stride=stride,
        collapse_consecutive=collapse,
        max_concepts=max_concepts,
    )

    if model:
        engine = LocalInferenceEngine()
        result = engine.infer(model, text, max_tokens=200, temperature=0.0, top_p=1.0)
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
                "category": concept.category.value,
                "confidence": concept.confidence,
                "characterSpan": {
                    "lowerBound": concept.character_span[0],
                    "upperBound": concept.character_span[1],
                },
                "triggerText": concept.trigger_text,
                "crossModalConfidence": concept.cross_modal_confidence,
            }
            for concept in detection.detected_concepts
        ],
        "meanConfidence": detection.mean_confidence,
        "meanCrossModalConfidence": detection.mean_cross_modal_confidence,
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
                f"  [{concept.category.value}] {concept.concept_id} confidence={concept.confidence:.2f}"
            )
            lines.append(f'    trigger: "{concept.trigger_text}"')
        lines.append("")
        lines.append(f"Mean Confidence: {detection.mean_confidence:.3f}")
        if detection.mean_cross_modal_confidence is not None:
            lines.append(f"Mean Cross-Modal Confidence: {detection.mean_cross_modal_confidence:.3f}")
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
    window_sizes: str | None = typer.Option(
        None, "--window-sizes", help="Comma-separated word window sizes (e.g., 10,20,30)"
    ),
    stride: int = typer.Option(5, "--stride", help="Word stride between windows"),
    max_concepts: int = typer.Option(30, "--max-concepts", help="Max concepts per response"),
    collapse: bool = typer.Option(
        True,
        "--collapse/--no-collapse",
        help="Collapse consecutive identical concepts",
    ),
    file: str | None = typer.Option(None, "--file", help="Optional output file"),
) -> None:
    """Compare conceptual sequences between two texts or models.

    Detection threshold is derived from concept embedding geometry,
    not user-provided. This ensures reliable, reproducible detection.
    """
    context = _context(ctx)
    detector = _build_detector(
        window_sizes=_parse_window_sizes(window_sizes),
        stride=stride,
        collapse_consecutive=collapse,
        max_concepts=max_concepts,
    )

    if text_a and text_b:
        text_to_analyze_a = text_a
        text_to_analyze_b = text_b
        model_id_a = "text-a"
        model_id_b = "text-b"
    elif model_a and model_b and prompt:
        engine = LocalInferenceEngine()
        response_a = engine.infer(model_a, prompt, max_tokens=200, temperature=0.0, top_p=1.0)
        response_b = engine.infer(model_b, prompt, max_tokens=200, temperature=0.0, top_p=1.0)
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
