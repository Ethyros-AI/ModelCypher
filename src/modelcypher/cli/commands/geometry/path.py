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

"""Geometry path CLI commands.

Provides commands for:
- Gate sequence detection
- Path comparison between texts or models

Commands:
    mc geometry path detect <text>
    mc geometry path compare --text-a <text> --text-b <text>
    mc geometry path compare --model-a <path> --model-b <path> --prompt <prompt>
"""

from __future__ import annotations

from pathlib import Path

import typer

from modelcypher.adapters.embedding_defaults import EmbeddingDefaults
from modelcypher.adapters.local_inference import LocalInferenceEngine
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output
from modelcypher.core.use_cases.geometry_service import GeometryService
from modelcypher.utils.json import dump_json

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("detect")
def geometry_path_detect(
    ctx: typer.Context,
    text: str = typer.Argument(...),
    model: str | None = typer.Option(None, "--model"),
    file: str | None = typer.Option(None, "--file"),
) -> None:
    """Detect computational gate sequences in text.

    The detection threshold is computed from the gate confidence distribution.
    No user-configurable threshold - the geometry determines it.

    Examples:
        mc geometry path detect "The sequence 1, 1, 2, 3, 5, 8..."
        mc geometry path detect "Hello world" --model ./model
    """
    context = _context(ctx)
    embedder = EmbeddingDefaults.make_default_embedder()
    service = GeometryService(embedder=embedder)

    if model:
        engine = LocalInferenceEngine()
        result = engine.infer(model, text, max_tokens=200, temperature=0.0, top_p=1.0)
        text_to_analyze = result.get("response", "")
        model_id = Path(model).name if Path(model).exists() else model
    else:
        text_to_analyze = text
        model_id = "input-text"

    # Threshold is computed from the data, not user-specified
    detection = service.detect_path(
        text_to_analyze,
        model_id=model_id,
        prompt_id="cli-input",
        threshold=None,  # Let the detector derive from confidence distribution
    )
    payload = service.detection_payload(detection)

    if file:
        Path(file).write_text(dump_json(payload, pretty=context.pretty), encoding="utf-8")

    if context.output_format == "text":
        gates = (
            " -> ".join(detection.gate_name_sequence) if detection.gate_name_sequence else "(none)"
        )
        lines = [
            f"Gate Sequence: {gates}",
            "",
            "Detected Gates:",
        ]
        for gate in detection.detected_gates:
            lines.append(f"  [{gate.gate_name}] confidence={gate.confidence:.2f}")
            lines.append(f'    trigger: "{gate.trigger_text}"')
        lines.append("")
        lines.append(f"Mean Confidence: {detection.mean_confidence:.3f}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("compare")
def geometry_path_compare(
    ctx: typer.Context,
    text_a: str | None = typer.Option(None, "--text-a"),
    text_b: str | None = typer.Option(None, "--text-b"),
    model_a: str | None = typer.Option(None, "--model-a"),
    model_b: str | None = typer.Option(None, "--model-b"),
    prompt: str | None = typer.Option(None, "--prompt"),
    file: str | None = typer.Option(None, "--file"),
) -> None:
    """Compare gate sequence paths between two texts or models.

    The detection threshold is computed from the gate confidence distribution.
    No user-configurable threshold - the geometry determines it.

    Examples:
        mc geometry path compare --text-a "First text" --text-b "Second text"
        mc geometry path compare --model-a ./model1 --model-b ./model2 --prompt "Hello"
    """
    context = _context(ctx)
    embedder = EmbeddingDefaults.make_default_embedder()
    service = GeometryService(embedder=embedder)

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
    else:
        raise typer.BadParameter(
            "Either --text-a and --text-b, or --model-a, --model-b, and --prompt are required."
        )

    # Threshold is computed from the data, not user-specified
    result = service.compare_paths(
        text_a=text_to_analyze_a,
        text_b=text_to_analyze_b,
        model_a=model_id_a,
        model_b=model_id_b,
        prompt_id="compare",
        threshold=None,  # Let the detector derive from confidence distribution
    )

    payload = service.path_comparison_payload(result)
    if file:
        Path(file).write_text(dump_json(payload, pretty=context.pretty), encoding="utf-8")

    if context.output_format == "text":
        path_a = " -> ".join(result.detection_a.gate_name_sequence) or "(none)"
        path_b = " -> ".join(result.detection_b.gate_name_sequence) or "(none)"
        lines = [
            f"Path A: {path_a}",
            f"Path B: {path_b}",
            "",
            "Path Comparison Results:",
            f"  Raw Distance: {result.comparison.total_distance:.3f}",
            f"  Normalized Distance: {result.comparison.normalized_distance:.3f}",
            "",
            "Alignment:",
        ]
        for step in result.comparison.alignment:
            op = step.op.value
            if op == "match":
                label = f"MATCH  {step.node_a.gate_id if step.node_a else '?'}"
            elif op == "substitute":
                left = step.node_a.gate_id if step.node_a else "?"
                right = step.node_b.gate_id if step.node_b else "?"
                label = f"SUBST  {left} -> {right} (cost: {step.cost:.2f})"
            elif op == "insert":
                label = f"INSERT {step.node_b.gate_id if step.node_b else '?'}"
            else:
                label = f"DELETE {step.node_a.gate_id if step.node_a else '?'}"
            lines.append(f"  {label}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
