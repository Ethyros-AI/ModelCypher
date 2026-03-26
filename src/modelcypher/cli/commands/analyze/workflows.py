from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import Any

import typer

from modelcypher.cli.composition import (
    get_observation_bundle_report_service,
    get_observation_service,
)
from modelcypher.cli.context import CLIContext
from modelcypher.cli.exit_codes import EXIT_INPUT, EXIT_RUNTIME
from modelcypher.cli.input_validation import (
    validate_file_exists,
    validate_model_path,
)
from modelcypher.cli.output import write_error, write_output
from modelcypher.core.use_cases.observation_service import (
    DEFAULT_ANALYSIS_SPACES,
    DEFAULT_MAX_TOKENS,
    ObservationTarget,
    PromptFamilyManifest,
)
from modelcypher.utils.errors import ErrorDetail


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


def _parse_spaces(raw_spaces: str) -> tuple[str, ...]:
    spaces = tuple(space.strip() for space in raw_spaces.split(",") if space.strip())
    return spaces or DEFAULT_ANALYSIS_SPACES


def _load_capture_prompts(
    *,
    prompt: str | None,
    prompt_file: str | None,
    prompt_stdin: bool,
    context: CLIContext,
) -> list[str]:
    provided = [value for value in (prompt, prompt_file) if value is not None]
    if len(provided) > 1 or ((prompt or prompt_file) and prompt_stdin):
        error = ErrorDetail(
            code="MC-1090",
            title="Conflicting prompt inputs",
            detail="Provide only one of --prompt, --prompt-file, or --prompt-stdin.",
            hint="Use a single prompt source for capture.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_INPUT)
        raise typer.Exit(code=EXIT_INPUT)

    if prompt is not None:
        return [prompt]

    if prompt_file is not None:
        path = validate_file_exists(prompt_file, description="Prompt file", context=context)
        return _parse_prompt_file(path)

    if prompt_stdin or not sys.stdin.isatty():
        content = sys.stdin.read().strip()
        if content:
            return [content]

    error = ErrorDetail(
        code="MC-1091",
        title="Missing prompt input",
        detail="Provide --prompt, --prompt-file, or --prompt-stdin.",
        hint="Capture needs a prompt or prompt file.",
        trace_id=context.trace_id,
    )
    write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_INPUT)
    raise typer.Exit(code=EXIT_INPUT)


def _parse_prompt_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, list):
            prompts = [str(item).strip() for item in data if str(item).strip()]
            if prompts:
                return prompts
        if isinstance(data, dict):
            raw_prompts = data.get("prompts")
            if isinstance(raw_prompts, list):
                prompts = [str(item).strip() for item in raw_prompts if str(item).strip()]
                if prompts:
                    return prompts
        raise ValueError(
            f"JSON prompt file must be an array of strings or an object with a prompts list: {path}"
        )

    prompts = [line.strip() for line in text.splitlines() if line.strip()]
    if prompts:
        return prompts
    raise ValueError(f"Prompt file did not contain any usable prompts: {path}")


def _emit_observation_result(
    *,
    result: Any,
    context: CLIContext,
) -> None:
    payload = result.to_dict()
    if context.output_format == "text":
        summary = payload["summary"]
        lines = [
            "ANALYZE WORKFLOW",
            f"Workflow: {payload['workflow']}",
            f"Bundle: {payload['outputDir']}",
            f"Report: {payload['files'].get('report')}",
            f"Targets: {summary['targetCount']}",
            f"Variants: {summary['variantCount']}",
            f"Comparisons: {summary['comparisonCount']}",
            f"Errors: {summary['errorCount']}",
            f"Spaces: {', '.join(summary['spaces'])}",
            (
                "Next: poetry run mc analyze report --bundle "
                + shlex.quote(payload["outputDir"])
            ),
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return
    write_output(payload, context.output_format, context.pretty)


def _emit_bundle_report_result(
    *,
    result: Any,
    context: CLIContext,
) -> None:
    if context.output_format == "text":
        write_output(result.markdown, context.output_format, context.pretty)
        return
    write_output(result.to_dict(), context.output_format, context.pretty)


def analyze_capture(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Model identifier or path"),
    adapter: str | None = typer.Option(None, "--adapter", help="Optional adapter directory"),
    label: str | None = typer.Option(None, "--label", help="Optional target label"),
    prompt: str | None = typer.Option(
        None,
        "--prompt",
        help="Single prompt to capture",
    ),
    prompt_file: str | None = typer.Option(
        None,
        "--prompt-file",
        help="File containing prompts (newline-separated or JSON array)",
    ),
    prompt_stdin: bool = typer.Option(
        False,
        "--prompt-stdin",
        help="Read a single prompt from stdin",
    ),
    name: str = typer.Option("capture", "--name", help="Bundle name"),
    spaces: str = typer.Option(
        ",".join(DEFAULT_ANALYSIS_SPACES),
        "--spaces",
        help="Comma-separated spaces to capture",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        help="Observation bundle output directory",
    ),
    max_tokens: int = typer.Option(
        DEFAULT_MAX_TOKENS,
        "--max-tokens",
        min=1,
        help="Maximum response tokens to generate per prompt",
    ),
) -> None:
    """Capture one prompt or prompt file into an observation bundle."""
    context = _context(ctx)
    validate_model_path(model, context=context)

    try:
        prompts = _load_capture_prompts(
            prompt=prompt,
            prompt_file=prompt_file,
            prompt_stdin=prompt_stdin,
            context=context,
        )
        manifest = PromptFamilyManifest.from_prompts(prompts, name=name)
        target = ObservationTarget(
            label=label or Path(model).expanduser().resolve().name or "target",
            model=str(Path(model).expanduser().resolve()),
            adapter=str(Path(adapter).expanduser().resolve()) if adapter else None,
        )
        service = get_observation_service()
        result = service.capture(
            target=target,
            manifest=manifest,
            output_dir=output,
            spaces=_parse_spaces(spaces),
            max_tokens=max_tokens,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        error = ErrorDetail(
            code="MC-1092",
            title="Analyze capture input error",
            detail=str(exc),
            hint="Check the prompt inputs, spaces list, and prompt file format.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_INPUT)
        raise typer.Exit(code=EXIT_INPUT)
    except RuntimeError as exc:
        error = ErrorDetail(
            code="MC-1093",
            title="Analyze capture failed",
            detail=str(exc),
            hint="Check model assets and backend readiness, then retry.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
        raise typer.Exit(code=EXIT_RUNTIME)

    _emit_observation_result(result=result, context=context)


def analyze_family(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Model identifier or path"),
    adapter: str | None = typer.Option(None, "--adapter", help="Optional adapter directory"),
    label: str | None = typer.Option(None, "--label", help="Optional target label"),
    manifest: str = typer.Option(
        ...,
        "--manifest",
        help="Prompt family manifest JSON file",
    ),
    spaces: str = typer.Option(
        ",".join(DEFAULT_ANALYSIS_SPACES),
        "--spaces",
        help="Comma-separated spaces to capture",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        help="Observation bundle output directory",
    ),
    max_tokens: int = typer.Option(
        DEFAULT_MAX_TOKENS,
        "--max-tokens",
        min=1,
        help="Maximum response tokens to generate per prompt",
    ),
) -> None:
    """Run a prompt-family manifest and persist pairwise deltas."""
    context = _context(ctx)
    validate_model_path(model, context=context)
    manifest_path = validate_file_exists(manifest, description="Prompt family manifest", context=context)

    try:
        prompt_family = PromptFamilyManifest.from_json_path(manifest_path)
        target = ObservationTarget(
            label=label or Path(model).expanduser().resolve().name or "target",
            model=str(Path(model).expanduser().resolve()),
            adapter=str(Path(adapter).expanduser().resolve()) if adapter else None,
        )
        service = get_observation_service()
        result = service.family(
            target=target,
            manifest=prompt_family,
            output_dir=output,
            spaces=_parse_spaces(spaces),
            max_tokens=max_tokens,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        error = ErrorDetail(
            code="MC-1094",
            title="Prompt family manifest error",
            detail=str(exc),
            hint="Ensure the manifest uses explicit case_id, variant_id, and text rows.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_INPUT)
        raise typer.Exit(code=EXIT_INPUT)
    except RuntimeError as exc:
        error = ErrorDetail(
            code="MC-1095",
            title="Analyze family failed",
            detail=str(exc),
            hint="Check model assets and backend readiness, then retry.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
        raise typer.Exit(code=EXIT_RUNTIME)

    _emit_observation_result(result=result, context=context)


def analyze_compare(
    ctx: typer.Context,
    left_model: str = typer.Option(..., "--left-model", help="Left-side model path"),
    right_model: str = typer.Option(..., "--right-model", help="Right-side model path"),
    left_adapter: str | None = typer.Option(None, "--left-adapter", help="Optional left adapter"),
    right_adapter: str | None = typer.Option(None, "--right-adapter", help="Optional right adapter"),
    left_label: str = typer.Option("left", "--left-label", help="Left target label"),
    right_label: str = typer.Option("right", "--right-label", help="Right target label"),
    manifest: str = typer.Option(
        ...,
        "--manifest",
        help="Prompt family manifest JSON file",
    ),
    spaces: str = typer.Option(
        ",".join(DEFAULT_ANALYSIS_SPACES),
        "--spaces",
        help="Comma-separated spaces to capture",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        help="Observation bundle output directory",
    ),
    max_tokens: int = typer.Option(
        DEFAULT_MAX_TOKENS,
        "--max-tokens",
        min=1,
        help="Maximum response tokens to generate per prompt",
    ),
) -> None:
    """Compare two targets on the same prompt-family manifest."""
    context = _context(ctx)
    validate_model_path(left_model, context=context)
    validate_model_path(right_model, context=context)
    manifest_path = validate_file_exists(manifest, description="Prompt family manifest", context=context)

    try:
        prompt_family = PromptFamilyManifest.from_json_path(manifest_path)
        service = get_observation_service()
        result = service.compare(
            left=ObservationTarget(
                label=left_label,
                model=str(Path(left_model).expanduser().resolve()),
                adapter=str(Path(left_adapter).expanduser().resolve()) if left_adapter else None,
            ),
            right=ObservationTarget(
                label=right_label,
                model=str(Path(right_model).expanduser().resolve()),
                adapter=str(Path(right_adapter).expanduser().resolve()) if right_adapter else None,
            ),
            manifest=prompt_family,
            output_dir=output,
            spaces=_parse_spaces(spaces),
            max_tokens=max_tokens,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        error = ErrorDetail(
            code="MC-1096",
            title="Analyze compare input error",
            detail=str(exc),
            hint="Ensure both targets are valid and the manifest is well-formed.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_INPUT)
        raise typer.Exit(code=EXIT_INPUT)
    except RuntimeError as exc:
        error = ErrorDetail(
            code="MC-1097",
            title="Analyze compare failed",
            detail=str(exc),
            hint="Check model assets and backend readiness, then retry.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
        raise typer.Exit(code=EXIT_RUNTIME)

    _emit_observation_result(result=result, context=context)


def analyze_report(
    ctx: typer.Context,
    bundle: str = typer.Option(
        ...,
        "--bundle",
        help="Observation bundle directory",
    ),
) -> None:
    """Read an existing observation bundle and render the shared report view."""
    context = _context(ctx)
    bundle_dir = Path(bundle).expanduser().resolve()

    if not bundle_dir.exists() or not bundle_dir.is_dir():
        error = ErrorDetail(
            code="MC-1098",
            title="Missing observation bundle",
            detail=f"Bundle directory not found: {bundle_dir}",
            hint="Point --bundle at a directory created by mc analyze capture/family/compare.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_INPUT)
        raise typer.Exit(code=EXIT_INPUT)

    try:
        service = get_observation_bundle_report_service()
        result = service.load(bundle_dir)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1099",
            title="Invalid observation bundle",
            detail=str(exc),
            hint=(
                "Check that the bundle contains manifest.json, summary.json, variants.jsonl, "
                "layer_metrics.jsonl, and comparisons.jsonl."
            ),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_INPUT)
        raise typer.Exit(code=EXIT_INPUT)
    except RuntimeError as exc:
        error = ErrorDetail(
            code="MC-1100",
            title="Analyze report failed",
            detail=str(exc),
            hint="Retry with a valid bundle directory or inspect the stored files directly.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
        raise typer.Exit(code=EXIT_RUNTIME)

    _emit_bundle_report_result(result=result, context=context)
