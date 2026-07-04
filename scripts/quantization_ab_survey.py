#!/usr/bin/env python3
"""Quantization A/B Survey: bf16 vs 4-bit geometric comparison.

Runs every relevant ModelCypher CLI tool on both bf16 and 4-bit quantized
versions of the same model, captures structured JSON output, computes
deltas for all numeric fields, and produces:
  1. comparison_report.md — A/B tables showing what quantization changes
  2. tool_health.md — stress test: what broke, what's unclear, what needs work

Usage:
    poetry run python scripts/quantization_ab_survey.py

    # Custom model paths
    poetry run python scripts/quantization_ab_survey.py \
        --bf16 /path/to/bf16 --q4 /path/to/4bit

    # Custom output directory
    poetry run python scripts/quantization_ab_survey.py \
        --output-dir results/custom_survey
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("quantization_ab_survey")

DEFAULT_BF16 = "/Volumes/CodeCypher/models/mlx-community/Qwen3.5-0.8B-bf16"
DEFAULT_Q4 = "/Volumes/CodeCypher/models/mlx-community/Qwen3.5-0.8B-4bit-g64"

# ── Probes ──────────────────────────────────────────────────────────────

SURVEY_PROBES = [
    # Factual (3)
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius at standard pressure.",
    "DNA carries genetic information in living organisms.",
    # Logical (3)
    "If all mammals are warm-blooded, and dolphins are mammals, then dolphins are warm-blooded.",
    "A number that is divisible by 6 must also be divisible by 2 and 3.",
    "The contrapositive of 'if P then Q' is 'if not Q then not P'.",
    # Creative (3)
    "Imagine a world where colors have sounds and music has texture.",
    "A poem about autumn leaves falling like memories fading.",
    "Music is mathematics made audible, patterns made beautiful.",
    # Technical (3)
    "The eigenvalues of a symmetric matrix are always real.",
    "Gradient descent minimizes a function by following the negative gradient.",
    "Singular value decomposition factors a matrix into rotation and scaling.",
    # Conversational (3)
    "Hello, how are you doing today?",
    "Could you explain this concept in simpler terms?",
    "What would you recommend for someone just starting out?",
    # Math (2)
    "What is 7 times 8?",
    "If a train leaves at 3pm going 60mph, how far does it travel in 2.5 hours?",
]

SINGLE_PROMPT_SET = [
    "The capital of France is Paris.",
    "If A implies B, and B implies C, then A implies C.",
    "What is 7 times 8?",
]

INFER_PROMPTS = [
    "Explain what a prime number is.",
    "What causes the seasons on Earth?",
    "Describe how a binary search works.",
    "What is the difference between a stack and a queue?",
    "How does photosynthesis work?",
]


# ── Benign Stderr Patterns ──────────────────────────────────────────────
# Lines matching any of these substrings are excluded from warning/error
# counts in assess_health(). Each pattern is documented with its source.
BENIGN_STDERR_PATTERNS: list[str] = [
    "HTTP/1.1 404 Not Found",       # HuggingFace dataset resolution probing
    "datasets-server.huggingface",   # HuggingFace datasets API info logs
    '"level": "info"',               # Structured JSON info logs (not warnings)
    "Could not load google/boolq from HuggingFace:",  # boolq split fallback warning
]

# ── Tool Specifications ─────────────────────────────────────────────────


@dataclass
class ToolSpec:
    """Specification for a CLI tool to run in the survey."""

    name: str
    # Command template. Placeholders: {model_path}, {probes_file}, {prompt}
    command_template: list[str]
    category: str  # "static", "activation", "behavioral"
    timeout: int  # seconds
    needs_probes: bool = False
    needs_prompt: bool = False
    prompts: list[str] | None = None  # for single-prompt tools


def build_tool_specs() -> list[ToolSpec]:
    """Build the list of tools to run."""
    return [
        # ── Static / Weight-Space ──
        ToolSpec(
            name="model-info",
            command_template=[
                "poetry", "run", "mc", "--json", "model", "info", "{model_path}",
            ],
            category="static",
            timeout=60,
        ),
        ToolSpec(
            name="model-capacity",
            command_template=[
                "poetry", "run", "mc", "--json", "model", "capacity", "{model_path}",
                "--top", "999",
            ],
            category="static",
            timeout=300,
        ),
        # ── Activation-Space (probes file) ──
        ToolSpec(
            name="dimension-profile",
            command_template=[
                "poetry", "run", "mc", "--json", "analyze", "dimension-profile",
                "--model", "{model_path}", "--probes", "{probes_file}", "--recovery",
            ],
            category="activation",
            timeout=300,
            needs_probes=True,
        ),
        ToolSpec(
            name="entropy-trajectory",
            command_template=[
                "poetry", "run", "mc", "--json", "analyze", "entropy-trajectory",
                "--model", "{model_path}", "--probes", "{probes_file}",
            ],
            category="activation",
            timeout=300,
            needs_probes=True,
        ),
        ToolSpec(
            name="spectral-trajectory",
            command_template=[
                "poetry", "run", "mc", "--json", "analyze", "spectral-trajectory",
                "--model", "{model_path}", "--probes", "{probes_file}",
            ],
            category="activation",
            timeout=300,
            needs_probes=True,
        ),
        ToolSpec(
            name="expansion-ratio",
            command_template=[
                "poetry", "run", "mc", "--json", "analyze", "expansion-ratio",
                "--model", "{model_path}", "--probes", "{probes_file}", "--trajectory",
            ],
            category="activation",
            timeout=300,
            needs_probes=True,
        ),
        ToolSpec(
            name="reasoning-flow",
            command_template=[
                "poetry", "run", "mc", "--json", "analyze", "reasoning-flow",
                "--model", "{model_path}", "--probes", "{probes_file}", "--trajectory",
            ],
            category="activation",
            timeout=300,
            needs_probes=True,
        ),
        ToolSpec(
            name="chain-profile",
            command_template=[
                "poetry", "run", "mc", "--json", "analyze", "chain-profile",
                "--model", "{model_path}", "--probes", "{probes_file}",
                "--samples", "60",
            ],
            category="activation",
            timeout=480,
            needs_probes=True,
        ),
        # ── Activation-Space (single prompt) ──
        ToolSpec(
            name="jacobian-trace",
            command_template=[
                "poetry", "run", "mc", "--json", "analyze", "jacobian-trace",
                "--model", "{model_path}", "--prompt", "{prompt}", "--trajectory",
            ],
            category="activation",
            timeout=300,
            needs_prompt=True,
            prompts=SINGLE_PROMPT_SET,
        ),
        ToolSpec(
            name="attention-collapse",
            command_template=[
                "poetry", "run", "mc", "--json", "analyze", "attention-collapse",
                "--model", "{model_path}", "--prompt", "{prompt}",
            ],
            category="activation",
            timeout=180,
            needs_prompt=True,
            prompts=SINGLE_PROMPT_SET,
        ),
        ToolSpec(
            name="attention-sink",
            command_template=[
                "poetry", "run", "mc", "--json", "analyze", "attention-sink",
                "--model", "{model_path}", "--prompt", "{prompt}",
            ],
            category="activation",
            timeout=180,
            needs_prompt=True,
            prompts=SINGLE_PROMPT_SET,
        ),
        # ── Behavioral ──
        ToolSpec(
            name="benchmark",
            command_template=[
                "poetry", "run", "mc", "--json", "analyze", "benchmark",
                "{model_path}", "--suite", "quick", "--limit", "20",
            ],
            category="behavioral",
            timeout=600,
        ),
        ToolSpec(
            name="infer",
            command_template=[
                "poetry", "run", "mc", "--json", "infer", "run",
                "--model", "{model_path}", "--prompt", "{prompt}",
            ],
            category="behavioral",
            timeout=120,
            needs_prompt=True,
            prompts=INFER_PROMPTS,
        ),
    ]


# ── Data Structures ─────────────────────────────────────────────────────


@dataclass
class RunResult:
    """Result of running a single CLI tool on one model."""

    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    parsed_json: dict | list | None = None
    parse_error: str | None = None
    duration_seconds: float = 0.0
    timed_out: bool = False
    crashed: bool = False


@dataclass
class ToolHealth:
    """Health assessment for a single tool."""

    json_valid: bool = False
    json_notes: str = ""
    bf16_crashed: bool = False
    q4_crashed: bool = False
    bf16_error: str = ""
    q4_error: str = ""
    bf16_timed_out: bool = False
    q4_timed_out: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class ToolComparison:
    """Complete comparison result for a single tool."""

    tool_name: str
    category: str
    prompt_index: int | None = None  # for single-prompt tools
    prompt_text: str | None = None
    bf16_result: RunResult = field(default_factory=RunResult)
    q4_result: RunResult = field(default_factory=RunResult)
    deltas: dict | None = None
    health: ToolHealth = field(default_factory=ToolHealth)


# ── Core Logic ──────────────────────────────────────────────────────────


def parse_json_output(stdout: str) -> tuple[dict | list | None, str | None]:
    """Parse JSON from CLI stdout, handling non-JSON prefix (warnings/progress).

    Returns (parsed_json, error_message).
    """
    if not stdout.strip():
        return None, "empty stdout"

    # Try parsing the whole thing first
    try:
        return json.loads(stdout), None
    except json.JSONDecodeError:
        pass

    # Try to find JSON object or array in the output
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        first = stdout.find(start_char)
        if first == -1:
            continue
        # Find the matching end by scanning backwards
        last = stdout.rfind(end_char)
        if last == -1 or last <= first:
            continue
        candidate = stdout[first : last + 1]
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError:
            continue

    return None, f"no valid JSON found in {len(stdout)} bytes of stdout"


def run_tool(
    model_path: str,
    spec: ToolSpec,
    probes_file: str | None = None,
    prompt: str | None = None,
) -> RunResult:
    """Run a single CLI tool and capture results."""
    # Build command from template
    cmd = []
    for part in spec.command_template:
        part = part.replace("{model_path}", model_path)
        if probes_file:
            part = part.replace("{probes_file}", probes_file)
        if prompt:
            part = part.replace("{prompt}", prompt)
        cmd.append(part)

    result = RunResult()
    model_name = Path(model_path).name
    prompt_preview = ""
    if prompt:
        prompt_preview = f" prompt={prompt[:40]!r}"

    logger.info("Running: %s on %s%s", spec.name, model_name, prompt_preview)
    logger.info("  cmd: %s", " ".join(cmd))

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=spec.timeout,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        result.exit_code = proc.returncode
        result.stdout = proc.stdout
        result.stderr = proc.stderr
        result.duration_seconds = time.monotonic() - start

        if proc.returncode != 0:
            result.crashed = True
            logger.warning(
                "  %s exited with code %d on %s",
                spec.name,
                proc.returncode,
                model_name,
            )
            # Still try to parse — some tools write JSON before crashing
            result.parsed_json, result.parse_error = parse_json_output(proc.stdout)
        else:
            result.parsed_json, result.parse_error = parse_json_output(proc.stdout)

    except subprocess.TimeoutExpired as e:
        result.duration_seconds = time.monotonic() - start
        result.timed_out = True
        result.stdout = e.stdout or ""
        result.stderr = e.stderr or ""
        logger.warning("  %s timed out after %ds on %s", spec.name, spec.timeout, model_name)

    except Exception as e:
        result.duration_seconds = time.monotonic() - start
        result.crashed = True
        result.stderr = str(e)
        logger.error("  %s failed with exception: %s", spec.name, e)

    status = "OK" if result.parsed_json is not None else "FAILED"
    logger.info(
        "  %s: %s (%.1fs, exit=%d, json=%s)",
        spec.name,
        status,
        result.duration_seconds,
        result.exit_code,
        "yes" if result.parsed_json is not None else "no",
    )
    return result


def compute_deltas(bf16: object, q4: object, path: str = "") -> dict:
    """Recursively compute deltas between two JSON structures.

    For numeric values: absolute_delta, relative_delta.
    For strings: match/mismatch.
    For lists: element-wise if same length.
    For dicts: recurse.
    """
    result: dict = {}

    if isinstance(bf16, dict) and isinstance(q4, dict):
        all_keys = set(bf16.keys()) | set(q4.keys())
        for key in sorted(all_keys):
            key_path = f"{path}.{key}" if path else key
            if key not in bf16:
                result[key] = {"status": "missing_in_bf16", "q4_value": q4[key]}
            elif key not in q4:
                result[key] = {"status": "missing_in_q4", "bf16_value": bf16[key]}
            else:
                sub = compute_deltas(bf16[key], q4[key], key_path)
                if sub:
                    result[key] = sub

    elif isinstance(bf16, list) and isinstance(q4, list):
        if len(bf16) == len(q4):
            items = {}
            for i, (b, q) in enumerate(zip(bf16, q4)):
                sub = compute_deltas(b, q, f"{path}[{i}]")
                if sub:
                    items[str(i)] = sub
            if items:
                result = {"_type": "list", "_length": len(bf16), "items": items}
        else:
            result = {
                "_type": "list_length_mismatch",
                "bf16_length": len(bf16),
                "q4_length": len(q4),
            }

    elif isinstance(bf16, (int, float)) and isinstance(q4, (int, float)):
        delta = q4 - bf16
        if delta != 0:
            entry: dict = {"bf16": bf16, "q4": q4, "delta": delta}
            if bf16 != 0:
                entry["relative"] = delta / abs(bf16)
            result = entry

    elif isinstance(bf16, str) and isinstance(q4, str):
        if bf16 != q4:
            result = {"bf16": bf16, "q4": q4, "match": False}

    elif isinstance(bf16, bool) and isinstance(q4, bool):
        if bf16 != q4:
            result = {"bf16": bf16, "q4": q4}

    elif bf16 is None and q4 is None:
        pass  # both null, no delta

    elif type(bf16) is not type(q4):
        result = {"bf16": str(bf16), "q4": str(q4), "type_mismatch": True}

    return result


def assess_health(
    spec: ToolSpec,
    bf16: RunResult,
    q4: RunResult,
) -> ToolHealth:
    """Assess tool health from run results."""
    health = ToolHealth()

    # JSON validity
    bf16_json_ok = bf16.parsed_json is not None and bf16.parse_error is None
    q4_json_ok = q4.parsed_json is not None and q4.parse_error is None
    health.json_valid = bf16_json_ok and q4_json_ok

    notes = []
    if not bf16_json_ok and not bf16.crashed and not bf16.timed_out:
        notes.append(f"bf16 JSON parse failed: {bf16.parse_error}")
    if not q4_json_ok and not q4.crashed and not q4.timed_out:
        notes.append(f"q4 JSON parse failed: {q4.parse_error}")

    # Check for warnings in stderr that might indicate issues
    for label, r in [("bf16", bf16), ("q4", q4)]:
        if r.stderr:
            # Count warning/error lines
            warn_lines = [
                ln for ln in r.stderr.splitlines()
                if ("warning" in ln.lower() or "error" in ln.lower())
                and not any(pat in ln for pat in BENIGN_STDERR_PATTERNS)
            ]
            if warn_lines:
                notes.append(
                    f"{label} stderr has {len(warn_lines)} warning/error lines"
                )

    health.bf16_crashed = bf16.crashed
    health.q4_crashed = q4.crashed
    health.bf16_timed_out = bf16.timed_out
    health.q4_timed_out = q4.timed_out

    if bf16.crashed:
        # First few lines of stderr for context
        err_preview = "\n".join(bf16.stderr.splitlines()[-5:]) if bf16.stderr else "no stderr"
        health.bf16_error = err_preview
        notes.append(f"bf16 crashed (exit={bf16.exit_code})")
    if q4.crashed:
        err_preview = "\n".join(q4.stderr.splitlines()[-5:]) if q4.stderr else "no stderr"
        health.q4_error = err_preview
        notes.append(f"q4 crashed (exit={q4.exit_code})")
    if bf16.timed_out:
        notes.append(f"bf16 timed out after {spec.timeout}s")
    if q4.timed_out:
        notes.append(f"q4 timed out after {spec.timeout}s")

    health.notes = notes
    return health


# ── Report Generation ───────────────────────────────────────────────────


def _extract_summary_metrics(comparisons: list[ToolComparison]) -> list[dict]:
    """Extract key metrics for the executive summary table."""
    metrics = []

    for comp in comparisons:
        bf16 = comp.bf16_result.parsed_json
        q4 = comp.q4_result.parsed_json
        if not isinstance(bf16, dict) or not isinstance(q4, dict):
            continue

        if comp.tool_name == "model-capacity":
            for key in ["meanEffectiveRank", "meanCapacityUtilization"]:
                if key in bf16 and key in q4:
                    metrics.append({
                        "metric": key,
                        "tool": comp.tool_name,
                        "bf16": bf16[key],
                        "q4": q4[key],
                    })

        elif comp.tool_name == "dimension-profile":
            for key in ["mean_intrinsic_dimension", "min_intrinsic_dimension",
                         "compression_ratio"]:
                if key in bf16 and key in q4:
                    metrics.append({
                        "metric": key,
                        "tool": comp.tool_name,
                        "bf16": bf16[key],
                        "q4": q4[key],
                    })

        elif comp.tool_name == "entropy-trajectory":
            for key in ["slope", "monotonicity", "early_late_ratio"]:
                if key in bf16 and key in q4:
                    metrics.append({
                        "metric": key,
                        "tool": comp.tool_name,
                        "bf16": bf16[key],
                        "q4": q4[key],
                    })

        elif comp.tool_name == "expansion-ratio" and comp.prompt_index is None:
            # Multi-probe expansion ratio — look for aggregate
            for key in ["expansion_ratio", "mean_expansion_ratio"]:
                if key in bf16 and key in q4:
                    metrics.append({
                        "metric": key,
                        "tool": comp.tool_name,
                        "bf16": bf16[key],
                        "q4": q4[key],
                    })

        elif comp.tool_name == "reasoning-flow" and comp.prompt_index is None:
            # Metrics are nested under results[i].overall, not at top level.
            # Average across all probes to get aggregate executive metrics.
            bf16_results = bf16.get("results", [])
            q4_results = q4.get("results", [])
            for key in ["mean_curvature", "smoothness", "directness", "total_arc_length"]:
                bf16_vals = [
                    r["overall"][key]
                    for r in bf16_results
                    if isinstance(r, dict) and "overall" in r and key in r["overall"]
                ]
                q4_vals = [
                    r["overall"][key]
                    for r in q4_results
                    if isinstance(r, dict) and "overall" in r and key in r["overall"]
                ]
                if bf16_vals and q4_vals:
                    metrics.append({
                        "metric": f"avg_{key}",
                        "tool": comp.tool_name,
                        "bf16": sum(bf16_vals) / len(bf16_vals),
                        "q4": sum(q4_vals) / len(q4_vals),
                    })

        elif comp.tool_name == "chain-profile" and comp.prompt_index is None:
            bf16_corr = bf16.get("correlations", {})
            q4_corr = q4.get("correlations", {})
            for key in ["entropyToCurvature", "cumulativeCurvatureToId", "meanAttnFraction"]:
                if key in bf16_corr and key in q4_corr:
                    metrics.append({
                        "metric": key,
                        "tool": comp.tool_name,
                        "bf16": bf16_corr[key],
                        "q4": q4_corr[key],
                    })

        elif comp.tool_name == "benchmark":
            # Look for accuracy or total scores
            for key in bf16:
                if isinstance(bf16.get(key), (int, float)) and key in q4:
                    if "accuracy" in key.lower() or "correct" in key.lower() or "score" in key.lower():
                        metrics.append({
                            "metric": key,
                            "tool": comp.tool_name,
                            "bf16": bf16[key],
                            "q4": q4[key],
                        })

        elif comp.tool_name == "infer" and comp.prompt_index == 0:
            for key in ["tokensPerSecond", "timeToFirstToken"]:
                if key in bf16 and key in q4:
                    metrics.append({
                        "metric": key,
                        "tool": comp.tool_name,
                        "bf16": bf16[key],
                        "q4": q4[key],
                    })

    return metrics


def _format_value(v: object) -> str:
    """Format a value for display in a markdown table."""
    if isinstance(v, float):
        if abs(v) < 0.001 and v != 0:
            return f"{v:.2e}"
        return f"{v:.4f}"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return v[:60]
    if v is None:
        return "null"
    return str(v)[:60]


def _format_relative(delta: float, baseline: float, precision: int = 1) -> str:
    """Format relative change against absolute baseline magnitude."""
    if baseline == 0:
        return "—"
    return f"{(100 * delta / abs(baseline)):+.{precision}f}%"


def generate_delta_summary(
    comparisons: list[ToolComparison],
    output_dir: Path,
) -> str:
    """Generate a compact delta summary table from the executive metrics.

    Writes to delta_summary.md and returns the table string for console output.
    """
    metrics = _extract_summary_metrics(comparisons)
    if not metrics:
        return "(no executive metrics extracted)"

    lines = ["| Metric | Tool | bf16 | 4-bit | Delta | Relative |"]
    lines.append("|--------|------|------|-------|-------|----------|")
    for m in metrics:
        bf16_v = m["bf16"]
        q4_v = m["q4"]
        if isinstance(bf16_v, (int, float)) and isinstance(q4_v, (int, float)):
            delta = q4_v - bf16_v
            rel = _format_relative(delta, bf16_v, precision=1)
            lines.append(
                f"| {m['metric']} | {m['tool']} "
                f"| {_format_value(bf16_v)} | {_format_value(q4_v)} "
                f"| {_format_value(delta)} | {rel} |"
            )
        else:
            lines.append(
                f"| {m['metric']} | {m['tool']} "
                f"| {_format_value(bf16_v)} | {_format_value(q4_v)} "
                f"| — | — |"
            )

    table = "\n".join(lines)
    (output_dir / "delta_summary.md").write_text(table + "\n")
    logger.info("Wrote delta summary to %s", output_dir / "delta_summary.md")
    return table


def generate_comparison_report(
    comparisons: list[ToolComparison],
    bf16_path: str,
    q4_path: str,
    timestamp: str,
) -> str:
    """Generate the markdown comparison report."""
    lines = [
        "# Quantization A/B Survey: bf16 vs 4-bit",
        "",
        f"**Generated:** {timestamp}",
        f"**bf16 model:** `{bf16_path}`",
        f"**4-bit model:** `{q4_path}`",
        "",
        "---",
        "",
    ]

    # Executive summary
    summary_metrics = _extract_summary_metrics(comparisons)
    if summary_metrics:
        lines.append("## Executive Summary")
        lines.append("")
        lines.append("| Metric | Tool | bf16 | 4-bit | Delta | Relative |")
        lines.append("|--------|------|------|-------|-------|----------|")
        for m in summary_metrics:
            bf16_v = m["bf16"]
            q4_v = m["q4"]
            if isinstance(bf16_v, (int, float)) and isinstance(q4_v, (int, float)):
                delta = q4_v - bf16_v
                lines.append(
                    f"| {m['metric']} | {m['tool']} | "
                    f"{_format_value(bf16_v)} | {_format_value(q4_v)} | "
                    f"{_format_value(delta)} | {_format_relative(delta, bf16_v, precision=2)} |"
                )
            else:
                lines.append(
                    f"| {m['metric']} | {m['tool']} | "
                    f"{_format_value(bf16_v)} | {_format_value(q4_v)} | — | — |"
                )
        lines.append("")
        lines.append("---")
        lines.append("")

    # Per-tool sections
    current_category = ""
    category_titles = {
        "static": "Static / Weight-Space Analysis",
        "activation": "Activation-Space Analysis",
        "behavioral": "Behavioral Analysis",
    }

    for comp in comparisons:
        if comp.category != current_category:
            current_category = comp.category
            lines.append(f"## {category_titles.get(current_category, current_category)}")
            lines.append("")

        # Section header
        title = comp.tool_name
        if comp.prompt_text:
            title += f" (prompt: {comp.prompt_text[:50]!r})"
        lines.append(f"### {title}")
        lines.append("")

        # Status line
        bf16_status = "OK" if comp.bf16_result.parsed_json is not None else "FAILED"
        q4_status = "OK" if comp.q4_result.parsed_json is not None else "FAILED"
        lines.append(
            f"**Status:** bf16={bf16_status} ({comp.bf16_result.duration_seconds:.1f}s) | "
            f"q4={q4_status} ({comp.q4_result.duration_seconds:.1f}s)"
        )
        lines.append("")

        if comp.bf16_result.crashed:
            lines.append(f"**bf16 ERROR:** exit code {comp.bf16_result.exit_code}")
            if comp.bf16_result.stderr:
                lines.append("```")
                lines.append(comp.bf16_result.stderr[-500:])
                lines.append("```")
            lines.append("")

        if comp.q4_result.crashed:
            lines.append(f"**q4 ERROR:** exit code {comp.q4_result.exit_code}")
            if comp.q4_result.stderr:
                lines.append("```")
                lines.append(comp.q4_result.stderr[-500:])
                lines.append("```")
            lines.append("")

        # Show deltas if both produced JSON
        bf16_data = comp.bf16_result.parsed_json
        q4_data = comp.q4_result.parsed_json
        if isinstance(bf16_data, dict) and isinstance(q4_data, dict):
            # Show top-level numeric fields as a table
            numeric_fields = []
            for key in sorted(set(bf16_data.keys()) | set(q4_data.keys())):
                bf16_v = bf16_data.get(key)
                q4_v = q4_data.get(key)
                if isinstance(bf16_v, (int, float)) and isinstance(q4_v, (int, float)):
                    numeric_fields.append((key, bf16_v, q4_v))
                elif isinstance(bf16_v, str) and isinstance(q4_v, str) and bf16_v != q4_v:
                    numeric_fields.append((key, bf16_v, q4_v))

            if numeric_fields:
                lines.append("| Field | bf16 | 4-bit | Delta | Relative |")
                lines.append("|-------|------|-------|-------|----------|")
                for key, bf16_v, q4_v in numeric_fields:
                    if isinstance(bf16_v, (int, float)) and isinstance(q4_v, (int, float)):
                        delta = q4_v - bf16_v
                        if bf16_v != 0:
                            rel = f"{delta / abs(bf16_v) * 100:+.2f}%"
                        else:
                            rel = "inf" if delta != 0 else "0%"
                        lines.append(
                            f"| {key} | {_format_value(bf16_v)} | "
                            f"{_format_value(q4_v)} | {_format_value(delta)} | {rel} |"
                        )
                    else:
                        lines.append(
                            f"| {key} | {_format_value(bf16_v)} | "
                            f"{_format_value(q4_v)} | — | — |"
                        )
                lines.append("")

            # Show per-layer data if present (common pattern: list of dicts)
            for key in sorted(bf16_data.keys()):
                bf16_v = bf16_data.get(key)
                q4_v = q4_data.get(key)
                if not (
                    isinstance(bf16_v, list)
                    and isinstance(q4_v, list)
                    and len(bf16_v) > 0
                    and isinstance(bf16_v[0], dict)
                ):
                    continue

                # Name-matched comparison: match items by "name" field
                # instead of position.  This handles model-info layers where
                # bf16 and q4 have different tensor counts and iteration order.
                bf16_has_name = all(
                    isinstance(x, dict) and "name" in x for x in bf16_v
                )
                q4_has_name = isinstance(q4_v, list) and all(
                    isinstance(x, dict) and "name" in x for x in q4_v
                )
                if bf16_has_name and q4_has_name:
                    bf16_map = {item["name"]: item for item in bf16_v}
                    q4_map = {item["name"]: item for item in q4_v}
                    # Filter out quantization metadata
                    q4_filtered = {
                        k: v for k, v in q4_map.items()
                        if not k.endswith((".scales", ".biases"))
                    }
                    common_names = sorted(set(bf16_map) & set(q4_filtered))
                    bf16_only = len(bf16_map) - len(common_names)
                    q4_only = len(q4_filtered) - len(common_names)
                    q4_quant_meta = len(q4_map) - len(q4_filtered)

                    lines.append(
                        f"**{key}** ({len(common_names)} matched, "
                        f"{bf16_only} bf16-only, {q4_only} q4-only, "
                        f"{q4_quant_meta} q4 quant metadata skipped):"
                    )
                    lines.append("")

                    if common_names:
                        sample = bf16_map[common_names[0]]
                        sample_keys = [
                            k for k in sample
                            if isinstance(sample.get(k), (int, float))
                        ][:4]
                        if sample_keys:
                            header = "| # | name |"
                            for sk in sample_keys:
                                header += f" {sk} (bf16) | {sk} (q4) | delta |"
                            lines.append(header)

                            sep = "|---|---|"
                            for _ in sample_keys:
                                sep += "---|---|---|"
                            lines.append(sep)

                            for i, name in enumerate(common_names[:20]):
                                b_item = bf16_map[name]
                                q_item = q4_filtered[name]
                                row = f"| {i} | {name} |"
                                for sk in sample_keys:
                                    bv = b_item.get(sk)
                                    qv = q_item.get(sk)
                                    if isinstance(bv, (int, float)) and isinstance(qv, (int, float)):
                                        d = qv - bv
                                        row += f" {_format_value(bv)} | {_format_value(qv)} | {_format_value(d)} |"
                                    else:
                                        row += f" {_format_value(bv)} | {_format_value(qv)} | — |"
                                lines.append(row)

                            if len(common_names) > 20:
                                lines.append(
                                    f"| ... | ({len(common_names) - 20} more) | ... |"
                                )
                        lines.append("")
                    continue

                # Positional comparison (same-length lists without name field)
                if len(bf16_v) != len(q4_v):
                    continue

                # Per-layer table — show first numeric field per item
                lines.append(f"**{key}** ({len(bf16_v)} items):")
                lines.append("")

                # Find common numeric keys in first element
                sample_keys = [
                    k for k in bf16_v[0]
                    if isinstance(bf16_v[0].get(k), (int, float))
                ][:6]  # limit columns
                if sample_keys:
                    # Find a label key
                    label_key = None
                    for candidate in ["layer_name", "name", "layer", "id", "index"]:
                        if candidate in bf16_v[0]:
                            label_key = candidate
                            break

                    header = "| # |"
                    if label_key:
                        header += f" {label_key} |"
                    for sk in sample_keys:
                        header += f" {sk} (bf16) | {sk} (q4) | delta |"
                    lines.append(header)

                    sep = "|---|"
                    if label_key:
                        sep += "---|"
                    for _ in sample_keys:
                        sep += "---|---|---|"
                    lines.append(sep)

                    # Limit rows to avoid huge tables
                    show_items = bf16_v[:20]
                    for i, (b_item, q_item) in enumerate(
                        zip(show_items, q4_v[:20])
                    ):
                        row = f"| {i} |"
                        if label_key:
                            row += f" {b_item.get(label_key, '')} |"
                        for sk in sample_keys:
                            bv = b_item.get(sk)
                            qv = q_item.get(sk)
                            if isinstance(bv, (int, float)) and isinstance(qv, (int, float)):
                                d = qv - bv
                                row += f" {_format_value(bv)} | {_format_value(qv)} | {_format_value(d)} |"
                            else:
                                row += f" {_format_value(bv)} | {_format_value(qv)} | — |"
                        lines.append(row)

                    if len(bf16_v) > 20:
                        lines.append(f"| ... | ({len(bf16_v) - 20} more rows) | ... |")
                    lines.append("")

            # Show per-layer data for simple numeric lists
            for key in sorted(bf16_data.keys()):
                bf16_v = bf16_data.get(key)
                q4_v = q4_data.get(key)
                if (
                    isinstance(bf16_v, list)
                    and isinstance(q4_v, list)
                    and len(bf16_v) == len(q4_v)
                    and len(bf16_v) > 0
                    and isinstance(bf16_v[0], (int, float))
                ):
                    lines.append(f"**{key}** ({len(bf16_v)} values):")
                    lines.append("")
                    lines.append("| Index | bf16 | 4-bit | Delta |")
                    lines.append("|-------|------|-------|-------|")
                    show_n = min(len(bf16_v), 30)
                    for i in range(show_n):
                        d = q4_v[i] - bf16_v[i]
                        lines.append(
                            f"| {i} | {_format_value(bf16_v[i])} | "
                            f"{_format_value(q4_v[i])} | {_format_value(d)} |"
                        )
                    if len(bf16_v) > show_n:
                        lines.append(f"| ... | ({len(bf16_v) - show_n} more) | | |")
                    lines.append("")

        elif bf16_data is None and q4_data is None:
            lines.append("Both models failed to produce output for this tool.")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def generate_tool_health_report(
    comparisons: list[ToolComparison],
    timestamp: str,
) -> str:
    """Generate the tool health / stress test report."""
    lines = [
        "# Tool Health Report: CLI Stress Test",
        "",
        f"**Generated:** {timestamp}",
        "",
        "## Summary",
        "",
        "| Tool | Prompt | JSON OK | bf16 | q4 | bf16 Time | q4 Time | Notes |",
        "|------|--------|---------|------|----|-----------|---------|---------| ",
    ]

    for comp in comparisons:
        h = comp.health
        prompt_label = comp.prompt_text[:30] if comp.prompt_text else "—"

        bf16_status = "CRASH" if h.bf16_crashed else ("TIMEOUT" if h.bf16_timed_out else "OK")
        q4_status = "CRASH" if h.q4_crashed else ("TIMEOUT" if h.q4_timed_out else "OK")
        json_status = "YES" if h.json_valid else "NO"

        notes = "; ".join(h.notes) if h.notes else "—"

        lines.append(
            f"| {comp.tool_name} | {prompt_label} | {json_status} | "
            f"{bf16_status} | {q4_status} | "
            f"{comp.bf16_result.duration_seconds:.1f}s | "
            f"{comp.q4_result.duration_seconds:.1f}s | {notes} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    # Detailed issues section
    issues = []
    for comp in comparisons:
        h = comp.health
        if h.notes or not h.json_valid or h.bf16_crashed or h.q4_crashed:
            prompt_label = f" (prompt: {comp.prompt_text[:40]!r})" if comp.prompt_text else ""
            issues.append(f"### {comp.tool_name}{prompt_label}")
            issues.append("")

            if h.bf16_crashed:
                issues.append("**bf16 crashed:**")
                issues.append("```")
                issues.append(h.bf16_error or "no error details")
                issues.append("```")
                issues.append("")

            if h.q4_crashed:
                issues.append("**q4 crashed:**")
                issues.append("```")
                issues.append(h.q4_error or "no error details")
                issues.append("```")
                issues.append("")

            if not h.json_valid:
                issues.append("**JSON output issue:**")
                if comp.bf16_result.parse_error:
                    issues.append(f"- bf16: {comp.bf16_result.parse_error}")
                if comp.q4_result.parse_error:
                    issues.append(f"- q4: {comp.q4_result.parse_error}")
                issues.append("")

            for note in h.notes:
                issues.append(f"- {note}")
            issues.append("")

    if issues:
        lines.append("## Detailed Issues")
        lines.append("")
        lines.extend(issues)
    else:
        lines.append("## Detailed Issues")
        lines.append("")
        lines.append("No issues found. All tools ran successfully on both models.")
        lines.append("")

    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantization A/B Survey: run all CLI tools on bf16 vs 4-bit"
    )
    parser.add_argument(
        "--bf16",
        default=DEFAULT_BF16,
        help=f"Path to bf16 model (default: {DEFAULT_BF16})",
    )
    parser.add_argument(
        "--q4",
        default=DEFAULT_Q4,
        help=f"Path to 4-bit model (default: {DEFAULT_Q4})",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: results/quantization_ab_survey/TIMESTAMP/)",
    )
    args = parser.parse_args()

    # Validate paths
    for label, path in [("bf16", args.bf16), ("q4", args.q4)]:
        if not Path(path).exists():
            logger.error("%s model not found: %s", label, path)
            sys.exit(1)

    # Setup output directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("results/quantization_ab_survey") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw" / "bf16").mkdir(parents=True, exist_ok=True)
    (output_dir / "raw" / "q4").mkdir(parents=True, exist_ok=True)

    # Write probe file
    probes_file = output_dir / "probes.txt"
    probes_file.write_text("\n".join(SURVEY_PROBES) + "\n")
    logger.info("Wrote %d probes to %s", len(SURVEY_PROBES), probes_file)

    # Build tool specs
    tool_specs = build_tool_specs()

    # Run survey
    comparisons: list[ToolComparison] = []
    total_start = time.monotonic()

    for spec in tool_specs:
        if spec.needs_prompt and spec.prompts:
            # Run once per prompt
            for i, prompt in enumerate(spec.prompts):
                comp = ToolComparison(
                    tool_name=spec.name,
                    category=spec.category,
                    prompt_index=i,
                    prompt_text=prompt,
                )

                comp.bf16_result = run_tool(
                    args.bf16, spec, str(probes_file), prompt
                )
                comp.q4_result = run_tool(
                    args.q4, spec, str(probes_file), prompt
                )

                # Save raw output
                for label, result in [("bf16", comp.bf16_result), ("q4", comp.q4_result)]:
                    raw_file = output_dir / "raw" / label / f"{spec.name}_prompt{i}.json"
                    raw_data = {
                        "exit_code": result.exit_code,
                        "duration_seconds": result.duration_seconds,
                        "timed_out": result.timed_out,
                        "crashed": result.crashed,
                        "parsed_json": result.parsed_json,
                        "parse_error": result.parse_error,
                        "stderr_tail": result.stderr[-1000:] if result.stderr else "",
                    }
                    raw_file.write_text(json.dumps(raw_data, indent=2, default=str))

                # Compute deltas
                if comp.bf16_result.parsed_json and comp.q4_result.parsed_json:
                    comp.deltas = compute_deltas(
                        comp.bf16_result.parsed_json,
                        comp.q4_result.parsed_json,
                    )

                comp.health = assess_health(spec, comp.bf16_result, comp.q4_result)
                comparisons.append(comp)
        else:
            # Run once (probes-file tools or no-input tools)
            comp = ToolComparison(
                tool_name=spec.name,
                category=spec.category,
            )

            comp.bf16_result = run_tool(args.bf16, spec, str(probes_file))
            comp.q4_result = run_tool(args.q4, spec, str(probes_file))

            # Save raw output
            for label, result in [("bf16", comp.bf16_result), ("q4", comp.q4_result)]:
                raw_file = output_dir / "raw" / label / f"{spec.name}.json"
                raw_data = {
                    "exit_code": result.exit_code,
                    "duration_seconds": result.duration_seconds,
                    "timed_out": result.timed_out,
                    "crashed": result.crashed,
                    "parsed_json": result.parsed_json,
                    "parse_error": result.parse_error,
                    "stderr_tail": result.stderr[-1000:] if result.stderr else "",
                }
                raw_file.write_text(json.dumps(raw_data, indent=2, default=str))

            # Compute deltas
            if comp.bf16_result.parsed_json and comp.q4_result.parsed_json:
                comp.deltas = compute_deltas(
                    comp.bf16_result.parsed_json,
                    comp.q4_result.parsed_json,
                )

            comp.health = assess_health(spec, comp.bf16_result, comp.q4_result)
            comparisons.append(comp)

    total_duration = time.monotonic() - total_start
    logger.info("Survey complete in %.1f seconds (%d tool runs)", total_duration, len(comparisons))

    # Generate reports
    comparison_report = generate_comparison_report(
        comparisons, args.bf16, args.q4, timestamp
    )
    (output_dir / "comparison_report.md").write_text(comparison_report)
    logger.info("Wrote comparison report to %s", output_dir / "comparison_report.md")

    tool_health_report = generate_tool_health_report(comparisons, timestamp)
    (output_dir / "tool_health.md").write_text(tool_health_report)
    logger.info("Wrote tool health report to %s", output_dir / "tool_health.md")

    # Save structured results
    survey_data = {
        "generated_at": timestamp,
        "bf16_model": args.bf16,
        "q4_model": args.q4,
        "total_duration_seconds": total_duration,
        "tool_count": len(tool_specs),
        "run_count": len(comparisons),
        "comparisons": [
            {
                "tool_name": c.tool_name,
                "category": c.category,
                "prompt_index": c.prompt_index,
                "prompt_text": c.prompt_text,
                "bf16_duration": c.bf16_result.duration_seconds,
                "q4_duration": c.q4_result.duration_seconds,
                "bf16_exit_code": c.bf16_result.exit_code,
                "q4_exit_code": c.q4_result.exit_code,
                "bf16_json_ok": c.bf16_result.parsed_json is not None,
                "q4_json_ok": c.q4_result.parsed_json is not None,
                "deltas": c.deltas,
                "health": {
                    "json_valid": c.health.json_valid,
                    "bf16_crashed": c.health.bf16_crashed,
                    "q4_crashed": c.health.q4_crashed,
                    "bf16_timed_out": c.health.bf16_timed_out,
                    "q4_timed_out": c.health.q4_timed_out,
                    "notes": c.health.notes,
                },
            }
            for c in comparisons
        ],
    }
    (output_dir / "survey_results.json").write_text(
        json.dumps(survey_data, indent=2, default=str)
    )
    logger.info("Wrote structured results to %s", output_dir / "survey_results.json")

    # Generate and print delta summary
    delta_table = generate_delta_summary(comparisons, output_dir)

    # Print summary
    ok_count = sum(
        1 for c in comparisons
        if c.bf16_result.parsed_json is not None and c.q4_result.parsed_json is not None
    )
    fail_count = len(comparisons) - ok_count
    print(f"\n{'=' * 60}")
    print(f"Survey complete: {ok_count}/{len(comparisons)} tool runs succeeded")
    print(f"Failures: {fail_count}")
    print(f"Total time: {total_duration:.1f}s")
    print(f"Results: {output_dir}")
    print(f"\n{delta_table}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
