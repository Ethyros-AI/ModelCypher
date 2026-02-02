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

"""Benchmark CLI commands.

Run benchmarks on models with geometric metrics.

Commands:
    mc benchmark run --model <model> --suite comprehensive
    mc benchmark list
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.core.use_cases.benchmark_service import SUITES
from modelcypher.utils.errors import ErrorDetail

benchmark_app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@benchmark_app.command("run")
def benchmark_run(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", "-m", help="Path to model"),
    adapter: str = typer.Option("", "--adapter", "-a", help="Path to LoRA adapter"),
    suite: str = typer.Option("quick", "--suite", "-s", help="Benchmark suite (quick, comprehensive)"),
    results_path: str = typer.Option("", "--results-path", "-o", help="Path to save results JSON"),
    failures_path: str = typer.Option("", "--failures-path", help="Path to save failure cases JSONL"),
    entropy_probe_path: str = typer.Option("", "--entropy-probe-path", help="Path to probe prompts for entropy/ID profiling"),
    entropy_profile_output: str = typer.Option("", "--entropy-profile-output", help="Path to save entropy profile JSON"),
    id_profile_output: str = typer.Option("", "--id-profile-output", help="Path to save intrinsic dimension profile JSON"),
    limit: int = typer.Option(0, "--limit", "-l", help="Limit samples per benchmark (0 = all)"),
    max_failures: int = typer.Option(10, "--max-failures", help="Max failures per benchmark (0 = all)"),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Max tokens for generation (let the model think)"),
    no_geometry: bool = typer.Option(False, "--no-geometry", help="Skip geometric metrics"),
) -> None:
    """Run benchmarks on a model.

    Examples:
        mc benchmark run --model /path/to/model --suite quick
        mc benchmark run --model /path/to/model --adapter /path/to/adapter --suite comprehensive
        mc benchmark run --model /path/to/model --suite reasoning --results-path results.json
    """
    context = _context(ctx)

    try:
        from mlx_lm import load, generate
        from modelcypher.core.use_cases.benchmark_service import BenchmarkService

        # Load model
        if adapter:
            from modelcypher.core.domain.training.self_reflection import load_self_reflection_adapters
            model_obj, tokenizer = load_self_reflection_adapters(model, adapter)
            typer.echo(f"Loaded model with adapter: {adapter}")
        else:
            model_obj, tokenizer = load(model)
            typer.echo(f"Loaded model: {model}")

        # Run benchmarks
        service = BenchmarkService()
        result = service.run_suite(
            model_obj,
            tokenizer,
            suite,
            generate,
            limit_per_benchmark=limit if limit > 0 else None,
            max_failures=None if max_failures == 0 else max_failures,
            max_tokens=max_tokens,
            entropy_probe_path=entropy_probe_path or None,
        )

        # Display results
        typer.echo("")
        typer.echo("=" * 60)
        typer.echo(f"BENCHMARK SUITE: {suite.upper()}")
        typer.echo("=" * 60)

        for r in result.benchmarks:
            status = "✓" if r.accuracy >= 0.8 else ("△" if r.accuracy >= 0.5 else "✗")
            typer.echo(f"{status} {r.benchmark}: {r.accuracy:.1%} ({r.correct}/{r.total})")
            if not no_geometry:
                typer.echo(f"   e/π matches: {r.geometric.avg_e_pi_matches:.1f}, "
                          f"strong alignment: {r.geometric.strong_alignment_pct:.1%}")

        typer.echo("-" * 60)
        typer.echo(f"OVERALL: {result.overall_accuracy:.1%}")
        typer.echo("=" * 60)

        # Save if output path provided
        if results_path:
            service.save_results(result, Path(results_path))

        if entropy_probe_path and entropy_profile_output:
            profile_path = Path(entropy_profile_output)
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            with profile_path.open("w") as f:
                json.dump(result.entropy_profile, f, indent=2)

        if entropy_probe_path and id_profile_output:
            id_path = Path(id_profile_output)
            id_path.parent.mkdir(parents=True, exist_ok=True)
            with id_path.open("w") as f:
                json.dump(result.intrinsic_dimension_profile, f, indent=2)

        if failures_path:
            failures_file = Path(failures_path)
            failures_file.parent.mkdir(parents=True, exist_ok=True)
            with failures_file.open("w") as f:
                for benchmark in result.benchmarks:
                    for failure in benchmark.failures:
                        f.write(
                            json.dumps(
                                {
                                    "benchmark": failure.benchmark,
                                    "prompt": failure.prompt,
                                    "expected": failure.expected,
                                    "actual": failure.actual,
                                    "e_pi_matches": failure.e_pi_matches,
                                    "expansion_ratio": failure.expansion_ratio,
                                }
                            )
                            + "\n"
                        )

        write_output(result.to_dict(), context.output_format, context.pretty)

    except Exception as exc:
        error = ErrorDetail(
            code="MC-6001",
            title="Benchmark failed",
            detail=str(exc),
            hint="Check model path and benchmark suite name",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)


@benchmark_app.command("list")
def benchmark_list(ctx: typer.Context) -> None:
    """List available benchmark suites.

    Examples:
        mc benchmark list
    """
    context = _context(ctx)

    suites_info = {
        name: {
            "benchmarks": benchmarks,
            "count": len(benchmarks),
        }
        for name, benchmarks in SUITES.items()
    }

    typer.echo("Available benchmark suites:")
    typer.echo("")
    for name, info in suites_info.items():
        typer.echo(f"  {name}:")
        for b in info["benchmarks"]:
            typer.echo(f"    - {b}")
        typer.echo("")

    write_output(suites_info, context.output_format, context.pretty)


@benchmark_app.command("analyze")
def benchmark_analyze(
    ctx: typer.Context,
    failures_path: str = typer.Option(..., "--failures-path", help="Path to failure cases JSONL"),
    benchmark: str = typer.Option("", "--benchmark", help="Filter to a single benchmark name"),
    limit: int = typer.Option(0, "--limit", help="Limit number of failure records to read (0 = all)"),
) -> None:
    """Analyze failure JSONL produced by `mc benchmark run`.

    Examples:
        mc benchmark analyze --failures-path data/v4_failures.jsonl
        mc benchmark analyze --failures-path data/v4_failures.jsonl --benchmark gsm8k
    """
    context = _context(ctx)

    try:
        path = Path(failures_path)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")

        counts = Counter()
        total = 0

        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                name = rec.get("benchmark", "unknown")
                if benchmark and name != benchmark:
                    continue
                counts[name] += 1
                total += 1
                if limit and total >= limit:
                    break

        summary = {
            "failures_path": str(path),
            "filter_benchmark": benchmark or None,
            "total_failures": total,
            "by_benchmark": dict(counts),
        }

        typer.echo("Failure counts:")
        for name, count in counts.most_common():
            typer.echo(f"  {name:15s} {count}")
        typer.echo(f"Total: {total}")

        write_output(summary, context.output_format, context.pretty)

    except Exception as exc:
        error = ErrorDetail(
            code="MC-6002",
            title="Benchmark analysis failed",
            detail=str(exc),
            hint="Check failures path and optional benchmark filter",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)


@benchmark_app.command("export-curriculum")
def benchmark_export_curriculum(
    ctx: typer.Context,
    failures_path: str = typer.Option(..., "--failures-path", help="Path to failure cases JSONL"),
    output_path: str = typer.Option(..., "--output-path", "-o", help="Path to write curriculum JSONL"),
    benchmark: str = typer.Option("", "--benchmark", help="Filter to a single benchmark name"),
    limit: int = typer.Option(0, "--limit", help="Limit number of failure records to export (0 = all)"),
    with_metadata: bool = typer.Option(False, "--with-metadata", help="Include benchmark/expected/actual metadata"),
) -> None:
    """Export failure JSONL to a curriculum JSONL in text-continuation format.

    Examples:
        mc benchmark export-curriculum --failures-path data/v4_failures.jsonl --output-path data/training/failures.jsonl
        mc benchmark export-curriculum --failures-path data/v4_failures.jsonl --output-path data/training/gsm8k_failures.jsonl --benchmark gsm8k
    """
    context = _context(ctx)

    try:
        in_path = Path(failures_path)
        if not in_path.exists():
            raise FileNotFoundError(f"No such file: {in_path}")

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        total = 0
        written = 0
        skipped = 0

        with in_path.open() as f, out_path.open("w") as out:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                name = rec.get("benchmark", "")
                if benchmark and name != benchmark:
                    continue

                prompt = (rec.get("prompt") or "").strip()
                expected = (rec.get("expected") or "").strip()
                if not prompt or not expected:
                    skipped += 1
                    continue

                text = f"{prompt} {expected}".strip()
                if with_metadata:
                    payload = {
                        "text": text,
                        "metadata": {
                            "benchmark": name,
                            "expected": expected,
                            "actual": rec.get("actual", ""),
                        },
                    }
                else:
                    payload = {"text": text}

                out.write(json.dumps(payload) + "\n")
                written += 1
                total += 1
                if limit and total >= limit:
                    break

        summary = {
            "failures_path": str(in_path),
            "output_path": str(out_path),
            "filter_benchmark": benchmark or None,
            "exported": written,
            "skipped": skipped,
        }

        typer.echo(f"Exported {written} records to {out_path}")
        if skipped:
            typer.echo(f"Skipped {skipped} records missing prompt/expected")

        write_output(summary, context.output_format, context.pretty)

    except Exception as exc:
        error = ErrorDetail(
            code="MC-6003",
            title="Curriculum export failed",
            detail=str(exc),
            hint="Check failures path and output path",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
