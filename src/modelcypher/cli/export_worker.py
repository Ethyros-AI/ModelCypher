from __future__ import annotations

import json
from pathlib import Path

import typer

from modelcypher.cli.composition import get_export_service
from modelcypher.core.use_cases.export_service import ExportRequest, ExportTargetKind

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model: str = typer.Option(..., "--model"),
    adapter: str = typer.Option(..., "--adapter"),
    output: str = typer.Option(..., "--output"),
    target: ExportTargetKind = typer.Option(..., "--target"),
    quantization_bits: int = typer.Option(4, "--quantization-bits"),
    quantization_group_size: int = typer.Option(64, "--quantization-group-size"),
    quantization_mode: str = typer.Option("nf4", "--quantization-mode"),
) -> None:
    service = get_export_service()
    outcome = service.export(
        ExportRequest(
            model_path=Path(model),
            adapter_path=Path(adapter),
            output_path=Path(output),
            target_kind=target,
            quantization_bits=quantization_bits,
            quantization_group_size=quantization_group_size,
            quantization_mode=quantization_mode,
        )
    )
    typer.echo(json.dumps(outcome.to_dict()))


if __name__ == "__main__":
    app()
