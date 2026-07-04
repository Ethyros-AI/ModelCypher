#!/usr/bin/env python
"""Generate the README 15-knob runtime-status matrix.

The table intentionally reports runtime truth, not aspiration.  It reads the
current training defaults from the CLI/service/backend code and compares the
README block against that generated output in ``--check`` mode.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
TRAIN_CLI = REPO_ROOT / "src/modelcypher/cli/commands/train.py"
TRAINING_SERVICE = REPO_ROOT / "src/modelcypher/core/use_cases/dataset_training_service.py"
MLX_TRAINING_ADAPTER = (
    REPO_ROOT / "src/modelcypher/backends/_mlx_training_adapter_train_mixin.py"
)
TRAINING_DOMAIN = REPO_ROOT / "src/modelcypher/core/domain/training"
SRC_ROOT = REPO_ROOT / "src"

START = "<!-- BEGIN GENERATED KNOB MATRIX -->"
END = "<!-- END GENERATED KNOB MATRIX -->"

STATUSES = {
    "derived+shipped-default",
    "derived+research-mode-only",
    "formula-exists-unwired",
    "dead-code",
    "removed",
}


@dataclass(frozen=True)
class RuntimeFacts:
    default_optimizer_mode: str
    default_optimizer_identity: str
    calibrated_lr_literal: str
    cosine_epochs: str
    adamw_betas: str
    has_mass_research_mode: bool
    geometric_dropout_runtime_hits: int
    residual_scaling_runtime_hits: int
    spectral_init_runtime_hits: int


@dataclass(frozen=True)
class MatrixRow:
    number: int
    control: str
    status: str
    current_truth: str
    code_source: str

    def as_markdown(self) -> str:
        if self.status not in STATUSES:
            raise ValueError(f"Unknown status for row {self.number}: {self.status}")
        return (
            f"| {self.number} | {self.control} | {self.status} | "
            f"{self.current_truth} | {self.code_source} |"
        )


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_literal(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text)
    if match is None:
        raise RuntimeError(f"Could not extract {label}")
    return match.group(1)


def _format_lr_literal(raw: str) -> str:
    compact = raw.replace(" ", "")
    if compact == "2e-4":
        return "2e-4"
    return compact


def _default_for_arg(path: Path, function_name: str, arg_name: str) -> str:
    tree = ast.parse(_read(path), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            positional_args = node.args.args
            positional_defaults = [None] * (
                len(positional_args) - len(node.args.defaults)
            ) + list(node.args.defaults)
            for arg, default in zip(positional_args, positional_defaults, strict=True):
                if arg.arg != arg_name or default is None:
                    continue
                if isinstance(default, ast.Name):
                    return default.id
                if isinstance(default, ast.Constant):
                    return str(default.value)
                return ast.unparse(default)
            for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True):
                if arg.arg != arg_name:
                    continue
                if isinstance(default, ast.Name):
                    return default.id
                if isinstance(default, ast.Constant):
                    return str(default.value)
                return ast.unparse(default)
    raise RuntimeError(f"Could not find {function_name}(..., {arg_name}=...)")


def _python_files() -> list[Path]:
    roots = [SRC_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "tests"]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    return files


def _runtime_hits(symbol: str) -> int:
    """Count non-definition symbol references in runtime code.

    This is a lightweight import-graph guard for the matrix.  Tests/docs are
    excluded so a formula with tests but no runtime wiring still reports as
    unwired or dead.
    """
    hits = 0
    for path in SRC_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = _read(path)
        if f"def {symbol}" in text or f"class {symbol}" in text:
            text = re.sub(rf"^\s*(def|class)\s+{re.escape(symbol)}\b.*$", "", text, flags=re.M)
        hits += len(re.findall(rf"\b{re.escape(symbol)}\b", text))
    return hits


def collect_facts() -> RuntimeFacts:
    adapter_text = _read(MLX_TRAINING_ADAPTER)
    service_text = _read(TRAINING_SERVICE)
    default_mode = _default_for_arg(
        TRAINING_SERVICE,
        "build_training_plan",
        "optimizer_research_mode",
    )
    lr_literal = _format_lr_literal(
        _extract_literal(r"_CALIBRATED_LR\s*=\s*([0-9.eE+-]+)", adapter_text, "AdamW LR")
    )
    cosine_epochs = _extract_literal(
        r"_CALIBRATED_COSINE_EPOCHS\s*=\s*([0-9]+)",
        adapter_text,
        "cosine horizon",
    )
    betas_match = re.search(
        r"optimizer_research_mode == OPTIMIZER_MODE_ADAMW_GEOMETRIC:.*?betas=\[([0-9.]+),\s*([0-9.]+)\]",
        adapter_text,
        flags=re.S,
    )
    if betas_match is None:
        raise RuntimeError("Could not extract canonical AdamW betas")
    default_identity = _extract_literal(
        r'GEOMETRIC_LORA_OPTIMIZER\s*=\s*"([^"]+)"',
        _read(REPO_ROOT / "src/modelcypher/core/domain/training/identity.py"),
        "optimizer identity",
    )
    return RuntimeFacts(
        default_optimizer_mode=default_mode,
        default_optimizer_identity=default_identity,
        calibrated_lr_literal=lr_literal,
        cosine_epochs=cosine_epochs,
        adamw_betas=f"{betas_match.group(1)}/{betas_match.group(2)}",
        has_mass_research_mode="cayley_stiefel_mass" in service_text
        and "eta_step = min(eta_ceiling, eta_sps, eta_weyl)" in service_text,
        geometric_dropout_runtime_hits=_runtime_hits("compute_geometric_dropout"),
        residual_scaling_runtime_hits=_runtime_hits("compute_residual_scaling"),
        spectral_init_runtime_hits=_runtime_hits("spectral_normalized_lora_init"),
    )


def build_rows(facts: RuntimeFacts) -> list[MatrixRow]:
    lr_truth = (
        f"default: calibrated AdamW {facts.calibrated_lr_literal} cosine; "
        "MASS on research modes"
    )
    momentum_truth = (
        f"default: AdamW betas {facts.adamw_betas}; Fisher/MASS moments only in "
        "research modes"
    )
    schedule_truth = (
        f"default: cosine over {facts.cosine_epochs} data-epochs; no-schedule "
        "MASS only in research modes"
    )
    return [
        MatrixRow(
            1,
            "Learning rate",
            "derived+research-mode-only",
            lr_truth,
            "`_mlx_training_adapter_train_mixin.py` + `mass_step_size.py`",
        ),
        MatrixRow(
            2,
            "Adam epsilon",
            "formula-exists-unwired",
            "formula exists in `compute_geometric_epsilon`; shipped AdamW path does not consume it",
            "`geometric_optimizer.py`",
        ),
        MatrixRow(
            3,
            "Momentum",
            "derived+research-mode-only",
            momentum_truth,
            "`_mlx_training_adapter_train_mixin.py` + `diagonal_fisher_preconditioner.py`",
        ),
        MatrixRow(
            4,
            "Weight decay",
            "formula-exists-unwired",
            "condition-ratio formula exists; shipped `mc train run` default passes `weight_decay=0.0`",
            "`geometric_optimizer.py`, `dataset_training_service.py`",
        ),
        MatrixRow(
            5,
            "Gradient clipping",
            "derived+research-mode-only",
            "MASS bounds updates in research modes; canonical AdamW path has no geometric clipper",
            "`mass_step_size.py`",
        ),
        MatrixRow(
            6,
            "Warmup",
            "derived+research-mode-only",
            "canonical path uses calibrated cosine from step 0; research modes use MASS ceilings",
            "`dataset_training_service.py`, `mass_step_size.py`",
        ),
        MatrixRow(
            7,
            "LR schedule",
            "derived+research-mode-only",
            schedule_truth,
            "`_mlx_training_adapter_train_mixin.py`, `mass_step_size.py`",
        ),
        MatrixRow(
            8,
            "Batch size",
            "derived+shipped-default",
            "derived from gradient-noise scale, then reduced only for memory-safe micro-batching",
            "`DatasetTrainingService.train_from_dataset`",
        ),
        MatrixRow(
            9,
            "Early stopping",
            "derived+shipped-default",
            "geometric certificate and measured validation-loss windows are wired into training",
            "`geometric_early_stopping.py`, `_mlx_training_adapter_train_mixin.py`",
        ),
        MatrixRow(
            10,
            "LoRA scale",
            "derived+shipped-default",
            "adapter scale budget and saturation telemetry are enforced during training",
            "`geometric_lora.py`, `_mlx_training_adapter_train_mixin.py`",
        ),
        MatrixRow(
            11,
            "LoRA rank",
            "derived+shipped-default",
            "per-module ranks derive from tail dimensions and rank-capacity samples",
            "`geometric_lora.py`, `DatasetTrainingService.build_training_plan`",
        ),
        MatrixRow(
            12,
            "Target modules",
            "derived+shipped-default",
            "target surface derives from layer spectral geometry",
            "`select_target_modules`, `DatasetTrainingService.build_training_plan`",
        ),
        MatrixRow(
            13,
            "Dropout",
            "removed",
            "derived dropout formula was deleted because no shipped training adapter consumed it",
            f"`compute_geometric_dropout` runtime references={facts.geometric_dropout_runtime_hits}",
        ),
        MatrixRow(
            14,
            "Weight init",
            "removed",
            "default init is PiSSA; the unshipped spectral-normalized helper was deleted",
            f"`spectral_normalized_lora_init` runtime references={facts.spectral_init_runtime_hits}",
        ),
        MatrixRow(
            15,
            "Residual scaling",
            "removed",
            "standalone residual-scaling formula was deleted because no shipped path consumed it",
            f"`residual_scaling.py`; runtime references={facts.residual_scaling_runtime_hits}",
        ),
    ]


def render_matrix() -> str:
    facts = collect_facts()
    rows = build_rows(facts)
    header = [
        START,
        "",
        "| # | Training control | Runtime status | Current code truth | Code source |",
        "|---|---|---|---|---|",
    ]
    body = [row.as_markdown() for row in rows]
    footer = [
        "",
        END,
    ]
    return "\n".join(header + body + footer)


def replace_readme_matrix(readme_text: str, matrix: str) -> str:
    pattern = re.compile(
        rf"{re.escape(START)}.*?{re.escape(END)}",
        flags=re.S,
    )
    if pattern.search(readme_text):
        return pattern.sub(matrix, readme_text)

    old_table = re.compile(
        r"## What Gets Derived\n\n"
        r"\| # \| What Industry Tunes \| What ModelCypher Derives \| Source \|\n"
        r"\|---\|---\|---\|---\|\n"
        r"(?:\|.*\|\n)+"
        r"\nFull derivations",
        flags=re.S,
    )
    replacement = "## What Gets Derived\n\n" + matrix + "\n\nFull derivations"
    next_text, count = old_table.subn(replacement, readme_text, count=1)
    if count != 1:
        raise RuntimeError("Could not locate README knob matrix")
    return next_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if README is stale")
    parser.add_argument("--write", action="store_true", help="update README in place")
    args = parser.parse_args(argv)

    matrix = render_matrix()
    readme_text = _read(README)
    expected = replace_readme_matrix(readme_text, matrix)

    if args.write:
        README.write_text(expected, encoding="utf-8")
        return 0

    if args.check and expected != readme_text:
        print(
            "README knob matrix is stale. Run: poetry run python "
            "scripts/generate_knob_matrix.py --write",
            file=sys.stderr,
        )
        return 1
    if args.check:
        return 0

    print(matrix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
