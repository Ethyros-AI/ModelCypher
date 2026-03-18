#!/usr/bin/env python3
"""Real-model smoke runner for the shipped training CLI workflow.

Validates the user-facing path:
  mc train run -> mc train evaluate -> mc train compare -> mc train export

This script is intentionally separate from service-level validation harnesses.
It shells out to the real CLI, captures raw command output, verifies runtime
ownership publication during training/export, and writes a compact report.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from modelcypher.core.use_cases.runtime_coordinator import RuntimeCoordinator

DEFAULT_MODEL_PATH = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
DEFAULT_TRAIN_DATA = "data/training/benchmark_train.jsonl"
DEFAULT_EVAL_DATA = "data/training/benchmark_val.jsonl"
DEFAULT_OUTPUT_ROOT = "results/canonical_train_workflow_smoke"

POLL_INTERVAL_SECONDS = 0.25
TRAIN_TIMEOUT_SECONDS = 7200
FOLLOWUP_TIMEOUT_SECONDS = 3600
EXPORT_TIMEOUT_SECONDS = 7200
RUNTIME_CLEAR_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class CommandRecord:
    stage: str
    command: list[str]
    exit_code: int
    duration_seconds: float
    stdout_path: str
    stderr_path: str
    stdout_tail: str
    stderr_tail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "command": list(self.command),
            "exit_code": self.exit_code,
            "duration_seconds": self.duration_seconds,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
        }


class CommandExecutionError(RuntimeError):
    """Raised when a subprocess-backed smoke stage fails."""

    def __init__(self, message: str, *, record: CommandRecord | None = None):
        super().__init__(message)
        self.record = record


@dataclass(frozen=True)
class WatchedCommandResult:
    record: CommandRecord
    observed_runtime: list[dict[str, Any]]
    contention_record: CommandRecord | None = None
    contention_payload: dict[str, Any] | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real-model smoke test for the canonical ModelCypher CLI workflow.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path to the local model directory used for the smoke run.",
    )
    parser.add_argument(
        "--train-data",
        default=DEFAULT_TRAIN_DATA,
        help="Path to the JSONL training dataset.",
    )
    parser.add_argument(
        "--eval-data",
        default=DEFAULT_EVAL_DATA,
        help="Path to the JSONL evaluation dataset.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where run artifacts are written.",
    )
    return parser.parse_args(argv)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _stage_slug(stage: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stage)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("No JSON payload found in empty stdout")
    for idx, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            payload = json.loads(stripped[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("Unable to locate a JSON object in command output")


def _has_json_output_flag(tokens: list[str]) -> bool:
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token in {"--json", "-j"}:
            return True
        if token.startswith("--output="):
            return token.split("=", 1)[1] == "json"
        if token == "--output" and idx + 1 < len(tokens):
            return tokens[idx + 1] == "json"
        idx += 1
    return False


def _command_to_argv(command: str, *, json_output: bool = True) -> list[str]:
    """Convert a generated `mc ...` command string into an executable argv.

    The generated command body is preserved verbatim via `shlex.split`; the only
    mutation is appending `--json` when structured output was not already
    requested.
    """

    tokens = shlex.split(command)
    argv = ["poetry", "run", *tokens]
    if json_output and not _has_json_output_flag(tokens):
        argv.append("--json")
    return argv


def _describe_process(pid: int) -> str:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid=,command="],
        cwd=str(_repo_root()),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or str(pid)


def _detect_gpu_process_conflicts() -> list[str]:
    result = subprocess.run(
        ["/bin/zsh", "-lc", "pgrep -af 'python|mlx' | grep -v grep || true"],
        cwd=str(_repo_root()),
        capture_output=True,
        text=True,
        check=False,
    )
    conflicts: list[str] = []
    ignored_pids = {os.getpid(), os.getppid()}
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_token = stripped.split(maxsplit=1)[0]
        if not pid_token.isdigit():
            continue
        pid = int(pid_token)
        if pid in ignored_pids:
            continue
        description = _describe_process(pid)
        if "canonical_train_workflow_smoke.py" in description:
            continue
        conflicts.append(description)
    return conflicts


def _command_env(modelcypher_home: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["MODELCYPHER_HOME"] = str(modelcypher_home)
    return env


def _tail(text: str, limit: int = 4000) -> str:
    return text[-limit:]


def _read_command_payload(record: CommandRecord) -> dict[str, Any]:
    return _extract_json_object(_read_text(record.stdout_path))


def _run_command(
    *,
    stage: str,
    command: list[str],
    run_dir: Path,
    modelcypher_home: Path,
    timeout_seconds: int | None = None,
    check: bool = True,
) -> CommandRecord:
    logs_dir = run_dir / "command_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    slug = _stage_slug(stage)
    stdout_path = logs_dir / f"{slug}.stdout.txt"
    stderr_path = logs_dir / f"{slug}.stderr.txt"

    start = time.monotonic()
    proc = subprocess.run(
        command,
        cwd=str(_repo_root()),
        env=_command_env(modelcypher_home),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    duration = time.monotonic() - start

    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")

    record = CommandRecord(
        stage=stage,
        command=command,
        exit_code=int(proc.returncode),
        duration_seconds=float(duration),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        stdout_tail=_tail(proc.stdout or ""),
        stderr_tail=_tail(proc.stderr or ""),
    )

    if check and proc.returncode != 0:
        raise CommandExecutionError(
            f"{stage} failed with exit_code={proc.returncode}. "
            f"stdout_tail={record.stdout_tail[-500:]} stderr_tail={record.stderr_tail[-500:]}",
            record=record,
        )
    return record


def _snapshot_runtime_status(coordinator: RuntimeCoordinator) -> dict[str, Any] | None:
    status = coordinator.status()
    return status.to_dict() if status is not None else None


def _append_runtime_observation(
    observations: list[dict[str, Any]],
    status: dict[str, Any] | None,
) -> None:
    if status is None:
        return
    if observations and observations[-1] == status:
        return
    observations.append(status)


def _wait_for_runtime_clear(
    coordinator: RuntimeCoordinator,
    *,
    timeout_seconds: float = RUNTIME_CLEAR_TIMEOUT_SECONDS,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if coordinator.status() is None:
            return True
        time.sleep(POLL_INTERVAL_SECONDS)
    return coordinator.status() is None


def _parse_busy_failure(text: str) -> dict[str, Any]:
    payload = _extract_json_object(text)
    error = payload.get("error")
    if not isinstance(error, dict):
        raise ValueError("Expected structured error payload")
    active_runtime = error.get("active_runtime")
    if not isinstance(active_runtime, dict):
        raise ValueError("Structured busy failure missing active_runtime")
    return payload


def _run_command_with_runtime_watch(
    *,
    stage: str,
    command: list[str],
    run_dir: Path,
    modelcypher_home: Path,
    expected_owner: str,
    timeout_seconds: int,
    contention_command: list[str] | None = None,
) -> WatchedCommandResult:
    logs_dir = run_dir / "command_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    slug = _stage_slug(stage)
    stdout_path = logs_dir / f"{slug}.stdout.txt"
    stderr_path = logs_dir / f"{slug}.stderr.txt"

    coordinator = RuntimeCoordinator(base_path=modelcypher_home)
    observed_runtime: list[dict[str, Any]] = []
    contention_record: CommandRecord | None = None
    contention_payload: dict[str, Any] | None = None
    saw_owner = False
    timed_out = False

    start = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w",
        encoding="utf-8",
    ) as stderr_handle:
        proc = subprocess.Popen(
            command,
            cwd=str(_repo_root()),
            env=_command_env(modelcypher_home),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )

        try:
            while proc.poll() is None:
                runtime_status = _snapshot_runtime_status(coordinator)
                _append_runtime_observation(observed_runtime, runtime_status)
                if runtime_status is not None and runtime_status.get("owner") == expected_owner:
                    saw_owner = True
                    if contention_command is not None and contention_record is None:
                        contention_record = _run_command(
                            stage=f"{stage}.contention",
                            command=contention_command,
                            run_dir=run_dir,
                            modelcypher_home=modelcypher_home,
                            timeout_seconds=FOLLOWUP_TIMEOUT_SECONDS,
                            check=False,
                        )
                        try:
                            contention_payload = _parse_busy_failure(
                                _read_text(contention_record.stdout_path),
                            )
                        except ValueError as exc:
                            raise CommandExecutionError(
                                f"{stage}.contention did not return a structured busy failure: {exc}",
                                record=contention_record,
                            ) from exc
                if time.monotonic() - start > timeout_seconds:
                    timed_out = True
                    proc.kill()
                    break
                time.sleep(POLL_INTERVAL_SECONDS)
        finally:
            proc.wait()

    duration = time.monotonic() - start
    stdout_text = _read_text(stdout_path)
    stderr_text = _read_text(stderr_path)
    record = CommandRecord(
        stage=stage,
        command=command,
        exit_code=int(proc.returncode),
        duration_seconds=float(duration),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        stdout_tail=_tail(stdout_text),
        stderr_tail=_tail(stderr_text),
    )

    if timed_out:
        raise CommandExecutionError(
            f"{stage} exceeded timeout={timeout_seconds}s",
            record=record,
        )
    if record.exit_code != 0:
        raise CommandExecutionError(
            f"{stage} failed with exit_code={record.exit_code}. "
            f"stdout_tail={record.stdout_tail[-500:]} stderr_tail={record.stderr_tail[-500:]}",
            record=record,
        )
    if not saw_owner:
        raise CommandExecutionError(
            f"{stage} never published runtime owner={expected_owner}",
            record=record,
        )
    if not _wait_for_runtime_clear(coordinator):
        raise CommandExecutionError(
            f"{stage} left runtime state uncleared after completion",
            record=record,
        )

    return WatchedCommandResult(
        record=record,
        observed_runtime=observed_runtime,
        contention_record=contention_record,
        contention_payload=contention_payload,
    )


def _normalize_next_actions(actions: Any) -> list[dict[str, str]]:
    if not isinstance(actions, list) or not actions:
        raise ValueError("Expected a non-empty next_actions list")
    normalized: list[dict[str, str]] = []
    for item in actions:
        if not isinstance(item, dict):
            raise ValueError("next_actions entries must be objects")
        name = item.get("name")
        reason = item.get("reason")
        command = item.get("command")
        if not all(isinstance(value, str) and value for value in (name, reason, command)):
            raise ValueError("next_actions entries must contain name, reason, and command")
        normalized.append({
            "name": name,
            "reason": reason,
            "command": command,
        })
    return normalized


def _extract_validated_next_actions(envelope: dict[str, Any]) -> list[dict[str, str]]:
    top_level = _normalize_next_actions(envelope.get("next_actions"))
    result = envelope.get("result")
    if not isinstance(result, dict):
        raise ValueError("Expected envelope.result to be an object")
    result_level = _normalize_next_actions(result.get("next_actions"))
    if top_level != result_level:
        raise ValueError("Top-level next_actions do not match result.next_actions")
    return top_level


def _find_next_action(
    actions: list[dict[str, str]],
    name: str,
) -> dict[str, str]:
    matches = [action for action in actions if action["name"] == name]
    if not matches:
        raise ValueError(f"Missing next action '{name}'")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous next action '{name}'")
    return matches[0]


def _validate_followup_envelope(
    envelope: dict[str, Any],
    *,
    expected_command: str,
) -> list[dict[str, str]]:
    if envelope.get("command") != expected_command:
        raise ValueError(
            f"Expected command={expected_command}, got {envelope.get('command')}",
        )
    return _extract_validated_next_actions(envelope)


def _validate_train_envelope(
    envelope: dict[str, Any],
    *,
    expected_eval_data_path: Path,
) -> dict[str, str]:
    if envelope.get("command") != "mc train run":
        raise ValueError(f"Unexpected train envelope command: {envelope.get('command')}")

    next_actions = _extract_validated_next_actions(envelope)
    result = envelope.get("result")
    if not isinstance(result, dict):
        raise ValueError("Training envelope missing result payload")

    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Training result missing artifacts block")

    artifact_dir = Path(str(artifacts.get("artifact_dir", ""))).expanduser().resolve()
    train_result_path = Path(str(artifacts.get("train_result_path", ""))).expanduser().resolve()
    eval_data_path = Path(str(artifacts.get("eval_data_path", ""))).expanduser().resolve()
    adapter_path = Path(str(result.get("adapter_path", ""))).expanduser().resolve()

    for path, label in (
        (artifact_dir, "artifact_dir"),
        (train_result_path, "train_result_path"),
        (eval_data_path, "eval_data_path"),
        (adapter_path, "adapter_path"),
    ):
        if not path.exists():
            raise ValueError(f"Training result {label} does not exist: {path}")

    if eval_data_path != expected_eval_data_path.expanduser().resolve():
        raise ValueError(
            f"Training result eval_data_path={eval_data_path} does not match "
            f"expected {expected_eval_data_path.expanduser().resolve()}",
        )

    runtime_status = result.get("runtime_status")
    if not isinstance(runtime_status, dict):
        raise ValueError("Training result missing runtime_status")
    if runtime_status.get("owner") != "training":
        raise ValueError("Training runtime_status owner must be 'training'")
    if runtime_status.get("phase") != "complete":
        raise ValueError("Training runtime_status phase must be 'complete'")
    if runtime_status.get("throughput_tokens_per_second") is None:
        raise ValueError("Training runtime_status missing throughput_tokens_per_second")
    memory = runtime_status.get("memory")
    if not isinstance(memory, dict):
        raise ValueError("Training runtime_status missing memory snapshot")
    if "active_gpu_memory_gb" not in memory or "peak_gpu_memory_gb" not in memory:
        raise ValueError("Training runtime_status memory missing active/peak GPU usage")

    capability_manifest = result.get("capability_manifest")
    if not isinstance(capability_manifest, dict):
        raise ValueError("Training result missing capability_manifest")

    evaluate_action = _find_next_action(next_actions, "evaluate")
    compare_action = _find_next_action(next_actions, "compare")
    export_action = _find_next_action(next_actions, "export")
    if str(eval_data_path) not in evaluate_action["command"]:
        raise ValueError("Evaluate next action does not reference the resolved eval dataset")
    if str(eval_data_path) not in compare_action["command"]:
        raise ValueError("Compare next action does not reference the resolved eval dataset")
    if str(adapter_path) not in compare_action["command"]:
        raise ValueError("Compare next action does not reference the trained adapter")
    if str(adapter_path) not in export_action["command"]:
        raise ValueError("Export next action does not reference the trained adapter")

    return {
        "artifact_dir": str(artifact_dir),
        "train_result_path": str(train_result_path),
        "eval_data_path": str(eval_data_path),
        "adapter_path": str(adapter_path),
    }


def _validate_export_payload(
    payload: dict[str, Any],
    *,
    expected_target_kind: str,
    expected_output_path: Path | None = None,
) -> dict[str, Any]:
    export_payload = payload.get("export")
    if not isinstance(export_payload, dict):
        raise ValueError("Export command missing export payload")
    if export_payload.get("target_kind") != expected_target_kind:
        raise ValueError(
            f"Expected target_kind={expected_target_kind}, got {export_payload.get('target_kind')}",
        )
    output_path = Path(str(export_payload.get("output_path", ""))).expanduser().resolve()
    if expected_output_path is not None and output_path != expected_output_path.expanduser().resolve():
        raise ValueError(
            f"Expected export output_path={expected_output_path.expanduser().resolve()}, "
            f"got {output_path}",
        )
    if not output_path.exists():
        raise ValueError(f"Export output_path does not exist: {output_path}")
    capability_manifest = export_payload.get("capability_manifest")
    if not isinstance(capability_manifest, dict):
        raise ValueError("Export payload missing capability_manifest")
    if expected_target_kind == "deployment_quantized":
        quantization = export_payload.get("quantization")
        if not isinstance(quantization, dict):
            raise ValueError("Quantized export missing quantization payload")
    return {
        "target_kind": str(export_payload["target_kind"]),
        "output_path": str(output_path),
    }


def _runtime_phases(observations: list[dict[str, Any]]) -> list[str]:
    phases: list[str] = []
    for observation in observations:
        phase = observation.get("phase")
        if isinstance(phase, str) and phase not in phases:
            phases.append(phase)
    return phases


def _write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Canonical CLI Workflow Smoke",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Started: `{summary['started_at']}`",
        f"- Finished: `{summary['finished_at']}`",
        f"- Success: `{summary['success']}`",
        f"- Model: `{summary['model_path']}`",
        f"- Train data: `{summary['train_data']}`",
        f"- Eval data: `{summary['eval_data']}`",
        f"- MODELCYPHER_HOME: `{summary['modelcypher_home']}`",
        "",
        "## Stages",
        "",
    ]
    for stage in summary.get("stages", []):
        lines.append(
            f"- `{stage['stage']}`: "
            f"{'pass' if stage['passed'] else 'fail'}"
            + (f" ({stage['note']})" if stage.get("note") else ""),
        )
    lines.extend(["", "## Commands", ""])
    for record in summary.get("command_records", []):
        lines.append(
            f"- `{record['stage']}`: `{shlex.join(record['command'])}` "
            f"(exit={record['exit_code']}, duration={record['duration_seconds']:.2f}s)",
        )
    lines.extend(["", "## Runtime Phases", ""])
    runtime_observations = summary.get("runtime_observations", {})
    for stage_name, observations in runtime_observations.items():
        phases = _runtime_phases(observations)
        lines.append(f"- `{stage_name}`: {', '.join(phases) if phases else 'none observed'}")
    lines.extend(["", "## Artifacts", ""])
    for name, artifact_path in sorted(summary.get("artifacts", {}).items()):
        lines.append(f"- `{name}`: `{artifact_path}`")
    contention_payload = summary.get("expected_contention_failure")
    if isinstance(contention_payload, dict):
        lines.extend(["", "## Contention", ""])
        active_runtime = contention_payload.get("error", {}).get("active_runtime", {})
        owner = active_runtime.get("owner", "unknown")
        phase = active_runtime.get("phase", "unknown")
        lines.append(
            f"- Expected busy failure observed while export owned the runtime "
            f"(owner=`{owner}`, phase=`{phase}`).",
        )
    failure = summary.get("failure")
    if isinstance(failure, dict):
        lines.extend(["", "## Failure", ""])
        lines.append(f"- Stage: `{failure.get('stage', 'unknown')}`")
        lines.append(f"- Detail: `{failure.get('detail', 'unknown')}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    model_path = Path(args.model_path).expanduser().resolve()
    train_data_path = Path(args.train_data).expanduser().resolve()
    eval_data_path = Path(args.eval_data).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    run_dir = output_root / _timestamp_slug()
    run_dir.mkdir(parents=True, exist_ok=True)

    modelcypher_home = run_dir / "modelcypher_home"
    payloads_dir = run_dir / "payloads"
    payloads_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()
    summary: dict[str, Any] = {
        "run_id": run_dir.name,
        "started_at": started_at,
        "finished_at": None,
        "success": False,
        "model_path": str(model_path),
        "train_data": str(train_data_path),
        "eval_data": str(eval_data_path),
        "modelcypher_home": str(modelcypher_home),
        "command_records": [],
        "stages": [],
        "runtime_observations": {},
        "artifacts": {},
        "expected_contention_failure": None,
        "failure_payloads": {},
        "failure": None,
    }
    current_stage = "setup"

    try:
        for path, label in (
            (model_path, "model_path"),
            (train_data_path, "train_data"),
            (eval_data_path, "eval_data"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} does not exist: {path}")

        active_conflicts = _detect_gpu_process_conflicts()
        if active_conflicts:
            raise RuntimeError(
                "GPU-heavy processes already active before smoke run: "
                + "; ".join(active_conflicts),
            )

        adapter_output_dir = run_dir / "artifacts" / "adapter"
        train_command = [
            "poetry",
            "run",
            "mc",
            "train",
            "run",
            "--model",
            str(model_path),
            "--data",
            str(train_data_path),
            "--eval-data",
            str(eval_data_path),
            "--output",
            str(adapter_output_dir),
            "--json",
        ]
        current_stage = "train.run"
        train_result = _run_command_with_runtime_watch(
            stage="train.run",
            command=train_command,
            run_dir=run_dir,
            modelcypher_home=modelcypher_home,
            expected_owner="training",
            timeout_seconds=TRAIN_TIMEOUT_SECONDS,
        )
        summary["command_records"].append(train_result.record.to_dict())
        summary["runtime_observations"]["train.run"] = train_result.observed_runtime
        train_payload = _read_command_payload(train_result.record)
        _write_json(payloads_dir / "train.run.json", train_payload)
        summary["artifacts"].update(
            _validate_train_envelope(
                train_payload,
                expected_eval_data_path=eval_data_path,
            ),
        )
        summary["stages"].append({"stage": "train.run", "passed": True})

        next_actions = _extract_validated_next_actions(train_payload)
        evaluate_command = _command_to_argv(_find_next_action(next_actions, "evaluate")["command"])
        compare_command = _command_to_argv(_find_next_action(next_actions, "compare")["command"])
        quantized_export_command = _command_to_argv(_find_next_action(next_actions, "export")["command"])

        current_stage = "train.evaluate"
        evaluate_record = _run_command(
            stage="train.evaluate",
            command=evaluate_command,
            run_dir=run_dir,
            modelcypher_home=modelcypher_home,
            timeout_seconds=FOLLOWUP_TIMEOUT_SECONDS,
        )
        summary["command_records"].append(evaluate_record.to_dict())
        evaluate_payload = _read_command_payload(evaluate_record)
        _write_json(payloads_dir / "train.evaluate.json", evaluate_payload)
        _validate_followup_envelope(evaluate_payload, expected_command="mc train evaluate")
        summary["stages"].append({"stage": "train.evaluate", "passed": True})

        current_stage = "train.compare"
        compare_record = _run_command(
            stage="train.compare",
            command=compare_command,
            run_dir=run_dir,
            modelcypher_home=modelcypher_home,
            timeout_seconds=FOLLOWUP_TIMEOUT_SECONDS,
        )
        summary["command_records"].append(compare_record.to_dict())
        compare_payload = _read_command_payload(compare_record)
        _write_json(payloads_dir / "train.compare.json", compare_payload)
        _validate_followup_envelope(compare_payload, expected_command="mc train compare")
        summary["stages"].append({"stage": "train.compare", "passed": True})

        adapter_path = Path(summary["artifacts"]["adapter_path"]).expanduser().resolve()
        contention_output_dir = run_dir / "exports" / "contention_busy"
        contention_command = [
            "poetry",
            "run",
            "mc",
            "train",
            "export",
            "--model",
            str(model_path),
            "--adapter",
            str(adapter_path),
            "--output",
            str(contention_output_dir),
            "--target",
            "merged_fp16",
            "--json",
        ]
        current_stage = "train.export.deployment_quantized"
        quantized_export_result = _run_command_with_runtime_watch(
            stage="train.export.deployment_quantized",
            command=quantized_export_command,
            run_dir=run_dir,
            modelcypher_home=modelcypher_home,
            expected_owner="export",
            timeout_seconds=EXPORT_TIMEOUT_SECONDS,
            contention_command=contention_command,
        )
        summary["command_records"].append(quantized_export_result.record.to_dict())
        summary["runtime_observations"]["train.export.deployment_quantized"] = (
            quantized_export_result.observed_runtime
        )
        if quantized_export_result.contention_record is None:
            raise RuntimeError("Quantized export did not execute the contention check")
        summary["command_records"].append(quantized_export_result.contention_record.to_dict())
        if quantized_export_result.contention_payload is None:
            raise RuntimeError("Quantized export contention check did not return active_runtime")
        summary["expected_contention_failure"] = quantized_export_result.contention_payload
        quantized_export_payload = _read_command_payload(quantized_export_result.record)
        _write_json(
            payloads_dir / "train.export.deployment_quantized.json",
            quantized_export_payload,
        )
        summary["artifacts"]["deployment_quantized_output"] = _validate_export_payload(
            quantized_export_payload,
            expected_target_kind="deployment_quantized",
        )["output_path"]
        summary["stages"].append({
            "stage": "train.export.deployment_quantized",
            "passed": True,
        })
        summary["stages"].append({
            "stage": "train.export.contention",
            "passed": True,
            "note": "structured busy failure observed",
        })

        merged_fp16_output_dir = run_dir / "exports" / "merged_fp16"
        merged_fp16_command = [
            "poetry",
            "run",
            "mc",
            "train",
            "export",
            "--model",
            str(model_path),
            "--adapter",
            str(adapter_path),
            "--output",
            str(merged_fp16_output_dir),
            "--target",
            "merged_fp16",
            "--json",
        ]
        current_stage = "train.export.merged_fp16"
        merged_fp16_record = _run_command(
            stage="train.export.merged_fp16",
            command=merged_fp16_command,
            run_dir=run_dir,
            modelcypher_home=modelcypher_home,
            timeout_seconds=EXPORT_TIMEOUT_SECONDS,
        )
        summary["command_records"].append(merged_fp16_record.to_dict())
        merged_fp16_payload = _read_command_payload(merged_fp16_record)
        _write_json(payloads_dir / "train.export.merged_fp16.json", merged_fp16_payload)
        summary["artifacts"]["merged_fp16_output"] = _validate_export_payload(
            merged_fp16_payload,
            expected_target_kind="merged_fp16",
            expected_output_path=merged_fp16_output_dir,
        )["output_path"]
        summary["stages"].append({"stage": "train.export.merged_fp16", "passed": True})

        summary["success"] = True
        return summary
    except Exception as exc:
        summary["failure"] = {
            "stage": current_stage,
            "detail": str(exc),
        }
        if not any(stage["stage"] == current_stage for stage in summary["stages"]):
            summary["stages"].append({"stage": current_stage, "passed": False})
        if isinstance(exc, CommandExecutionError) and exc.record is not None:
            if not any(
                record["stage"] == exc.record.stage and record["stdout_path"] == exc.record.stdout_path
                for record in summary["command_records"]
            ):
                summary["command_records"].append(exc.record.to_dict())
            try:
                failure_payload = _read_command_payload(exc.record)
            except ValueError:
                failure_payload = None
            if isinstance(failure_payload, dict):
                summary["failure_payloads"][exc.record.stage] = failure_payload
                _write_json(payloads_dir / f"{_stage_slug(exc.record.stage)}.failure.json", failure_payload)
        raise
    finally:
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(run_dir / "summary.json", summary)
        _write_report(summary, run_dir / "REPORT.md")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_smoke(args)
    except Exception as exc:
        sys.stderr.write(f"canonical_train_workflow_smoke failed: {exc}\n")
        return 1

    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
