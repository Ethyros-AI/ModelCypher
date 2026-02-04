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

"""
Training Service for LoRA adapter fine-tuning.

Orchestrates model training jobs including preflight checks, job management,
and progress monitoring. Supports pause/resume and checkpoint recovery.

Example:
    service = TrainingService(engine=training_engine)
    preflight = service.preflight(config)
    if preflight["canProceed"]:
        job, events = service.start(config, stream=True)
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    detect_model_dtype,
    machine_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.training import ComputePrecision, Hyperparameters, TrainingSpec

if TYPE_CHECKING:
    from modelcypher.ports.training import TrainingEngine
    from modelcypher.ports.model_loader import ModelLoaderPort


class TrainingService:
    def __init__(self, engine: "TrainingEngine", model_loader: "ModelLoaderPort | None" = None) -> None:
        self.engine = engine
        self._model_loader = model_loader

    def preflight(self, config: TrainingSpec) -> dict:
        result = self.engine.preflight(config)
        return {
            "predictedBatchSize": result.predicted_batch_size,
            "estimatedVRAMUsageBytes": result.estimated_vram_bytes,
            "availableVRAMBytes": result.available_vram_bytes,
            "canProceed": result.can_proceed,
        }

    def start(
        self, config: TrainingSpec, stream: bool = False, detach: bool = False
    ) -> tuple[dict, list[dict]]:
        job, events = self.engine.start(config, stream_events=stream, detach=detach)
        # Support both old config.batch_size and new config.hyperparameters.batch_size
        batch_size = (
            config.hyperparameters.batch_size
            if hasattr(config, "hyperparameters")
            else getattr(config, "batch_size", 1)
        )
        return {
            "jobId": job.job_id,
            "batchSize": batch_size,
        }, events

    def derive_spec(
        self,
        model: str,
        dataset: str,
        output_path: str,
        resume_from: str | None = None,
    ) -> TrainingSpec:
        """Derive training spec from model/dataset geometry (no user knobs)."""
        if self._model_loader is None:
            raise RuntimeError("TrainingService requires model_loader for geometry-derived specs.")

        model_dir = Path(model).expanduser().resolve()
        if not model_dir.exists():
            raise ValueError(f"Model path does not exist: {model_dir}")

        dataset_path = Path(dataset).expanduser().resolve()
        if not dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")

        model_obj, tokenizer = self._model_loader.load_model_for_training(str(model_dir))
        hidden_dim = _resolve_hidden_dim(model_dir, model_obj)
        sample_count, max_token_len = _dataset_token_stats(dataset_path, tokenizer)
        if sample_count <= 0 or max_token_len <= 0:
            raise ValueError("Dataset contains no usable samples.")

        context_limit = _resolve_context_limit(model_dir, tokenizer)
        if context_limit is None:
            sequence_length = max_token_len
        else:
            sequence_length = min(context_limit, max_token_len)
        if sequence_length < 2:
            raise ValueError("Derived sequence_length must be >= 2.")

        backend = get_default_backend()
        params = _flatten_model_params(model_obj)
        if not params:
            raise ValueError("Model exposes no parameters for geometry derivation.")

        weights = {name: param for name, param in params}
        model_dtype = detect_model_dtype(weights, backend)
        precision = _precision_from_dtype(model_dtype)
        mixed_precision = precision is not ComputePrecision.FLOAT32

        eps_ref = params[0][1]
        eps = float(machine_epsilon(backend, eps_ref))
        sqrt_eps = sqrt_scalar(eps, backend)
        param_rms = _parameter_rms(params, backend)
        denom = max(param_rms, sqrt_eps)
        learning_rate = sqrt_eps / denom

        batch_size = 1  # Algebraic minimum; no heuristic batching.
        grad_accum = 1
        warmup_steps = 0
        weight_decay = eps
        epochs = max(1, int(math.ceil(hidden_dim / float(sample_count))))
        seed = _stable_seed(dataset_path)

        hyperparams = Hyperparameters(
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            sequence_length=sequence_length,
            gradient_accumulation_steps=grad_accum,
            gradient_checkpointing=False,
            mixed_precision=mixed_precision,
            compute_precision=precision,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            seed=seed,
            deterministic=True,
            optimizer_type="adamw",
        )

        return TrainingSpec(
            model_id=str(model_dir),
            dataset_path=str(dataset_path),
            output_path=output_path,
            hyperparameters=hyperparams,
            lora_config=None,
            resume_from_checkpoint_path=resume_from,
        )

    def status(self, job_id: str) -> dict:
        job = self.engine.status(job_id)
        return {
            "jobId": job.job_id,
            "status": job.status.value,
            "currentStep": job.current_step,
            "totalSteps": job.total_steps,
            "currentEpoch": job.current_epoch,
            "totalEpochs": job.total_epochs,
            "loss": job.loss,
            "learningRate": job.learning_rate,
            "createdAt": job.created_at.isoformat() + "Z",
            "updatedAt": job.updated_at.isoformat() + "Z",
            "modelId": job.model_id,
            "datasetPath": job.dataset_path,
        }

    def pause(self, job_id: str) -> dict:
        job = self.engine.pause(job_id)
        return {"jobId": job.job_id, "status": job.status.value}

    def resume(self, job_id: str) -> dict:
        job = self.engine.resume(job_id)
        return {"jobId": job.job_id, "status": job.status.value}

    def cancel(self, job_id: str) -> dict:
        job = self.engine.cancel(job_id)
        return {"jobId": job.job_id, "status": job.status.value}

    def logs(self, job_id: str, tail: int = 100) -> list[str]:
        return self.engine.logs(job_id, tail=tail)


def _stable_seed(dataset_path: Path) -> int:
    stat = dataset_path.stat()
    payload = f"{dataset_path}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16)


def _precision_from_dtype(dtype: object) -> ComputePrecision:
    name = getattr(dtype, "name", None) or getattr(dtype, "__name__", None) or str(dtype)
    name = name.lower()
    if "bfloat16" in name:
        return ComputePrecision.BFLOAT16
    if "float16" in name:
        return ComputePrecision.FLOAT16
    return ComputePrecision.FLOAT32


def _flatten_model_params(model: Any) -> list[tuple[str, Any]]:
    """Flatten model parameters to list of (name, array) tuples.

    Works with any model that has a parameters() method returning nested dicts.
    """
    params = getattr(model, "parameters", None)
    if params is None:
        return []

    def _flatten_dict(d: Any, prefix: str = "") -> list[tuple[str, Any]]:
        """Recursively flatten nested dicts/lists of parameters."""
        result = []
        if isinstance(d, dict):
            for k, v in d.items():
                new_key = f"{prefix}.{k}" if prefix else k
                result.extend(_flatten_dict(v, new_key))
        elif isinstance(d, (list, tuple)):
            for i, v in enumerate(d):
                new_key = f"{prefix}.{i}" if prefix else str(i)
                result.extend(_flatten_dict(v, new_key))
        elif hasattr(d, "shape"):  # Array-like object
            result.append((prefix, d))
        return result

    try:
        if callable(params):
            params = params()
        return _flatten_dict(params)
    except Exception:
        return []


def _array_size(arr: Any) -> int:
    if hasattr(arr, "size"):
        try:
            size = int(arr.size)
        except Exception:
            size = None
        if size is not None:
            return size
    shape = getattr(arr, "shape", None)
    if shape is None:
        return 0
    size = 1
    for dim in shape:
        size *= int(dim)
    return int(size)


def _parameter_rms(params: list[tuple[str, Any]], backend) -> float:
    total_sq = 0.0
    total_count = 0
    for _, param in params:
        arr = param if hasattr(param, "shape") else backend.array(param)
        sq = backend.sum(arr * arr)
        backend.eval(sq)
        total_sq += float(backend.to_scalar(sq))
        total_count += _array_size(arr)
    if total_count <= 0:
        return 0.0
    mean_sq = total_sq / float(total_count)
    return sqrt_scalar(mean_sq, backend)


def _resolve_context_limit(model_dir: Path, tokenizer: Any) -> int | None:
    candidates: list[int] = []
    for attr in (
        "model_max_length",
        "max_length",
        "max_seq_len",
        "max_sequence_length",
        "n_ctx",
        "context_length",
        "max_context_length",
    ):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, (int, float)):
            int_value = int(value)
            if int_value > 0:
                candidates.append(int_value)

    config_path = model_dir / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            config = {}
        for key in (
            "max_position_embeddings",
            "max_seq_len",
            "max_sequence_length",
            "n_ctx",
            "context_length",
            "seq_length",
        ):
            value = config.get(key)
            if isinstance(value, (int, float)) and value > 0:
                candidates.append(int(value))

    if not candidates:
        return None
    return min(candidates)


def _resolve_hidden_dim(model_dir: Path, model: Any) -> int:
    config_path = model_dir / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            config = {}
        for key in (
            "hidden_size",
            "hidden_dim",
            "d_model",
            "n_embd",
            "model_dim",
        ):
            value = config.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return int(value)

    dims: list[int] = []
    for _, param in _flatten_model_params(model):
        shape = getattr(param, "shape", None)
        if shape is None or len(shape) < 2:
            continue
        dims.append(int(shape[-1]))
    if dims:
        return max(dims)
    raise ValueError("Unable to resolve hidden dimension from model/config.")


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    if not hasattr(tokenizer, "encode"):
        return []
    try:
        encoded = tokenizer.encode(text)
    except Exception:
        return []
    if isinstance(encoded, list):
        return encoded
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    return []


def _extract_text(line: str, tokenizer: Any) -> str | None:
    line = line.strip()
    if not line:
        return None
    if line.startswith("{"):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            if isinstance(payload.get("text"), str):
                return payload["text"].strip()
            messages = payload.get("messages")
            if isinstance(messages, list):
                if hasattr(tokenizer, "apply_chat_template"):
                    return tokenizer.apply_chat_template(messages)
                return " ".join(
                    msg.get("content", "")
                    for msg in messages
                    if isinstance(msg, dict)
                ).strip()
    return line


def _dataset_token_stats(dataset_path: Path, tokenizer: Any) -> tuple[int, int]:
    sample_count = 0
    max_token_len = 0
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = _extract_text(line, tokenizer)
            if not text:
                continue
            token_ids = _encode_text(tokenizer, text)
            if not token_ids:
                continue
            sample_count += 1
            if len(token_ids) > max_token_len:
                max_token_len = len(token_ids)
    return sample_count, max_token_len
