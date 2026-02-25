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

"""System status and readiness service."""

from __future__ import annotations

import json
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend
    from modelcypher.backends import BackendDescriptor


class _StorePaths(Protocol):
    base: Path


class _ModelStore(Protocol):
    paths: _StorePaths


class SystemService:
    def __init__(
        self,
        model_store: "_ModelStore",
        backend: "Backend | None" = None,
    ) -> None:
        self._model_store = model_store
        self._backend = backend

    def status(self) -> dict:
        return self.readiness()

    def readiness(self) -> dict:
        from modelcypher.backends import detect_default_backend_type, probe_backends

        probes = probe_backends(explicit=False)
        preferred_backend = detect_default_backend_type()
        preferred_probe = next(
            (probe for probe in probes if probe.key == preferred_backend),
            None,
        )
        has_backend = any(probe.available for probe in probes)
        system_memory = self._system_memory_bytes()
        memory_gb = int(system_memory / (1024**3)) if system_memory else 0
        backend_versions = {
            probe.key: probe.system_info.get("version")
            for probe in probes
        }

        disk_total, disk_used, disk_free = self._disk_usage(self._model_store.paths.base)
        disk_free_gb = int(disk_free / (1024**3))

        score = 0
        score += 40 if has_backend else 0
        score += 20 if memory_gb >= 16 else (10 if memory_gb >= 8 else 0)
        score += 20 if disk_free_gb >= 50 else (10 if disk_free_gb >= 20 else 0)
        if preferred_probe and preferred_probe.available:
            score += 20

        readiness_score = min(score, 100)

        backend_health = {
            probe.key: 100 if probe.available else 0
            for probe in probes
        }

        return {
            "machineName": platform.node(),
            "preferredBackend": preferred_backend,
            "readinessScore": readiness_score,
            "scoreBreakdown": {
                "totalScore": readiness_score,
                "datasetScore": 100,
                "memoryFitScore": 100 if memory_gb >= 16 else 50,
                "systemPressureScore": 100,
                "backendHealth": backend_health,
                "storageScore": 100 if disk_free_gb > 100 else 50,
                "preflightScore": readiness_score,
            },
            "resources": {
                "gpuMemoryBytes": system_memory // 2 if system_memory else 0,
                "systemMemoryBytes": system_memory,
                "diskFreeBytes": disk_free,
            },
            "backends": [self._probe_payload(probe) for probe in probes],
            "backendVersions": backend_versions,
            "blockers": [] if has_backend else ["No backend available"],
        }

    def _disk_usage(self, path: Path) -> tuple[int, int, int]:
        try:
            import shutil
            total, used, free = shutil.disk_usage(path)
            return total, used, free
        except Exception:
            return 0, 0, 0

    def probe(self, target: str) -> dict:
        from modelcypher.backends import probe_backends

        probes = probe_backends(explicit=True)
        system_memory = self._system_memory_bytes()
        gpu_memory = system_memory // 2 if system_memory else 0
        memory_payload = {"systemBytes": system_memory, "gpuBytes": gpu_memory}
        backend_payloads = [self._probe_payload(probe) for probe in probes]

        if target == "memory":
            return {"target": target, "memory": memory_payload}
        for probe in probes:
            if target == probe.key:
                return {
                    "target": target,
                    "backend": self._probe_payload(probe),
                    "memory": memory_payload,
                }
        if target in ("backends", "all"):
            return {"target": target, "backends": backend_payloads, "memory": memory_payload}
        return {"target": target, "backends": backend_payloads, "memory": memory_payload}

    def memory_profile(
        self,
        model: str,
        prompt: str | None = None,
        train_probe: bool = False,
        decode_tokens: int = 32,
    ) -> dict[str, Any]:
        """Profile stage-wise GPU memory for model load/inference.

        Stages: baseline -> load -> tokenize -> forward -> bounded decode windows
        -> optional train probe surrogate.
        """
        backend = self._ensure_backend()
        model_path = Path(model).expanduser().resolve()
        config_path = model_path / "config.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json in model directory: {model_path}")
        if decode_tokens < 1:
            raise ValueError("decode_tokens must be >= 1")

        prompt_text = prompt or "Summarize the geometric structure of this prompt."
        memory_stages: list[dict[str, Any]] = []
        runtime_stages: list[dict[str, Any]] = []
        started = time.perf_counter()

        def _now() -> str:
            return datetime.now(timezone.utc).isoformat()

        def _capture(stage: str) -> dict[str, Any]:
            active_gb = float(backend.get_active_memory_gb())
            peak_gb = float(backend.get_peak_memory_gb())
            payload = {
                "stage": stage,
                "timestamp": _now(),
                "active_gb": active_gb,
                "peak_gb": max(peak_gb, active_gb),
                "elapsed_sec": float(time.perf_counter() - started),
            }
            memory_stages.append(payload)
            return payload

        def _timed_stage(stage: str, fn: Any) -> Any:
            backend.reset_peak_memory()
            t0 = time.perf_counter()
            result = fn()
            runtime_stages.append({
                "stage": stage,
                "timestamp": _now(),
                "duration_sec": float(time.perf_counter() - t0),
            })
            return result

        backend.clear_cache()
        backend.reset_peak_memory()
        _capture("baseline")

        model_obj, tokenizer = _timed_stage(
            "load",
            lambda: backend.load_model(str(model_path)),
        )
        _capture("load")

        param_count = self._count_parameters(backend, model_obj)
        quantization_mode, precision_bits = self._read_precision_metadata(config_path)

        token_ids = _timed_stage(
            "tokenize",
            lambda: backend.encode_tokens(tokenizer, prompt_text),
        )
        _capture("tokenize")

        def _run_forward() -> Any:
            logits = backend.collect_logits(
                model_obj,
                tokenizer,
                prompt_text,
                token_ids=token_ids,
            )
            backend.eval(logits)
            return logits

        _timed_stage("forward", _run_forward)
        _capture("forward")

        decode_windows = sorted(set([
            max(1, decode_tokens // 4),
            max(1, decode_tokens // 2),
            decode_tokens,
        ]))
        decode_points: list[dict[str, Any]] = []
        for max_toks in decode_windows:
            response = _timed_stage(
                f"decode_{max_toks}",
                lambda max_toks=max_toks: backend.generate(
                    model_obj,
                    tokenizer,
                    prompt_text,
                    max_tokens=max_toks,
                ),
            )
            stage = _capture(f"decode_{max_toks}")
            generated_token_count = len(backend.encode_tokens(tokenizer, response))
            decode_points.append({
                "max_tokens": int(max_toks),
                "active_gb": float(stage["active_gb"]),
                "peak_gb": float(stage["peak_gb"]),
                "generated_tokens": int(generated_token_count),
            })

        decode_slope = self._decode_slope(decode_points)

        train_probe_payload: dict[str, Any] | None = None
        if train_probe:
            try:
                probe_result = _timed_stage(
                    "train_probe_nblora_step",
                    lambda: self._run_nb_lora_train_probe(
                        backend=backend,
                        model=model_obj,
                        tokenizer=tokenizer,
                        prompt=prompt_text,
                    ),
                )
                train_stage = _capture("train_probe_nblora_step")
                train_probe_payload = {
                    "enabled": True,
                    "mode": "nblora_step",
                    "n_trainable_params": int(probe_result.get("n_trainable_params", 0)),
                    "spectral_bounds_ok": probe_result.get("spectral_bounds_ok"),
                    "max_spectral_ratio": probe_result.get("max_spectral_ratio"),
                    "train_iters": probe_result.get("train_iters"),
                    "last_loss": probe_result.get("last_loss"),
                    "stop_reason": probe_result.get("stop_reason"),
                    "geometry_mode": probe_result.get("geometry_mode"),
                    "seq_length": probe_result.get("seq_length"),
                    "target_module_count": probe_result.get("target_module_count"),
                    "n_lora_layers": probe_result.get("n_lora_layers"),
                    "train_step_active_gb": float(train_stage["active_gb"]),
                    "train_step_peak_gb": float(train_stage["peak_gb"]),
                }
            except Exception as exc:
                trainable_params = self._count_trainable_parameters(backend, model_obj)

                def _train_surrogate() -> None:
                    logits = backend.collect_logits(
                        model_obj,
                        tokenizer,
                        prompt_text,
                        token_ids=token_ids,
                    )
                    backend.eval(logits)

                _timed_stage("train_probe_forward", _train_surrogate)
                train_stage = _capture("train_probe_forward")
                train_probe_payload = {
                    "enabled": True,
                    "mode": "forward_surrogate",
                    "probe_error": str(exc),
                    "n_trainable_params": int(trainable_params),
                    "spectral_bounds_ok": None,
                    "train_step_active_gb": float(train_stage["active_gb"]),
                    "train_step_peak_gb": float(train_stage["peak_gb"]),
                }

        return {
            "model": str(model_path),
            "model_id": model_path.name,
            "param_count": int(param_count),
            "precision_bits": int(precision_bits),
            "quantization_mode": quantization_mode,
            "prompt": prompt_text,
            "prompt_token_count": int(len(token_ids)),
            "decode_tokens": int(decode_tokens),
            "memory_stages": memory_stages,
            "runtime_stages": runtime_stages,
            "decode_slope": {
                "gb_per_token": float(decode_slope),
                "windows": decode_points,
            },
            "train_probe": train_probe_payload,
        }

    @staticmethod
    def _probe_payload(probe: "BackendDescriptor") -> dict:
        return {
            "key": probe.key,
            "displayName": probe.display_name,
            "available": probe.available,
            "error": probe.error,
            "systemInfo": probe.system_info,
        }

    @staticmethod
    def _system_memory_bytes() -> int:
        try:
            import os
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages * page_size)
        except Exception:
            return 0

    def _ensure_backend(self) -> "Backend":
        if self._backend is None:
            from modelcypher.core.domain._backend import get_default_backend

            self._backend = get_default_backend()
        return self._backend

    @staticmethod
    def _read_precision_metadata(config_path: Path) -> tuple[str | None, int]:
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return None, 16

        quantization = payload.get("quantization")
        torch_dtype = str(payload.get("torch_dtype") or "").lower()

        if isinstance(quantization, dict):
            bits = quantization.get("bits")
            if isinstance(bits, int) and bits > 0:
                mode = str(quantization.get("type") or quantization.get("scheme") or "quantized")
                return mode, bits
            mode = str(quantization.get("type") or quantization.get("scheme") or "quantized")
            inferred = SystemService._bits_from_text(mode)
            return mode, inferred if inferred is not None else 16

        if isinstance(quantization, str):
            inferred = SystemService._bits_from_text(quantization)
            return quantization, inferred if inferred is not None else 16

        if "float32" in torch_dtype:
            return "float32", 32
        if "float16" in torch_dtype or "bfloat16" in torch_dtype:
            return torch_dtype or "float16", 16

        return None, 16

    @staticmethod
    def _bits_from_text(text: str) -> int | None:
        match = re.search(r"(\d+)\s*-?\s*bit", text.lower())
        if not match:
            match = re.search(r"\b(\d+)\b", text.lower())
        if match:
            value = int(match.group(1))
            if value > 0:
                return value
        return None

    @staticmethod
    def _count_parameters(backend: "Backend", model: Any) -> int:
        params_obj = getattr(model, "parameters", None)
        if callable(params_obj):
            try:
                flattened = backend.tree_flatten(params_obj())
            except Exception:
                return 0
            return SystemService._sum_tensor_elements(flattened, backend)
        return 0

    @staticmethod
    def _count_trainable_parameters(backend: "Backend", model: Any) -> int:
        params_obj = getattr(model, "trainable_parameters", None)
        if callable(params_obj):
            try:
                flattened = backend.tree_flatten(params_obj())
            except Exception:
                return 0
            return SystemService._sum_tensor_elements(flattened, backend)
        return SystemService._count_parameters(backend, model)

    @staticmethod
    def _sum_tensor_elements(
        flattened: list[tuple[str, Any]],
        backend: "Backend",
    ) -> int:
        total = 0
        for _, tensor in flattened:
            try:
                shape = backend.shape(tensor)
            except Exception:
                continue
            count = 1
            for dim in shape:
                count *= int(dim)
            total += count
        return total

    @staticmethod
    def _decode_slope(points: list[dict[str, Any]]) -> float:
        if len(points) < 2:
            return 0.0
        first = points[0]
        last = points[-1]
        token_delta = int(last["max_tokens"]) - int(first["max_tokens"])
        if token_delta <= 0:
            return 0.0
        active_delta = float(last["active_gb"]) - float(first["active_gb"])
        return active_delta / float(token_delta)

    @staticmethod
    def _run_nb_lora_train_probe(
        *,
        backend: "Backend",
        model: Any,
        tokenizer: Any,
        prompt: str,
    ) -> dict[str, Any]:
        # NB-LoRA adapter probe currently targets MLX backend only.
        if not hasattr(backend, "mx"):
            raise RuntimeError("nblora train probe requires MLX backend")

        from modelcypher.backends.mlx_training_adapter import MLXTrainingAdapter

        adapter = MLXTrainingAdapter(backend)
        return adapter.run_train_probe_step(
            model,
            tokenizer,
            prompt=prompt,
            use_randomized_geometry=True,
        )
