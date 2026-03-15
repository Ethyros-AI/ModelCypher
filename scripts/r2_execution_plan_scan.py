#!/usr/bin/env python3
"""R2 falsifier harness for MLX-first RYS execution-plan scans."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import sys
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from modelcypher.adapters.execution_plan_applicator import apply_execution_plan
from modelcypher.adapters.model_loader import ModelLoader
from modelcypher.core.domain.atlas.probe_loader import load_all_probes
from modelcypher.core.domain.atlas.unified_atlas import AtlasProbe, AtlasSource
from modelcypher.core.domain.domains import AtlasDomain
from modelcypher.core.domain.geometry.cka import compute_geodesic_cka
from modelcypher.core.domain.geometry.gram_spectrum import compute_gram_spectrum
from modelcypher.core.domain.geometry.perturbation_bound import (
    compute_readout_effective_rank,
)
from modelcypher.core.domain.inference import LayerExecutionPlan
from modelcypher.core.domain.training.degeneration import (
    derive_ngram_order,
    ngram_repetition_rate,
)
from modelcypher.core.use_cases.benchmark_service import BenchmarkService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "nblora_vs_standard" / "r2_execution_plan_scan"
BENCHMARK_SUITE = ("gsm8k", "arc_easy", "boolq")
ARTIFACT_SCHEMA = "mc.r2.execution_plan_scan.v1"
PREDICTION_CONTRACT = (
    "If layer reuse can relieve inference-representation collapse without changing "
    "weights, then some repeated blocks will increase canonical trajectory rank or "
    "canonical intrinsic dimension while preserving or improving inference-manifold "
    "CKA and the frozen quick-suite readout relative to the identity plan."
)


@dataclass(frozen=True)
class PlanCandidate:
    """One execution-plan candidate in the scan."""

    key: str
    plan: LayerExecutionPlan
    rys_start: int | None
    rys_end: int | None

    def repeated_block(self) -> dict[str, int] | None:
        if self.rys_start is None or self.rys_end is None:
            return None
        return {
            "start": self.rys_start,
            "end": self.rys_end,
            "width": self.rys_end - self.rys_start,
        }


@dataclass(frozen=True)
class ScanConfig:
    """Immutable runtime configuration for the scan."""

    model_path: Path
    output_dir: Path
    top_k: int
    probe_manifest_path: Path | None = None
    max_probes: int | None = None
    behavior_limit_per_benchmark: int = 20
    seed: int = 42
    start_min: int | None = None
    start_max: int | None = None
    end_min: int | None = None
    end_max: int | None = None
    max_tokens: int = 512


@dataclass(frozen=True)
class ScanServices:
    """Injected services for the scan."""

    backend: Any
    activation_provider: Any
    verification_depth_service: Any
    benchmark_service: BenchmarkService
    model_loader: ModelLoader


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, default=_json_default) + "\n" for row in rows
    )
    path.write_text(payload, encoding="utf-8")


def _append_ledger(path: Path, event: str, **payload: Any) -> None:
    record = {
        "timestamp": _utc_now_iso(),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_json_default) + "\n")


def build_services() -> ScanServices:
    """Build production services for the scan."""
    from modelcypher.cli.composition import (
        get_activation_provider,
        get_backend,
        get_verification_depth_profile_service,
    )

    backend = get_backend()
    return ScanServices(
        backend=backend,
        activation_provider=get_activation_provider(),
        verification_depth_service=get_verification_depth_profile_service(),
        benchmark_service=BenchmarkService(),
        model_loader=ModelLoader(backend),
    )


def _probe_text(probe: AtlasProbe) -> str | None:
    for text in probe.support_texts:
        if text and text.strip():
            return text.strip()
    if probe.description and probe.description.strip():
        return probe.description.strip()
    if probe.name and probe.name.strip():
        return probe.name.strip()
    return None


def _probe_to_manifest_entry(probe: AtlasProbe) -> dict[str, Any]:
    selected_text = _probe_text(probe)
    if selected_text is None:
        raise ValueError(f"Probe {probe.probe_id} has no usable text")
    return {
        "probe_id": probe.probe_id,
        "id": probe.id,
        "source": probe.source.value,
        "domain": probe.domain.value,
        "name": probe.name,
        "description": probe.description,
        "category_name": probe.category_name,
        "verification_depth": probe.verification_depth,
        "selected_text": selected_text,
    }


def _probe_from_manifest_entry(entry: dict[str, Any]) -> AtlasProbe:
    return AtlasProbe(
        id=str(entry["id"]),
        source=AtlasSource(str(entry["source"])),
        domain=AtlasDomain(str(entry["domain"])),
        name=str(entry.get("name", "")),
        description=str(entry.get("description", "")),
        cross_domain_weight=1.0,
        category_name=str(entry.get("category_name", "")),
        support_texts=(str(entry["selected_text"]),),
        verification_depth=(
            int(entry["verification_depth"])
            if entry.get("verification_depth") is not None
            else None
        ),
    )


def _select_probes_round_robin(
    probes: list[AtlasProbe],
    max_probes: int | None,
) -> list[AtlasProbe]:
    selected = [
        probe for probe in probes if probe.verification_depth is not None and _probe_text(probe)
    ]
    selected.sort(
        key=lambda probe: (
            int(probe.verification_depth or 0),
            probe.source.value,
            probe.probe_id,
        )
    )
    if max_probes is None or max_probes >= len(selected):
        return selected

    buckets: dict[int, deque[AtlasProbe]] = defaultdict(deque)
    for probe in selected:
        buckets[int(probe.verification_depth or 0)].append(probe)

    ordered_levels = sorted(buckets.keys())
    chosen: list[AtlasProbe] = []
    while len(chosen) < max_probes:
        progress = False
        for level in ordered_levels:
            if len(chosen) >= max_probes:
                break
            if buckets[level]:
                chosen.append(buckets[level].popleft())
                progress = True
        if not progress:
            break
    return chosen


def load_or_create_probe_manifest(
    *,
    output_dir: Path,
    explicit_manifest_path: Path | None,
    max_probes: int | None,
) -> tuple[list[AtlasProbe], dict[str, Any], Path]:
    """Load a frozen probe manifest, or create one deterministically."""
    frozen_manifest_path = output_dir / "probe_manifest.json"

    if explicit_manifest_path is not None:
        manifest_path = explicit_manifest_path.expanduser().resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        probes = [_probe_from_manifest_entry(entry) for entry in payload["probes"]]
        if manifest_path != frozen_manifest_path:
            _write_json(frozen_manifest_path, payload)
        return probes, payload, frozen_manifest_path

    if frozen_manifest_path.exists():
        payload = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
        probes = [_probe_from_manifest_entry(entry) for entry in payload["probes"]]
        return probes, payload, frozen_manifest_path

    probes = _select_probes_round_robin(load_all_probes(), max_probes)
    payload = {
        "schema": f"{ARTIFACT_SCHEMA}.probe_manifest",
        "created_at": _utc_now_iso(),
        "selection_strategy": "verification_depth_round_robin",
        "requested_max_probes": max_probes,
        "selected_probe_count": len(probes),
        "probes": [_probe_to_manifest_entry(probe) for probe in probes],
    }
    _write_json(frozen_manifest_path, payload)
    return probes, payload, frozen_manifest_path


def build_plan_candidates(
    *,
    base_layer_count: int,
    start_min: int | None,
    start_max: int | None,
    end_min: int | None,
    end_max: int | None,
) -> list[PlanCandidate]:
    """Build the identity plan plus all valid RYS plans inside the bounds."""
    resolved_start_min = 0 if start_min is None else max(0, start_min)
    resolved_start_max = (
        base_layer_count - 1
        if start_max is None
        else min(base_layer_count - 1, start_max)
    )
    resolved_end_min = 1 if end_min is None else max(1, end_min)
    resolved_end_max = (
        base_layer_count if end_max is None else min(base_layer_count, end_max)
    )

    candidates = [
        PlanCandidate(
            key="identity",
            plan=LayerExecutionPlan.identity(base_layer_count),
            rys_start=None,
            rys_end=None,
        )
    ]

    for start in range(resolved_start_min, resolved_start_max + 1):
        for end in range(max(start + 1, resolved_end_min), resolved_end_max + 1):
            plan = LayerExecutionPlan.from_rys(base_layer_count, start, end)
            candidates.append(
                PlanCandidate(
                    key=f"rys:{start}:{end}",
                    plan=plan,
                    rys_start=start,
                    rys_end=end,
                )
            )

    return candidates


def _layer_profiles_from_trajectories(
    backend: Any,
    plan: LayerExecutionPlan,
    positions: dict[int, Any],
    velocities: dict[int, Any],
) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for execution_step in sorted(positions.keys()):
        position_matrix = positions[execution_step]
        velocity_matrix = velocities.get(execution_step)
        if velocity_matrix is not None and int(velocity_matrix.shape[0]) > 0:
            trajectory_matrix = backend.concatenate([position_matrix, velocity_matrix], axis=0)
        else:
            trajectory_matrix = position_matrix

        backend.eval(position_matrix, trajectory_matrix)
        activation_spectrum = compute_gram_spectrum(position_matrix, backend=backend)
        trajectory_spectrum = compute_gram_spectrum(trajectory_matrix, backend=backend)
        hidden_dim = int(activation_spectrum.d_features)
        trajectory_rank = int(trajectory_spectrum.numeric_rank)
        profiles.append(
            {
                "executionStep": execution_step,
                "sourceLayerIndex": int(plan.layer_indices[execution_step]),
                "activationRank": int(activation_spectrum.numeric_rank),
                "trajectoryRank": trajectory_rank,
                "intrinsicDimension": float(trajectory_spectrum.intrinsic_dimension),
                "conditionNumber": float(trajectory_spectrum.condition_number),
                "hiddenDim": hidden_dim,
                "probeSampleCount": int(activation_spectrum.n_samples),
                "trajectorySampleCount": int(trajectory_spectrum.n_samples),
                "nullRank": hidden_dim - trajectory_rank,
            }
        )
    return profiles


def _norm_summary(
    backend: Any,
    positions: dict[int, Any],
) -> dict[str, Any]:
    mean_layer_norms: list[float] = []
    for execution_step in sorted(positions.keys()):
        position_matrix = positions[execution_step]
        row_norms = backend.sqrt(backend.sum(position_matrix * position_matrix, axis=1))
        mean_norm = backend.mean(row_norms)
        backend.eval(mean_norm)
        mean_layer_norms.append(float(backend.to_scalar(mean_norm)))

    jumps = [
        mean_layer_norms[idx] - mean_layer_norms[idx - 1]
        for idx in range(1, len(mean_layer_norms))
    ]
    return {
        "meanLayerNorms": mean_layer_norms,
        "consecutiveNormJumps": jumps,
        "maxConsecutiveNormJump": max((abs(value) for value in jumps), default=0.0),
    }


def evaluate_stage1_candidate(
    *,
    candidate: PlanCandidate,
    model: Any,
    tokenizer: Any,
    probe_texts: list[str],
    services: ScanServices,
) -> dict[str, Any]:
    """Evaluate the cheap geometry bundle for one candidate."""
    with apply_execution_plan(model, candidate.plan):
        trajectories = services.activation_provider.collect_trajectory_batch(
            model, tokenizer, probe_texts
        )

    profiles = _layer_profiles_from_trajectories(
        services.backend,
        candidate.plan,
        trajectories.positions,
        trajectories.velocities,
    )
    finite_ids = [
        profile["intrinsicDimension"]
        for profile in profiles
        if math.isfinite(profile["intrinsicDimension"])
    ]
    finite_conditions = [
        profile["conditionNumber"]
        for profile in profiles
        if math.isfinite(profile["conditionNumber"])
    ]
    canonical_trajectory_rank = max(
        (profile["trajectoryRank"] for profile in profiles),
        default=None,
    )
    hidden_dim = profiles[0]["hiddenDim"] if profiles else None
    canonical_null_rank = (
        int(hidden_dim - canonical_trajectory_rank)
        if hidden_dim is not None and canonical_trajectory_rank is not None
        else None
    )
    return {
        "planKey": candidate.key,
        "plan": candidate.plan.to_dict(),
        "repeatedBlock": candidate.repeated_block(),
        "stage1": {
            "layerProfiles": profiles,
            "canonicalTrajectoryRank": canonical_trajectory_rank,
            "canonicalIntrinsicDimension": max(finite_ids) if finite_ids else None,
            "canonicalNullRank": canonical_null_rank,
            "canonicalConditionNumber": max(finite_conditions) if finite_conditions else None,
            "normSummary": _norm_summary(services.backend, trajectories.positions),
        },
        "stage2": None,
    }


def _condition_number_inflation(value: float | None, baseline: float | None) -> float:
    if value is None:
        return float("inf")
    if baseline is None:
        return value
    if not math.isfinite(value) or not math.isfinite(baseline):
        return float("inf")
    return value - baseline


def annotate_stage1_rankings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach deltas and deterministic rankings to stage-1 rows."""
    identity = next(row for row in rows if row["planKey"] == "identity")
    baseline = identity["stage1"]
    baseline_rank = baseline["canonicalTrajectoryRank"] or 0
    baseline_id = baseline["canonicalIntrinsicDimension"] or 0.0
    baseline_cond = baseline["canonicalConditionNumber"]
    baseline_max_jump = baseline["normSummary"]["maxConsecutiveNormJump"]

    for row in rows:
        stage1 = row["stage1"]
        stage1["deltasVsIdentity"] = {
            "canonicalTrajectoryRank": (stage1["canonicalTrajectoryRank"] or 0) - baseline_rank,
            "canonicalIntrinsicDimension": (
                (stage1["canonicalIntrinsicDimension"] or 0.0) - baseline_id
            ),
            "canonicalConditionNumberInflation": _condition_number_inflation(
                stage1["canonicalConditionNumber"],
                baseline_cond,
            ),
            "maxNormJumpInflation": (
                stage1["normSummary"]["maxConsecutiveNormJump"] - baseline_max_jump
            ),
        }

    non_identity = [row for row in rows if row["planKey"] != "identity"]
    non_identity.sort(
        key=lambda row: (
            -(row["stage1"]["deltasVsIdentity"]["canonicalTrajectoryRank"]),
            -(row["stage1"]["deltasVsIdentity"]["canonicalIntrinsicDimension"]),
            row["stage1"]["deltasVsIdentity"]["canonicalConditionNumberInflation"],
            row["stage1"]["deltasVsIdentity"]["maxNormJumpInflation"],
            row["repeatedBlock"]["start"] if row["repeatedBlock"] else -1,
            row["repeatedBlock"]["end"] if row["repeatedBlock"] else -1,
        )
    )
    identity["stage1"]["rank"] = 0
    for idx, row in enumerate(non_identity, start=1):
        row["stage1"]["rank"] = idx
    return [identity, *non_identity]


def select_stage2_plan_keys(rows: list[dict[str, Any]], top_k: int) -> list[str]:
    """Select identity, top-K, and bottom-2 stage-2 plans."""
    ordered = [row["planKey"] for row in rows if row["planKey"] != "identity"]
    selected = ["identity"]
    selected.extend(ordered[: max(0, top_k)])
    selected.extend(ordered[-2:])

    deduped: list[str] = []
    for key in selected:
        if key not in deduped:
            deduped.append(key)
    return deduped


def _final_positions_by_original_layer(
    candidate: PlanCandidate,
    positions: dict[int, Any],
) -> dict[int, Any]:
    final_positions: dict[int, Any] = {}
    for execution_step, source_layer_index in enumerate(candidate.plan.layer_indices):
        final_positions[int(source_layer_index)] = positions[execution_step]
    return final_positions


def _compute_inference_cka(
    *,
    candidate: PlanCandidate,
    positions: dict[int, Any],
    identity_positions_by_layer: dict[int, Any],
    services: ScanServices,
) -> dict[str, Any]:
    candidate_positions = _final_positions_by_original_layer(candidate, positions)
    per_layer: dict[str, float] = {}
    for layer_idx in sorted(identity_positions_by_layer.keys()):
        if layer_idx not in candidate_positions:
            continue
        per_layer[str(layer_idx)] = float(
            compute_geodesic_cka(
                identity_positions_by_layer[layer_idx],
                candidate_positions[layer_idx],
                backend=services.backend,
            )
        )

    values = list(per_layer.values())
    return {
        "perLayer": per_layer,
        "mean": sum(values) / len(values) if values else None,
        "min": min(values) if values else None,
    }


def _response_degeneration(text: str, readout_effective_rank: float) -> dict[str, float | int]:
    tokens = text.split()
    generation_length = max(2, len(tokens))
    ngram_order = derive_ngram_order(readout_effective_rank, generation_length)
    return {
        "ngramOrder": ngram_order,
        "repetitionRate": ngram_repetition_rate(text, n=ngram_order),
    }


def run_behavioral_readout(
    *,
    model: Any,
    tokenizer: Any,
    services: ScanServices,
    behavior_limit_per_benchmark: int,
    max_tokens: int,
    readout_effective_rank: float,
) -> dict[str, Any]:
    """Run the frozen quick-suite readout using the existing benchmark service."""
    benchmark_rows: list[dict[str, Any]] = []
    total_correct = 0
    total_questions = 0
    all_degenerations: list[float] = []

    for benchmark_name in BENCHMARK_SUITE:
        captured_responses: list[str] = []

        def generate_fn(
            model_obj: Any,
            tokenizer_obj: Any,
            prompt: str,
            max_tokens: int,
            verbose: bool = False,
        ) -> str:
            _ = verbose
            response = services.backend.generate(
                model_obj,
                tokenizer_obj,
                prompt,
                max_tokens=max_tokens,
            )
            captured_responses.append(response)
            return response

        result = services.benchmark_service.run_benchmark(
            model,
            tokenizer,
            benchmark_name,
            generate_fn,
            limit=behavior_limit_per_benchmark,
            compute_geometry=False,
            max_tokens=max_tokens,
        )
        degeneration = [
            _response_degeneration(response, readout_effective_rank)
            for response in captured_responses
        ]
        repetition_rates = [entry["repetitionRate"] for entry in degeneration]
        benchmark_rows.append(
            {
                "benchmark": benchmark_name,
                "rawResult": asdict(result),
                "degeneration": {
                    "meanRepetitionRate": (
                        sum(repetition_rates) / len(repetition_rates)
                        if repetition_rates
                        else 0.0
                    ),
                    "maxRepetitionRate": max(repetition_rates, default=0.0),
                    "responseCount": len(captured_responses),
                    "responses": degeneration,
                },
            }
        )
        total_correct += int(result.correct)
        total_questions += int(result.total)
        all_degenerations.extend(repetition_rates)

    return {
        "suite": "quick",
        "suiteBenchmarks": list(BENCHMARK_SUITE),
        "overallAccuracy": (
            total_correct / total_questions if total_questions > 0 else 0.0
        ),
        "correct": total_correct,
        "total": total_questions,
        "readoutEffectiveRank": readout_effective_rank,
        "degenerationMeanRepetitionRate": (
            sum(all_degenerations) / len(all_degenerations) if all_degenerations else 0.0
        ),
        "degenerationMaxRepetitionRate": max(all_degenerations, default=0.0),
        "benchmarks": benchmark_rows,
    }


def evaluate_stage2_candidate(
    *,
    candidate: PlanCandidate,
    model: Any,
    tokenizer: Any,
    probes: list[AtlasProbe],
    probe_texts: list[str],
    services: ScanServices,
    behavior_limit_per_benchmark: int,
    max_tokens: int,
    readout_effective_rank: float,
    identity_positions_by_layer: dict[int, Any] | None,
) -> tuple[dict[str, Any], dict[int, Any]]:
    """Evaluate the heavy stage-2 bundle for one candidate."""
    with apply_execution_plan(model, candidate.plan):
        trajectories = services.activation_provider.collect_trajectory_batch(
            model, tokenizer, probe_texts
        )
        verification = services.verification_depth_service.profile(
            model=model,
            tokenizer=tokenizer,
            probes=probes,
        )
        behavioral = run_behavioral_readout(
            model=model,
            tokenizer=tokenizer,
            services=services,
            behavior_limit_per_benchmark=behavior_limit_per_benchmark,
            max_tokens=max_tokens,
            readout_effective_rank=readout_effective_rank,
        )

    final_positions = _final_positions_by_original_layer(candidate, trajectories.positions)
    cka = None
    if identity_positions_by_layer is not None:
        cka = _compute_inference_cka(
            candidate=candidate,
            positions=trajectories.positions,
            identity_positions_by_layer=identity_positions_by_layer,
            services=services,
        )
    return (
        {
            "verificationDepthProfile": verification.to_dict(),
            "inferenceCkaVsIdentity": cka,
            "behavioralQuickSuite": behavioral,
        },
        final_positions,
    )


def _safe_get(row: dict[str, Any], *path: str) -> float | None:
    cursor: Any = row
    for key in path:
        if cursor is None or key not in cursor:
            return None
        cursor = cursor[key]
    if cursor is None or not isinstance(cursor, (int, float)) or not math.isfinite(float(cursor)):
        return None
    return float(cursor)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denom_x = math.sqrt(sum(value * value for value in centered_x))
    denom_y = math.sqrt(sum(value * value for value in centered_y))
    if denom_x == 0.0 or denom_y == 0.0:
        return None
    numerator = sum(x * y for x, y in zip(centered_x, centered_y, strict=False))
    return numerator / (denom_x * denom_y)


def compute_stage2_correlations(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """Correlate stage-1 geometry with stage-2 behavior and CKA."""
    non_identity = [
        row
        for row in rows
        if row["planKey"] != "identity" and row["stage2"] is not None
    ]
    metrics = {
        "trajectoryRankDelta_vs_behaviorDelta": (
            "stage1",
            "deltasVsIdentity",
            "canonicalTrajectoryRank",
            "stage2",
            "behavioralQuickSuite",
            "overallAccuracyDeltaVsIdentity",
        ),
        "intrinsicDimensionDelta_vs_behaviorDelta": (
            "stage1",
            "deltasVsIdentity",
            "canonicalIntrinsicDimension",
            "stage2",
            "behavioralQuickSuite",
            "overallAccuracyDeltaVsIdentity",
        ),
        "conditionInflation_vs_behaviorDelta": (
            "stage1",
            "deltasVsIdentity",
            "canonicalConditionNumberInflation",
            "stage2",
            "behavioralQuickSuite",
            "overallAccuracyDeltaVsIdentity",
        ),
        "trajectoryRankDelta_vs_inferenceCkaMean": (
            "stage1",
            "deltasVsIdentity",
            "canonicalTrajectoryRank",
            "stage2",
            "inferenceCkaVsIdentity",
            "mean",
        ),
        "intrinsicDimensionDelta_vs_inferenceCkaMean": (
            "stage1",
            "deltasVsIdentity",
            "canonicalIntrinsicDimension",
            "stage2",
            "inferenceCkaVsIdentity",
            "mean",
        ),
    }

    correlations: dict[str, float | None] = {}
    for name, path in metrics.items():
        xs: list[float] = []
        ys: list[float] = []
        left_path = path[:3]
        right_path = path[3:]
        for row in non_identity:
            left = _safe_get(row, *left_path)
            right = _safe_get(row, *right_path)
            if left is None or right is None:
                continue
            xs.append(left)
            ys.append(right)
        correlations[name] = _pearson(xs, ys)
    return correlations


def build_run_manifest(
    *,
    config: ScanConfig,
    model_metadata: dict[str, Any],
    base_layer_count: int,
    probe_manifest_payload: dict[str, Any],
    stage2_keys: list[str],
) -> dict[str, Any]:
    """Build the run manifest for the falsifier spend."""
    return {
        "schema": ARTIFACT_SCHEMA,
        "created_at": _utc_now_iso(),
        "roadmap_item": "R2",
        "open_question": "Q1",
        "output_dir": str(config.output_dir),
        "model": model_metadata,
        "prediction_contract": {
            "observable": (
                "observable = f(geometry_state, architecture_state, scale_state, "
                "precision_state, measurement_operator)"
            ),
            "geometry_state": (
                "canonical trajectory rank, intrinsic dimension, condition-number "
                "surface, max norm-jump surface, inference-manifold CKA"
            ),
            "architecture_state": (
                "fixed-weight execution-plan routing via Python layer-list reassignment "
                "with shared layer objects"
            ),
            "scale_state": {
                "base_layer_count": base_layer_count,
                "scan_bounds": {
                    "start_min": config.start_min,
                    "start_max": config.start_max,
                    "end_min": config.end_min,
                    "end_max": config.end_max,
                },
            },
            "precision_state": {
                "backend": model_metadata["backend"],
                "torch_dtype": model_metadata["torch_dtype"],
                "backend_default_eps": model_metadata["backend_default_eps"],
            },
            "measurement_operator": {
                "stage1": (
                    "collect_trajectory_batch + compute_gram_spectrum + norm summary"
                ),
                "stage2": (
                    "verification-depth profile + geodesic RBF CKA + quick-suite "
                    "benchmark service"
                ),
            },
        },
        "falsifier_prediction": PREDICTION_CONTRACT,
        "frozen_contract": {
            "two_stage": True,
            "seed": config.seed,
            "behavioral_suite": "quick",
            "behavior_limit_per_benchmark": config.behavior_limit_per_benchmark,
            "stage2_plan_keys": stage2_keys,
            "ranking_rule": [
                "maximize canonical trajectory-rank delta vs identity",
                "maximize canonical intrinsic-dimension delta vs identity",
                "minimize canonical condition-number inflation vs identity",
                "minimize max norm-jump inflation vs identity",
            ],
            "probe_manifest_path": str(config.output_dir / "probe_manifest.json"),
            "probe_selection": {
                "strategy": probe_manifest_payload["selection_strategy"],
                "requested_max_probes": probe_manifest_payload["requested_max_probes"],
                "selected_probe_count": probe_manifest_payload["selected_probe_count"],
            },
        },
    }


def _model_metadata(model_path: Path, backend: Any) -> dict[str, Any]:
    config_path = model_path / "config.json"
    config_payload: dict[str, Any] = {}
    if config_path.exists():
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))

    return {
        "path": str(model_path),
        "name": model_path.name,
        "model_type": config_payload.get("model_type"),
        "architectures": config_payload.get("architectures", []),
        "torch_dtype": config_payload.get("torch_dtype"),
        "backend": type(backend).__name__,
        "backend_default_eps": float(backend.finfo().eps),
    }


def build_summary(
    *,
    config: ScanConfig,
    rows: list[dict[str, Any]],
    stage2_keys: list[str],
    probe_manifest_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build the summary artifact from the raw row table."""
    identity = next(row for row in rows if row["planKey"] == "identity")
    stage2_rows = [row for row in rows if row["planKey"] in stage2_keys]
    return {
        "schema": f"{ARTIFACT_SCHEMA}.summary",
        "created_at": _utc_now_iso(),
        "output_dir": str(config.output_dir),
        "selected_probe_count": probe_manifest_payload["selected_probe_count"],
        "stage1PlanCount": len(rows),
        "stage2PlanKeys": stage2_keys,
        "identityBaseline": {
            "planKey": identity["planKey"],
            "stage1": identity["stage1"],
            "stage2": identity["stage2"],
        },
        "stage1Ordering": [
            {
                "planKey": row["planKey"],
                "rank": row["stage1"]["rank"],
                "repeatedBlock": row["repeatedBlock"],
                "trajectoryRankDelta": row["stage1"]["deltasVsIdentity"]["canonicalTrajectoryRank"],
                "intrinsicDimensionDelta": row["stage1"]["deltasVsIdentity"]["canonicalIntrinsicDimension"],
                "conditionNumberInflation": row["stage1"]["deltasVsIdentity"]["canonicalConditionNumberInflation"],
                "maxNormJumpInflation": row["stage1"]["deltasVsIdentity"]["maxNormJumpInflation"],
            }
            for row in rows
        ],
        "stage2Comparisons": [
            {
                "planKey": row["planKey"],
                "repeatedBlock": row["repeatedBlock"],
                "behaviorOverallAccuracy": _safe_get(
                    row, "stage2", "behavioralQuickSuite", "overallAccuracy"
                ),
                "behaviorOverallAccuracyDeltaVsIdentity": _safe_get(
                    row, "stage2", "behavioralQuickSuite", "overallAccuracyDeltaVsIdentity"
                ),
                "inferenceCkaMean": _safe_get(
                    row, "stage2", "inferenceCkaVsIdentity", "mean"
                ),
                "inferenceCkaMin": _safe_get(
                    row, "stage2", "inferenceCkaVsIdentity", "min"
                ),
            }
            for row in stage2_rows
        ],
        "stage2Correlations": compute_stage2_correlations(stage2_rows),
    }


def write_report(
    *,
    path: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    """Write a concise bedrock-oriented markdown report."""
    ordering = summary["stage1Ordering"]
    stage2 = summary["stage2Comparisons"]
    top_candidates = ordering[1: min(6, len(ordering))]
    lines = [
        "# R2 Execution-Plan Scan",
        "",
        f"- Schema: `{manifest['schema']}`",
        f"- Roadmap / question: `{manifest['roadmap_item']}` / `{manifest['open_question']}`",
        f"- Model: `{manifest['model']['name']}`",
        f"- Selected probes: `{summary['selected_probe_count']}`",
        f"- Stage-1 plans: `{summary['stage1PlanCount']}`",
        f"- Stage-2 plans: `{', '.join(summary['stage2PlanKeys'])}`",
        "",
        "## Prediction",
        "",
        manifest["falsifier_prediction"],
        "",
        "## Stage 1 Ordering",
        "",
        "| Rank | Plan | Block | Δ rank | Δ ID | Δ cond | Δ max jump |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in top_candidates:
        block = row["repeatedBlock"]
        block_label = (
            f"[{block['start']}, {block['end']})"
            if block is not None
            else "identity"
        )
        lines.append(
            f"| {row['rank']} | `{row['planKey']}` | `{block_label}` | "
            f"{row['trajectoryRankDelta']:.0f} | {row['intrinsicDimensionDelta']:.4f} | "
            f"{row['conditionNumberInflation']:.4f} | {row['maxNormJumpInflation']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Stage 2 Outcomes",
            "",
            "| Plan | Accuracy | Δ accuracy | mean CKA | min CKA |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stage2:
        accuracy = row["behaviorOverallAccuracy"]
        accuracy_delta = row["behaviorOverallAccuracyDeltaVsIdentity"]
        cka_mean = row["inferenceCkaMean"]
        cka_min = row["inferenceCkaMin"]
        lines.append(
            f"| `{row['planKey']}` | "
            f"{accuracy if accuracy is not None else float('nan'):.4f} | "
            f"{accuracy_delta if accuracy_delta is not None else float('nan'):.4f} | "
            f"{cka_mean if cka_mean is not None else float('nan'):.4f} | "
            f"{cka_min if cka_min is not None else float('nan'):.4f} |"
        )

    correlations = summary["stage2Correlations"]
    lines.extend(
        [
            "",
            "## Bedrock Questions",
            "",
            "### What changed under layer reuse?",
            "",
            "The table above reports how repeated blocks changed canonical trajectory rank, "
            "intrinsic dimension, conditioning, and the frozen quick-suite relative to the identity plan.",
            "",
            "### Where did inference collapse reduce or worsen?",
            "",
            "Compare `mean CKA` and `min CKA` across stage-2 plans. Higher values indicate "
            "closer inference-manifold agreement with the identity baseline on the frozen probe manifest.",
            "",
            "### Which repeated blocks were favored?",
            "",
            "The stage-1 ordering reports the lexicographic candidate ranking used to choose "
            "the stage-2 spend. The favored blocks are the top entries in that table.",
            "",
            "### What falsifier survived or failed?",
            "",
            "The falsifier survives if no repeated block improves or preserves inference-manifold "
            "CKA and the frozen quick-suite simultaneously. It fails if one or more blocks do.",
            "",
            "## Stage 2 Correlations",
            "",
        ]
    )
    for name, value in correlations.items():
        value_text = "null" if value is None else f"{value:.6f}"
        lines.append(f"- `{name}` = `{value_text}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_scan(config: ScanConfig, services: ScanServices | None = None) -> dict[str, Any]:
    """Execute the two-stage execution-plan scan."""
    services = services or build_services()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = config.output_dir / "ledger.jsonl"
    _append_ledger(
        ledger_path,
        "run_started",
        model_path=str(config.model_path),
        top_k=config.top_k,
    )

    probes, probe_manifest_payload, _probe_manifest_path = load_or_create_probe_manifest(
        output_dir=config.output_dir,
        explicit_manifest_path=config.probe_manifest_path,
        max_probes=config.max_probes,
    )
    probe_texts = [_probe_text(probe) for probe in probes]
    probe_texts = [text for text in probe_texts if text is not None]
    if not probe_texts:
        raise ValueError("Probe manifest does not contain any usable probe texts")

    model, tokenizer = services.model_loader.load_model(str(config.model_path))
    base_layer_count = services.backend.get_num_layers(model)
    candidates = build_plan_candidates(
        base_layer_count=base_layer_count,
        start_min=config.start_min,
        start_max=config.start_max,
        end_min=config.end_min,
        end_max=config.end_max,
    )

    stage1_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row = evaluate_stage1_candidate(
            candidate=candidate,
            model=model,
            tokenizer=tokenizer,
            probe_texts=probe_texts,
            services=services,
        )
        stage1_rows.append(row)
        _append_ledger(
            ledger_path,
            "stage1_variant_measured",
            plan_key=candidate.key,
            repeated_block=candidate.repeated_block(),
            canonical_trajectory_rank=row["stage1"]["canonicalTrajectoryRank"],
            canonical_intrinsic_dimension=row["stage1"]["canonicalIntrinsicDimension"],
        )
        gc.collect()

    rows = annotate_stage1_rankings(stage1_rows)
    stage2_keys = select_stage2_plan_keys(rows, config.top_k)
    model_metadata = _model_metadata(config.model_path, services.backend)
    manifest = build_run_manifest(
        config=config,
        model_metadata=model_metadata,
        base_layer_count=base_layer_count,
        probe_manifest_payload=probe_manifest_payload,
        stage2_keys=stage2_keys,
    )
    _write_json(config.output_dir / "run_manifest.json", manifest)

    readout_effective_rank = compute_readout_effective_rank(model, services.backend)
    identity_candidate = next(
        candidate for candidate in candidates if candidate.key == "identity"
    )
    identity_stage2, identity_positions = evaluate_stage2_candidate(
        candidate=identity_candidate,
        model=model,
        tokenizer=tokenizer,
        probes=probes,
        probe_texts=probe_texts,
        services=services,
        behavior_limit_per_benchmark=config.behavior_limit_per_benchmark,
        max_tokens=config.max_tokens,
        readout_effective_rank=readout_effective_rank,
        identity_positions_by_layer=None,
    )
    identity_row = next(row for row in rows if row["planKey"] == "identity")
    identity_row["stage2"] = identity_stage2
    _append_ledger(
        ledger_path,
        "stage2_variant_measured",
        plan_key="identity",
        overall_accuracy=identity_stage2["behavioralQuickSuite"]["overallAccuracy"],
    )

    identity_accuracy = identity_stage2["behavioralQuickSuite"]["overallAccuracy"]
    for row in rows:
        if row["stage2"] is None:
            continue
        row["stage2"]["behavioralQuickSuite"]["overallAccuracyDeltaVsIdentity"] = (
            row["stage2"]["behavioralQuickSuite"]["overallAccuracy"] - identity_accuracy
        )

    selected_candidates = {
        candidate.key: candidate for candidate in candidates if candidate.key in stage2_keys
    }
    for plan_key in stage2_keys:
        if plan_key == "identity":
            continue
        candidate = selected_candidates[plan_key]
        row = next(item for item in rows if item["planKey"] == plan_key)
        stage2, _ = evaluate_stage2_candidate(
            candidate=candidate,
            model=model,
            tokenizer=tokenizer,
            probes=probes,
            probe_texts=probe_texts,
            services=services,
            behavior_limit_per_benchmark=config.behavior_limit_per_benchmark,
            max_tokens=config.max_tokens,
            readout_effective_rank=readout_effective_rank,
            identity_positions_by_layer=identity_positions,
        )
        stage2["behavioralQuickSuite"]["overallAccuracyDeltaVsIdentity"] = (
            stage2["behavioralQuickSuite"]["overallAccuracy"] - identity_accuracy
        )
        row["stage2"] = stage2
        _append_ledger(
            ledger_path,
            "stage2_variant_measured",
            plan_key=plan_key,
            repeated_block=row["repeatedBlock"],
            overall_accuracy=stage2["behavioralQuickSuite"]["overallAccuracy"],
            inference_cka_mean=_safe_get(
                {"stage2": stage2},
                "stage2",
                "inferenceCkaVsIdentity",
                "mean",
            ),
        )
        gc.collect()

    summary = build_summary(
        config=config,
        rows=rows,
        stage2_keys=stage2_keys,
        probe_manifest_payload=probe_manifest_payload,
    )
    _write_json(config.output_dir / "summary.json", summary)
    _write_jsonl(config.output_dir / "variant_results.jsonl", rows)
    write_report(
        path=config.output_dir / "REPORT.md",
        manifest=manifest,
        summary=summary,
    )
    _append_ledger(
        ledger_path,
        "run_completed",
        stage1_plan_count=len(rows),
        stage2_plan_keys=stage2_keys,
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the R2 MLX-first execution-plan scan."
    )
    parser.add_argument("--model", required=True, type=Path, help="Path to the base model.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for REPORT.md, summary.json, run_manifest.json, "
            "probe_manifest.json, ledger.jsonl, and variant_results.jsonl. "
            "Defaults to results/nblora_vs_standard/r2_execution_plan_scan/<run_id>/."
        ),
    )
    parser.add_argument("--top-k", required=True, type=int, help="Number of top-ranked plans to promote into stage 2.")
    parser.add_argument("--probe-manifest", type=Path, help="Reuse an existing frozen probe manifest JSON.")
    parser.add_argument("--max-probes", type=int, default=None, help="Optional cap on the frozen verification-depth probe set.")
    parser.add_argument(
        "--behavior-limit-per-benchmark",
        type=int,
        default=20,
        help="Directional quick-suite sample budget per benchmark.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed recorded in the manifest.")
    parser.add_argument("--start-min", type=int, default=None, help="Optional minimum RYS start index.")
    parser.add_argument("--start-max", type=int, default=None, help="Optional maximum RYS start index.")
    parser.add_argument("--end-min", type=int, default=None, help="Optional minimum RYS end index.")
    parser.add_argument("--end-max", type=int, default=None, help="Optional maximum RYS end index.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum generation tokens for the quick-suite readout.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    run_id = _run_id()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (DEFAULT_RESULTS_ROOT / run_id).resolve()
    )
    config = ScanConfig(
        model_path=args.model.expanduser().resolve(),
        output_dir=output_dir,
        top_k=args.top_k,
        probe_manifest_path=(
            args.probe_manifest.expanduser().resolve()
            if args.probe_manifest is not None
            else None
        ),
        max_probes=args.max_probes,
        behavior_limit_per_benchmark=args.behavior_limit_per_benchmark,
        seed=args.seed,
        start_min=args.start_min,
        start_max=args.start_max,
        end_min=args.end_min,
        end_max=args.end_max,
        max_tokens=args.max_tokens,
    )
    summary = run_scan(config)
    print(summary["output_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
