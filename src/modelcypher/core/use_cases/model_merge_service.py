from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file

from modelcypher.core.domain.manifold_stitcher import intersection_map_from_dict
from modelcypher.core.use_cases.anchor_extractor import AnchorExtractionConfig, AnchorExtractor
from modelcypher.core.use_cases.merge_engine import (
    AnchorMode,
    ModuleScope,
    RotationalMergeOptions,
    RotationalMerger,
)
from modelcypher.ports.storage import ModelStore
from modelcypher.utils.paths import ensure_dir, expand_path


logger = logging.getLogger(__name__)


class ModelMergeService:
    def __init__(
        self,
        store: ModelStore,
        merger: RotationalMerger | None = None,
        anchor_extractor: AnchorExtractor | None = None,
    ) -> None:
        if store is None:
            raise ValueError("Model store is required")
        self.store = store
        self.anchor_extractor = anchor_extractor or AnchorExtractor()
        if merger is None:
            from modelcypher.backends import default_backend

            merger = RotationalMerger(default_backend())
        self.merger = merger

    def merge(
        self,
        source_id: str,
        target_id: str,
        output_dir: str,
        alpha: float = 0.5,
        alignment_rank: int = 32,
        module_scope: str | None = None,
        anchor_mode: str = "semantic-primes",
        intersection_path: str | None = None,
        fisher_source: str | None = None,
        fisher_target: str | None = None,
        fisher_strength: float = 0.0,
        fisher_epsilon: float = 1e-6,
        adaptive_alpha: bool = False,
        dry_run: bool = False,
    ) -> dict:
        source_path = self._resolve_model_path(source_id)
        target_path = self._resolve_model_path(target_id)

        source_payload = self._load_weights(source_path)
        target_payload = self._load_weights(target_path)

        normalized_mode = self._parse_anchor_mode(anchor_mode)
        if normalized_mode == "unified":
            raise NotImplementedError("Unified merge pipeline is not implemented yet.")

        if fisher_strength > 0:
            logger.warning(
                "Fisher blending is only available in unified mode. Ignoring fisher inputs."
            )
        del fisher_source, fisher_target, fisher_strength, fisher_epsilon

        intersection = None
        if intersection_path:
            payload = json.loads(Path(intersection_path).read_text(encoding="utf-8"))
            intersection = intersection_map_from_dict(payload)

        scope = self._parse_module_scope(module_scope, normalized_mode)
        mode = AnchorMode(normalized_mode)
        options = RotationalMergeOptions(
            alignment_rank=alignment_rank,
            alpha=alpha,
            anchor_mode=mode,
            module_scope=scope,
            use_enriched_primes=True,
            intersection_map=intersection,
            use_adaptive_alpha=adaptive_alpha and intersection is not None,
        )

        anchor_config = AnchorExtractionConfig(use_enriched_primes=True)
        source_anchors, source_confidence = self.anchor_extractor.extract(
            str(source_payload.model_dir),
            source_payload.weights,
            config=anchor_config,
            backend=self.merger.backend,
        )
        target_anchors, target_confidence = self.anchor_extractor.extract(
            str(target_payload.model_dir),
            target_payload.weights,
            config=anchor_config,
            backend=self.merger.backend,
        )
        shared = self.merger.build_shared_anchors(
            source_anchors,
            target_anchors,
            source_confidence,
            target_confidence,
            alignment_rank=alignment_rank,
        )

        merged, analysis = self.merger.merge(
            source_payload.weights,
            target_payload.weights,
            options,
            shared,
            source_id=source_id,
            target_id=target_id,
        )

        output_path = expand_path(output_dir) if dry_run else ensure_dir(output_dir)
        if not dry_run:
            self._save_weights(output_path, merged, target_payload.format)
            self._copy_support_files(target_payload.model_dir, output_path)

        report = {
            "sourceModel": source_id,
            "targetModel": target_id,
            "anchorMode": options.anchor_mode.value,
            "timestamp": analysis.timestamp.isoformat() + "Z",
            "meanProcrustesError": analysis.mean_procrustes_error,
            "maxProcrustesError": analysis.max_procrustes_error,
            "rotationFieldRoughness": analysis.rotation_field_roughness,
            "anchorCoverage": analysis.anchor_coverage,
            "layerMetrics": [self._layer_metric_payload(metric) for metric in analysis.layer_metrics],
        }
        if analysis.mlp_blocks_aligned:
            report["mlpRebasinQuality"] = analysis.mlp_rebasin_quality
            report["mlpBlocksAligned"] = analysis.mlp_blocks_aligned
        return report

    @staticmethod
    def _layer_metric_payload(metric: Any) -> dict[str, Any]:
        return {
            "layerIndex": metric.layer_index,
            "moduleName": metric.module_name,
            "moduleKind": metric.module_kind,
            "procrustesError": metric.procrustes_error,
            "conditionNumber": metric.condition_number,
            "rotationDeviation": metric.rotation_deviation,
            "spectralRatio": metric.spectral_ratio,
        }

    def _resolve_model_path(self, model_id: str) -> Path:
        model = self.store.get_model(model_id)
        candidate = model.path if model is not None else model_id
        resolved = expand_path(candidate)
        if not resolved.exists():
            raise RuntimeError(f"Model not found: {model_id}")
        return resolved

    def _parse_anchor_mode(self, mode: str) -> str:
        normalized = mode.strip().lower().replace("_", "-")
        aliases = {
            "semantic-primes": "semantic-primes",
            "semanticprimes": "semantic-primes",
            "geometric": "geometric",
            "intersection": "intersection",
            "rebasin": "rebasin",
            "unified": "unified",
        }
        if normalized not in aliases:
            raise ValueError(
                "Invalid anchor mode. Use: semantic-primes, geometric, intersection, rebasin, unified."
            )
        return aliases[normalized]

    def _parse_module_scope(self, scope: str | None, anchor_mode: str) -> ModuleScope:
        if scope is None:
            return ModuleScope.all if anchor_mode == "rebasin" else ModuleScope.attention_only
        normalized = scope.strip().lower().replace("_", "-")
        if normalized in {"attention-only", "attention", "attn"}:
            return ModuleScope.attention_only
        if normalized in {"all", "full", "everything", "attention-mlp", "attention+mlp"}:
            return ModuleScope.all
        raise ValueError("Invalid module scope. Use: attention-only or all.")

    def _load_weights(self, path: Path) -> _WeightsPayload:
        resolved = expand_path(str(path))
        model_dir = resolved if resolved.is_dir() else resolved.parent

        weight_files: list[Path] = []
        fmt = ""
        if resolved.is_dir():
            safetensors = sorted(resolved.glob("*.safetensors"))
            npz_files = sorted(resolved.glob("*.npz"))
            if safetensors:
                weight_files = safetensors
                fmt = "safetensors"
            elif npz_files:
                weight_files = npz_files
                fmt = "npz"
        else:
            if resolved.suffix == ".safetensors":
                weight_files = [resolved]
                fmt = "safetensors"
            elif resolved.suffix == ".npz":
                weight_files = [resolved]
                fmt = "npz"

        if not weight_files or not fmt:
            raise RuntimeError(f"Weights not found at: {resolved}")

        weights: dict[str, np.ndarray] = {}
        for weight_file in weight_files:
            if fmt == "safetensors":
                payload = load_file(weight_file)
                weights.update({key: np.asarray(value) for key, value in payload.items()})
            else:
                payload = np.load(weight_file)
                weights.update({key: np.asarray(payload[key]) for key in payload.files})

        return _WeightsPayload(weights=weights, format=fmt, model_dir=model_dir)

    def _save_weights(self, output_dir: Path, weights: dict[str, Any], fmt: str) -> None:
        if fmt == "safetensors":
            path = output_dir / "model.safetensors"
            save_file(weights, str(path))
            return
        path = output_dir / "weights.npz"
        np.savez(path, **weights)

    @staticmethod
    def _copy_support_files(source_dir: Path, output_dir: Path) -> None:
        if not source_dir.exists():
            return
        for item in source_dir.iterdir():
            if not item.is_file():
                continue
            if item.suffix in {".safetensors", ".bin", ".pt", ".npz"}:
                continue
            destination = output_dir / item.name
            if destination.exists():
                continue
            shutil.copy2(item, destination)


@dataclass(frozen=True)
class _WeightsPayload:
    weights: dict[str, np.ndarray]
    format: str
    model_dir: Path
