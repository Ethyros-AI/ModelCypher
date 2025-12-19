from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from modelcypher.adapters.filesystem_storage import FileSystemStore
from modelcypher.core.use_cases.merge_engine import MergeAnalysisResult, RotationalMerger
from modelcypher.utils.paths import ensure_dir, expand_path


class ModelMergeService:
    def __init__(self, store: FileSystemStore | None = None, merger: RotationalMerger | None = None) -> None:
        self.store = store or FileSystemStore()
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
        anchor_mode: str = "semantic-primes",
        dry_run: bool = False,
    ) -> dict:
        source_model = self.store.get_model(source_id)
        target_model = self.store.get_model(target_id)
        if source_model is None:
            raise RuntimeError(f"Model not found: {source_id}")
        if target_model is None:
            raise RuntimeError(f"Model not found: {target_id}")

        source_weights = self._load_weights(source_model.path)
        target_weights = self._load_weights(target_model.path)

        merged, analysis = self.merger.merge(source_weights, target_weights, alpha=alpha, anchor_mode=anchor_mode)

        output_path = ensure_dir(output_dir)
        if not dry_run:
            self._save_weights(output_path, merged)

        report = {
            "sourceModel": source_id,
            "targetModel": target_id,
            "anchorMode": anchor_mode,
            "timestamp": analysis.timestamp.isoformat() + "Z",
            "meanProcrustesError": analysis.mean_procrustes_error,
            "maxProcrustesError": analysis.max_procrustes_error,
            "rotationFieldRoughness": analysis.rotation_field_roughness,
            "anchorCoverage": analysis.anchor_coverage,
            "layerMetrics": [asdict(metric) for metric in analysis.layer_metrics],
            "outputDir": str(output_path),
            "dryRun": dry_run,
        }
        return report

    @staticmethod
    def _load_weights(path: str) -> dict[str, Any]:
        resolved = expand_path(path)
        if resolved.is_dir():
            resolved = resolved / "weights.npz"
        if not resolved.exists():
            raise RuntimeError(f"Weights not found: {resolved}")
        data = np.load(resolved)
        return {name: data[name] for name in data.files}

    @staticmethod
    def _save_weights(output_dir: Path, weights: dict[str, Any]) -> None:
        path = output_dir / "weights.npz"
        np.savez(path, **weights)
