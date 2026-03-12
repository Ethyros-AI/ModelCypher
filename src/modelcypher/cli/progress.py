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

"""Structured progress reporting for AI interpretation.

This module provides progress events that AI assistants can read and explain
to humans. Each event contains:
- What is happening (technical)
- Why it matters (context for explanation)
- Progress metrics (numbers with meaning)
- Geometry context (what the math means)

Example output:
{
  "_type": "merge_progress",
  "stage": "probe",
  "substage": "manifold_mapping",
  "model": "source",
  "what": "Mapping the source model's representation space",
  "why": "Need to discover which directions the model uses before finding unused space",
  "progress": {"batch": 5, "probes": 75, "layers_saturated": 0, "layers_total": 36},
  "geometry": {"explanation": "Saturation = manifold fully mapped, no new dimensions"}
}

An AI reading this can explain to a human:
"The merge is mapping your source model's representation space - discovering which
directions it actively uses. Once it finds all used directions, it can identify
unused space where new knowledge can be added. So far 75 probes processed, but
the manifold isn't fully mapped yet (0/36 layers saturated)."
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TextIO


@dataclass
class MergeStage:
    """Describes a merge pipeline stage for AI explanation."""

    name: str
    description: str
    why: str
    substages: list[str] = field(default_factory=list)


# Stage definitions with explanations
MERGE_STAGES = {
    "load": MergeStage(
        name="load",
        description="Loading model weights and tokenizers",
        why="Need to access the neural network parameters before analysis",
        substages=["source_weights", "target_weights", "source_model", "target_model"],
    ),
    "probe": MergeStage(
        name="probe",
        description="Mapping representation manifolds via activation probing",
        why="Must discover which directions each model uses before finding transfer space",
        substages=[
            "load_probes",
            "source_manifold",
            "target_manifold",
            "compute_alignment",
            "compute_transforms",
        ],
    ),
    "density": MergeStage(
        name="density",
        description="Analyzing knowledge density to identify transfer regions",
        why="Determines WHERE to transfer - regions where source is dense but target is sparse",
        substages=["source_density", "target_density", "compute_mask"],
    ),
    "transplant": MergeStage(
        name="transplant",
        description="Projecting source knowledge into target's null space",
        why="The actual knowledge transfer - adds source knowledge where target has unused capacity",
        substages=["compute_null_space", "project_deltas", "apply_transforms", "save_weights"],
    ),
}


@dataclass
class ProgressEvent:
    """A structured progress event for AI interpretation."""

    stage: str
    substage: str
    model: str | None  # "source", "target", or None
    what: str  # Technical description
    why: str  # Why this matters (for AI to explain)
    progress: dict[str, Any]  # Numeric progress metrics
    geometry: dict[str, Any] | None = None  # Geometric context
    event_type: str = "merge_progress"  # "merge_progress" or "training_progress"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON output."""
        result = {
            "_type": self.event_type,
            "timestamp": self.timestamp,
            "stage": self.stage,
            "substage": self.substage,
            "what": self.what,
            "why": self.why,
            "progress": self.progress,
        }
        if self.model:
            result["model"] = self.model
        if self.geometry:
            result["geometry"] = self.geometry
        return result


class ProgressReporter:
    """Reports structured progress events for AI interpretation.

    Usage:
        reporter = ProgressReporter()

        # In probe stage:
        reporter.probe_mapping_progress(
            model="source",
            batch=5,
            probes=75,
            layers_saturated=0,
            layers_total=36,
            ranks={"layer_0": 128, "layer_1": 256},
        )

        # In transplant stage:
        reporter.transplant_progress(
            layer=5,
            layer_total=32,
            preserved_fraction=0.85,
            null_rank=512,
        )
    """

    def __init__(
        self,
        output: TextIO | None = None,
        callback: Callable[[ProgressEvent], None] | None = None,
    ):
        """Initialize reporter.

        Args:
            output: Where to write JSON events (default: stderr)
            callback: Optional callback for each event (for testing/integration)
        """
        self._output = output or sys.stderr
        self._callback = callback
        self._start_time = time.time()
        self._current_stage: str | None = None

    def _emit(self, event: ProgressEvent) -> None:
        """Emit a progress event."""
        if self._callback:
            self._callback(event)
        self._output.write(json.dumps(event.to_dict()) + "\n")
        self._output.flush()

    # =========================================================================
    # LOAD STAGE
    # =========================================================================

    def load_started(self, source_path: str, target_path: str) -> None:
        """Report merge load started."""
        self._current_stage = "load"
        self._emit(
            ProgressEvent(
                stage="load",
                substage="started",
                model=None,
                what="Starting model merge pipeline",
                why="Loading both models to analyze their representation spaces",
                progress={
                    "source_path": source_path,
                    "target_path": target_path,
                },
            )
        )

    def load_weights_progress(self, model: str, tensors_loaded: int, total_tensors: int) -> None:
        """Report weight loading progress."""
        self._emit(
            ProgressEvent(
                stage="load",
                substage=f"{model}_weights",
                model=model,
                what=f"Loading {model} model weights",
                why="Neural network parameters must be loaded before analysis",
                progress={
                    "tensors_loaded": tensors_loaded,
                    "total_tensors": total_tensors,
                    "percent": round(100 * tensors_loaded / max(1, total_tensors), 1),
                },
            )
        )

    def load_model_complete(
        self,
        model: str,
        architecture: str,
        layers: int,
        hidden_dim: int,
        vocab_size: int,
    ) -> None:
        """Report model loaded with key dimensions."""
        self._emit(
            ProgressEvent(
                stage="load",
                substage=f"{model}_model",
                model=model,
                what=f"Loaded {model} model: {architecture}",
                why="Model architecture determines how knowledge is organized",
                progress={
                    "architecture": architecture,
                    "layers": layers,
                    "hidden_dim": hidden_dim,
                    "vocab_size": vocab_size,
                },
                geometry={
                    "explanation": f"Hidden dimension {hidden_dim} = the size of the representation space at each layer",
                    "layers_meaning": f"{layers} transformer layers, each transforms representations",
                },
            )
        )

    # =========================================================================
    # PROBE STAGE
    # =========================================================================

    def probe_started(self, total_probes: int, domains: int, min_required: int) -> None:
        """Report probe stage started."""
        self._current_stage = "probe"
        self._emit(
            ProgressEvent(
                stage="probe",
                substage="started",
                model=None,
                what="Starting manifold mapping via activation probing",
                why="Must discover which directions each model uses in representation space",
                progress={
                    "total_probes_available": total_probes,
                    "semantic_domains": domains,
                    "geometry_minimum": min_required,
                },
                geometry={
                    "explanation": "Probes are texts that activate different regions of the representation manifold",
                    "domains_meaning": f"{domains} semantic domains ensure coverage of different knowledge types",
                    "minimum_meaning": f"Need at least {min_required} samples to span the space (algebraic requirement)",
                },
            )
        )

    def probe_mapping_progress(
        self,
        model: str,
        batch: int,
        probes: int,
        layers_saturated: int,
        layers_total: int,
        ranks: dict[int, int] | None = None,
        hidden_dims: dict[int, int] | None = None,
    ) -> None:
        """Report manifold mapping progress."""
        progress = {
            "batch": batch,
            "probes_processed": probes,
            "layers_saturated": layers_saturated,
            "layers_total": layers_total,
            "percent_saturated": round(100 * layers_saturated / max(1, layers_total), 1),
        }

        geometry = {
            "explanation": "Saturation = manifold fully mapped, no new dimensions being discovered",
            "saturation_meaning": f"{layers_saturated}/{layers_total} layers have stable rank (geometry fully characterized)",
        }

        if ranks:
            geometry["current_ranks"] = ranks
        if hidden_dims:
            geometry["hidden_dims"] = hidden_dims
            # Calculate null space available
            null_available = {
                layer: hidden_dims.get(layer, 0) - rank for layer, rank in (ranks or {}).items()
            }
            geometry["null_space_available"] = null_available
            geometry["null_space_meaning"] = "Unused directions where new knowledge can be added"

        self._emit(
            ProgressEvent(
                stage="probe",
                substage="manifold_mapping",
                model=model,
                what=f"Mapping {model} model's representation manifold",
                why="Discovering which directions the model actively uses",
                progress=progress,
                geometry=geometry,
            )
        )

    def probe_layer_saturated(
        self,
        model: str,
        layer: int,
        rank: int,
        hidden_dim: int,
        batch: int,
    ) -> None:
        """Report a layer reached saturation."""
        null_rank = hidden_dim - rank
        self._emit(
            ProgressEvent(
                stage="probe",
                substage="layer_saturated",
                model=model,
                what=f"Layer {layer} manifold fully mapped",
                why="This layer's geometry is now characterized - we know its effective dimension",
                progress={
                    "layer": layer,
                    "batch_at_saturation": batch,
                    "activation_rank": rank,
                    "hidden_dim": hidden_dim,
                    "null_rank": null_rank,
                },
                geometry={
                    "explanation": f"Layer {layer} uses {rank}/{hidden_dim} dimensions",
                    "null_space": f"{null_rank} unused dimensions available for knowledge transfer",
                    "utilization": f"{100 * rank / hidden_dim:.1f}% of representation space actively used",
                },
            )
        )

    def probe_mapping_complete(
        self,
        model: str,
        total_probes: int,
        total_batches: int,
        all_saturated: bool,
        layer_ranks: dict[int, int],
    ) -> None:
        """Report manifold mapping complete for a model."""
        self._emit(
            ProgressEvent(
                stage="probe",
                substage="mapping_complete",
                model=model,
                what=f"Completed {model} manifold mapping",
                why="Full geometry characterized - ready for alignment computation",
                progress={
                    "probes_processed": total_probes,
                    "batches": total_batches,
                    "all_layers_saturated": all_saturated,
                    "layer_count": len(layer_ranks),
                },
                geometry={
                    "layer_ranks": layer_ranks,
                    "explanation": "Each layer's rank = intrinsic dimension of its representation manifold",
                },
            )
        )

    def probe_alignment_progress(
        self,
        layer: int,
        layer_total: int,
        cka_raw: float,
        cka_aligned: float,
        gram_condition: float,
    ) -> None:
        """Report alignment computation progress."""
        self._emit(
            ProgressEvent(
                stage="probe",
                substage="compute_alignment",
                model=None,
                what=f"Computing alignment transform for layer {layer}",
                why="Finding the rotation that maps source coordinates to target coordinates",
                progress={
                    "layer": layer,
                    "layer_total": layer_total,
                    "percent": round(100 * (layer + 1) / layer_total, 1),
                },
                geometry={
                    "cka_raw": round(cka_raw, 4),
                    "cka_aligned": round(cka_aligned, 4),
                    "gram_condition": round(gram_condition, 2),
                    "explanation": f"CKA: {cka_raw:.2f} raw → {cka_aligned:.2f} aligned (1.0 = perfect structural match)",
                    "condition_meaning": f"Gram condition {gram_condition:.0f} measures numerical stability of alignment",
                },
            )
        )

    # =========================================================================
    # DENSITY STAGE
    # =========================================================================

    def density_started(self, layers: int) -> None:
        """Report density analysis started."""
        self._current_stage = "density"
        self._emit(
            ProgressEvent(
                stage="density",
                substage="started",
                model=None,
                what="Starting knowledge density analysis",
                why="Identifying WHERE to transfer - regions where source has knowledge target lacks",
                progress={"layers_to_analyze": layers},
                geometry={
                    "explanation": "Density = how much knowledge is encoded in each region of the manifold",
                },
            )
        )

    def density_layer_complete(
        self,
        layer: int,
        source_density: float,
        target_density: float,
        transfer_weight: float,
    ) -> None:
        """Report density analysis for a layer."""
        self._emit(
            ProgressEvent(
                stage="density",
                substage="layer_analysis",
                model=None,
                what=f"Layer {layer} density analysis complete",
                why="Determines how much knowledge to transfer at this layer",
                progress={
                    "layer": layer,
                    "source_density": round(source_density, 4),
                    "target_density": round(target_density, 4),
                    "transfer_weight": round(transfer_weight, 4),
                },
                geometry={
                    "explanation": f"Source density {source_density:.2f} vs target {target_density:.2f}",
                    "transfer_meaning": f"Transfer weight {transfer_weight:.2f} = how much source knowledge to add",
                },
            )
        )

    # =========================================================================
    # TRANSPLANT STAGE
    # =========================================================================

    def transplant_started(self, layers: int, total_weights: int) -> None:
        """Report transplant stage started."""
        self._current_stage = "transplant"
        self._emit(
            ProgressEvent(
                stage="transplant",
                substage="started",
                model=None,
                what="Starting null-space knowledge transplant",
                why="Projecting source knowledge into target's unused representation space",
                progress={
                    "layers": layers,
                    "total_weights": total_weights,
                },
                geometry={
                    "explanation": "Null-space projection adds knowledge without disrupting existing capabilities",
                    "method": "Source deltas are rotated into directions the target doesn't use",
                },
            )
        )

    def transplant_layer_progress(
        self,
        layer: int,
        layer_total: int,
        null_rank: int,
        hidden_dim: int,
        preserved_fraction: float,
        weights_modified: int,
    ) -> None:
        """Report transplant progress for a layer."""
        self._emit(
            ProgressEvent(
                stage="transplant",
                substage="layer_transplant",
                model=None,
                what=f"Transplanting knowledge to layer {layer}",
                why="Adding source knowledge to target's null space at this layer",
                progress={
                    "layer": layer,
                    "layer_total": layer_total,
                    "percent": round(100 * (layer + 1) / layer_total, 1),
                    "weights_modified": weights_modified,
                },
                geometry={
                    "null_rank": null_rank,
                    "hidden_dim": hidden_dim,
                    "null_fraction": round(null_rank / max(1, hidden_dim), 4),
                    "preserved_fraction": round(preserved_fraction, 4),
                    "explanation": f"Layer {layer}: {null_rank}/{hidden_dim} dimensions available for transfer",
                    "preserved_meaning": f"{preserved_fraction:.1%} of source knowledge survived projection",
                },
            )
        )

    def transplant_complete(
        self,
        layers_transplanted: int,
        weights_modified: int,
        mean_preserved: float,
        output_path: str,
    ) -> None:
        """Report transplant complete."""
        self._emit(
            ProgressEvent(
                stage="transplant",
                substage="complete",
                model=None,
                what="Knowledge transplant complete",
                why="Merged model now contains knowledge from both source and target",
                progress={
                    "layers_transplanted": layers_transplanted,
                    "weights_modified": weights_modified,
                    "output_path": output_path,
                },
                geometry={
                    "mean_preserved_fraction": round(mean_preserved, 4),
                    "explanation": f"On average, {mean_preserved:.1%} of source knowledge was added to each layer",
                    "result": "Merged model should have source capabilities while preserving target behavior",
                },
            )
        )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    def merge_complete(
        self,
        source_path: str,
        target_path: str,
        output_path: str,
        total_duration: float,
        probe_duration: float,
        transplant_duration: float,
        layers: int,
        mean_cka: float,
        mean_preserved: float,
    ) -> None:
        """Report merge pipeline complete with summary."""
        self._emit(
            ProgressEvent(
                stage="complete",
                substage="summary",
                model=None,
                what="Merge pipeline complete",
                why="Models have been geometrically combined",
                progress={
                    "source": source_path,
                    "target": target_path,
                    "output": output_path,
                    "total_duration_s": round(total_duration, 2),
                    "probe_duration_s": round(probe_duration, 2),
                    "transplant_duration_s": round(transplant_duration, 2),
                    "layers_merged": layers,
                },
                geometry={
                    "mean_cka_alignment": round(mean_cka, 4),
                    "mean_preserved_fraction": round(mean_preserved, 4),
                    "explanation": f"CKA {mean_cka:.2f} = structural similarity after alignment",
                    "preserved_meaning": f"{mean_preserved:.1%} of source knowledge transferred on average",
                    "recommendation": "Run inference to verify coherence before deployment",
                },
            )
        )


    # =========================================================================
    # TRAINING: LOAD
    # =========================================================================

    def training_started(self, model_path: str, dataset_path: str) -> None:
        """Report training pipeline started."""
        self._current_stage = "load"
        self._emit(
            ProgressEvent(
                stage="load",
                substage="started",
                model=None,
                what="Starting geometry-derived LoRA training pipeline",
                why="Loading model and dataset to derive the plan before training; there is no fixed LR and MASS will choose step sizes online",
                progress={"model_path": model_path, "dataset_path": dataset_path},
                event_type="training_progress",
            )
        )

    def training_model_loaded(self, model_path: str, n_layers: int) -> None:
        """Report model loaded for training."""
        self._emit(
            ProgressEvent(
                stage="load",
                substage="model_loaded",
                model=None,
                what="Model loaded for training",
                why="Model weights are needed for geometric analysis and adapter injection",
                progress={"model_path": model_path, "layers": n_layers},
                event_type="training_progress",
            )
        )

    def training_dataset_loaded(
        self, n_train: int, n_eval: int, seq_length: int
    ) -> None:
        """Report dataset loaded and split."""
        self._emit(
            ProgressEvent(
                stage="load",
                substage="dataset_loaded",
                model=None,
                what="Dataset loaded and split",
                why="Training and evaluation samples prepared for geometry-derived training",
                progress={
                    "n_train": n_train,
                    "n_eval": n_eval,
                    "seq_length": seq_length,
                },
                event_type="training_progress",
            )
        )

    # =========================================================================
    # TRAINING: GEOMETRY ANALYSIS
    # =========================================================================

    def training_geometry_started(self) -> None:
        """Report geometry analysis started."""
        self._current_stage = "geometry"
        self._emit(
            ProgressEvent(
                stage="geometry",
                substage="started",
                model=None,
                what="Analyzing weight geometry via SVD",
                why="SVD of each weight matrix determines rank, null-space, and spectral bounds for training",
                progress={},
                event_type="training_progress",
            )
        )

    def training_geometry_complete(
        self,
        n_target_modules: int,
        n_trainable_params: int,
        batch_size: int,
        rank_min: int,
        rank_max: int,
        split_method: str,
    ) -> None:
        """Report geometry analysis and adapter injection complete."""
        self._emit(
            ProgressEvent(
                stage="geometry",
                substage="complete",
                model=None,
                what="Geometry analysis complete, adapter surface injected",
                why="All hyperparameters derived from model geometry — training configuration is fully determined",
                progress={
                    "target_modules": n_target_modules,
                    "trainable_params": n_trainable_params,
                    "batch_size": batch_size,
                    "rank_min": rank_min,
                    "rank_max": rank_max,
                    "split_method": split_method,
                },
                geometry={
                    "explanation": (
                        "Target surface, ranks, split method, and spectral ceilings are "
                        "resolved before training; only online controller terms remain "
                        "to be measured."
                    ),
                },
                event_type="training_progress",
            )
        )

    # =========================================================================
    # TRAINING: TRAIN LOOP
    # =========================================================================

    def training_loop_started(
        self,
        max_iters: int,
        *,
        iters_per_epoch: int | None = None,
        precision_floor_epochs: int | None = None,
    ) -> None:
        """Report training loop started."""
        progress: dict[str, Any] = {"max_iters": max_iters}
        geometry: dict[str, Any] | None = {
            "explanation": (
                "max_iters is the precision-derived safety cap, not the expected "
                "runtime; the geometric stopping certificate can terminate much earlier."
            ),
        }
        if iters_per_epoch is not None:
            progress["iters_per_epoch"] = iters_per_epoch
        if precision_floor_epochs is not None:
            progress["precision_floor_epochs"] = precision_floor_epochs
            geometry["precision_floor_meaning"] = (
                "The safety cap spans the number of epochs before loss changes fall "
                "below the IEEE-754 resolvability floor."
            )
        self._current_stage = "train"
        self._emit(
            ProgressEvent(
                stage="train",
                substage="started",
                model=None,
                what="Starting geometric LoRA training loop",
                why="Optimizing geometry-derived LoRA adapters; there is no fixed LR and MASS will choose step sizes online",
                progress=progress,
                geometry=geometry,
                event_type="training_progress",
            )
        )

    def training_loop_complete(
        self,
        train_iters: int,
        initial_loss: float,
        final_loss: float,
        stop_reason: str,
        training_time_seconds: float,
    ) -> None:
        """Report training loop complete."""
        self._emit(
            ProgressEvent(
                stage="train",
                substage="complete",
                model=None,
                what="Training loop complete",
                why="Adapter weights optimized — proceeding to verification",
                progress={
                    "train_iters": train_iters,
                    "initial_loss": round(initial_loss, 4),
                    "final_loss": round(final_loss, 4),
                    "stop_reason": stop_reason,
                    "training_time_s": round(training_time_seconds, 2),
                },
                event_type="training_progress",
            )
        )

    # =========================================================================
    # TRAINING: VERIFICATION
    # =========================================================================

    def training_verification_started(self, gates: list[str] | None = None) -> None:
        """Report post-training verification started."""
        self._current_stage = "verify"
        self._emit(
            ProgressEvent(
                stage="verify",
                substage="started",
                model=None,
                what="Starting post-training verification",
                why="Checking the exact post-training gates that determine whether the adapter is promotable on this run",
                progress={"gates": list(gates or [])},
                event_type="training_progress",
            )
        )

    def training_verification_complete(
        self,
        spectral_bounds_ok: bool,
        min_cka: float | None,
        mean_cka: float | None,
        gates: list[str] | None = None,
    ) -> None:
        """Report verification complete."""
        progress: dict[str, Any] = {"spectral_bounds_ok": spectral_bounds_ok}
        if min_cka is not None:
            progress["min_cka"] = round(min_cka, 4)
        if mean_cka is not None:
            progress["mean_cka"] = round(mean_cka, 4)
        if gates:
            progress["gates"] = list(gates)
        self._emit(
            ProgressEvent(
                stage="verify",
                substage="complete",
                model=None,
                what="Post-training verification complete",
                why="Verification completed against the same explicit gates that were announced before training",
                progress=progress,
                geometry={
                    "explanation": (
                        "CKA measures how well adapted model preserves base "
                        "representations (1.0 = perfect); promotability also checks "
                        "degeneration and the pipeline gate."
                    ),
                },
                event_type="training_progress",
            )
        )

    # =========================================================================
    # TRAINING: COMPLETE
    # =========================================================================

    def training_complete(
        self,
        adapter_path: str | None,
        training_time_seconds: float,
        final_loss: float,
        post_loss: float,
    ) -> None:
        """Report training pipeline complete."""
        self._emit(
            ProgressEvent(
                stage="complete",
                substage="summary",
                model=None,
                what="Training pipeline complete",
                why="Adapter trained and verified",
                progress={
                    "adapter_path": adapter_path,
                    "training_time_s": round(training_time_seconds, 2),
                    "final_loss": round(final_loss, 4),
                    "post_eval_loss": round(post_loss, 4),
                },
                event_type="training_progress",
            )
        )


# Global reporter instance (can be replaced for testing)
_reporter: ProgressReporter | None = None


def get_reporter() -> ProgressReporter:
    """Get the global progress reporter."""
    global _reporter
    if _reporter is None:
        _reporter = ProgressReporter()
    return _reporter


def set_reporter(reporter: ProgressReporter) -> None:
    """Set the global progress reporter (for testing)."""
    global _reporter
    _reporter = reporter
