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

"""LoRA stacker for cumulative self-improvement.

This module manages stacked LoRA adapters for iterative self-improvement:
- Tracks cumulative geometry changes across adapter stack
- Decides when to merge vs continue stacking
- Provides composed model access for evaluation

The stacking loop:
    1. Train LoRA on identified gaps
    2. Add to stack, measure cumulative geometry
    3. Check if merge needed (barrier > threshold OR drift > threshold)
    4. If merge: consolidate all adapters
    5. Repeat with increased difficulty

Policy thresholds must be specified explicitly - there are no defaults.
The caller decides merge policy based on their use case.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)


@dataclass
class AdapterInfo:
    """Information about a single adapter in the stack."""

    path: Path
    added_at: str
    barrier_contribution: float
    cka_from_base: float
    difficulty_level: int
    training_samples: int = 0
    target_modules: List[str] = field(default_factory=list)
    exit_convergence: float = 0.0  # exit_mean_norm / exit_dev_norm

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "path": str(self.path),
            "added_at": self.added_at,
            "barrier_contribution": self.barrier_contribution,
            "cka_from_base": self.cka_from_base,
            "difficulty_level": self.difficulty_level,
            "training_samples": self.training_samples,
            "target_modules": self.target_modules,
            "exit_convergence": self.exit_convergence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdapterInfo":
        """Deserialize from dict."""
        return cls(
            path=Path(data["path"]),
            added_at=data["added_at"],
            barrier_contribution=data["barrier_contribution"],
            cka_from_base=data["cka_from_base"],
            difficulty_level=data["difficulty_level"],
            training_samples=data.get("training_samples", 0),
            target_modules=data.get("target_modules", []),
            exit_convergence=data.get("exit_convergence", 0.0),
        )


@dataclass
class StackerPolicy:
    """Policy thresholds for LoRA stacking decisions.

    All values must be explicitly specified - no defaults.
    The caller decides policy based on their use case.
    """

    barrier_merge_threshold: float
    """Merge if cumulative mode connectivity barrier exceeds this."""

    cka_drift_threshold: float
    """Merge if CKA drift from base exceeds this."""

    max_adapters: int
    """Hard limit on stack depth before forced merge."""

    convergence_ratio_threshold: float
    """Trigger convergence detection if adapter/base convergence exceeds this."""

    convergence_barrier_multiplier: float
    """When convergence detected, multiply barrier threshold by this factor."""


@dataclass
class StackedLoRAState:
    """Track cumulative state across LoRA stack."""

    base_model_path: Path
    policy: StackerPolicy
    adapters: List[AdapterInfo] = field(default_factory=list)
    cumulative_barrier: float = 0.0
    cumulative_cka_drift: float = 0.0
    current_difficulty: int = 0
    merges_performed: int = 0
    convergence_detected: bool = False  # Training saturation signal
    base_exit_convergence: float = 0.0  # Reference from base model

    @property
    def n_adapters(self) -> int:
        """Number of adapters currently stacked."""
        return len(self.adapters)

    @property
    def effective_barrier_threshold(self) -> float:
        """Get barrier threshold, adjusted for convergence.

        When convergence is detected (training saturated), merge earlier
        to consolidate gains before continuing.
        """
        if self.convergence_detected:
            return self.policy.barrier_merge_threshold * self.policy.convergence_barrier_multiplier
        return self.policy.barrier_merge_threshold

    @property
    def should_merge(self) -> bool:
        """Decide if stack should be consolidated.

        Thresholds are reduced when convergence is detected, causing
        earlier merging when training shows signs of saturation.
        """
        # Merge if any threshold exceeded
        if self.cumulative_barrier > self.effective_barrier_threshold:
            return True
        if self.cumulative_cka_drift > self.policy.cka_drift_threshold:
            return True
        if self.n_adapters >= self.policy.max_adapters:
            return True
        return False

    @property
    def merge_reason(self) -> str:
        """Get reason for merge recommendation."""
        threshold = self.effective_barrier_threshold
        convergence_note = " [convergence detected]" if self.convergence_detected else ""

        if self.cumulative_barrier > threshold:
            return f"barrier_exceeded ({self.cumulative_barrier:.4f} > {threshold}){convergence_note}"
        if self.cumulative_cka_drift > self.policy.cka_drift_threshold:
            return f"cka_drift_exceeded ({self.cumulative_cka_drift:.4f} > {self.policy.cka_drift_threshold})"
        if self.n_adapters >= self.policy.max_adapters:
            return f"adapter_count ({self.n_adapters} >= {self.policy.max_adapters})"
        return "none"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "base_model_path": str(self.base_model_path),
            "policy": {
                "barrier_merge_threshold": self.policy.barrier_merge_threshold,
                "cka_drift_threshold": self.policy.cka_drift_threshold,
                "max_adapters": self.policy.max_adapters,
                "convergence_ratio_threshold": self.policy.convergence_ratio_threshold,
                "convergence_barrier_multiplier": self.policy.convergence_barrier_multiplier,
            },
            "adapters": [a.to_dict() for a in self.adapters],
            "cumulative_barrier": self.cumulative_barrier,
            "cumulative_cka_drift": self.cumulative_cka_drift,
            "current_difficulty": self.current_difficulty,
            "merges_performed": self.merges_performed,
            "convergence_detected": self.convergence_detected,
            "base_exit_convergence": self.base_exit_convergence,
            "n_adapters": self.n_adapters,
            "should_merge": self.should_merge,
            "merge_reason": self.merge_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StackedLoRAState":
        """Deserialize from dict."""
        policy_data = data["policy"]  # Required - no defaults
        policy = StackerPolicy(
            barrier_merge_threshold=policy_data["barrier_merge_threshold"],
            cka_drift_threshold=policy_data["cka_drift_threshold"],
            max_adapters=policy_data["max_adapters"],
            convergence_ratio_threshold=policy_data["convergence_ratio_threshold"],
            convergence_barrier_multiplier=policy_data["convergence_barrier_multiplier"],
        )
        return cls(
            base_model_path=Path(data["base_model_path"]),
            policy=policy,
            adapters=[AdapterInfo.from_dict(a) for a in data.get("adapters", [])],
            cumulative_barrier=data.get("cumulative_barrier", 0.0),
            cumulative_cka_drift=data.get("cumulative_cka_drift", 0.0),
            current_difficulty=data.get("current_difficulty", 0),
            merges_performed=data.get("merges_performed", 0),
            convergence_detected=data.get("convergence_detected", False),
            base_exit_convergence=data.get("base_exit_convergence", 0.0),
        )

    def save(self, path: Path) -> None:
        """Save state to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved stacker state to %s", path)

    @classmethod
    def load(cls, path: Path) -> "StackedLoRAState":
        """Load state from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class StackResult:
    """Result of adding an adapter to the stack."""

    success: bool
    adapter_info: Optional[AdapterInfo]
    cumulative_barrier: float
    cumulative_cka_drift: float
    should_merge: bool
    merge_reason: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "success": self.success,
            "adapter_info": self.adapter_info.to_dict() if self.adapter_info else None,
            "cumulative_barrier": self.cumulative_barrier,
            "cumulative_cka_drift": self.cumulative_cka_drift,
            "should_merge": self.should_merge,
            "merge_reason": self.merge_reason,
            "message": self.message,
        }


@dataclass
class LoraStackMergeResult:
    """Result of merging the adapter stack."""

    success: bool
    merged_model_path: Optional[Path]
    adapters_merged: int
    previous_barrier: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "success": self.success,
            "merged_model_path": str(self.merged_model_path) if self.merged_model_path else None,
            "adapters_merged": self.adapters_merged,
            "previous_barrier": self.previous_barrier,
            "message": self.message,
        }


MergeResult = LoraStackMergeResult


class LoRAStacker:
    """Manage stacked LoRA adapters for cumulative self-improvement.

    Example usage:
        policy = StackerPolicy(
            barrier_merge_threshold=0.03,
            cka_drift_threshold=0.1,
            max_adapters=5,
            convergence_ratio_threshold=1.0,
            convergence_barrier_multiplier=0.5,
        )
        stacker = LoRAStacker(Path("/path/to/base/model"), policy=policy)

        # Add adapters as they're trained
        result = stacker.add_adapter(
            adapter_path=Path("/path/to/adapter1"),
            barrier=0.008,
            cka_from_base=0.95,
            difficulty_level=1,
        )

        if result.should_merge:
            merge_result = stacker.merge_stack(output_path)

        # Save state for persistence
        stacker.save_state(Path("stack_state.json"))
    """

    def __init__(
        self,
        base_model_path: Path,
        policy: StackerPolicy,
        backend: "Backend | None" = None,
        state_path: Optional[Path] = None,
        base_exit_convergence: float = 0.0,
    ) -> None:
        """Initialize the LoRA stacker.

        Args:
            base_model_path: Path to the base model
            policy: Stacking policy thresholds (required, no defaults)
            backend: Compute backend (optional, loads default if needed)
            state_path: Path to existing state file (optional, for resuming)
            base_exit_convergence: Exit convergence of the base model
                                  (mean_norm / dev_norm at exit layer).
                                  Used as reference for saturation detection.
                                  If not provided, convergence detection is disabled.
        """
        self.base_model_path = Path(base_model_path)
        self._backend = backend
        self._policy = policy

        # Load existing state or create new
        if state_path and state_path.exists():
            self.state = StackedLoRAState.load(state_path)
            logger.info(
                "Loaded existing stack: %d adapters, barrier=%.4f, drift=%.4f",
                self.state.n_adapters,
                self.state.cumulative_barrier,
                self.state.cumulative_cka_drift,
            )
        else:
            self.state = StackedLoRAState(
                base_model_path=self.base_model_path,
                policy=policy,
                base_exit_convergence=base_exit_convergence,
            )
            if base_exit_convergence > 0:
                logger.info(
                    "Created new stacker for %s (base_convergence=%.2f)",
                    base_model_path, base_exit_convergence
                )
            else:
                logger.info("Created new stacker for %s", base_model_path)

    @property
    def backend(self) -> "Backend":
        """Get compute backend, loading default if needed."""
        if self._backend is None:
            from modelcypher.core.domain._backend import get_default_backend
            self._backend = get_default_backend()
        return self._backend

    def add_adapter(
        self,
        adapter_path: Path,
        barrier: float,
        cka_from_base: float,
        difficulty_level: int,
        training_samples: int = 0,
        target_modules: Optional[List[str]] = None,
        exit_convergence: float = 0.0,
    ) -> StackResult:
        """Add a new adapter to the stack.

        Args:
            adapter_path: Path to the trained adapter
            barrier: Mode connectivity barrier for this adapter
            cka_from_base: CKA similarity to base model (1.0 = identical)
            difficulty_level: Curriculum difficulty level
            training_samples: Number of samples used for training
            target_modules: Which modules were targeted
            exit_convergence: Exit layer convergence (mean_norm / dev_norm).
                             Values > 1.5 indicate training saturation, which
                             triggers earlier merging to consolidate gains.

        Returns:
            StackResult with cumulative metrics and merge recommendation
        """
        adapter_path = Path(adapter_path)
        if not adapter_path.exists():
            return StackResult(
                success=False,
                adapter_info=None,
                cumulative_barrier=self.state.cumulative_barrier,
                cumulative_cka_drift=self.state.cumulative_cka_drift,
                should_merge=False,
                merge_reason="none",
                message=f"Adapter path does not exist: {adapter_path}",
            )

        # Create adapter info
        adapter_info = AdapterInfo(
            path=adapter_path,
            added_at=datetime.now().isoformat(),
            barrier_contribution=barrier,
            cka_from_base=cka_from_base,
            difficulty_level=difficulty_level,
            training_samples=training_samples,
            target_modules=target_modules or [],
            exit_convergence=exit_convergence,
        )

        # Update cumulative metrics
        # Barrier accumulates (conservative: assume additive)
        self.state.cumulative_barrier += barrier

        # CKA drift: use max drift from any adapter (worst case)
        cka_drift = 1.0 - cka_from_base
        if cka_drift > self.state.cumulative_cka_drift:
            self.state.cumulative_cka_drift = cka_drift

        # Check for convergence signal (training saturation)
        # Relational: compare adapter convergence to base model convergence
        # If adapter is MORE converged than base, training is saturating
        base_conv = self.state.base_exit_convergence
        if base_conv > 0 and exit_convergence > 0:
            convergence_ratio = exit_convergence / base_conv
            if convergence_ratio > self.state.policy.convergence_ratio_threshold:
                self.state.convergence_detected = True
                logger.info(
                    "Convergence detected: adapter/base ratio=%.2f > %.2f "
                    "(adapter=%.2f, base=%.2f, threshold reduced to %.4f)",
                    convergence_ratio,
                    self.state.policy.convergence_ratio_threshold,
                    exit_convergence,
                    base_conv,
                    self.state.effective_barrier_threshold,
                )

        # Add to stack
        self.state.adapters.append(adapter_info)
        self.state.current_difficulty = max(self.state.current_difficulty, difficulty_level)

        logger.info(
            "Added adapter %s: barrier=%.4f (cumulative=%.4f), cka_drift=%.4f, convergence=%s",
            adapter_path.name,
            barrier,
            self.state.cumulative_barrier,
            self.state.cumulative_cka_drift,
            "detected" if self.state.convergence_detected else "none",
        )

        return StackResult(
            success=True,
            adapter_info=adapter_info,
            cumulative_barrier=self.state.cumulative_barrier,
            cumulative_cka_drift=self.state.cumulative_cka_drift,
            should_merge=self.state.should_merge,
            merge_reason=self.state.merge_reason,
            message=f"Added adapter {adapter_path.name} (stack depth: {self.state.n_adapters})",
        )

    def merge_stack(self, output_path: Path) -> MergeResult:
        """Merge all stacked adapters into a single model.

        This consolidates the stack, resetting cumulative metrics.
        The merged model becomes the new base for future adapters.

        Args:
            output_path: Where to save the merged model

        Returns:
            MergeResult with merge details
        """
        if self.state.n_adapters == 0:
            return MergeResult(
                success=False,
                merged_model_path=None,
                adapters_merged=0,
                previous_barrier=0.0,
                message="No adapters to merge",
            )

        # Import merge infrastructure
        try:
            from modelcypher.adapters.merging.lora_adapter_merger import LoraAdapterMerger
        except ImportError:
            return MergeResult(
                success=False,
                merged_model_path=None,
                adapters_merged=self.state.n_adapters,
                previous_barrier=self.state.cumulative_barrier,
                message="LoraAdapterMerger not available",
            )

        output_path = Path(output_path)
        adapter_paths = [a.path for a in self.state.adapters]
        previous_barrier = self.state.cumulative_barrier
        n_adapters = self.state.n_adapters

        logger.info(
            "Merging %d adapters into %s (barrier=%.4f)",
            n_adapters,
            output_path,
            previous_barrier,
        )

        try:
            # Use existing merger with Fisher weighting
            merger = LoraAdapterMerger(backend=self.backend)
            merger.merge_adapters(
                base_model_path=self.state.base_model_path,
                adapter_paths=adapter_paths,
                output_path=output_path,
                weight_method="fisher",  # Use Fisher-weighted merge
            )

            # Reset state with merged model as new base
            self.state = StackedLoRAState(
                base_model_path=output_path,
                policy=self.state.policy,
                merges_performed=self.state.merges_performed + 1,
                current_difficulty=self.state.current_difficulty,
            )

            return MergeResult(
                success=True,
                merged_model_path=output_path,
                adapters_merged=n_adapters,
                previous_barrier=previous_barrier,
                message=f"Merged {n_adapters} adapters (barrier was {previous_barrier:.4f})",
            )

        except Exception as e:
            logger.error("Merge failed: %s", e)
            return MergeResult(
                success=False,
                merged_model_path=None,
                adapters_merged=0,
                previous_barrier=previous_barrier,
                message=f"Merge failed: {e}",
            )

    def get_adapter_paths(self) -> List[Path]:
        """Get ordered list of adapter paths for sequential application."""
        return [a.path for a in self.state.adapters]

    def save_state(self, path: Path) -> None:
        """Save stacker state to JSON file."""
        self.state.save(path)

    def get_status(self) -> dict[str, Any]:
        """Get current stacker status."""
        return {
            "base_model": str(self.state.base_model_path),
            "n_adapters": self.state.n_adapters,
            "cumulative_barrier": self.state.cumulative_barrier,
            "cumulative_cka_drift": self.state.cumulative_cka_drift,
            "current_difficulty": self.state.current_difficulty,
            "should_merge": self.state.should_merge,
            "merge_reason": self.state.merge_reason,
            "merges_performed": self.state.merges_performed,
            "policy": {
                "barrier_merge": self.state.policy.barrier_merge_threshold,
                "cka_drift": self.state.policy.cka_drift_threshold,
                "max_adapters": self.state.policy.max_adapters,
            },
        }


__all__ = [
    "LoRAStacker",
    "StackedLoRAState",
    "StackerPolicy",
    "StackResult",
    "MergeResult",
    "AdapterInfo",
]
