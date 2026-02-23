#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ModelCypher
"""Shared types and dataclasses for self-improvement system.

This module defines the core data structures used throughout the
autonomous self-improvement pipeline:

- CapabilityStatus: Classification of capability state
- Capability: Definition of a testable capability
- CapabilityAnalysis: Results of scanning a capability
- VerifiedSample: Training sample verified by oracle
- ImprovementAction: Action taken during improvement loop
- ImprovementLog: Full log of improvement session
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class CapabilityStatus(Enum):
    """Classification of a capability's state.

    - WORKING: accuracy_raw >= threshold (capability works without intervention)
    - DISCONNECTED: accuracy_primed >= threshold but accuracy_raw < threshold
                    (capability exists but needs activation via priming)
    - TRUE_GAP: both accuracies < threshold (capability is missing, needs training)
    """

    WORKING = "working"
    DISCONNECTED = "disconnected"
    TRUE_GAP = "true_gap"


@dataclass(frozen=True)
class Capability:
    """Definition of a testable capability.

    Attributes:
        name: Human-readable name (e.g., "arithmetic", "word_problems")
        prompts: Prompts for activation analysis (compute κ)
        problems: Test problems as (prompt, expected_answer) pairs
    """

    name: str
    prompts: Tuple[str, ...]
    problems: Tuple[Tuple[str, str], ...]

    @classmethod
    def from_lists(
        cls,
        name: str,
        prompts: List[str],
        problems: List[Tuple[str, str]],
    ) -> "Capability":
        """Create from mutable lists (convenience constructor)."""
        return cls(
            name=name,
            prompts=tuple(prompts),
            problems=tuple(tuple(p) for p in problems),
        )


@dataclass
class CapabilityAnalysis:
    """Results of scanning a capability.

    Attributes:
        capability: The capability that was scanned
        status: Classification result (WORKING/DISCONNECTED/TRUE_GAP)
        accuracy_raw: Accuracy without any priming
        accuracy_primed: Accuracy with best-performing prime
        kappa_raw: Condition number of Gram matrix (raw activations)
        kappa_primed: Condition number with priming
        best_prime: The prime that achieved highest accuracy
    """

    capability: Capability
    status: CapabilityStatus
    accuracy_raw: float
    accuracy_primed: float
    kappa_raw: float
    kappa_primed: float
    best_prime: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        return {
            "name": self.capability.name,
            "status": self.status.value,
            "accuracy_raw": self.accuracy_raw,
            "accuracy_primed": self.accuracy_primed,
            "kappa_raw": self.kappa_raw,
            "kappa_primed": self.kappa_primed,
            "best_prime": self.best_prime,
        }


@dataclass(frozen=True)
class VerifiedSample:
    """A training sample verified by the oracle.

    Attributes:
        input_text: The input (e.g., "I have 3 apples. I get 2 more. Total:")
        output_text: The expected output (e.g., "3+2=")
        answer: The verified answer (e.g., "5")
        oracle_computed: What the oracle actually computed
    """

    input_text: str
    output_text: str
    answer: str
    oracle_computed: str

    def to_training_format(self) -> Dict[str, str]:
        """Convert to training format (prompt/completion)."""
        return {
            "prompt": self.input_text,
            "completion": f"{self.output_text}{self.answer}",
        }

    def to_dict(self) -> Dict[str, str]:
        """Serialize to dictionary for JSON output."""
        return {
            "input": self.input_text,
            "output": self.output_text,
            "answer": self.answer,
            "verified_computed": self.oracle_computed,
        }


@dataclass
class ImprovementAction:
    """An action taken during the improvement loop.

    Attributes:
        capability: Name of the capability being improved
        action_type: Type of action ("apply_prime", "generate_training", "specify_lora")
        details: Additional details about the action
    """

    capability: str
    action_type: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        return {
            "capability": self.capability,
            "action_type": self.action_type,
            "details": self.details,
        }


@dataclass
class SelfImprovementConfig:
    """Configuration for self-improvement pipeline.

    Attributes:
        loop_preservation: If True, add loop preservation loss during training.
            The loss weight (λ) is derived from geometry (1/σ_max), not configurable.
        geometric_self_awareness: If True, augment training data with
            geometric context ([GEOMETRY] prefix).
        max_rounds: Maximum improvement rounds.
        n_samples_per_round: Training samples per round.
    """

    loop_preservation: bool = True
    geometric_self_awareness: bool = True
    max_rounds: int
    n_samples_per_round: int


@dataclass
class ImprovementLog:
    """Log of an improvement session.

    Attributes:
        iterations: Number of improvement iterations run
        capabilities_scanned: Names of capabilities that were scanned
        capabilities_working: Names of capabilities classified as WORKING
        capabilities_bridged: Names of capabilities that were bridged via priming
        true_gaps: Names of capabilities classified as TRUE_GAP
        actions: List of improvement actions taken
        training_data_path: Path to generated training data (if any)
        training_spec: LoRA training specification (if any)
    """

    iterations: int = 0
    capabilities_scanned: List[str] = field(default_factory=list)
    capabilities_working: List[str] = field(default_factory=list)
    capabilities_bridged: List[str] = field(default_factory=list)
    true_gaps: List[str] = field(default_factory=list)
    actions: List[ImprovementAction] = field(default_factory=list)
    training_data_path: Optional[str] = None
    training_spec: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        return {
            "iterations": self.iterations,
            "capabilities_scanned": self.capabilities_scanned,
            "capabilities_working": self.capabilities_working,
            "capabilities_bridged": self.capabilities_bridged,
            "true_gaps": self.true_gaps,
            "actions": [a.to_dict() for a in self.actions],
            "training_data_path": self.training_data_path,
            "training_spec": self.training_spec,
        }


# Default primes discovered during experiments
DEFAULT_PRIMES: Tuple[str, ...] = (
    "say",
    "Arithmetic means calculating numbers.",
    "One less is",
)


__all__ = [
    "CapabilityStatus",
    "Capability",
    "CapabilityAnalysis",
    "VerifiedSample",
    "ImprovementAction",
    "ImprovementLog",
    "SelfImprovementConfig",
    "DEFAULT_PRIMES",
]
