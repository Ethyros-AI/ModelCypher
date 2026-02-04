#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ModelCypher
"""Autonomous self-improvement system for language models.

This module provides a complete autonomous self-improvement pipeline that:
1. Scans model capabilities via geometry (κ) and behavior
2. Classifies capabilities as WORKING, DISCONNECTED, or TRUE_GAP
3. Bridges disconnected capabilities via semantic priming
4. Generates oracle-verified training data for true gaps
5. Outputs LoRA training specifications

Key Components:
    CapabilityScanner: Analyzes model capabilities
    VerificationOracle: Validates learning using verified capabilities
    SafeSelfPlayGenerator: Creates ground-truth verified training data
    AutonomousSelfImprover: Orchestrates the complete improvement loop

Example Usage:
    >>> from modelcypher.core.use_cases.self_improve import (
    ...     AutonomousSelfImprover,
    ...     Capability,
    ... )
    >>> from modelcypher.adapters.model_loader import ModelLoader
    >>>
    >>> loader = ModelLoader()
    >>> model, tokenizer = loader.load_model("path/to/model")
    >>> improver = AutonomousSelfImprover(model, tokenizer)
    >>>
    >>> capabilities = [
    ...     Capability.from_lists(
    ...         "arithmetic",
    ...         prompts=["1+1=", "2+2="],
    ...         problems=[("1+1=", "2"), ("2+2=", "4")],
    ...     ),
    ...     Capability.from_lists(
    ...         "word_problems",
    ...         prompts=["I have 3 apples. I get 2 more. Total:"],
    ...         problems=[("I have 3 apples. I get 2 more. Total:", "5")],
    ...     ),
    ... ]
    >>>
    >>> log = improver.improve(capabilities)
    >>> print(f"Bridged: {log.capabilities_bridged}")
    >>> print(f"True gaps: {log.true_gaps}")

Safety Guarantee:
    The system uses verified capabilities as oracles to check new learning.
    Every training sample is verified against ground truth before use.
    This prevents the model from learning nonsense or incorrect mappings.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Lazy loading configuration: (module_name, attribute_name)
# NOTE: Scanner and Oracle are in adapters (MLX-specific inference code)
_ATTR_TO_MODULE = {
    # Types
    "CapabilityStatus": ("types", "CapabilityStatus"),
    "Capability": ("types", "Capability"),
    "CapabilityAnalysis": ("types", "CapabilityAnalysis"),
    "VerifiedSample": ("types", "VerifiedSample"),
    "ImprovementAction": ("types", "ImprovementAction"),
    "ImprovementLog": ("types", "ImprovementLog"),
    "DEFAULT_PRIMES": ("types", "DEFAULT_PRIMES"),
    "DEFAULT_ACCURACY_THRESHOLD": ("types", "DEFAULT_ACCURACY_THRESHOLD"),
    # Generator
    "SafeSelfPlayGenerator": ("generator", "SafeSelfPlayGenerator"),
    # Improver
    "AutonomousSelfImprover": ("improver", "AutonomousSelfImprover"),
    # Stacker
    "LoRAStacker": ("lora_stacker", "LoRAStacker"),
    "StackedLoRAState": ("lora_stacker", "StackedLoRAState"),
    "StackResult": ("lora_stacker", "StackResult"),
    "MergeResult": ("lora_stacker", "MergeResult"),
    "AdapterInfo": ("lora_stacker", "AdapterInfo"),
}

# Adapter imports (MLX-specific)
_ADAPTER_IMPORTS = {
    "CapabilityScanner": "modelcypher.adapters.self_improve.mlx.scanner",
    "VerificationOracle": "modelcypher.adapters.self_improve.mlx.oracle",
}


def __getattr__(name: str):
    """Lazy load module attributes on first access."""
    if name in _ATTR_TO_MODULE:
        module_name, attr_name = _ATTR_TO_MODULE[name]
        module = importlib.import_module(f".{module_name}", __name__)
        return getattr(module, attr_name)
    if name in _ADAPTER_IMPORTS:
        module = importlib.import_module(_ADAPTER_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List available attributes for tab completion."""
    return list(_ATTR_TO_MODULE.keys()) + list(_ADAPTER_IMPORTS.keys())


if TYPE_CHECKING:
    # For static type checkers and IDEs
    from .generator import SafeSelfPlayGenerator
    from .improver import AutonomousSelfImprover
    from .lora_stacker import (
        AdapterInfo,
        LoRAStacker,
        MergeResult,
        StackedLoRAState,
        StackResult,
    )
    from .types import (
        DEFAULT_ACCURACY_THRESHOLD,
        DEFAULT_PRIMES,
        Capability,
        CapabilityAnalysis,
        CapabilityStatus,
        ImprovementAction,
        ImprovementLog,
        VerifiedSample,
    )
    # MLX-specific adapters
    from modelcypher.adapters.self_improve.mlx.oracle import VerificationOracle
    from modelcypher.adapters.self_improve.mlx.scanner import CapabilityScanner


__all__ = [
    # Types
    "CapabilityStatus",
    "Capability",
    "CapabilityAnalysis",
    "VerifiedSample",
    "ImprovementAction",
    "ImprovementLog",
    "DEFAULT_PRIMES",
    "DEFAULT_ACCURACY_THRESHOLD",
    # Classes
    "CapabilityScanner",
    "VerificationOracle",
    "SafeSelfPlayGenerator",
    "AutonomousSelfImprover",
    # Stacker
    "LoRAStacker",
    "StackedLoRAState",
    "StackResult",
    "MergeResult",
    "AdapterInfo",
]
