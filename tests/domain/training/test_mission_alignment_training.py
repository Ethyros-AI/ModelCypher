# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]

_FILE_PATTERNS: dict[str, list[str]] = {
    "src/modelcypher/backends/_mlx_training_adapter_train_mixin.py": [
        "_ARMIJO_C =",
        "_ARMIJO_BETA =",
        "_ARMIJO_MAX_BACKTRACKS =",
        " + 1e-10",
        " + 1e-8",
        # REINFORCE ⊥ CE derivation (2026-03-06): outcome training is dead
        "outcome_training",
        "make_outcome_loss",
    ],
    "src/modelcypher/backends/_mlx_training_adapter_core_mixin.py": [
        "safety_margin: float = 0.9",
        "n_samples = min(len(paired_dataset), 50)",
        " + 1e-10",
    ],
    "src/modelcypher/core/use_cases/dataset_training_service.py": [
        "safety_margin: float = 0.9",
        "min(4, len(eval_dataset) //",
        "max(batch_size, 8)",
        "target_count = min(len(train_samples), 200)",
        "[:128]",
        "lr_override",
        "scale_bound_override",
        "research_allow_quantization_frontier_invalid",
        "constraint_state_override",
        # Removed in deep audit: these are geometry-derived, not user knobs
        "deep: bool",
        "eval_batches: int",
        "adaptive_lr: bool",
        "lr_monotonic: bool",
        # REINFORCE ⊥ CE derivation (2026-03-06): outcome training is dead
        "outcome_training",
        "auto_regime",
    ],
    "src/modelcypher/cli/commands/train.py": [
        "--safety-margin",
        "--lr",
        # Removed in deep audit: geometry derives these
        "--deep",
        "--eval-batches",
        "--adaptive-lr",
        "--lr-monotonic",
        "--max-iters",
        # REINFORCE ⊥ CE derivation (2026-03-06)
        "--auto-regime",
    ],
    "src/modelcypher/adapters/model_loader.py": [
        "def load_model_for_training(",
        "def get_model_loader(",
        "def load_model_weights_only(",
        "backwards compatibility",
    ],
    "src/modelcypher/experimental/merge/lora_adapter_merger.py": [
        'capability_transfer: bool = True',
        'get("capability_transfer", "true")',
        'get("training_objective", "unknown")',
    ],
    "src/modelcypher/experimental/merge/pipeline.py": [
        "MC_INJECTION_LAYER",
    ],
    "src/modelcypher/experimental/merge/stages/probe_from_profile.py": [
        "MC_INJECTION_LAYER",
    ],
    "src/modelcypher/experimental/merge/stages/transplant_embeddings.py": [
        "MC_FORCE_EMBEDDING_TRANSPLANT",
        "MC_SKIP_EMBEDDING_TRANSPLANT",
    ],
}


def test_training_path_omits_legacy_guess_constants() -> None:
    """Guardrail for mission-critical training files.

    This test blocks reintroduction of legacy heuristic literals/flags that were
    removed in favor of geometry-, dtype-, or measured-data-derived values.
    """
    violations: list[str] = []
    for rel_path, banned_fragments in _FILE_PATTERNS.items():
        content = (_ROOT / rel_path).read_text(encoding="utf-8")
        for fragment in banned_fragments:
            if fragment in content:
                violations.append(f"{rel_path}: contains '{fragment}'")

    assert not violations, "\n".join(violations)
