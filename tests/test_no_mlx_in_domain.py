"""Guard test to prevent direct MLX imports in domain layer.

This test enforces the hexagonal architecture rule: the domain layer must
not depend on infrastructure (MLX is platform-specific infrastructure).

All domain files should use the Backend protocol via:
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.ports.backend import Array, Backend

Files that still import MLX directly are tracked below and should be migrated.
"""

from pathlib import Path

import pytest

# Files already migrated to use Backend protocol
MIGRATED_FILES = {
    "semantics/vector_space.py",
    "geometry/types.py",
    "geometry/intrinsic_dimension.py",
    "geometry/fingerprints.py",
    "entropy/logit_entropy_calculator.py",
    "geometry/generalized_procrustes.py",
    "entropy/conflict_score.py",
    "entropy/entropy_tracker.py",
    "entropy/entropy_delta_tracker.py",
    "dynamics/regime_state_detector.py",
    "entropy/sep_probe.py",
    "entropy/hidden_state_extractor.py",
    "inference/entropy_dynamics.py",
    "geometry/topological_fingerprint.py",
    "geometry/probes.py",
    "geometry/compositional_probes.py",
    "geometry/dora_decomposition.py",
    "geometry/tangent_space_alignment.py",
    "geometry/fisher_blending.py",
    "geometry/refusal_direction_detector.py",
    "geometry/metaphor_convergence_analyzer.py",
    "geometry/neuron_sparsity_analyzer.py",
    "geometry/invariant_convergence_analyzer.py",
    "geometry/refinement_density.py",
    "geometry/sparse_region_prober.py",
    "geometry/manifold_fidelity_sweep.py",
    "geometry/permutation_aligner.py",
    "training/gradient_smoothness_estimator.py",
    "merging/gradient_boundary_smoother.py",
    "merging/rotational_merger.py",
    "merging/unified_manifold_merger.py",  # Fully migrated to Backend
    "geometry/manifold_stitcher.py",  # Fully migrated to Backend
}

# Files still to be migrated (tracked for progress monitoring)
# Remove from this set as files are migrated
# NOTE: Some files have infrastructure dependencies (mlx.nn, mlx_lm) that
# cannot be fully abstracted via Backend protocol. These remain here until
# a full training/inference abstraction layer is implemented.
PENDING_MIGRATION = {
    # Full MLX dependencies (infrastructure: mlx.nn, file I/O)
    "merging/lora_adapter_merger.py",  # mx.load, mx.save_safetensors for file I/O
    "training/loss_landscape.py",
    "training/evaluation.py",
    "training/engine.py",
    "training/lora.py",  # mlx.nn.Module for LoRA layers
    "training/checkpoints.py",  # mx.save_safetensors, mx.load
    # Partial migration (math ops use Backend, model loading uses mlx_lm)
    "inference/dual_path.py",  # mlx_lm for model loading
    "thermo/linguistic_calorimeter.py",  # mlx_lm for model loading
}


def get_mlx_imports_in_domain():
    """Find all files in domain that import mlx directly."""
    domain_path = Path("src/modelcypher/core/domain")
    mlx_files = []

    for py_file in domain_path.rglob("*.py"):
        # Skip the backend manager itself
        if py_file.name == "_backend.py":
            continue

        content = py_file.read_text()
        if "import mlx" in content or "from mlx" in content:
            relative = py_file.relative_to(domain_path)
            mlx_files.append(str(relative))

    return set(mlx_files)


def test_migrated_files_have_no_mlx():
    """Verify that files marked as migrated don't import MLX."""
    mlx_files = get_mlx_imports_in_domain()
    violations = mlx_files & MIGRATED_FILES

    if violations:
        pytest.fail(
            f"Files marked as migrated still import MLX: {violations}\n"
            "Either complete the migration or remove from MIGRATED_FILES."
        )


def test_pending_files_are_tracked():
    """Verify pending migration files are accurately tracked."""
    mlx_files = get_mlx_imports_in_domain()
    untracked = mlx_files - PENDING_MIGRATION - MIGRATED_FILES

    if untracked:
        pytest.fail(
            f"New MLX imports found in domain that are not tracked: {untracked}\n"
            "Add to PENDING_MIGRATION set to track progress."
        )


def test_migration_progress():
    """Report current migration progress."""
    total = len(MIGRATED_FILES) + len(PENDING_MIGRATION)
    migrated = len(MIGRATED_FILES)
    print(f"\n[Migration Progress] {migrated}/{total} files migrated")
    print(f"  - Migrated: {migrated}")
    print(f"  - Pending:  {len(PENDING_MIGRATION)}")
