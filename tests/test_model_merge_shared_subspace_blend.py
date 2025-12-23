from modelcypher.core.use_cases.model_merge_service import (
    DEFAULT_SHARED_SUBSPACE_BLEND,
    ModelMergeService,
)


def test_shared_subspace_blend_defaults_to_zero_when_disabled() -> None:
    resolved = ModelMergeService._resolve_shared_subspace_blend(False, None)
    assert resolved == 0.0


def test_shared_subspace_blend_defaults_to_full_when_enabled() -> None:
    resolved = ModelMergeService._resolve_shared_subspace_blend(True, None)
    assert resolved == DEFAULT_SHARED_SUBSPACE_BLEND


def test_shared_subspace_blend_respects_explicit_value() -> None:
    resolved = ModelMergeService._resolve_shared_subspace_blend(True, 0.25)
    assert resolved == 0.25


def test_shared_subspace_blend_disabled_ignores_explicit_value() -> None:
    """When disabled, explicit value is ignored."""
    resolved = ModelMergeService._resolve_shared_subspace_blend(False, 0.75)
    assert resolved == 0.0


def test_shared_subspace_blend_boundary_values() -> None:
    """Test edge values 0.0 and 1.0."""
    assert ModelMergeService._resolve_shared_subspace_blend(True, 0.0) == 0.0
    assert ModelMergeService._resolve_shared_subspace_blend(True, 1.0) == 1.0


def test_shared_subspace_blend_default_constant_is_positive() -> None:
    """Default blend constant should be between 0 and 1."""
    assert 0.0 < DEFAULT_SHARED_SUBSPACE_BLEND <= 1.0


def test_shared_subspace_blend_midpoint_value() -> None:
    """Test a common midpoint value."""
    resolved = ModelMergeService._resolve_shared_subspace_blend(True, 0.5)
    assert resolved == 0.5
