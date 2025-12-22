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
