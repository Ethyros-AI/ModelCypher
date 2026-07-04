# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for TransplantManifest contracts.

Covers:
  M1 — record() increments the correct counters for each status category.
  M2 — WeightStatus predicates (is_success, is_expected_skip, is_failure) are correct.
  M3 — validate(strict=True) raises on failures and weight-count mismatches.
  M4 — validate(strict=True) passes on a clean manifest.
  M5 — validate(strict=False) never raises regardless of failures.
  M6 — get_mean_preserved_fraction() returns None with no data, mean otherwise.
  M7 — to_dict() serialises all fields.
  M8 — get_failure_summary() groups failures by status name.
"""

from __future__ import annotations

import pytest

from modelcypher.experimental.merge.exceptions import (
    PostconditionError,
    WeightCountMismatchError,
)
from modelcypher.experimental.merge.stages.manifest import (
    TransplantManifest,
    WeightStatus,
    WeightTransformRecord,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record(key: str, status: WeightStatus, **kwargs) -> WeightTransformRecord:
    return WeightTransformRecord(key=key, status=status, **kwargs)


def _transformed(key: str = "w0", **kwargs) -> WeightTransformRecord:
    return _record(key, WeightStatus.TRANSFORMED, **kwargs)


def _failed(key: str = "w0", msg: str = "nan") -> WeightTransformRecord:
    return _record(key, WeightStatus.FAILED_NUMERICAL, error_message=msg)


def _skipped(key: str = "w0") -> WeightTransformRecord:
    return _record(key, WeightStatus.SKIPPED_NON_2D)


# ---------------------------------------------------------------------------
# M1: record() counter tracking
# ---------------------------------------------------------------------------

class TestRecordTracking:
    def test_transformed_increments_weights_transformed(self):
        m = TransplantManifest()
        m.record("w0", _transformed("w0"))
        assert m.weights_transformed == 1
        assert m.weights_failed == 0
        assert m.weights_skipped_expected == 0
        assert m.total_weights_considered == 1

    def test_failure_increments_weights_failed(self):
        m = TransplantManifest()
        m.record("w0", _failed("w0"))
        assert m.weights_failed == 1
        assert m.weights_transformed == 0
        assert m.weights_skipped_expected == 0

    def test_expected_skip_increments_skipped(self):
        m = TransplantManifest()
        m.record("w0", _skipped("w0"))
        assert m.weights_skipped_expected == 1
        assert m.weights_transformed == 0
        assert m.weights_failed == 0

    def test_mixed_records_accumulate_independently(self):
        m = TransplantManifest()
        m.record("w0", _transformed("w0"))
        m.record("w1", _failed("w1"))
        m.record("w2", _skipped("w2"))
        assert m.weights_transformed == 1
        assert m.weights_failed == 1
        assert m.weights_skipped_expected == 1
        assert m.total_weights_considered == 3

    def test_identity_counts_as_transformed(self):
        m = TransplantManifest()
        m.record("w0", _record("w0", WeightStatus.IDENTITY))
        assert m.weights_transformed == 1
        assert m.weights_failed == 0


# ---------------------------------------------------------------------------
# M2: WeightStatus predicates
# ---------------------------------------------------------------------------

class TestStatusPredicates:
    def test_transformed_is_success(self):
        assert WeightStatus.TRANSFORMED.is_success() is True

    def test_identity_is_success(self):
        assert WeightStatus.IDENTITY.is_success() is True

    def test_failure_is_not_success(self):
        assert WeightStatus.FAILED_NUMERICAL.is_success() is False

    def test_skip_is_not_success(self):
        assert WeightStatus.SKIPPED_NON_2D.is_success() is False

    def test_skipped_non_2d_is_expected_skip(self):
        assert WeightStatus.SKIPPED_NON_2D.is_expected_skip() is True

    def test_skipped_missing_activations_is_expected_skip(self):
        assert WeightStatus.SKIPPED_MISSING_ACTIVATIONS.is_expected_skip() is True

    def test_failure_is_not_expected_skip(self):
        assert WeightStatus.FAILED_NUMERICAL.is_expected_skip() is False

    def test_transformed_is_not_failure(self):
        assert WeightStatus.TRANSFORMED.is_failure() is False

    def test_failed_numerical_is_failure(self):
        assert WeightStatus.FAILED_NUMERICAL.is_failure() is True

    def test_failed_alignment_is_failure(self):
        assert WeightStatus.FAILED_ALIGNMENT.is_failure() is True


# ---------------------------------------------------------------------------
# M3: validate(strict=True) raises on problems
# ---------------------------------------------------------------------------

class TestManifestValidateStrict:
    def test_raises_postcondition_on_failure(self):
        m = TransplantManifest()
        m.record("w0", _failed("w0"))
        with pytest.raises(PostconditionError):
            m.validate(target_weight_count=1, strict=True, min_preserved_fraction=0.0)

    def test_raises_weight_count_mismatch(self):
        """Two TRANSFORMED records but target_weight_count=3 → mismatch."""
        m = TransplantManifest()
        m.record("w0", _transformed("w0"))
        m.record("w1", _transformed("w1"))
        # target has 3 weights, 0 skipped → expect 3 transformed, got 2
        with pytest.raises(WeightCountMismatchError):
            m.validate(target_weight_count=3, strict=True, min_preserved_fraction=0.0)

    def test_raises_postcondition_on_low_preserved_fraction(self):
        """Mean preserved_fraction below explicit threshold raises."""
        m = TransplantManifest()
        m.record("w0", _transformed("w0", preserved_fraction=0.001))
        # target_weight_count=1 matches weights_transformed=1, check 2 passes
        # min_preserved_fraction=0.5 is above 0.001 → check 3 raises
        with pytest.raises(PostconditionError):
            m.validate(
                target_weight_count=1,
                strict=True,
                min_preserved_fraction=0.5,
            )


# ---------------------------------------------------------------------------
# M4: validate(strict=True) passes on clean manifest
# ---------------------------------------------------------------------------

class TestManifestValidateClean:
    def test_passes_all_transformed(self):
        m = TransplantManifest()
        m.record("w0", _transformed("w0"))
        m.record("w1", _transformed("w1"))
        # target_weight_count=2, skipped=0 → expected_transformed=2 == weights_transformed=2
        m.validate(target_weight_count=2, strict=True, min_preserved_fraction=0.0)

    def test_passes_with_expected_skips(self):
        """Skipped weights are correctly excluded from the count comparison."""
        m = TransplantManifest()
        m.record("w0", _transformed("w0"))
        m.record("w1", _skipped("w1"))
        # target_weight_count=2, skipped=1 → expected_transformed=1 == weights_transformed=1
        m.validate(target_weight_count=2, strict=True, min_preserved_fraction=0.0)

    def test_passes_with_sufficient_preserved_fraction(self):
        m = TransplantManifest()
        m.record("w0", _transformed("w0", preserved_fraction=0.9))
        m.validate(
            target_weight_count=1,
            strict=True,
            min_preserved_fraction=0.5,
        )


# ---------------------------------------------------------------------------
# M5: validate(strict=False) never raises
# ---------------------------------------------------------------------------

class TestManifestValidateNonStrict:
    def test_non_strict_ignores_failures(self):
        m = TransplantManifest()
        m.record("w0", _failed("w0"))
        # strict=False → no exception even with failures
        m.validate(target_weight_count=1, strict=False, min_preserved_fraction=0.0)

    def test_non_strict_ignores_count_mismatch(self):
        m = TransplantManifest()
        m.record("w0", _transformed("w0"))
        m.validate(target_weight_count=5, strict=False, min_preserved_fraction=0.0)


# ---------------------------------------------------------------------------
# M6: get_mean_preserved_fraction
# ---------------------------------------------------------------------------

class TestPreservedFraction:
    def test_returns_none_when_no_fractions(self):
        m = TransplantManifest()
        m.record("w0", _transformed("w0"))  # no preserved_fraction
        assert m.get_mean_preserved_fraction() is None

    def test_computes_mean_across_records(self):
        m = TransplantManifest()
        m.record("w0", _transformed("w0", preserved_fraction=0.9))
        m.record("w1", _transformed("w1", preserved_fraction=0.8))
        result = m.get_mean_preserved_fraction()
        assert result is not None
        assert abs(result - 0.85) < 1e-9

    def test_ignores_records_without_fraction(self):
        """Records with preserved_fraction=None don't affect the mean."""
        m = TransplantManifest()
        m.record("w0", _transformed("w0", preserved_fraction=0.6))
        m.record("w1", _transformed("w1"))  # no fraction
        assert m.get_mean_preserved_fraction() == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# M7: to_dict serialisation
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict_contains_summary(self):
        m = TransplantManifest()
        m.record("w0", _transformed("w0"))
        d = m.to_dict()
        assert "summary" in d
        assert d["summary"]["transformed"] == 1
        assert d["summary"]["failed"] == 0

    def test_to_dict_round_trips_record_fields(self):
        m = TransplantManifest()
        m.record(
            "model.layers.0.mlp.weight",
            WeightTransformRecord(
                key="model.layers.0.mlp.weight",
                status=WeightStatus.TRANSFORMED,
                source_shape=(960, 960),
                target_shape=(896, 896),
                stitch_type="hidden",
                preserved_fraction=0.73,
                cka_achieved=0.9999,
            ),
        )
        d = m.to_dict()
        wr = d["weights"]["model.layers.0.mlp.weight"]
        assert wr["status"] == "transformed"
        assert wr["stitch_type"] == "hidden"
        assert wr["preserved_fraction"] == pytest.approx(0.73)
        assert wr["cka_achieved"] == pytest.approx(0.9999)
        assert wr["source_shape"] == [960, 960]

    def test_to_dict_includes_failed_weights_list(self):
        m = TransplantManifest()
        m.record("w0", _failed("w0"))
        d = m.to_dict()
        assert "w0" in d["failed_weights"]


# ---------------------------------------------------------------------------
# M8: get_failure_summary groups by status name
# ---------------------------------------------------------------------------

class TestFailureSummary:
    def test_groups_by_status_type(self):
        m = TransplantManifest()
        m.record("w0", _record("w0", WeightStatus.FAILED_NUMERICAL))
        m.record("w1", _record("w1", WeightStatus.FAILED_NUMERICAL))
        m.record("w2", _record("w2", WeightStatus.FAILED_ALIGNMENT))
        summary = m.get_failure_summary()
        assert "failed_numerical" in summary
        assert "failed_alignment" in summary
        assert set(summary["failed_numerical"]) == {"w0", "w1"}
        assert summary["failed_alignment"] == ["w2"]

    def test_empty_summary_on_no_failures(self):
        m = TransplantManifest()
        m.record("w0", _transformed("w0"))
        assert m.get_failure_summary() == {}
