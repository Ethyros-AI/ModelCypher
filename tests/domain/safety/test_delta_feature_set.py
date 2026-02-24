# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import pytest

from modelcypher.core.domain.safety.delta_feature_set import (
    DEFAULT_FEATURE_VERSION,
    DeltaFeatureSet,
)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestImports:
    def test_delta_feature_set_importable(self) -> None:
        assert DeltaFeatureSet is not None

    def test_default_feature_version_importable(self) -> None:
        assert DEFAULT_FEATURE_VERSION is not None

    def test_default_feature_version_value(self) -> None:
        assert DEFAULT_FEATURE_VERSION == "P0-lite"


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_default_instantiation(self) -> None:
        fs = DeltaFeatureSet()
        assert fs.geodesic_spreads == ()
        assert fs.sparsity == ()
        assert fs.cosine_to_aligned == ()
        assert fs.outlier_layer_indices == ()
        assert fs.feature_version == DEFAULT_FEATURE_VERSION

    def test_instantiation_with_data(self) -> None:
        fs = DeltaFeatureSet(
            geodesic_spreads=(0.1, 0.2, 0.3),
            sparsity=(0.5, 0.6),
            cosine_to_aligned=(0.9, 0.95),
            outlier_layer_indices=(1,),
            feature_version="P1",
        )
        assert fs.geodesic_spreads == (0.1, 0.2, 0.3)
        assert fs.sparsity == (0.5, 0.6)
        assert fs.cosine_to_aligned == (0.9, 0.95)
        assert fs.outlier_layer_indices == (1,)
        assert fs.feature_version == "P1"

    def test_frozen(self) -> None:
        fs = DeltaFeatureSet()
        with pytest.raises(AttributeError):
            fs.feature_version = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Computed Properties
# ---------------------------------------------------------------------------


class TestComputedProperties:
    def test_layer_count_with_data(self) -> None:
        fs = DeltaFeatureSet(geodesic_spreads=(0.1, 0.2, 0.3))
        assert fs.layer_count == 3

    def test_layer_count_empty(self) -> None:
        fs = DeltaFeatureSet()
        assert fs.layer_count == 0

    def test_has_outlier_layers_true(self) -> None:
        fs = DeltaFeatureSet(outlier_layer_indices=(0, 2))
        assert fs.has_outlier_layers is True

    def test_has_outlier_layers_false(self) -> None:
        fs = DeltaFeatureSet()
        assert fs.has_outlier_layers is False

    def test_outlier_layer_fraction_with_data(self) -> None:
        fs = DeltaFeatureSet(
            geodesic_spreads=(0.1, 0.2, 0.3, 0.4),
            outlier_layer_indices=(1, 3),
        )
        assert fs.outlier_layer_fraction == pytest.approx(0.5)

    def test_outlier_layer_fraction_zero_layers(self) -> None:
        fs = DeltaFeatureSet()
        assert fs.outlier_layer_fraction == 0.0

    def test_outlier_layer_fraction_no_outliers(self) -> None:
        fs = DeltaFeatureSet(geodesic_spreads=(0.1, 0.2))
        assert fs.outlier_layer_fraction == 0.0

    def test_mean_geodesic_spread_with_data(self) -> None:
        fs = DeltaFeatureSet(geodesic_spreads=(1.0, 2.0, 3.0))
        assert fs.mean_geodesic_spread == pytest.approx(2.0)

    def test_mean_geodesic_spread_empty(self) -> None:
        fs = DeltaFeatureSet()
        assert fs.mean_geodesic_spread == 0.0

    def test_max_geodesic_spread_with_data(self) -> None:
        fs = DeltaFeatureSet(geodesic_spreads=(1.0, 5.0, 3.0))
        assert fs.max_geodesic_spread == pytest.approx(5.0)

    def test_max_geodesic_spread_empty(self) -> None:
        fs = DeltaFeatureSet()
        assert fs.max_geodesic_spread == 0.0

    def test_mean_sparsity_with_data(self) -> None:
        fs = DeltaFeatureSet(sparsity=(0.2, 0.4, 0.6))
        assert fs.mean_sparsity == pytest.approx(0.4)

    def test_mean_sparsity_empty(self) -> None:
        fs = DeltaFeatureSet()
        assert fs.mean_sparsity == 0.0


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_dict(self) -> None:
        fs = DeltaFeatureSet(
            geodesic_spreads=(0.1, 0.2),
            sparsity=(0.5,),
            cosine_to_aligned=(0.9,),
            outlier_layer_indices=(0,),
            feature_version="P0-lite",
        )
        d = fs.to_dict()
        assert d["geodesic_spreads"] == [0.1, 0.2]
        assert d["sparsity"] == [0.5]
        assert d["cosine_to_aligned"] == [0.9]
        assert d["outlier_layer_indices"] == [0]
        assert d["feature_version"] == "P0-lite"

    def test_to_dict_empty(self) -> None:
        fs = DeltaFeatureSet()
        d = fs.to_dict()
        assert d["geodesic_spreads"] == []
        assert d["sparsity"] == []
        assert d["cosine_to_aligned"] == []
        assert d["outlier_layer_indices"] == []
        assert d["feature_version"] == DEFAULT_FEATURE_VERSION

    def test_from_dict(self) -> None:
        data = {
            "geodesic_spreads": [0.1, 0.2],
            "sparsity": [0.5],
            "cosine_to_aligned": [0.9],
            "outlier_layer_indices": [0],
            "feature_version": "P1",
        }
        fs = DeltaFeatureSet.from_dict(data)
        assert fs.geodesic_spreads == (0.1, 0.2)
        assert fs.sparsity == (0.5,)
        assert fs.cosine_to_aligned == (0.9,)
        assert fs.outlier_layer_indices == (0,)
        assert fs.feature_version == "P1"

    def test_from_dict_empty(self) -> None:
        fs = DeltaFeatureSet.from_dict({})
        assert fs.geodesic_spreads == ()
        assert fs.sparsity == ()
        assert fs.cosine_to_aligned == ()
        assert fs.outlier_layer_indices == ()
        assert fs.feature_version == DEFAULT_FEATURE_VERSION

    def test_round_trip(self) -> None:
        original = DeltaFeatureSet(
            geodesic_spreads=(0.1, 0.2, 0.3),
            sparsity=(0.5, 0.6),
            cosine_to_aligned=(0.9, 0.95),
            outlier_layer_indices=(1, 2),
            feature_version="P0-lite",
        )
        restored = DeltaFeatureSet.from_dict(original.to_dict())
        assert restored == original

    def test_from_dict_legacy_l2_norms_key(self) -> None:
        data = {
            "l2_norms": [1.0, 2.0, 3.0],
            "sparsity": [0.1],
        }
        fs = DeltaFeatureSet.from_dict(data)
        assert fs.geodesic_spreads == (1.0, 2.0, 3.0)

    def test_from_dict_geodesic_spreads_takes_precedence_over_l2_norms(self) -> None:
        data = {
            "geodesic_spreads": [4.0, 5.0],
            "l2_norms": [1.0, 2.0, 3.0],
        }
        fs = DeltaFeatureSet.from_dict(data)
        assert fs.geodesic_spreads == (4.0, 5.0)
