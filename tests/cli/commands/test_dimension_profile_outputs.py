"""Raw global/local intrinsic-dimension output contracts."""

from __future__ import annotations

from types import SimpleNamespace

from modelcypher.cli.commands.analyze.geometric import _dimension_layer_result


class _Backend:
    def eval(self, *_values) -> None:
        return None

    def tolist(self, value) -> list[float]:
        return list(value)


class _Service:
    def compute_intrinsic_dimension(self, _stacked, *, with_ci: bool):
        ci = SimpleNamespace(lower=1.5, upper=2.5, resamples=17) if with_ci else None
        return SimpleNamespace(
            intrinsic_dimension=2.0,
            sample_count=8,
            usable_count=7,
            ci=ci,
        )

    def compute_local_dimension_map(self, _stacked):
        return SimpleNamespace(
            dimensions=[1.0, 2.0, 3.0],
            mean_dimension=2.0,
            std_dimension=1.0,
            modal_dimension=2.0,
            deficient_indices=[0],
            k_neighbors=2,
        )

    def compute_mle_intrinsic_dimension(self, _stacked):
        return SimpleNamespace(
            intrinsic_dimension=2.25,
            sample_count=8,
            usable_count=8,
            k_neighbors=3,
        )


def test_dimension_layer_result_emits_raw_global_local_and_ci_values() -> None:
    result = _dimension_layer_result(
        service=_Service(),
        backend=_Backend(),
        stacked=[[1.0]],
        layer_idx=3,
        local=True,
        with_ci=True,
        with_mle=True,
    )

    assert result == {
        "layer": 3,
        "intrinsic_dimension": 2.0,
        "sample_count": 8,
        "usable_count": 7,
        "confidence_interval": {
            "lower": 1.5,
            "upper": 2.5,
            "resamples": 17,
        },
        "mle_dimension": {
            "intrinsic_dimension": 2.25,
            "sample_count": 8,
            "usable_count": 8,
            "k_neighbors": 3,
            "estimator": "levina_bickel_eq_8",
            "neighborhood_policy": "minimum_connected_geodesic_graph",
        },
        "local_dimension": {
            "mean": 2.0,
            "std": 1.0,
            "modal": 2.0,
            "k_neighbors": 2,
            "dimensions": [1.0, 2.0, 3.0],
            "deficient_indices": [0],
        },
    }
