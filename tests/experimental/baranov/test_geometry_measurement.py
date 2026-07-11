"""Unit tests for geometry_measurement module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from modelcypher.experimental.baranov.geometry_measurement import (
    CKADriftResult,
    GeometrySnapshot,
    collect_probe_activations,
    compute_cka_drift,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeArray:
    """Minimal fake array for testing backend operations."""

    def __init__(self, values: list[float], shape: tuple[int, ...] | None = None):
        self.values = values
        self.shape = shape or (len(values),)


def _make_mock_backend():
    """Create a mock backend that returns predictable arrays."""
    backend = MagicMock()

    # collect_hidden_activations returns {layer_idx: array}
    def mock_collect(model, tokenizer, texts, layer_indices=None):
        # Return 2 layers, each with shape [1, seq_len, hidden_dim]
        return {
            0: FakeArray([1.0, 2.0, 3.0], shape=(1, 3, 3)),
            1: FakeArray([4.0, 5.0, 6.0], shape=(1, 3, 3)),
        }

    backend.collect_hidden_activations = mock_collect
    backend.mean.side_effect = lambda arr, axis: FakeArray([1.0, 2.0, 3.0], shape=(1, 3))
    backend.reshape.side_effect = lambda arr, shape: FakeArray([1.0, 2.0, 3.0], shape=(3,))
    backend.eval.return_value = None
    backend.stack.side_effect = lambda arrs: FakeArray([1.0, 2.0, 3.0], shape=(len(arrs), 3))
    return backend


# ---------------------------------------------------------------------------
# GeometrySnapshot
# ---------------------------------------------------------------------------


class TestGeometrySnapshot:
    def test_frozen(self):
        snap = GeometrySnapshot(
            activations={0: [FakeArray([1.0])]},
            probe_texts=("hello",),
            n_layers=1,
        )
        with pytest.raises(AttributeError):
            snap.n_layers = 5  # type: ignore[misc]

    def test_stores_data(self):
        acts = {0: [FakeArray([1.0])], 1: [FakeArray([2.0])]}
        snap = GeometrySnapshot(
            activations=acts,
            probe_texts=("a", "b"),
            n_layers=2,
        )
        assert snap.n_layers == 2
        assert snap.probe_texts == ("a", "b")
        assert len(snap.activations) == 2


# ---------------------------------------------------------------------------
# CKADriftResult
# ---------------------------------------------------------------------------


class TestCKADriftResult:
    def test_as_dict(self):
        result = CKADriftResult(
            per_layer_cka={0: 0.95, 1: 0.90},
            min_cka=0.90,
            mean_cka=0.925,
            cka_drift=0.10,
            preserved_fraction=0.925,
        )
        d = result.as_dict()
        assert d["min_cka"] == 0.90
        assert d["mean_cka"] == 0.925
        assert d["cka_drift"] == 0.10
        assert d["preserved_fraction"] == 0.925
        assert d["per_layer_cka"] == {0: 0.95, 1: 0.90}

    def test_frozen(self):
        result = CKADriftResult(
            per_layer_cka={}, min_cka=1.0, mean_cka=1.0,
            cka_drift=0.0, preserved_fraction=1.0,
        )
        with pytest.raises(AttributeError):
            result.min_cka = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# collect_probe_activations
# ---------------------------------------------------------------------------


class TestCollectProbeActivations:
    def test_returns_geometry_snapshot(self):
        backend = _make_mock_backend()
        model = MagicMock()
        tokenizer = MagicMock()

        snap = collect_probe_activations(
            model, tokenizer, ["probe1", "probe2"], backend,
        )

        assert isinstance(snap, GeometrySnapshot)
        assert snap.probe_texts == ("probe1", "probe2")
        assert snap.n_layers == 2

    def test_passes_layer_indices(self):
        backend = _make_mock_backend()

        # Override to verify layer_indices are passed through
        calls = []
        original = backend.collect_hidden_activations

        def tracking_collect(model, tokenizer, texts, layer_indices=None):
            calls.append(layer_indices)
            return original(model, tokenizer, texts, layer_indices=layer_indices)

        backend.collect_hidden_activations = tracking_collect

        collect_probe_activations(
            MagicMock(), MagicMock(), ["p1"], backend, layer_indices=[0, 5, 10],
        )

        assert calls[0] == [0, 5, 10]

    def test_collects_one_text_at_a_time(self):
        """Each text is collected separately (not batched)."""
        backend = _make_mock_backend()
        texts_seen = []

        def tracking_collect(model, tokenizer, texts, layer_indices=None):
            texts_seen.append(texts)
            return {0: FakeArray([1.0], shape=(1, 1, 1))}

        backend.collect_hidden_activations = tracking_collect

        collect_probe_activations(
            MagicMock(), MagicMock(), ["a", "b", "c"], backend,
        )

        assert len(texts_seen) == 3
        assert all(len(t) == 1 for t in texts_seen)

    def test_mean_pools_over_sequence(self):
        """Backend.mean is called with axis=1 for sequence pooling."""
        backend = _make_mock_backend()
        axes_used = []

        def tracking_mean(arr, axis):
            axes_used.append(axis)
            return FakeArray([1.0], shape=(1, 3))

        backend.mean = tracking_mean

        collect_probe_activations(
            MagicMock(), MagicMock(), ["p1"], backend,
        )

        assert all(a == 1 for a in axes_used)


# ---------------------------------------------------------------------------
# compute_cka_drift
# ---------------------------------------------------------------------------


class TestComputeCkaDrift:
    def test_identical_snapshots_perfect_cka(self):
        """When pre == post activations, CKA should be 1.0 per layer."""
        backend = _make_mock_backend()

        # Mock compute_linear_cka_from_activations to return 1.0
        import modelcypher.experimental.baranov.geometry_measurement as gm

        # Simpler: just create snapshots and mock the CKA function
        acts = {0: [FakeArray([1.0])], 1: [FakeArray([2.0])]}
        pre = GeometrySnapshot(activations=acts, probe_texts=("p1",), n_layers=2)
        post = GeometrySnapshot(activations=acts, probe_texts=("p1",), n_layers=2)

        # Patch the CKA import
        from unittest.mock import patch
        with patch(
            "modelcypher.core.domain.geometry.cka.compute_linear_cka_from_activations",
            return_value=1.0,
        ):
            result = compute_cka_drift(pre, post, backend)

        assert isinstance(result, CKADriftResult)
        assert result.min_cka == 1.0
        assert result.mean_cka == 1.0
        assert result.cka_drift == 0.0
        assert result.preserved_fraction == 1.0

    def test_partial_drift(self):
        """CKA drift with mixed per-layer values."""
        backend = _make_mock_backend()

        acts_pre = {0: [FakeArray([1.0])], 1: [FakeArray([2.0])]}
        acts_post = {0: [FakeArray([3.0])], 1: [FakeArray([4.0])]}
        pre = GeometrySnapshot(activations=acts_pre, probe_texts=("p1",), n_layers=2)
        post = GeometrySnapshot(activations=acts_post, probe_texts=("p1",), n_layers=2)

        # Layer 0: CKA=0.9, Layer 1: CKA=0.7
        cka_values = iter([0.9, 0.7])

        from unittest.mock import patch
        with patch(
            "modelcypher.core.domain.geometry.cka.compute_linear_cka_from_activations",
            side_effect=lambda *a, **kw: next(cka_values),
        ):
            result = compute_cka_drift(pre, post, backend)

        assert result.per_layer_cka == {0: 0.9, 1: 0.7}
        assert result.min_cka == 0.7
        assert result.mean_cka == pytest.approx(0.8)
        assert result.cka_drift == pytest.approx(0.3)
        assert result.preserved_fraction == pytest.approx(0.8)

    def test_no_common_layers_raises(self):
        pre = GeometrySnapshot(activations={0: []}, probe_texts=(), n_layers=1)
        post = GeometrySnapshot(activations={5: []}, probe_texts=(), n_layers=1)

        with pytest.raises(ValueError, match="No common layers"):
            compute_cka_drift(pre, post, _make_mock_backend())

    def test_common_layer_subset(self):
        """Only common layers are compared."""
        backend = _make_mock_backend()

        pre = GeometrySnapshot(
            activations={0: [FakeArray([1.0])], 1: [FakeArray([2.0])], 2: [FakeArray([3.0])]},
            probe_texts=("p1",), n_layers=3,
        )
        post = GeometrySnapshot(
            activations={1: [FakeArray([4.0])], 2: [FakeArray([5.0])], 3: [FakeArray([6.0])]},
            probe_texts=("p1",), n_layers=3,
        )

        from unittest.mock import patch
        with patch(
            "modelcypher.core.domain.geometry.cka.compute_linear_cka_from_activations",
            return_value=0.95,
        ):
            result = compute_cka_drift(pre, post, backend)

        # Only layers 1 and 2 are common
        assert set(result.per_layer_cka.keys()) == {1, 2}
