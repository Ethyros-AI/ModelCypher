"""Unit tests for OuterProductEditor.

Tests use a mock backend and mock model to verify the editor's control
flow, snapshot/rollback logic, and EditState transitions without
requiring a real model or GPU.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from modelcypher.experimental.baranov.edit_applicator import EditApplicator
from modelcypher.experimental.baranov.models import (
    EditState,
    EditStatus,
    FactTriple,
)
from modelcypher.experimental.baranov.outer_product_editor import (
    OuterProductEditor,
    _navigate_to_module,
)

# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------


class MockArray:
    """Minimal array mock that supports shape, dtype, arithmetic, and indexing."""

    def __init__(self, shape: tuple[int, ...], value: float = 0.0) -> None:
        self.shape = shape
        self.dtype = "float32"
        self._value = value

    def __add__(self, other: Any) -> MockArray:
        return MockArray(self.shape, self._value + getattr(other, "_value", 0.0))

    def __truediv__(self, other: Any) -> MockArray:
        return MockArray(self.shape, self._value)

    def __mul__(self, other: Any) -> MockArray:
        return MockArray(self.shape, self._value)

    def __getitem__(self, key: Any) -> MockArray:
        if isinstance(key, int):
            # Single element access -> reduce first dim
            if len(self.shape) > 1:
                return MockArray(self.shape[1:], self._value)
            return MockArray((), self._value)
        if isinstance(key, tuple):
            # Slicing: return a smaller array
            return MockArray(self.shape, self._value)
        return MockArray(self.shape, self._value)


class MockModule:
    """Mock model sub-module with a weight attribute."""

    def __init__(self, weight: MockArray) -> None:
        self.weight = weight


class MockModel:
    """Mock model with model.layers[i].mlp.down_proj structure."""

    def __init__(self, n_layers: int = 4, hidden_dim: int = 32) -> None:
        self.model = self._ModelInner(n_layers, hidden_dim)

    class _ModelInner:
        def __init__(self, n_layers: int, hidden_dim: int) -> None:
            self.embed_tokens = MockModule(MockArray((100, hidden_dim)))
            self.layers = [
                self._Layer(hidden_dim) for _ in range(n_layers)
            ]

        class _Layer:
            def __init__(self, hidden_dim: int) -> None:
                self.mlp = self._MLP(hidden_dim)

            class _MLP:
                def __init__(self, hidden_dim: int) -> None:
                    self.down_proj = MockModule(MockArray((hidden_dim, hidden_dim)))


def _make_mock_backend(hidden_dim: int = 32, n_facts: int = 2) -> MagicMock:
    """Create a mock backend with the methods OuterProductEditor uses."""
    backend = MagicMock()

    # collect_hidden_activations -> {layer_id: [n_facts, seq_len, hidden_dim]}
    def collect_hidden(model, tokenizer, prompts, layer_indices=None):
        layers = layer_indices or [0]
        return {
            lid: MockArray((len(prompts), 10, hidden_dim))
            for lid in layers
        }

    backend.collect_hidden_activations.side_effect = collect_hidden
    backend.zeros.side_effect = lambda shape, dtype=None: MockArray(shape)
    backend.stack.side_effect = lambda arrs: MockArray(
        (len(arrs),) + arrs[0].shape if arrs else (0,),
    )
    backend.mean.side_effect = lambda arr, axis=None: MockArray(
        arr.shape[1:] if axis == 0 else arr.shape,
    )
    backend.sum.side_effect = lambda arr, **kw: MockArray((), 1.0)
    backend.reshape.side_effect = lambda arr, shape: MockArray(
        tuple(
            d if d != -1 else arr.shape[0]
            for d in shape
        ),
    )
    backend.matmul.side_effect = lambda a, b: MockArray(
        (a.shape[0], b.shape[1]),
    )
    backend.norm.side_effect = lambda arr, **kw: MockArray((), 1.0)
    backend.to_scalar.side_effect = lambda arr: 1.0
    backend.eval.return_value = None

    return backend


def _make_facts(n: int = 2) -> list[FactTriple]:
    return [
        FactTriple(
            subject=f"subject_{i}",
            relation="is_a",
            object=f"object_{i}",
            fact_id=f"fact_{i}",
        )
        for i in range(n)
    ]


def _make_mock_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda text: [1, 2, 3]
    return tokenizer


# ---------------------------------------------------------------------------
# _navigate_to_module
# ---------------------------------------------------------------------------


class TestNavigateToModule:
    def test_simple_path(self) -> None:
        model = MockModel()
        module = _navigate_to_module(model, "model.layers.0.mlp.down_proj")
        assert hasattr(module, "weight")

    def test_numeric_index(self) -> None:
        model = MockModel(n_layers=3)
        module = _navigate_to_module(model, "model.layers.2.mlp.down_proj")
        assert hasattr(module, "weight")

    def test_bad_path_raises(self) -> None:
        model = MockModel()
        with pytest.raises(AttributeError):
            _navigate_to_module(model, "model.nonexistent.path")


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_satisfies_protocol(self) -> None:
        backend = _make_mock_backend()
        tokenizer = _make_mock_tokenizer()
        editor = OuterProductEditor(backend, tokenizer)
        assert isinstance(editor, EditApplicator)


# ---------------------------------------------------------------------------
# apply_edit
# ---------------------------------------------------------------------------


class TestApplyEdit:
    def test_returns_applied_state(self) -> None:
        backend = _make_mock_backend()
        tokenizer = _make_mock_tokenizer()
        editor = OuterProductEditor(backend, tokenizer)
        model = MockModel()
        facts = _make_facts(2)

        result = editor.apply_edit(facts, [0, 1], model)
        assert isinstance(result, EditState)
        assert result.status == EditStatus.applied
        assert result.fact_ids == ("fact_0", "fact_1")
        assert result.layer_ids == (0, 1)

    def test_edit_id_is_unique(self) -> None:
        backend = _make_mock_backend()
        tokenizer = _make_mock_tokenizer()
        editor = OuterProductEditor(backend, tokenizer)

        r1 = editor.apply_edit(_make_facts(), [0], MockModel())
        r2 = editor.apply_edit(_make_facts(), [0], MockModel())
        assert r1.edit_id != r2.edit_id

    def test_metrics_contain_relative_norm(self) -> None:
        backend = _make_mock_backend()
        tokenizer = _make_mock_tokenizer()
        editor = OuterProductEditor(backend, tokenizer)
        model = MockModel()

        result = editor.apply_edit(_make_facts(), [0, 2], model)
        metrics = result.metrics_dict
        assert "relative_edit_norm_layer_0" in metrics
        assert "relative_edit_norm_layer_2" in metrics

    def test_collects_activations_at_target_layers(self) -> None:
        backend = _make_mock_backend()
        tokenizer = _make_mock_tokenizer()
        editor = OuterProductEditor(backend, tokenizer)
        model = MockModel()

        editor.apply_edit(_make_facts(3), [1, 3], model)
        backend.collect_hidden_activations.assert_called_once()
        call_args = backend.collect_hidden_activations.call_args
        assert call_args.kwargs.get("layer_indices") == [1, 3] or call_args[1].get("layer_indices") == [1, 3]

    def test_failure_returns_failed_state(self) -> None:
        backend = _make_mock_backend()
        backend.collect_hidden_activations.side_effect = RuntimeError("GPU error")
        tokenizer = _make_mock_tokenizer()
        editor = OuterProductEditor(backend, tokenizer)
        model = MockModel()

        result = editor.apply_edit(_make_facts(), [0], model)
        assert result.status == EditStatus.failed


# ---------------------------------------------------------------------------
# rollback_edit
# ---------------------------------------------------------------------------


class TestRollbackEdit:
    def test_rollback_restores_state(self) -> None:
        backend = _make_mock_backend()
        tokenizer = _make_mock_tokenizer()
        editor = OuterProductEditor(backend, tokenizer)
        model = MockModel()

        # Capture original weight reference
        original_weight = model.model.layers[0].mlp.down_proj.weight

        # Apply edit (weight gets replaced)
        edit_state = editor.apply_edit(_make_facts(), [0], model)
        assert edit_state.status == EditStatus.applied

        # Rollback
        rolled_back = editor.rollback_edit(edit_state, model)
        assert rolled_back.status == EditStatus.rolled_back

        # Weight should be restored to original reference
        assert model.model.layers[0].mlp.down_proj.weight is original_weight

    def test_rollback_unknown_edit_raises(self) -> None:
        backend = _make_mock_backend()
        tokenizer = _make_mock_tokenizer()
        editor = OuterProductEditor(backend, tokenizer)

        fake_state = EditState.from_metrics_dict(
            edit_id="nonexistent",
            fact_ids=("f1",),
            layer_ids=(0,),
            status=EditStatus.applied,
            metrics_dict={},
        )
        with pytest.raises(ValueError, match="No snapshot found"):
            editor.rollback_edit(fake_state, MockModel())

    def test_double_rollback_raises(self) -> None:
        backend = _make_mock_backend()
        tokenizer = _make_mock_tokenizer()
        editor = OuterProductEditor(backend, tokenizer)
        model = MockModel()

        edit_state = editor.apply_edit(_make_facts(), [0], model)
        editor.rollback_edit(edit_state, model)

        # Second rollback should fail (snapshot consumed)
        with pytest.raises(ValueError, match="No snapshot found"):
            editor.rollback_edit(edit_state, model)

    def test_rollback_does_not_affect_other_edits(self) -> None:
        backend = _make_mock_backend()
        tokenizer = _make_mock_tokenizer()
        editor = OuterProductEditor(backend, tokenizer)

        model1 = MockModel()
        model2 = MockModel()

        edit1 = editor.apply_edit(_make_facts(), [0], model1)
        edit2 = editor.apply_edit(_make_facts(), [0], model2)

        # Rollback edit1 only
        editor.rollback_edit(edit1, model1)

        # edit2 snapshot should still exist
        rolled2 = editor.rollback_edit(edit2, model2)
        assert rolled2.status == EditStatus.rolled_back


# ---------------------------------------------------------------------------
# Custom projection
# ---------------------------------------------------------------------------


class TestCustomProjection:
    def test_custom_projection_name(self) -> None:
        """Editor respects the projection parameter."""
        backend = _make_mock_backend()
        tokenizer = _make_mock_tokenizer()
        editor = OuterProductEditor(backend, tokenizer, projection="down_proj")
        assert editor._get_weight_key(3) == "model.layers.3.mlp.down_proj"
