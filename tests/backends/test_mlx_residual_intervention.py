"""Framework-free contract tests for the MLX residual intervention hook."""

from __future__ import annotations

import numpy as np
import pytest

from modelcypher.backends._mlx_backend_activation_mixin import (
    _MLXBackendActivationMixin,
)


class _FakeMX:
    array = staticmethod(np.asarray)
    concatenate = staticmethod(np.concatenate)
    reshape = staticmethod(np.reshape)

    @staticmethod
    def eval(*_values) -> None:
        return None


class _Embedding:
    def __call__(self, input_ids):
        return np.repeat(input_ids[..., None], 2, axis=-1).astype(float)


class _Layer:
    def __call__(self, hidden, *, mask, cache):
        assert mask is None
        assert cache is None
        return hidden + 1.0


class _Base:
    def __init__(self) -> None:
        self.embed_tokens = _Embedding()
        self.layers = [_Layer(), _Layer()]


class _Model:
    def __init__(self) -> None:
        self.model = _Base()

    @staticmethod
    def lm_head(hidden):
        return hidden


class _Tokenizer:
    @staticmethod
    def encode(_text: str) -> list[int]:
        return [1, 2, 3]


class _Harness(_MLXBackendActivationMixin):
    def __init__(self) -> None:
        self.mx = _FakeMX()


def test_residual_intervention_changes_only_declared_token_before_continuation() -> None:
    harness = _Harness()
    model = _Model()
    tokenizer = _Tokenizer()

    baseline = harness.collect_logits_with_residual_intervention(
        model,
        tokenizer,
        "prompt",
    )
    perturbed = harness.collect_logits_with_residual_intervention(
        model,
        tokenizer,
        "prompt",
        target_layer=0,
        token_position=1,
        delta=np.array([3.0, -2.0]),
    )

    np.testing.assert_allclose(perturbed[:, 0, :], baseline[:, 0, :])
    np.testing.assert_allclose(perturbed[:, 2, :], baseline[:, 2, :])
    np.testing.assert_allclose(
        perturbed[:, 1, :] - baseline[:, 1, :],
        np.array([[3.0, -2.0]]),
    )


def test_residual_intervention_rejects_partial_or_out_of_range_declarations() -> None:
    harness = _Harness()
    model = _Model()
    tokenizer = _Tokenizer()

    with pytest.raises(ValueError, match="requires"):
        harness.collect_logits_with_residual_intervention(
            model,
            tokenizer,
            "prompt",
            target_layer=0,
        )
    with pytest.raises(ValueError, match="token_position"):
        harness.collect_logits_with_residual_intervention(
            model,
            tokenizer,
            "prompt",
            target_layer=0,
            token_position=3,
            delta=np.ones(2),
        )
