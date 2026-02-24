# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for signature_base module.

Covers SignatureMixin and LabeledSignatureMixin with concrete test
dataclass implementations. Backend-dependent methods use the any_backend
fixture with monkeypatching.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

import modelcypher.core.domain.geometry.signature_base as sig_mod
from modelcypher.core.domain.geometry.signature_base import (
    LabeledSignatureMixin,
    SignatureMixin,
)

# ---------------------------------------------------------------------------
# Concrete test implementations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimpleSignature(SignatureMixin):
    """Minimal concrete signature for testing."""

    values: list[float]


@dataclass(frozen=True)
class LabeledSignature(LabeledSignatureMixin):
    """Concrete labeled signature for testing."""

    labels: list[str]
    values: list[float]


# ---------------------------------------------------------------------------
# SignatureMixin._has_same_dimensions
# ---------------------------------------------------------------------------


class TestHasSameDimensions:
    """Tests for _has_same_dimensions (pure Python, no backend needed)."""

    def test_same_length(self):
        a = SimpleSignature(values=[1.0, 2.0, 3.0])
        b = SimpleSignature(values=[4.0, 5.0, 6.0])
        assert a._has_same_dimensions(b) is True

    def test_different_length(self):
        a = SimpleSignature(values=[1.0, 2.0])
        b = SimpleSignature(values=[1.0, 2.0, 3.0])
        assert a._has_same_dimensions(b) is False

    def test_both_empty(self):
        a = SimpleSignature(values=[])
        b = SimpleSignature(values=[])
        assert a._has_same_dimensions(b) is True

    def test_one_empty(self):
        a = SimpleSignature(values=[])
        b = SimpleSignature(values=[1.0])
        assert a._has_same_dimensions(b) is False

    def test_no_values_attribute(self):
        a = SimpleSignature(values=[1.0])

        class NoValues:
            pass

        assert a._has_same_dimensions(NoValues()) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# SignatureMixin.l2_norm (requires backend)
# ---------------------------------------------------------------------------


class TestL2Norm:
    """Tests for l2_norm which uses backend for geodesic norm."""

    def test_known_vector(self, any_backend, monkeypatch):
        monkeypatch.setattr(sig_mod, "get_default_backend", lambda: any_backend)
        sig = SimpleSignature(values=[3.0, 4.0])
        norm = sig.l2_norm()
        assert norm is not None
        # Geodesic norm of [3,4] with 2-point k-NN equals Euclidean distance = 5.0
        assert norm == pytest.approx(5.0, rel=1e-3)

    def test_unit_vector(self, any_backend, monkeypatch):
        monkeypatch.setattr(sig_mod, "get_default_backend", lambda: any_backend)
        sig = SimpleSignature(values=[1.0, 0.0])
        norm = sig.l2_norm()
        assert norm is not None
        assert norm == pytest.approx(1.0, rel=1e-3)

    def test_non_negative(self, any_backend, monkeypatch):
        monkeypatch.setattr(sig_mod, "get_default_backend", lambda: any_backend)
        sig = SimpleSignature(values=[-3.0, -4.0])
        norm = sig.l2_norm()
        assert norm is not None
        assert norm >= 0.0

    def test_empty_vector_raises(self, any_backend, monkeypatch):
        monkeypatch.setattr(sig_mod, "get_default_backend", lambda: any_backend)
        sig = SimpleSignature(values=[])
        with pytest.raises(ValueError, match="empty"):
            sig.l2_norm()


# ---------------------------------------------------------------------------
# SignatureMixin.l2_normalized (requires backend)
# ---------------------------------------------------------------------------


class TestL2Normalized:
    """Tests for l2_normalized which returns a unit-norm signature."""

    def test_result_is_unit_norm(self, any_backend, monkeypatch):
        monkeypatch.setattr(sig_mod, "get_default_backend", lambda: any_backend)
        sig = SimpleSignature(values=[3.0, 4.0])
        normed = sig.l2_normalized()
        assert isinstance(normed, SimpleSignature)
        # Re-compute norm of normalized vector
        norm_of_normed = normed.l2_norm()
        assert norm_of_normed == pytest.approx(1.0, rel=1e-2)

    def test_preserves_type(self, any_backend, monkeypatch):
        monkeypatch.setattr(sig_mod, "get_default_backend", lambda: any_backend)
        sig = SimpleSignature(values=[1.0, 0.0])
        normed = sig.l2_normalized()
        assert type(normed) is SimpleSignature

    def test_empty_vector_raises(self, any_backend, monkeypatch):
        monkeypatch.setattr(sig_mod, "get_default_backend", lambda: any_backend)
        sig = SimpleSignature(values=[])
        with pytest.raises(ValueError, match="empty"):
            sig.l2_normalized()


# ---------------------------------------------------------------------------
# SignatureMixin.cosine_similarity (requires backend)
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    """Tests for cosine_similarity between two signatures."""

    def test_parallel_vectors(self, any_backend, monkeypatch):
        monkeypatch.setattr(sig_mod, "get_default_backend", lambda: any_backend)
        a = SimpleSignature(values=[1.0, 0.0])
        b = SimpleSignature(values=[2.0, 0.0])
        sim = a.cosine_similarity(b)
        assert sim == pytest.approx(1.0, abs=0.05)

    def test_antiparallel_vectors(self, any_backend, monkeypatch):
        monkeypatch.setattr(sig_mod, "get_default_backend", lambda: any_backend)
        a = SimpleSignature(values=[1.0, 0.0])
        b = SimpleSignature(values=[-1.0, 0.0])
        sim = a.cosine_similarity(b)
        assert sim == pytest.approx(-1.0, abs=0.05)

    def test_orthogonal_vectors(self, any_backend, monkeypatch):
        monkeypatch.setattr(sig_mod, "get_default_backend", lambda: any_backend)
        a = SimpleSignature(values=[1.0, 0.0])
        b = SimpleSignature(values=[0.0, 1.0])
        sim = a.cosine_similarity(b)
        assert sim == pytest.approx(0.0, abs=0.05)

    def test_self_similarity(self, any_backend, monkeypatch):
        monkeypatch.setattr(sig_mod, "get_default_backend", lambda: any_backend)
        a = SimpleSignature(values=[3.0, 4.0])
        sim = a.cosine_similarity(a)
        assert sim == pytest.approx(1.0, abs=0.05)

    def test_different_dimensions_truncated(self, any_backend, monkeypatch):
        """Different-length signatures should use truncated comparison."""
        monkeypatch.setattr(sig_mod, "get_default_backend", lambda: any_backend)
        a = SimpleSignature(values=[1.0, 0.0, 0.0])
        b = SimpleSignature(values=[1.0, 0.0])
        # Truncated to 2 dims: [1,0] vs [1,0] -> similarity = 1.0
        sim = a.cosine_similarity(b)
        assert sim == pytest.approx(1.0, abs=0.05)

    def test_bounded_range(self, any_backend, monkeypatch):
        monkeypatch.setattr(sig_mod, "get_default_backend", lambda: any_backend)
        a = SimpleSignature(values=[1.0, 2.0, 3.0])
        b = SimpleSignature(values=[4.0, -1.0, 2.0])
        sim = a.cosine_similarity(b)
        assert -1.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# LabeledSignatureMixin
# ---------------------------------------------------------------------------


class TestLabeledSignatureMixin:
    """Tests for LabeledSignatureMixin label-aware comparisons."""

    def test_get_labels(self):
        sig = LabeledSignature(labels=["a", "b", "c"], values=[1.0, 2.0, 3.0])
        assert sig._get_labels() == ["a", "b", "c"]

    def test_same_dimensions_same_labels(self):
        a = LabeledSignature(labels=["x", "y"], values=[1.0, 2.0])
        b = LabeledSignature(labels=["x", "y"], values=[3.0, 4.0])
        assert a._has_same_dimensions(b) is True

    def test_same_dimensions_different_labels(self):
        a = LabeledSignature(labels=["x", "y"], values=[1.0, 2.0])
        b = LabeledSignature(labels=["a", "b"], values=[3.0, 4.0])
        assert a._has_same_dimensions(b) is False

    def test_different_dimensions(self):
        a = LabeledSignature(labels=["x"], values=[1.0])
        b = LabeledSignature(labels=["x", "y"], values=[1.0, 2.0])
        assert a._has_same_dimensions(b) is False

    def test_against_unlabeled(self):
        """Comparing a labeled signature against an unlabeled one."""
        labeled = LabeledSignature(labels=["x", "y"], values=[1.0, 2.0])
        unlabeled = SimpleSignature(values=[3.0, 4.0])
        # unlabeled has no _get_labels, so labels check is skipped -> True
        assert labeled._has_same_dimensions(unlabeled) is True

    def test_cosine_similarity_with_matching_labels(self, any_backend, monkeypatch):
        monkeypatch.setattr(sig_mod, "get_default_backend", lambda: any_backend)
        a = LabeledSignature(labels=["x", "y"], values=[1.0, 0.0])
        b = LabeledSignature(labels=["x", "y"], values=[0.0, 1.0])
        sim = a.cosine_similarity(b)
        assert sim == pytest.approx(0.0, abs=0.05)

    def test_l2_norm(self, any_backend, monkeypatch):
        monkeypatch.setattr(sig_mod, "get_default_backend", lambda: any_backend)
        sig = LabeledSignature(labels=["a", "b"], values=[3.0, 4.0])
        norm = sig.l2_norm()
        assert norm is not None
        assert norm == pytest.approx(5.0, rel=1e-3)
