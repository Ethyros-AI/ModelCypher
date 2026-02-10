# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from uuid import uuid4

import pytest

import modelcypher.core.domain.safety.calibration.semantic_prime_baseline as sp_mod
from modelcypher.core.domain.safety.calibration.semantic_prime_baseline import (
    BaselineSemanticPrimeSignature,
    SemanticPrimeBaseline,
)


def _patch_geometry(monkeypatch, any_backend) -> None:
    b = any_backend
    monkeypatch.setattr(sp_mod, "get_default_backend", lambda: b)
    monkeypatch.setattr(
        sp_mod,
        "SemanticPrimeSignature",
        BaselineSemanticPrimeSignature,
        raising=False,
    )

    def _geodesic_pairwise_metrics(lhs, rhs, backend):
        dot = backend.sum(lhs * rhs, axis=1)
        lhs_norm = backend.sqrt(backend.sum(lhs * lhs, axis=1))
        rhs_norm = backend.sqrt(backend.sum(rhs * rhs, axis=1))
        denom = lhs_norm * rhs_norm + backend.full(lhs_norm.shape, 1e-12)
        cos = dot / denom
        return cos, backend.zeros_like(cos)

    def _geodesic_norms(matrix, backend):
        return backend.sqrt(backend.sum(matrix * matrix, axis=1))

    monkeypatch.setattr(sp_mod, "geodesic_pairwise_metrics", _geodesic_pairwise_metrics)
    monkeypatch.setattr(sp_mod, "geodesic_norms", _geodesic_norms)


def test_signature_validation_similarity_normalization_and_mean(monkeypatch, any_backend) -> None:
    _patch_geometry(monkeypatch, any_backend)

    sig = BaselineSemanticPrimeSignature(prime_ids=("I", "YOU"), values=(3.0, 4.0))
    other = BaselineSemanticPrimeSignature(prime_ids=("I", "YOU"), values=(3.0, 4.0))
    mismatch = BaselineSemanticPrimeSignature(prime_ids=("YOU", "I"), values=(3.0, 4.0))

    assert sig.dimension == 2
    assert sig.cosine_similarity(other) == pytest.approx(1.0)
    assert sig.cosine_similarity(mismatch) is None

    normalized = sig.l2_normalized()
    assert normalized.values[0] == pytest.approx(0.6)
    assert normalized.values[1] == pytest.approx(0.8)

    mean = BaselineSemanticPrimeSignature.mean([sig, other])
    assert mean is not None
    assert mean.dimension == 2

    assert BaselineSemanticPrimeSignature.mean([sig, mismatch]) is None
    assert BaselineSemanticPrimeSignature.mean([]) is None


def test_signature_serialization_and_length_guard(monkeypatch, any_backend) -> None:
    _patch_geometry(monkeypatch, any_backend)

    with pytest.raises(ValueError, match="prime_ids length"):
        BaselineSemanticPrimeSignature(prime_ids=("I",), values=(1.0, 2.0))

    sig = BaselineSemanticPrimeSignature(prime_ids=("A", "B"), values=(1.0, 2.0))
    decoded = BaselineSemanticPrimeSignature.from_dict(sig.to_dict())
    assert decoded == sig


def test_baseline_similarity_and_roundtrip(monkeypatch, any_backend) -> None:
    _patch_geometry(monkeypatch, any_backend)

    sig = BaselineSemanticPrimeSignature(prime_ids=("A", "B"), values=(1.0, 0.0))
    baseline = SemanticPrimeBaseline(
        adapter_id=uuid4(),
        sample_count=3,
        signature=sig,
        base_model_id="base/model",
        source="test",
    )

    assert baseline.similarity_to(sig) == pytest.approx(1.0)

    encoded = baseline.to_dict()
    decoded = SemanticPrimeBaseline.from_dict(encoded)

    assert decoded.adapter_id == baseline.adapter_id
    assert decoded.sample_count == 3
    assert decoded.base_model_id == "base/model"
    assert decoded.source == "test"
    assert decoded.signature.prime_ids == ("A", "B")

    decoded_no_time = SemanticPrimeBaseline.from_dict(
        {
            "adapter_id": str(uuid4()),
            "sample_count": 1,
            "signature": sig.to_dict(),
        }
    )
    assert decoded_no_time.sample_count == 1
