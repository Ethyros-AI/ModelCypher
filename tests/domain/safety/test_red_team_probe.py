# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from pathlib import Path

import pytest

import modelcypher.core.domain.safety.red_team_probe as red_mod
from modelcypher.core.domain.safety.adapter_safety_models import (
    AdapterSafetyTier,
    AdapterSafetyTrigger,
)
from modelcypher.core.domain.safety.adapter_safety_probe import ProbeContext
from modelcypher.core.domain.safety.red_team_probe import (
    MetadataDistance,
    RedTeamProbe,
    RedTeamScanner,
)


class _EmbedderStub:
    pass


def _context(**kwargs) -> ProbeContext:
    base = dict(
        adapter_path=Path("/tmp/adapter"),
        tier=AdapterSafetyTier.QUICK,
        trigger=AdapterSafetyTrigger.MANUAL_RESCAN,
    )
    base.update(kwargs)
    return ProbeContext(**base)


def test_collect_metadata_items_collects_all_fields() -> None:
    context = _context(
        adapter_name="adapter",
        adapter_description="desc",
        skill_tags=("tag1", "tag2"),
        creator="creator",
        base_model_id="base/model",
        target_modules=("q_proj",),
        training_datasets=("dataset1",),
    )

    items = red_mod._collect_metadata_items(context)

    assert ("adapter_name", "adapter") in items
    assert ("adapter_description", "desc") in items
    assert ("skill_tag", "tag1") in items
    assert ("creator", "creator") in items
    assert ("base_model_id", "base/model") in items
    assert ("target_module", "q_proj") in items
    assert ("training_dataset", "dataset1") in items


async def test_probe_evaluate_branches(monkeypatch) -> None:
    probe = RedTeamProbe()

    missing = await probe.evaluate(_context(embedder=None))
    assert missing.finding_counts == {"metadata_items": 0, "missing_embedder": 1}

    insufficient = await probe.evaluate(_context(embedder=_EmbedderStub(), adapter_name="only-name"))
    assert insufficient.finding_counts == {"metadata_items": 1, "insufficient_metadata": 1}

    monkeypatch.setattr(
        red_mod,
        "_metadata_outliers",
        lambda _items, _embedder: (
            [MetadataDistance(field="adapter_name", text="x", mean_distance=1.2)],
            [MetadataDistance(field="adapter_name", text="x", mean_distance=1.2)],
            1.0,
            1.2,
            1.2,
        ),
    )

    result = await probe.evaluate(
        _context(
            embedder=_EmbedderStub(),
            adapter_name="name",
            adapter_description="desc",
        )
    )

    assert result.findings == ("adapter_name: mean_distance=1.2",)
    assert result.finding_counts is not None
    assert result.finding_counts["outlier_items"] == 1
    assert result.finding_counts["distance_threshold"] == pytest.approx(1.0)


def test_metadata_distance_and_outlier_helpers(monkeypatch, any_backend) -> None:
    b = any_backend
    monkeypatch.setattr(red_mod, "get_default_backend", lambda: b)

    items = [
        ("a", "alpha"),
        ("b", "beta"),
        ("c", "gamma"),
    ]

    monkeypatch.setattr(
        red_mod,
        "get_or_compute_embeddings_sync",
        lambda _embedder, _backend, _ns, _texts: b.array([]),
    )
    distances, threshold = red_mod._metadata_distances(items, _EmbedderStub())
    assert distances == []
    assert threshold == 0.0

    monkeypatch.setattr(
        red_mod,
        "get_or_compute_embeddings_sync",
        lambda _embedder, _backend, _ns, _texts: b.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        ),
    )

    class _RG:
        def __init__(self, _backend):
            pass

        def geodesic_distances(self, _points):
            return type(
                "_Geo",
                (),
                {"distances": b.array([[0.0, 2.0, 4.0], [2.0, 0.0, 2.0], [4.0, 2.0, 0.0]])},
            )()

    monkeypatch.setattr(red_mod, "RiemannianGeometry", _RG)
    monkeypatch.setattr(red_mod, "_distance_threshold", lambda values: 2.5)

    distances, threshold = red_mod._metadata_distances(items, _EmbedderStub())
    assert threshold == pytest.approx(2.5)
    assert [d.mean_distance for d in distances] == pytest.approx([3.0, 2.0, 3.0])

    all_dist, outliers, threshold, mean_distance, max_distance = red_mod._metadata_outliers(
        items,
        _EmbedderStub(),
    )
    assert len(all_dist) == 3
    assert [o.field for o in outliers] == ["a", "c"]
    assert threshold == pytest.approx(2.5)
    assert mean_distance == pytest.approx((3.0 + 2.0 + 3.0) / 3.0)
    assert max_distance == pytest.approx(3.0)


def test_red_team_scanner_paths(monkeypatch) -> None:
    scanner_none = RedTeamScanner(embedder=None)
    assert scanner_none.scan_adapter(name="adapter") == []

    scanner = RedTeamScanner(embedder=_EmbedderStub())

    assert scanner.scan_adapter(name="adapter") == []

    monkeypatch.setattr(
        red_mod,
        "_metadata_outliers",
        lambda _items, _embedder: (
            [MetadataDistance(field="adapter_name", text="adapter", mean_distance=1.0)],
            [MetadataDistance(field="adapter_name", text="adapter", mean_distance=1.0)],
            0.5,
            1.0,
            1.0,
        ),
    )

    indicators = scanner.scan_adapter(name="adapter", description="desc")
    assert len(indicators) == 1
    assert indicators[0].field == "adapter_name"
    assert indicators[0].mean_distance == pytest.approx(1.0)
