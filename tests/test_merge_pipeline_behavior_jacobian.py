# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import sys
from types import ModuleType
from types import SimpleNamespace

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.experimental.merge import pipeline
from modelcypher.experimental.merge.models import FingerprintComparison
from modelcypher.experimental.merge.stages.compression_descent import (
    CompressionDescentResult,
)


class _DummyModelLoader:
    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer
        self.load_model_calls: list[tuple[str, str | None]] = []

    def load_model(self, model_path: str, adapter_path: str | None = None):
        self.load_model_calls.append((model_path, adapter_path))
        return self._model, self._tokenizer


class _DummyLayerProfile:
    def __init__(self):
        self.gram_ranks: dict[int, int] = {}
        self.variance_concentrations: dict[int, float] = {}
        self.effective_ranks: dict[int, float] = {}
        self.bottleneck_layer: int | None = None
        self.layer_sparsity: dict[int, float] = {}
        self.sparse_layers: list[int] = []
        self.skip_layers: list[int] = []
        self.boundary_radii: dict[int, float] = {}

    def compute_bottleneck_layers(self) -> list[int]:
        return [0]

    def compute_ramp_layers(self) -> list[int]:
        return []

    def set_cross_architecture_skip_layers(self) -> None:
        return None

    def compute_best_injection_layer(self) -> int:
        return 0

    def compute_transmission_layers(self) -> list[int]:
        return [0]


class _DummyProfileStore:
    def load(self, _model_path: str):
        return SimpleNamespace(
            has_activations=True,
            probe_ids=["probe-0", "probe-1"],
            probe_domains=["facts", "math"],
        )


class _DummyProfileService:
    def __init__(self, **_kwargs):
        pass

    def compute_profile(self, *_args, **_kwargs):
        raise AssertionError("compute_profile should not run in this wiring test")


def test_run_merge_behavior_jacobian_loads_model_and_passes_context(monkeypatch):
    backend = get_default_backend()
    eye = backend.eye(2)
    acts = backend.array([[1.0, 0.0], [0.0, 1.0]])
    backend.eval(eye, acts)

    source_weights = {"model.layers.0.mlp.down_proj.weight": eye}
    target_weights = {"model.layers.0.mlp.down_proj.weight": eye}

    profile_alignment = SimpleNamespace(
        probe_result={
            "probe_ids": ["probe-0", "probe-1"],
            "probe_domains": ["facts", "math"],
            "confidences": {},
        },
        probe_metrics={
            "probe_failed": False,
            "perfect_alignment": True,
            "rank_augmentation": {"final_coverage": {}},
            "mean_cka": 1.0,
            "layer_count": 1,
        },
        source_activations={0: acts},
        target_activations={0: acts},
        source_intermediate_activations={},
        target_intermediate_activations={},
        feature_transforms={0: eye},
        scale_ratios={0: 1.0},
        embedding_transform=None,
        attention_transforms={},
        k_transforms={},
        v_transforms={},
        intermediate_transforms={},
        gate_transforms={},
        layer_mapping={0: 0},
        source_embedding_activations=None,
        target_embedding_activations=None,
        source_mean_pooled={},
        target_mean_pooled={},
        injection_layer=0,
    )

    loaded_model = object()
    loaded_tokenizer = object()
    model_loader = _DummyModelLoader(loaded_model, loaded_tokenizer)

    import modelcypher.core.domain.profile as profile_module

    monkeypatch.setattr(profile_module, "GeometricProfileStore", _DummyProfileStore)
    fake_profile_service = ModuleType("modelcypher.core.use_cases.profile_service")
    fake_profile_service.ProfileService = _DummyProfileService
    monkeypatch.setitem(
        sys.modules,
        "modelcypher.core.use_cases.profile_service",
        fake_profile_service,
    )
    monkeypatch.setattr(
        "modelcypher.experimental.merge.stages.probe_from_profile.check_profiles_available",
        lambda *_args, **_kwargs: (True, "/tmp/source-profile", "/tmp/target-profile"),
    )
    monkeypatch.setattr(
        "modelcypher.experimental.merge.stages.probe_from_profile.compute_alignment_from_profiles",
        lambda *_args, **_kwargs: profile_alignment,
    )
    monkeypatch.setattr(
        pipeline,
        "extract_layer_indices",
        lambda _weights: [0],
    )
    monkeypatch.setattr(
        pipeline,
        "create_layer_profile",
        lambda _layer_indices: _DummyLayerProfile(),
    )

    dummy_var = SimpleNamespace(var_top1=0.95, effective_rank=1.0)
    monkeypatch.setattr(
        "modelcypher.core.domain.geometry.variance_concentration.compute_variance_concentration",
        lambda _acts, _backend: dummy_var,
    )
    monkeypatch.setattr(
        "modelcypher.core.domain.geometry.variance_concentration.identify_bottleneck_layers",
        lambda _metrics, seed=42: ([0], {}),
    )
    monkeypatch.setattr(
        "modelcypher.core.domain.geometry.manifold_boundary.compute_boundary_radii_from_weights",
        lambda **_kwargs: {0: 1.0},
    )

    class _DummySparseRegionLocator:
        def analyze_from_activations(self, **_kwargs):
            return SimpleNamespace(layer_sparsity={0: 0.0}, sparse_layers=[], skip_layers=[])

    monkeypatch.setattr(pipeline, "SparseRegionLocator", _DummySparseRegionLocator)

    monkeypatch.setattr(
        "modelcypher.experimental.merge.metrics.compute_fingerprint_comparison",
        lambda **_kwargs: FingerprintComparison(
            source_gram_hash="a",
            target_gram_hash="b",
            source_condition_number=1.0,
            target_condition_number=1.0,
            source_effective_dim=1.0,
            target_effective_dim=1.0,
            condition_number_ratio=1.0,
            effective_dim_delta=0.0,
        ),
    )
    monkeypatch.setattr(
        "modelcypher.experimental.merge.metrics.compute_geometric_metrics_from_transplant",
        lambda transplant_metrics: {
            "mean_preserved_fraction": transplant_metrics.get("mean_preserved_fraction", 0.0),
        },
    )
    monkeypatch.setattr(
        pipeline,
        "_serialize_density_detail",
        lambda *_args, **_kwargs: {},
    )

    monkeypatch.setattr(
        pipeline,
        "stage_density",
        lambda **_kwargs: SimpleNamespace(
            graft_mask={"probe-0": {0: True}, "probe-1": {0: False}},
            density_weights={0: backend.array([1.0, 0.5])},
            metrics={
                "overall_source_density": 1.1,
                "overall_target_density": 1.0,
                "overall_opportunity": 0.1,
                "concepts_analyzed": 2,
                "positive_opportunity_count": 1,
                "nonpositive_opportunity_count": 1,
            },
        ),
    )

    captured: dict[str, object] = {}

    def _fake_stage_transplant(**kwargs):
        captured["behavior_jacobian_ctx"] = kwargs.get("behavior_jacobian_ctx")
        return kwargs["target_weights"], {
            "projection_losses": [0.0],
            "mean_preserved_fraction": 1.0,
            "layers_transplanted": 1,
            "weights_transplanted": 1,
            "mlp_reverted_keys": [],
        }

    monkeypatch.setattr(pipeline, "stage_transplant", _fake_stage_transplant)
    monkeypatch.setattr(
        pipeline,
        "stage_compression_descent",
        lambda **_kwargs: CompressionDescentResult(),
    )
    monkeypatch.setattr(
        pipeline,
        "apply_compression_descent_to_weights",
        lambda merged_weights, _result: merged_weights,
    )

    from modelcypher.core.domain.atlas.unified_atlas import UnifiedAtlasInventory

    probes = [
        SimpleNamespace(name="probe-name-fallback", support_texts=[]),
        SimpleNamespace(name="probe-name-2", support_texts=["probe support text"]),
    ]
    monkeypatch.setattr(
        UnifiedAtlasInventory,
        "all_probes",
        classmethod(lambda cls: probes),
    )

    result = pipeline.run_merge(
        model_loader=model_loader,
        backend=backend,
        source_path="/tmp/source-model",
        target_path="/tmp/target-model",
        source_weights=source_weights,
        target_weights=target_weights,
        source_model=object(),
        target_model=None,  # Critical: must trigger load_model in behavior_jacobian mode
        source_tokenizer=object(),
        target_tokenizer=object(),
        activation_provider=object(),
        behavior_jacobian=True,
    )

    assert model_loader.load_model_calls == [("/tmp/target-model", None)]

    ctx = captured.get("behavior_jacobian_ctx")
    assert ctx is not None, "Expected behavior_jacobian_ctx to be passed to stage_transplant"
    assert ctx.model is loaded_model
    assert ctx.tokenizer is loaded_tokenizer
    assert ctx.probe_texts == ["probe-name-fallback", "probe support text"]

    assert result.merge_strategy == "transplant"
    assert result.transplant_metrics["layers_transplanted"] == 1
