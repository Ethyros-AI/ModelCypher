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

from modelcypher.core.use_cases.merge import pipeline


def test_pipeline_forwards_graft_mask_to_transplant(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_load_weights(_loader, _path):
        weights = {"model.layers.0.mlp.down_proj.weight": object()}
        return weights, "safetensors"

    def fake_load_tokenizer(_path):
        return object()

    def fake_load_model_for_probing(_path):
        return object()

    def fake_stage_probe(**_kwargs):
        return (
            {
                "confidences": {0: 1.0},
                "intersection_map": None,
                "probe_ids": ["p0"],
                "probe_domains": ["math"],
                "dimension_correlations": {},
            },
            {"probe_failed": False, "perfect_alignment": True},
            {0: ["s0"]},  # source_activations
            {0: ["t0"]},  # target_activations
            None,  # source_intermediate_activations
            None,  # target_intermediate_activations
            None,  # source_attention_activations
            None,  # target_attention_activations
            None,  # source_k_activations
            None,  # target_k_activations
            None,  # feature_transforms
            None,  # scale_ratios
            None,  # embedding_transform
            None,  # attention_transforms
            None,  # k_transforms
            None,  # v_transforms
            None,  # intermediate_transforms
            None,  # layer_mapping
        )

    def fake_stage_density(**_kwargs):
        calls["density_called"] = True
        return {"p0": {0: True}}, {"positive_opportunity_count": 1}

    # Note: fake_stage_permute removed - stage_permute no longer exists in pipeline

    def fake_stage_transplant(*, graft_mask, **_kwargs):
        calls["graft_mask"] = graft_mask
        return {}, {"preserved_fractions": [], "cka_after": []}

    def fake_infer_hidden_dim(_weights):
        return 2

    monkeypatch.setattr(pipeline, "load_weights", fake_load_weights)
    monkeypatch.setattr(pipeline, "load_tokenizer", fake_load_tokenizer)
    monkeypatch.setattr(pipeline, "load_model_for_probing", fake_load_model_for_probing)
    monkeypatch.setattr(pipeline, "stage_probe", fake_stage_probe)
    monkeypatch.setattr(pipeline, "stage_density", fake_stage_density)
    # Note: stage_permute was removed from the pipeline - no longer needed
    monkeypatch.setattr(pipeline, "stage_transplant", fake_stage_transplant)
    monkeypatch.setattr(pipeline, "infer_hidden_dim", fake_infer_hidden_dim)

    pipeline.run_merge(
        model_loader=object(),
        backend=pipeline.get_default_backend(),
        source_path="/source",
        target_path="/target",
        dry_run=True,
    )

    assert calls.get("density_called") is True
    assert calls.get("graft_mask") == {"p0": {0: True}}