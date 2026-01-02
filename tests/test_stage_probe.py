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

from modelcypher.core.use_cases.merge import stages as merge_stages
from modelcypher.core.use_cases.merge.stages.probe import ProbeResult


def test_stage_probe_wrapper_maps_probe_result(monkeypatch) -> None:
    def fake_stage_probe_impl(**_kwargs) -> ProbeResult:
        return ProbeResult(
            correlations={"p0": 0.9},
            confidences={0: 0.8},
            intersection_map=None,
            dimension_correlations={"p0": 0.7},
            metrics={"mean_cka": 0.8},
            source_activations={0: ["s0"]},
            target_activations={0: ["t0"]},
            probe_ids=["p0"],
            probe_domains=["math"],
        )

    monkeypatch.setattr(merge_stages, "stage_probe_impl", fake_stage_probe_impl)

    result = merge_stages.stage_probe(
        source_weights={},
        target_weights={},
        source_model=None,
        target_model=None,
        source_tokenizer=None,
        target_tokenizer=None,
        alignment_map=None,
        extract_layer_index_fn=lambda _key: None,
    )

    probe_payload, metrics, source_acts, target_acts, *_ = result

    assert probe_payload["correlations"] == {"p0": 0.9}
    assert probe_payload["confidences"] == {0: 0.8}
    assert probe_payload["dimension_correlations"] == {"p0": 0.7}
    assert probe_payload["probe_ids"] == ["p0"]
    assert probe_payload["probe_domains"] == ["math"]
    assert metrics == {"mean_cka": 0.8}
    assert source_acts == {0: ["s0"]}
    assert target_acts == {0: ["t0"]}
