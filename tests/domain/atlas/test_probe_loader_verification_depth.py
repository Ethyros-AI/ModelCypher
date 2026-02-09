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

import json

from modelcypher.core.domain.atlas.probe_loader import load_probes_from_file


def test_file_level_verification_depth_default_applies(tmp_path) -> None:
    probe_file = tmp_path / "probes.json"
    payload = {
        "domain": "logical",
        "verification_depth_default": 2,
        "probe_count": 1,
        "probes": [
            {
                "id": "semantic_prime:TEST",
                "name": "test",
                "description": "test probe",
                "support_texts": ["alpha", "beta"],
            }
        ],
    }
    probe_file.write_text(json.dumps(payload), encoding="utf-8")

    probes = list(load_probes_from_file(probe_file))
    assert len(probes) == 1
    assert probes[0].verification_depth == 2
    assert isinstance(probes[0].support_texts, tuple)
    assert probes[0].support_texts == ("alpha", "beta")


def test_probe_level_verification_depth_override_applies(tmp_path) -> None:
    probe_file = tmp_path / "probes.json"
    payload = {
        "domain": "logical",
        "verification_depth_default": 1,
        "probe_count": 2,
        "probes": [
            {
                "id": "semantic_prime:A",
                "name": "a",
                "description": "probe a",
                "support_texts": ["a"],
                "verification_depth": 4,
            },
            {
                "id": "semantic_prime:B",
                "name": "b",
                "description": "probe b",
                "support_texts": ["b"],
            },
        ],
    }
    probe_file.write_text(json.dumps(payload), encoding="utf-8")

    probes = list(load_probes_from_file(probe_file))
    assert len(probes) == 2
    assert probes[0].verification_depth == 4
    assert probes[1].verification_depth == 1


def test_absent_verification_depth_metadata_defaults_to_none(tmp_path) -> None:
    probe_file = tmp_path / "probes.json"
    payload = {
        "domain": "logical",
        "probe_count": 1,
        "probes": [
            {
                "id": "semantic_prime:C",
                "name": "c",
                "description": "probe c",
                "support_texts": ["c"],
            }
        ],
    }
    probe_file.write_text(json.dumps(payload), encoding="utf-8")

    probes = list(load_probes_from_file(probe_file))
    assert len(probes) == 1
    assert probes[0].verification_depth is None
    assert isinstance(probes[0].support_texts, tuple)
