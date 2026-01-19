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

"""Hypothesis property tests for geometry service payloads."""

from __future__ import annotations

from datetime import datetime, timezone

from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain.geometry.gate_detector import DetectionResult, DetectedGate
from modelcypher.core.use_cases.geometry_service import GeometryService


_finite_float = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False, width=32)


@settings(max_examples=10, deadline=None)
@given(
    model_id=st.text(min_size=1, max_size=12),
    prompt_id=st.text(min_size=1, max_size=12),
    response_text=st.text(min_size=0, max_size=64),
    gate_count=st.integers(min_value=0, max_value=5),
    timestamp=st.datetimes(timezones=st.just(timezone.utc)),
)
def test_detection_payload_identity(
    model_id: str,
    prompt_id: str,
    response_text: str,
    gate_count: int,
    timestamp: datetime,
) -> None:
    gate_ids = [f"gate_{idx}" for idx in range(gate_count)]
    gates: list[DetectedGate] = []
    for idx, gate_id in enumerate(gate_ids):
        gates.append(
            DetectedGate(
                gate_id=gate_id,
                gate_name=f"Gate {idx}",
                similarity=float(idx),
                character_span=(idx, idx + 1),
                trigger_text=f"t{idx}",
                local_entropy=float(idx) if idx % 2 == 0 else None,
            )
        )

    result = DetectionResult(
        model_id=model_id,
        prompt_id=prompt_id,
        response_text=response_text,
        detected_gates=gates,
        timestamp=timestamp,
    )

    payload = GeometryService.detection_payload(result)
    assert payload["modelID"] == model_id
    assert payload["promptID"] == prompt_id
    assert payload["responseText"] == response_text
    assert payload["meanSimilarity"] == result.mean_similarity
    assert payload["timestamp"] == GeometryService._iso_timestamp(result.timestamp)
    assert len(payload["detectedGates"]) == gate_count
    for gate_payload, gate in zip(payload["detectedGates"], gates):
        assert gate_payload["gateID"] == gate.gate_id
        assert gate_payload["gateName"] == gate.gate_name
        assert gate_payload["similarity"] == gate.similarity
        assert gate_payload["characterSpan"]["lowerBound"] == gate.character_span[0]
        assert gate_payload["characterSpan"]["upperBound"] == gate.character_span[1]
        assert gate_payload["triggerText"] == gate.trigger_text
        assert gate_payload["localEntropy"] == gate.local_entropy
