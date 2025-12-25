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

from unittest.mock import MagicMock

import pytest

from modelcypher.core.use_cases.dataset_service import DatasetService


@pytest.fixture
def mock_dataset_store():
    """Provide a mock DatasetStore for DatasetService tests."""
    return MagicMock()


def test_validate_dataset_missing_text_field_marks_invalid(
    tmp_path, monkeypatch, mock_dataset_store
):
    monkeypatch.setenv("MODELCYPHER_HOME", str(tmp_path / "home"))
    dataset_path = tmp_path / "chat.jsonl"
    dataset_path.write_text(
        '{"messages":[{"role":"system","content":"[Environment context.]"},'
        '{"role":"user","content":"hi"},'
        '{"role":"assistant","content":"ok"}]}\n',
        encoding="utf-8",
    )

    result = DatasetService(store=mock_dataset_store).validate_dataset(str(dataset_path))
    assert result["valid"] is False
    assert "Missing required field 'text'" in result["errors"]


def test_validate_dataset_token_estimate(tmp_path, monkeypatch, mock_dataset_store):
    monkeypatch.setenv("MODELCYPHER_HOME", str(tmp_path / "home"))
    dataset_path = tmp_path / "data.jsonl"
    line1 = '{"text":"abcd"}'
    line2 = '{"text":"abcdefgh"}'
    dataset_path.write_text(f"{line1}\n{line2}\n", encoding="utf-8")

    result = DatasetService(store=mock_dataset_store).validate_dataset(str(dataset_path))
    avg_length = (len(line1) + len(line2)) // 2

    def round_half_away_from_zero(value: float) -> int:
        return int(value + 0.5) if value >= 0 else int(value - 0.5)

    expected_avg_tokens = max(1, round_half_away_from_zero(avg_length / 4.0))
    expected_min_tokens = max(1, round_half_away_from_zero(min(len(line1), len(line2)) / 4.0))
    expected_max_tokens = max(1, round_half_away_from_zero(max(len(line1), len(line2)) / 4.0))

    assert result["averageTokens"] == float(expected_avg_tokens)
    assert result["minTokens"] == expected_min_tokens
    assert result["maxTokens"] == expected_max_tokens
    assert result["totalExamples"] == 2


def test_validate_dataset_reports_invalid_json(tmp_path, monkeypatch, mock_dataset_store):
    monkeypatch.setenv("MODELCYPHER_HOME", str(tmp_path / "home"))
    dataset_path = tmp_path / "bad.jsonl"
    dataset_path.write_text("{not_json}\n", encoding="utf-8")

    result = DatasetService(store=mock_dataset_store).validate_dataset(str(dataset_path))
    assert result["valid"] is False
    assert result["errors"]
    assert "Not valid JSON" in result["errors"][0]
