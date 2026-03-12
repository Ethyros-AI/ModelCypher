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

from modelcypher.core.use_cases.curriculum.benchmark_loader import BenchmarkLoader


def test_boolq_test_split_uses_validation_without_retry(monkeypatch, tmp_path):
    loader = BenchmarkLoader(cache_dir=tmp_path)
    requested_splits: list[str] = []

    def _fake_load(dataset_name: str, split: str, config: str | None = None):
        assert dataset_name == "google/boolq"
        assert config is None
        requested_splits.append(split)
        if split != "validation":
            return None
        return [
            {
                "passage": "The sky is blue.",
                "question": "Is the sky blue?",
                "answer": True,
            },
        ]

    monkeypatch.setattr(loader, "_try_load_huggingface", _fake_load)

    benchmark = loader.load("boolq", split="test", limit=1)

    assert requested_splits == ["validation"]
    assert len(benchmark.samples) == 1
    assert benchmark.samples[0].answer == "yes"


def test_boolq_train_split_keeps_requested_split(monkeypatch, tmp_path):
    loader = BenchmarkLoader(cache_dir=tmp_path)
    requested_splits: list[str] = []

    def _fake_load(dataset_name: str, split: str, config: str | None = None):
        assert dataset_name == "google/boolq"
        assert config is None
        requested_splits.append(split)
        return []

    monkeypatch.setattr(loader, "_try_load_huggingface", _fake_load)

    loader.load("boolq", split="train", limit=1)

    assert requested_splits == ["train"]
