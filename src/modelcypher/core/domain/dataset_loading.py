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

"""Dataset loading for LoRA training.

Reads JSONL format where each line contains at least ``{"text": "..."}``.
Tokenization happens in adapter code; this module is pure Python.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_jsonl_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL dataset from disk.

    Args:
        path: Path to a JSONL file.

    Returns:
        List of parsed sample dictionaries containing at least a ``text`` key.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If no valid ``{"text": ...}`` samples are found.
    """
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    samples: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict) and "text" in payload:
                samples.append(payload)

    if not samples:
        raise ValueError(f"No valid samples found in {dataset_path}")

    logger.info("Loaded %d samples from %s", len(samples), dataset_path)
    return samples


__all__ = ["load_jsonl_dataset"]
