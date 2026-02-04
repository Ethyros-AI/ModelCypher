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

"""Checkpoint utilities for transplant stage."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _save_checkpoint(
    checkpoint_dir: Path,
    layer_idx: int,
    metrics: dict[str, Any],
) -> None:
    """Save transplant progress checkpoint for resume capability."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    meta_path = checkpoint_dir / "transplant_checkpoint.json"
    meta = {
        "last_completed_layer": layer_idx,
        "timestamp": time.time(),
        "weights_transplanted": metrics.get("weights_transplanted", 0),
        "layers_transplanted": metrics.get("layers_transplanted", 0),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("CHECKPOINT: Saved progress at layer %d to %s", layer_idx, checkpoint_dir)


def _load_checkpoint(checkpoint_dir: Path) -> tuple[int, dict[str, Any]] | None:
    """Load transplant checkpoint if available."""
    meta_path = checkpoint_dir / "transplant_checkpoint.json"
    if not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text())
        last_layer = meta.get("last_completed_layer", -1)
        logger.info(
            "CHECKPOINT: Resuming from layer %d (weights=%d, layers=%d)",
            last_layer,
            meta.get("weights_transplanted", 0),
            meta.get("layers_transplanted", 0),
        )
        return last_layer, meta
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("CHECKPOINT: Failed to load checkpoint: %s", e)
        return None
