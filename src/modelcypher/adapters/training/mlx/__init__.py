# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""MLX-specific training adapters.

This package contains all MLX (Apple Metal) specific training implementations:
- checkpoints: MLX checkpoint save/load
- engine: MLX training engine
- evaluation: MLX model evaluation
- lora: MLX LoRA implementation
- loss_landscape: MLX loss landscape analysis
"""

from __future__ import annotations

__all__: list[str] = []
