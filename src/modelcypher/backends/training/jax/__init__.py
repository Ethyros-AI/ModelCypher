# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""JAX-specific training adapters.

This package contains all JAX (TPU/GPU) specific training implementations:
- checkpoints: JAX checkpoint save/load
- engine: JAX training engine
- evaluation: JAX model evaluation
- lora: JAX LoRA implementation
- loss_landscape: JAX loss landscape analysis
"""

from __future__ import annotations

__all__: list[str] = []
