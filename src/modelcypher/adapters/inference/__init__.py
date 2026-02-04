# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Inference adapters for different compute backends.

This package contains framework-specific inference implementations:
- mlx: MLX (Apple Metal) inference adapters
- jax: JAX inference adapters (planned)
- cuda: PyTorch/CUDA inference adapters (planned)
"""

from __future__ import annotations

__all__: list[str] = []
