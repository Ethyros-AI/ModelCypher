# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""CUDA/PyTorch-specific training adapters.

This package contains all PyTorch (CUDA) specific training implementations:
- checkpoints: PyTorch checkpoint save/load
- engine: PyTorch training engine
- evaluation: PyTorch model evaluation
- lora: PyTorch LoRA implementation
- loss_landscape: PyTorch loss landscape analysis
"""

from __future__ import annotations

__all__: list[str] = []
