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

"""Training adapters for framework-specific training operations.

Each adapter implements the TrainingPort protocol for a specific ML framework:
- MLXTrainingAdapter: Apple MLX (macOS Metal)
- JAXTrainingAdapter: Google JAX (TPU/GPU)  [TODO]
- TorchTrainingAdapter: PyTorch (CUDA)  [TODO]

Domain code uses the Backend protocol for numeric operations.
These adapters handle framework-specific training infrastructure.
"""

from __future__ import annotations

__all__: list[str] = []
