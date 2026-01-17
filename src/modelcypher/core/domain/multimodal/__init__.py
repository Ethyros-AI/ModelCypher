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

"""Multi-modal domain components.

Provides alignment and injection utilities for cross-modal workflows.
"""

from modelcypher.core.domain.multimodal.types import ModalityEmbeddings, ModalityType
from modelcypher.core.domain.multimodal.channel_adapter import (
    MultiModalChannelAdapter,
    MultiModalOfframpResult,
    OfframpProjection,
)
from modelcypher.core.domain.multimodal.attention_memory import (
    AttentionMemoryInjector,
    LayerType,
    LayerTypeConfig,
    MemoryTokenContent,
    get_architecture_config,
    register_architecture,
)

__all__ = [
    "ModalityType",
    "ModalityEmbeddings",
    "MultiModalChannelAdapter",
    "MultiModalOfframpResult",
    "OfframpProjection",
    # Attention memory injection
    "AttentionMemoryInjector",
    "LayerType",
    "LayerTypeConfig",
    "MemoryTokenContent",
    "get_architecture_config",
    "register_architecture",
]
