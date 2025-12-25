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


from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EmbedderPort(Protocol):
    """
    Interface for semantic text embedding.
    """

    async def embed(self, texts: list[str]) -> Any:
        """
        Embeds a list of texts into a matrix of shape [N, D].
        Returns MLX array or list of lists.
        """
        ...

    async def dimension(self) -> int:
        """
        Returns the embedding dimension.
        """
        ...
