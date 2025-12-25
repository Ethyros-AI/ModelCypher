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


from typing import Protocol, runtime_checkable

from modelcypher.core.domain.geometry.types import (
    ConceptComparisonResult,
    ConceptConfiguration,
    DetectionResult,
)


@runtime_checkable
class ConceptDiscoveryPort(Protocol):
    """
    Interface for semantic concept detection in generated text.
    """

    async def detect_concepts(
        self, response: str, model_id: str, prompt_id: str, config: ConceptConfiguration
    ) -> DetectionResult: ...

    async def compare_results(
        self, result_a: DetectionResult, result_b: DetectionResult
    ) -> ConceptComparisonResult: ...
