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

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from modelcypher.core.domain.agents.semantic_prime_atlas import SemanticPrimeAtlas
from modelcypher.core.domain.agents.semantic_primes import SemanticPrimeSignature


class DriftMethod(str, Enum):
    prime_signature = "primeSignature"
    skipped = "skipped"


@dataclass(frozen=True)
class SemanticPrimeDriftAssessment:
    """Assessment of semantic drift via prime signatures.

    Raw measurements:
    - cosine_similarity: Similarity between baseline and observed signatures
    - threshold: The configured threshold (for reference, not interpretation)

    Callers should interpret similarity relative to their own baselines.
    """

    method: DriftMethod
    cosine_similarity: float | None = None
    threshold: float | None = None
    note: str | None = None


@dataclass(frozen=True)
class SemanticPrimeDriftConfig:
    enabled: bool = True
    minimum_cosine_similarity: float = 0.65
    fail_closed: bool = False


class SemanticPrimeDriftDetector:
    def __init__(
        self,
        configuration: SemanticPrimeDriftConfig | None = None,
        atlas: SemanticPrimeAtlas | None = None,
    ) -> None:
        self._config = configuration or SemanticPrimeDriftConfig()
        self._atlas = atlas or SemanticPrimeAtlas()

    def assess(
        self, baseline: SemanticPrimeSignature, observed_text: str
    ) -> SemanticPrimeDriftAssessment:
        if not self._config.enabled:
            return SemanticPrimeDriftAssessment(
                method=DriftMethod.skipped, note="disabled"
            )

        observed = self._atlas.signature(observed_text)
        if observed is None:
            return SemanticPrimeDriftAssessment(
                method=DriftMethod.skipped,
                note="no_signature",
            )

        similarity = baseline.cosine_similarity(observed)
        if similarity is None:
            return SemanticPrimeDriftAssessment(
                method=DriftMethod.skipped,
                note="signature_computation_failed",
            )

        return SemanticPrimeDriftAssessment(
            method=DriftMethod.prime_signature,
            cosine_similarity=float(similarity),
            threshold=self._config.minimum_cosine_similarity,
        )
