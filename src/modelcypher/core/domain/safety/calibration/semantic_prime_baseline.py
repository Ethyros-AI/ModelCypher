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

"""Persisted baseline signature for semantic prime drift monitoring.

Each adapter should have a "home region" in prime-space, derived from reference
task outputs, so drift can be detected without reading intent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID

from modelcypher.core.domain.geometry.vector_math import VectorMath


@dataclass(frozen=True)
class SemanticPrimeSignature:
    """Signature vector in semantic prime space.

    Represents an adapter's position in the space of semantic primes,
    used to detect drift from expected behavior.
    """

    prime_ids: tuple[str, ...]
    """Identifiers for each prime dimension."""

    values: tuple[float, ...]
    """Values for each prime dimension."""

    def __post_init__(self) -> None:
        """Validate that prime_ids and values have the same length."""
        if len(self.prime_ids) != len(self.values):
            raise ValueError(
                f"prime_ids length ({len(self.prime_ids)}) must match "
                f"values length ({len(self.values)})"
            )

    @property
    def dimension(self) -> int:
        """Number of dimensions in the signature."""
        return len(self.values)

    def cosine_similarity(self, other: SemanticPrimeSignature) -> float | None:
        """Compute cosine similarity with another signature.

        Args:
            other: Another signature to compare.

        Returns:
            Cosine similarity in [-1, 1], or None if incompatible.
        """
        if self.prime_ids != other.prime_ids:
            return None
        if len(self.values) != len(other.values):
            return None
        return VectorMath.cosine_similarity(list(self.values), list(other.values))

    def l2_normalized(self) -> SemanticPrimeSignature:
        """Return L2-normalized version of this signature."""
        normalized = VectorMath.l2_normalized(list(self.values))
        return SemanticPrimeSignature(
            prime_ids=self.prime_ids, values=tuple(normalized)
        )

    @classmethod
    def mean(cls, signatures: list[SemanticPrimeSignature]) -> SemanticPrimeSignature | None:
        """Compute mean signature from multiple signatures.

        Args:
            signatures: List of signatures to average.

        Returns:
            Mean signature (L2-normalized), or None if incompatible.
        """
        if not signatures:
            return None

        first = signatures[0]
        if not all(
            s.prime_ids == first.prime_ids and len(s.values) == len(first.values)
            for s in signatures
        ):
            return None

        dim = len(first.values)
        summed = [0.0] * dim

        for sig in signatures:
            for i, v in enumerate(sig.values):
                summed[i] += v

        inv_count = 1.0 / len(signatures)
        mean_values = tuple(v * inv_count for v in summed)

        return SemanticPrimeSignature(
            prime_ids=first.prime_ids, values=mean_values
        ).l2_normalized()

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "prime_ids": list(self.prime_ids),
            "values": list(self.values),
        }

    @classmethod
    def from_dict(cls, data: dict) -> SemanticPrimeSignature:
        """Create from dictionary."""
        return cls(
            prime_ids=tuple(data["prime_ids"]),
            values=tuple(data["values"]),
        )


@dataclass(frozen=True)
class SemanticPrimeBaseline:
    """Persisted baseline signature for semantic prime drift monitoring.

    Each adapter should have a "home region" in prime-space, derived from
    reference task outputs, so drift can be detected without reading intent.
    """

    adapter_id: UUID
    """Unique identifier for the adapter."""

    sample_count: int
    """Number of samples used to derive this baseline."""

    signature: SemanticPrimeSignature
    """The baseline signature in prime space."""

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When this baseline was created."""

    base_model_id: str | None = None
    """Optional base model identifier."""

    source: str | None = None
    """Source of the baseline data."""

    def similarity_to(self, other_signature: SemanticPrimeSignature) -> float | None:
        """Compute similarity to another signature.

        Args:
            other_signature: Signature to compare against.

        Returns:
            Cosine similarity, or None if incompatible.
        """
        return self.signature.cosine_similarity(other_signature)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "adapter_id": str(self.adapter_id),
            "sample_count": self.sample_count,
            "signature": self.signature.to_dict(),
            "created_at": self.created_at.isoformat(),
            "base_model_id": self.base_model_id,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SemanticPrimeBaseline:
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        return cls(
            adapter_id=UUID(data["adapter_id"]),
            sample_count=data["sample_count"],
            signature=SemanticPrimeSignature.from_dict(data["signature"]),
            created_at=created_at,
            base_model_id=data.get("base_model_id"),
            source=data.get("source"),
        )
