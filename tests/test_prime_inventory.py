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

from modelcypher.core.domain.agents.computational_gate_atlas import (
    ComputationalGateInventory,
    ComputationalGateSignature,
)
from modelcypher.core.domain.agents.semantic_prime_atlas import SemanticPrimeAtlas
from modelcypher.core.domain.agents.semantic_prime_frames import SemanticPrimeFrames
from modelcypher.core.domain.agents.semantic_prime_multilingual import (
    SemanticPrimeMultilingualInventoryLoader,
)
from modelcypher.core.domain.agents.semantic_primes import (
    SemanticPrimeInventory,
    SemanticPrimeSignature,
)


def test_semantic_prime_inventory_count() -> None:
    primes = SemanticPrimeInventory.english2014()
    assert len(primes) == 65
    assert primes[0].id == "I"


def test_semantic_prime_frames_count() -> None:
    enriched = SemanticPrimeFrames.enriched()
    assert len(enriched) == 65
    assert enriched[0].id == "I"


def test_multilingual_inventory_ordered_texts() -> None:
    inventory = SemanticPrimeMultilingualInventoryLoader.global_diverse()
    prime_ids = [prime.id for prime in SemanticPrimeInventory.english2014()]
    ordered = inventory.ordered_texts(prime_ids_in_order=prime_ids, languages=None)
    assert len(ordered) == len(prime_ids)
    assert ordered[0][0] == "I"


def test_computational_gate_counts() -> None:
    core = ComputationalGateInventory.core_gates()
    composite = ComputationalGateInventory.composite_gates()
    all_gates = ComputationalGateInventory.all_gates()
    probe = ComputationalGateInventory.probe_gates()
    assert len(core) == 66
    assert len(composite) == 10
    assert len(all_gates) == 76
    assert len(probe) == 61


def test_signature_cosine_similarity() -> None:
    sig_a = SemanticPrimeSignature(prime_ids=["A", "B"], values=[1.0, 0.0])
    sig_b = SemanticPrimeSignature(prime_ids=["A", "B"], values=[1.0, 0.0])
    assert sig_a.cosine_similarity(sig_b) == 1.0

    gate_a = ComputationalGateSignature(gate_ids=["1", "2"], values=[1.0, 0.0])
    gate_b = ComputationalGateSignature(gate_ids=["1", "2"], values=[0.0, 1.0])
    assert gate_a.cosine_similarity(gate_b) == 0.0


def test_normalized_entropy_uniform() -> None:
    atlas = SemanticPrimeAtlas()
    entropy = atlas._normalized_entropy([1.0, 1.0, 1.0])  # pylint: disable=protected-access
    assert entropy is not None
    assert abs(entropy - 1.0) < 1e-6
