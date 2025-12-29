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

"""Tests for SyntaxAtlas."""

from __future__ import annotations

from modelcypher.core.domain.agents.syntax_atlas import (
    ALL_SYNTAX_PROBES,
    SyntaxCategory,
    SyntaxConceptInventory,
)


class TestSyntaxConceptInventory:
    """Tests for SyntaxConceptInventory."""

    def test_total_concept_count(self) -> None:
        """Should have 24 total syntax concepts."""
        all_concepts = SyntaxConceptInventory.all_concepts()
        assert len(all_concepts) == 24

    def test_count_method(self) -> None:
        """Count method should return 24."""
        assert SyntaxConceptInventory.count() == 24

    def test_all_concept_ids_unique(self) -> None:
        """All syntax concept IDs should be unique."""
        all_concepts = SyntaxConceptInventory.all_concepts()
        ids = [c.id for c in all_concepts]
        assert len(ids) == len(set(ids))

    def test_count_by_category(self) -> None:
        """Category counts should match the probe inventory."""
        counts = SyntaxConceptInventory.count_by_category()
        assert counts[SyntaxCategory.PART_OF_SPEECH] == 6
        assert counts[SyntaxCategory.MORPHOLOGY] == 5
        assert counts[SyntaxCategory.FUNCTION_WORD] == 4
        assert counts[SyntaxCategory.WORD_ORDER] == 3
        assert counts[SyntaxCategory.CLAUSE_STRUCTURE] == 3
        assert counts[SyntaxCategory.PUNCTUATION] == 2
        assert counts[SyntaxCategory.ORTHOGRAPHY] == 1


class TestSyntaxConceptProperties:
    """Tests for individual syntax concept properties."""

    def test_all_concepts_have_support_texts(self) -> None:
        """All concepts should have at least 2 support texts."""
        for concept in ALL_SYNTAX_PROBES:
            assert len(concept.support_texts) >= 2, (
                f"{concept.id} has only {len(concept.support_texts)} support texts"
            )

    def test_all_concepts_have_descriptions(self) -> None:
        """All concepts should have non-empty descriptions."""
        for concept in ALL_SYNTAX_PROBES:
            assert concept.description
