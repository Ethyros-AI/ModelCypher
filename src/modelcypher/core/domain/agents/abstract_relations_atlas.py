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

"""
Abstract Relations Atlas - Relational concepts for cross-dimensional alignment.

Covers abstract relational concepts:
- Causality (cause, effect, result)
- Similarity (like, same, different)
- Containment (inside, outside, contain)
- Ordering (before, after, between)
- Connection (connect, separate, join)
- Dependency (depend, require, support)

Total: ~50 probes for relational grounding.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RelationCategory(str, Enum):
    """Category of abstract relation."""
    
    CAUSALITY = "causality"
    SIMILARITY = "similarity"
    CONTAINMENT = "containment"
    ORDERING = "ordering"
    CONNECTION = "connection"
    DEPENDENCY = "dependency"


@dataclass(frozen=True)
class RelationConcept:
    """An abstract relational concept."""
    
    id: str
    name: str
    category: RelationCategory
    description: str
    support_texts: tuple[str, ...]
    cross_domain_weight: float = 1.0


class AbstractRelationInventory:
    """Inventory of abstract relational concepts."""
    
    # Causality relations
    CAUSALITY = (
        RelationConcept("cause", "cause", RelationCategory.CAUSALITY, "Origin of effect", ("cause", "source", "origin")),
        RelationConcept("effect", "effect", RelationCategory.CAUSALITY, "Result of cause", ("effect", "outcome", "impact")),
        RelationConcept("result", "result", RelationCategory.CAUSALITY, "Consequence", ("result", "consequence", "product")),
        RelationConcept("consequence", "consequence", RelationCategory.CAUSALITY, "Following outcome", ("consequence", "aftermath", "outcome")),
        RelationConcept("reason", "reason", RelationCategory.CAUSALITY, "Explanatory cause", ("reason", "rationale", "motive")),
        RelationConcept("because", "because", RelationCategory.CAUSALITY, "Causal connector", ("because", "since", "as")),
        RelationConcept("therefore", "therefore", RelationCategory.CAUSALITY, "Consequence marker", ("therefore", "thus", "hence")),
        RelationConcept("influence", "influence", RelationCategory.CAUSALITY, "Partial causation", ("influence", "affect", "impact")),
    )
    
    # Similarity relations
    SIMILARITY = (
        RelationConcept("like", "like", RelationCategory.SIMILARITY, "Similar to", ("like", "similar", "comparable")),
        RelationConcept("same", "same", RelationCategory.SIMILARITY, "Identical", ("same", "identical", "equal")),
        RelationConcept("different", "different", RelationCategory.SIMILARITY, "Not alike", ("different", "distinct", "unlike")),
        RelationConcept("similar", "similar", RelationCategory.SIMILARITY, "Alike", ("similar", "alike", "resembling")),
        RelationConcept("opposite", "opposite", RelationCategory.SIMILARITY, "Contrary", ("opposite", "contrary", "inverse")),
        RelationConcept("match", "match", RelationCategory.SIMILARITY, "Corresponding", ("match", "correspond", "fit")),
        RelationConcept("differ", "differ", RelationCategory.SIMILARITY, "Be different", ("differ", "vary", "diverge")),
        RelationConcept("compare", "compare", RelationCategory.SIMILARITY, "Assess similarity", ("compare", "contrast", "analyze")),
    )
    
    # Containment relations
    CONTAINMENT = (
        RelationConcept("inside", "inside", RelationCategory.CONTAINMENT, "Within bounds", ("inside", "within", "interior")),
        RelationConcept("outside", "outside", RelationCategory.CONTAINMENT, "Beyond bounds", ("outside", "exterior", "external")),
        RelationConcept("contain", "contain", RelationCategory.CONTAINMENT, "Hold within", ("contain", "hold", "include")),
        RelationConcept("include", "include", RelationCategory.CONTAINMENT, "Have as part", ("include", "comprise", "encompass")),
        RelationConcept("exclude", "exclude", RelationCategory.CONTAINMENT, "Leave out", ("exclude", "omit", "reject")),
        RelationConcept("within", "within", RelationCategory.CONTAINMENT, "Inside of", ("within", "inside", "in")),
        RelationConcept("surround", "surround", RelationCategory.CONTAINMENT, "Encircle", ("surround", "encircle", "enclose")),
        RelationConcept("boundary", "boundary", RelationCategory.CONTAINMENT, "Edge limit", ("boundary", "border", "limit")),
    )
    
    # Ordering relations
    ORDERING = (
        RelationConcept("before_rel", "before", RelationCategory.ORDERING, "Earlier position", ("before", "prior", "preceding")),
        RelationConcept("after_rel", "after", RelationCategory.ORDERING, "Later position", ("after", "following", "subsequent")),
        RelationConcept("between", "between", RelationCategory.ORDERING, "In the middle", ("between", "amid", "among")),
        RelationConcept("among", "among", RelationCategory.ORDERING, "In group", ("among", "amongst", "within")),
        RelationConcept("order", "order", RelationCategory.ORDERING, "Sequence", ("order", "sequence", "arrangement")),
        RelationConcept("rank", "rank", RelationCategory.ORDERING, "Position level", ("rank", "position", "level")),
        RelationConcept("sequence", "sequence", RelationCategory.ORDERING, "Ordered series", ("sequence", "series", "chain")),
        RelationConcept("priority", "priority", RelationCategory.ORDERING, "Importance order", ("priority", "precedence", "importance")),
    )
    
    # Connection relations
    CONNECTION = (
        RelationConcept("connect", "connect", RelationCategory.CONNECTION, "Join together", ("connect", "link", "join")),
        RelationConcept("separate", "separate", RelationCategory.CONNECTION, "Keep apart", ("separate", "divide", "split")),
        RelationConcept("join", "join", RelationCategory.CONNECTION, "Come together", ("join", "unite", "merge")),
        RelationConcept("link", "link", RelationCategory.CONNECTION, "Create connection", ("link", "connect", "tie")),
        RelationConcept("attach", "attach", RelationCategory.CONNECTION, "Fasten to", ("attach", "fasten", "fix")),
        RelationConcept("detach", "detach", RelationCategory.CONNECTION, "Unfasten from", ("detach", "disconnect", "remove")),
        RelationConcept("relate", "relate", RelationCategory.CONNECTION, "Have connection", ("relate", "associate", "connect")),
        RelationConcept("associate", "associate", RelationCategory.CONNECTION, "Link mentally", ("associate", "connect", "link")),
    )
    
    # Dependency relations
    DEPENDENCY = (
        RelationConcept("depend", "depend", RelationCategory.DEPENDENCY, "Rely on", ("depend", "rely", "hinge")),
        RelationConcept("independent", "independent", RelationCategory.DEPENDENCY, "Not reliant", ("independent", "autonomous", "self-sufficient")),
        RelationConcept("require", "require", RelationCategory.DEPENDENCY, "Need for function", ("require", "need", "demand")),
        RelationConcept("need", "need", RelationCategory.DEPENDENCY, "Necessity", ("need", "require", "must have")),
        RelationConcept("support", "support", RelationCategory.DEPENDENCY, "Hold up", ("support", "sustain", "uphold")),
        RelationConcept("enable", "enable", RelationCategory.DEPENDENCY, "Make possible", ("enable", "allow", "permit")),
        RelationConcept("prevent", "prevent", RelationCategory.DEPENDENCY, "Stop happening", ("prevent", "block", "stop")),
        RelationConcept("condition", "condition", RelationCategory.DEPENDENCY, "Requirement", ("condition", "requirement", "prerequisite")),
        RelationConcept("prerequisite", "prerequisite", RelationCategory.DEPENDENCY, "Prior requirement", ("prerequisite", "precondition", "requirement")),
        RelationConcept("optional", "optional", RelationCategory.DEPENDENCY, "Not required", ("optional", "elective", "discretionary")),
    )
    
    @classmethod
    def all_concepts(cls) -> list[RelationConcept]:
        """Get all relation concepts."""
        concepts: list[RelationConcept] = []
        concepts.extend(cls.CAUSALITY)
        concepts.extend(cls.SIMILARITY)
        concepts.extend(cls.CONTAINMENT)
        concepts.extend(cls.ORDERING)
        concepts.extend(cls.CONNECTION)
        concepts.extend(cls.DEPENDENCY)
        return concepts
    
    @classmethod
    def concepts_by_category(cls, category: RelationCategory) -> list[RelationConcept]:
        """Get concepts for a specific category."""
        return [c for c in cls.all_concepts() if c.category == category]
