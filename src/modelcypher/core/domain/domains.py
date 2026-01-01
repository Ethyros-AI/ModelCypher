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

"""Unified domain definitions - SINGLE SOURCE OF TRUTH.

All domain enums and domain resolution logic live here.
No other file should define domain enums.

AtlasDomain is THE canonical domain enum. All other code should import from here.

The domain taxonomy in data/domain_taxonomy.yaml provides aliases that map
to AtlasDomain values. Use resolve_domain() to convert user input to AtlasDomain.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import yaml


class AtlasDomain(str, Enum):
    """
    Unified domain enum for all ModelCypher operations.

    This is THE canonical domain enum. All probes, waypoints, merge operations,
    and geometry analysis should use these values.

    Domains map to probe categories in the UnifiedAtlas (499 total probes).
    """

    # Mathematical/logical domains
    MATHEMATICAL = "mathematical"  # Sequences, ratios, patterns (70 probes)
    LOGICAL = "logical"  # Logic, conditionals, causality (40 probes)

    # Language/semantic domains
    LINGUISTIC = "linguistic"  # Semantic primes, speech acts (89 probes)
    MENTAL = "mental"  # Mental predicates, cognitive

    # Computational domains
    COMPUTATIONAL = "computational"  # Code gates, algorithms (76 probes)
    STRUCTURAL = "structural"  # Data types, modularity

    # Affective domains
    AFFECTIVE = "affective"  # Emotions, valence (32 probes)
    RELATIONAL = "relational"  # Social, interpersonal (50 probes)

    # Temporal/spatial domains
    TEMPORAL = "temporal"  # Time concepts (25 probes)
    SPATIAL = "spatial"  # Place, location (23 probes)

    # Moral/ethical domains
    MORAL = "moral"  # Ethics, virtue, vice (30 probes)

    # Safety domains (AI safety critical concepts)
    SAFETY = "safety"  # Consent, autonomy, coercion, boundaries, vulnerability

    # Philosophical domains (fundamental categories of thought)
    PHILOSOPHICAL = "philosophical"  # Ontology, epistemology, logic, modality, mereology

    # Factual domain (uses LINGUISTIC + PHILOSOPHICAL probes)
    FACTUAL = "factual"  # Factual knowledge


# =============================================================================
# Domain Aliases
# =============================================================================

# Static alias map for common names that map to AtlasDomain values.
# This is the fallback if domain_taxonomy.yaml is not available.
_DOMAIN_ALIASES: dict[str, AtlasDomain] = {
    # Mathematical
    "math": AtlasDomain.MATHEMATICAL,
    "numeric": AtlasDomain.MATHEMATICAL,
    "quantitative": AtlasDomain.MATHEMATICAL,
    # Logical
    "logic": AtlasDomain.LOGICAL,
    "boolean": AtlasDomain.LOGICAL,
    "inference": AtlasDomain.LOGICAL,
    # Computational
    "coding": AtlasDomain.COMPUTATIONAL,
    "code": AtlasDomain.COMPUTATIONAL,
    "programming": AtlasDomain.COMPUTATIONAL,
    # Linguistic
    "language": AtlasDomain.LINGUISTIC,
    "semantics": AtlasDomain.LINGUISTIC,
    # Affective
    "emotional": AtlasDomain.AFFECTIVE,
    "sentiment": AtlasDomain.AFFECTIVE,
    # Relational (includes "social" alias for backward compatibility)
    "social": AtlasDomain.RELATIONAL,
    "interpersonal": AtlasDomain.RELATIONAL,
    # Temporal
    "time": AtlasDomain.TEMPORAL,
    # Spatial
    "space": AtlasDomain.SPATIAL,
    "location": AtlasDomain.SPATIAL,
    # Moral
    "ethical": AtlasDomain.MORAL,
    "ethics": AtlasDomain.MORAL,
    # Philosophical
    "philosophy": AtlasDomain.PHILOSOPHICAL,
    "ontological": AtlasDomain.PHILOSOPHICAL,
    # Factual
    "knowledge": AtlasDomain.FACTUAL,
    "facts": AtlasDomain.FACTUAL,
}


@lru_cache(maxsize=1)
def _load_taxonomy_aliases() -> dict[str, AtlasDomain]:
    """Load domain aliases from domain_taxonomy.yaml.

    Returns static aliases if YAML file is not available.
    """
    aliases = dict(_DOMAIN_ALIASES)

    # Try to load from YAML
    taxonomy_path = Path(__file__).parent.parent.parent / "data" / "domain_taxonomy.yaml"
    if taxonomy_path.exists():
        try:
            with open(taxonomy_path) as f:
                taxonomy = yaml.safe_load(f)

            # Extract core domain aliases
            for domain_name, domain_info in taxonomy.get("core_domains", {}).items():
                atlas_domain_str = domain_info.get("atlas_domain", "").upper()
                try:
                    atlas_domain = AtlasDomain(domain_name.lower())
                except ValueError:
                    # Try to find by name
                    atlas_domain = getattr(AtlasDomain, atlas_domain_str, None)
                    if atlas_domain is None:
                        continue

                # Add the domain name itself
                aliases[domain_name.lower()] = atlas_domain

                # Add all aliases
                for alias in domain_info.get("aliases", []):
                    aliases[alias.lower()] = atlas_domain

            # Extract composite domain aliases
            for domain_name, domain_info in taxonomy.get("composite_domains", {}).items():
                # Composite domains map to their first component
                components = domain_info.get("components", [])
                if components:
                    first_component = components[0].upper()
                    atlas_domain = getattr(AtlasDomain, first_component, None)
                    if atlas_domain:
                        aliases[domain_name.lower()] = atlas_domain
                        for alias in domain_info.get("aliases", []):
                            aliases[alias.lower()] = atlas_domain

        except Exception:
            # Fall back to static aliases on any error
            pass

    return aliases


def resolve_domain(name: str) -> AtlasDomain | None:
    """Resolve a domain name or alias to an AtlasDomain.

    Args:
        name: Domain name or alias (case-insensitive)

    Returns:
        AtlasDomain if found, None otherwise

    Examples:
        >>> resolve_domain("mathematical")
        AtlasDomain.MATHEMATICAL
        >>> resolve_domain("math")
        AtlasDomain.MATHEMATICAL
        >>> resolve_domain("social")
        AtlasDomain.RELATIONAL
        >>> resolve_domain("coding")
        AtlasDomain.COMPUTATIONAL
    """
    name_lower = name.strip().lower()

    # Try direct enum lookup first
    try:
        return AtlasDomain(name_lower)
    except ValueError:
        pass

    # Try alias lookup
    aliases = _load_taxonomy_aliases()
    return aliases.get(name_lower)


def resolve_domains(names: Sequence[str]) -> list[AtlasDomain]:
    """Resolve multiple domain names to AtlasDomain values.

    Unknown domains are logged as warnings and skipped.

    Args:
        names: Sequence of domain names or aliases

    Returns:
        List of resolved AtlasDomain values (may be empty)
    """
    import logging

    logger = logging.getLogger(__name__)
    result = []

    for name in names:
        domain = resolve_domain(name)
        if domain is not None:
            if domain not in result:
                result.append(domain)
        else:
            logger.warning("Unknown domain: %s, skipping", name)

    return result


def list_all_domains() -> list[AtlasDomain]:
    """Return all AtlasDomain values."""
    return list(AtlasDomain)


def list_domain_aliases() -> dict[str, AtlasDomain]:
    """Return all domain aliases."""
    return dict(_load_taxonomy_aliases())


__all__ = [
    "AtlasDomain",
    "resolve_domain",
    "resolve_domains",
    "list_all_domains",
    "list_domain_aliases",
]
