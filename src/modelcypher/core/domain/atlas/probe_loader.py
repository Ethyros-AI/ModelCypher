# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Probe Loader - Load probes from structured JSON data files.

This module provides functions to load probe definitions from JSON files
in data/probes/, allowing easy expansion without modifying Python code.

Usage:
    from modelcypher.core.domain.atlas.probe_loader import load_all_probes
    probes = load_all_probes()  # Returns list of AtlasProbe objects
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from modelcypher.core.domain.domains import AtlasDomain
from .unified_atlas import AtlasProbe, AtlasSource

if TYPE_CHECKING:
    from typing import Iterator

logger = logging.getLogger(__name__)

# Default path to probe data files (relative to package root)
_PROBE_DATA_DIR = Path(__file__).parent.parent.parent.parent.parent.parent / "data" / "probes"


def _domain_from_string(domain_str: str) -> AtlasDomain:
    """Convert domain string to AtlasDomain enum."""
    domain_map = {
        "mathematical": AtlasDomain.MATHEMATICAL,
        "logical": AtlasDomain.LOGICAL,
        "linguistic": AtlasDomain.LINGUISTIC,
        "mental": AtlasDomain.MENTAL,
        "computational": AtlasDomain.COMPUTATIONAL,
        "structural": AtlasDomain.STRUCTURAL,
        "affective": AtlasDomain.AFFECTIVE,
        "relational": AtlasDomain.RELATIONAL,
        "temporal": AtlasDomain.TEMPORAL,
        "spatial": AtlasDomain.SPATIAL,
        "moral": AtlasDomain.MORAL,
        "safety": AtlasDomain.SAFETY,
        "philosophical": AtlasDomain.PHILOSOPHICAL,
        "physical": AtlasDomain.PHYSICAL,
        "factual": AtlasDomain.FACTUAL,
    }
    return domain_map.get(domain_str.lower(), AtlasDomain.FACTUAL)


def load_probes_from_file(filepath: Path) -> Iterator[AtlasProbe]:
    """Load probes from a single JSON file.
    
    Args:
        filepath: Path to JSON file with probe definitions
        
    Yields:
        AtlasProbe objects from the file
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        
        domain_str = data.get("domain", "factual")
        domain = _domain_from_string(domain_str)
        
        for probe_data in data.get("probes", []):
            # Get support texts as tuple
            support_texts = tuple(probe_data.get("support_texts", []))
            
            # Parse probe_id to extract source and id
            full_id = probe_data.get("id", "")
            if ":" in full_id:
                source_str, probe_id = full_id.split(":", 1)
            else:
                source_str, probe_id = "external", full_id
            
            # Map source string to AtlasSource enum
            source_map = {
                "sequence_invariant": AtlasSource.SEQUENCE_INVARIANT,
                "semantic_prime": AtlasSource.SEMANTIC_PRIME,
                "computational_gate": AtlasSource.COMPUTATIONAL_GATE,
                "emotion_concept": AtlasSource.EMOTION_CONCEPT,
                "domain_specific": AtlasSource.DOMAIN_SPECIFIC,
                "temporal_concept": AtlasSource.TEMPORAL_CONCEPT,
                "spatial_concept": AtlasSource.SPATIAL_CONCEPT,
                "social_concept": AtlasSource.SOCIAL_CONCEPT,
                "moral_concept": AtlasSource.MORAL_CONCEPT,
                "philosophical_concept": AtlasSource.PHILOSOPHICAL_CONCEPT,
                "safety_ethics": AtlasSource.SAFETY_ETHICS,
                "physical_existence": AtlasSource.PHYSICAL_EXISTENCE,
                "compositional": AtlasSource.COMPOSITIONAL,
                "conceptual_genealogy": AtlasSource.CONCEPTUAL_GENEALOGY,
                "metaphor_invariant": AtlasSource.METAPHOR_INVARIANT,
                "conceptual_metaphor": AtlasSource.CONCEPTUAL_METAPHOR,
                "syntax_concept": AtlasSource.SYNTAX_CONCEPT,
                "perceptual": AtlasSource.PERCEPTUAL,
                "numeric": AtlasSource.NUMERIC,
                "common_object": AtlasSource.COMMON_OBJECT,
                "action_verb": AtlasSource.ACTION_VERB,
                "abstract_relation": AtlasSource.ABSTRACT_RELATION,
                "pronoun_perspective": AtlasSource.PRONOUN_PERSPECTIVE,
                "prime_number": AtlasSource.PRIME_NUMBER,
            }
            source = source_map.get(source_str, AtlasSource.DOMAIN_SPECIFIC)
            
            yield AtlasProbe(
                id=probe_id,
                source=source,
                domain=domain,
                name=probe_data.get("name", ""),
                description=probe_data.get("description", ""),
                cross_domain_weight=1.0,  # Default weight
                category_name=domain_str,
                support_texts=support_texts,
            )
    except Exception as e:
        logger.warning("Failed to load probes from %s: %s", filepath, e)


def load_all_probes(data_dir: Path | None = None) -> list[AtlasProbe]:
    """Load all probes from all JSON files in the data directory.
    
    Args:
        data_dir: Optional path to probe data directory.
                  Defaults to data/probes/ relative to package root.
    
    Returns:
        List of all AtlasProbe objects from all domain files.
    """
    if data_dir is None:
        data_dir = _PROBE_DATA_DIR
    
    if not data_dir.exists():
        logger.warning("Probe data directory not found: %s", data_dir)
        return []
    
    all_probes = []
    json_files = sorted(data_dir.glob("*.json"))
    
    for filepath in json_files:
        probes = list(load_probes_from_file(filepath))
        all_probes.extend(probes)
        logger.debug("Loaded %d probes from %s", len(probes), filepath.name)
    
    logger.info("Loaded %d probes from %d files in %s", 
                len(all_probes), len(json_files), data_dir)
    return all_probes


def get_probe_count_by_domain(data_dir: Path | None = None) -> dict[str, int]:
    """Get count of probes per domain from data files.
    
    Returns:
        Dictionary mapping domain name to probe count.
    """
    if data_dir is None:
        data_dir = _PROBE_DATA_DIR
    
    counts = {}
    for filepath in data_dir.glob("*.json"):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            counts[data.get("domain", filepath.stem)] = data.get("probe_count", 0)
        except Exception as exc:
            logger.warning("Failed to load probe metadata from %s: %s", filepath, exc)
    
    return counts
