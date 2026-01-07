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
Perceptual Atlas - Sensory concepts for cross-dimensional alignment.

Covers the fundamental sensory modalities that ground language in embodied experience:
- Colors (visual wavelengths)
- Sounds (auditory phenomena)
- Textures (tactile properties)
- Tastes (gustatory categories)
- Smells (olfactory qualities)
- Temperature (thermal sensation)
- Visual Properties (light/surface)
- Body Sensations (interoception)

Total: ~60 probes for perceptual grounding.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PerceptualCategory(str, Enum):
    """Category of perceptual concept."""
    
    COLOR = "color"
    SOUND = "sound"
    TEXTURE = "texture"
    TASTE = "taste"
    SMELL = "smell"
    TEMPERATURE = "temperature"
    VISUAL_PROPERTY = "visual_property"
    BODY_SENSATION = "body_sensation"


@dataclass(frozen=True)
class PerceptualConcept:
    """A perceptual/sensory concept."""
    
    id: str
    name: str
    category: PerceptualCategory
    description: str
    support_texts: tuple[str, ...]
    cross_domain_weight: float = 1.0


class PerceptualConceptInventory:
    """Inventory of perceptual concepts."""
    
    # Colors - basic color terms (Berlin & Kay universals + common)
    COLORS = (
        PerceptualConcept("red", "red", PerceptualCategory.COLOR, "The color red", ("red", "crimson", "scarlet")),
        PerceptualConcept("blue", "blue", PerceptualCategory.COLOR, "The color blue", ("blue", "azure", "navy")),
        PerceptualConcept("green", "green", PerceptualCategory.COLOR, "The color green", ("green", "emerald", "lime")),
        PerceptualConcept("yellow", "yellow", PerceptualCategory.COLOR, "The color yellow", ("yellow", "gold", "lemon")),
        PerceptualConcept("orange", "orange", PerceptualCategory.COLOR, "The color orange", ("orange", "amber", "tangerine")),
        PerceptualConcept("purple", "purple", PerceptualCategory.COLOR, "The color purple", ("purple", "violet", "lavender")),
        PerceptualConcept("white", "white", PerceptualCategory.COLOR, "The color white", ("white", "ivory", "snow")),
        PerceptualConcept("black", "black", PerceptualCategory.COLOR, "The color black", ("black", "ebony", "midnight")),
        PerceptualConcept("gray", "gray", PerceptualCategory.COLOR, "The color gray", ("gray", "grey", "silver")),
        PerceptualConcept("brown", "brown", PerceptualCategory.COLOR, "The color brown", ("brown", "tan", "chocolate")),
        PerceptualConcept("pink", "pink", PerceptualCategory.COLOR, "The color pink", ("pink", "rose", "blush")),
    )
    
    # Sounds - auditory phenomena
    SOUNDS = (
        PerceptualConcept("loud", "loud", PerceptualCategory.SOUND, "High volume", ("loud", "noisy", "deafening")),
        PerceptualConcept("quiet", "quiet", PerceptualCategory.SOUND, "Low volume", ("quiet", "soft", "gentle")),
        PerceptualConcept("music", "music", PerceptualCategory.SOUND, "Organized sound", ("music", "melody", "song")),
        PerceptualConcept("noise", "noise", PerceptualCategory.SOUND, "Unwanted sound", ("noise", "din", "racket")),
        PerceptualConcept("silence", "silence", PerceptualCategory.SOUND, "Absence of sound", ("silence", "quiet", "hush")),
        PerceptualConcept("rhythm", "rhythm", PerceptualCategory.SOUND, "Temporal pattern", ("rhythm", "beat", "tempo")),
        PerceptualConcept("melody", "melody", PerceptualCategory.SOUND, "Pitch sequence", ("melody", "tune", "air")),
        PerceptualConcept("harmony", "harmony", PerceptualCategory.SOUND, "Simultaneous pitches", ("harmony", "chord", "consonance")),
    )
    
    # Textures - tactile properties
    TEXTURES = (
        PerceptualConcept("smooth", "smooth", PerceptualCategory.TEXTURE, "Even surface", ("smooth", "sleek", "polished")),
        PerceptualConcept("rough", "rough", PerceptualCategory.TEXTURE, "Uneven surface", ("rough", "coarse", "bumpy")),
        PerceptualConcept("soft", "soft", PerceptualCategory.TEXTURE, "Yielding to pressure", ("soft", "plush", "cushioned")),
        PerceptualConcept("hard", "hard", PerceptualCategory.TEXTURE, "Resistant to pressure", ("hard", "firm", "solid")),
        PerceptualConcept("wet", "wet", PerceptualCategory.TEXTURE, "Covered with liquid", ("wet", "moist", "damp")),
        PerceptualConcept("dry", "dry", PerceptualCategory.TEXTURE, "Without moisture", ("dry", "arid", "parched")),
        PerceptualConcept("sticky", "sticky", PerceptualCategory.TEXTURE, "Adhering surface", ("sticky", "tacky", "gummy")),
        PerceptualConcept("slippery", "slippery", PerceptualCategory.TEXTURE, "Low friction", ("slippery", "slick", "greasy")),
    )
    
    # Tastes - gustatory categories
    TASTES = (
        PerceptualConcept("sweet", "sweet", PerceptualCategory.TASTE, "Sugar-like taste", ("sweet", "sugary", "honeyed")),
        PerceptualConcept("sour", "sour", PerceptualCategory.TASTE, "Acidic taste", ("sour", "tart", "tangy")),
        PerceptualConcept("bitter", "bitter", PerceptualCategory.TASTE, "Alkaloid taste", ("bitter", "acrid", "sharp")),
        PerceptualConcept("salty", "salty", PerceptualCategory.TASTE, "Sodium taste", ("salty", "briny", "saline")),
        PerceptualConcept("savory", "savory", PerceptualCategory.TASTE, "Umami taste", ("savory", "umami", "meaty")),
        PerceptualConcept("spicy", "spicy", PerceptualCategory.TASTE, "Capsaicin heat", ("spicy", "hot", "peppery")),
        PerceptualConcept("bland", "bland", PerceptualCategory.TASTE, "Without flavor", ("bland", "tasteless", "plain")),
    )
    
    # Smells - olfactory qualities
    SMELLS = (
        PerceptualConcept("fresh", "fresh", PerceptualCategory.SMELL, "Clean smell", ("fresh", "clean", "crisp")),
        PerceptualConcept("stale", "stale", PerceptualCategory.SMELL, "Old smell", ("stale", "musty", "stuffy")),
        PerceptualConcept("fragrant", "fragrant", PerceptualCategory.SMELL, "Pleasant smell", ("fragrant", "aromatic", "perfumed")),
        PerceptualConcept("pungent", "pungent", PerceptualCategory.SMELL, "Strong sharp smell", ("pungent", "acrid", "sharp")),
        PerceptualConcept("musky", "musky", PerceptualCategory.SMELL, "Animal-like smell", ("musky", "earthy", "animalistic")),
        PerceptualConcept("floral", "floral", PerceptualCategory.SMELL, "Flower-like smell", ("floral", "flowery", "rose-like")),
    )
    
    # Temperature - thermal sensation
    TEMPERATURE = (
        PerceptualConcept("hot", "hot", PerceptualCategory.TEMPERATURE, "High temperature", ("hot", "burning", "scorching")),
        PerceptualConcept("cold", "cold", PerceptualCategory.TEMPERATURE, "Low temperature", ("cold", "freezing", "icy")),
        PerceptualConcept("warm", "warm", PerceptualCategory.TEMPERATURE, "Moderate high temp", ("warm", "cozy", "toasty")),
        PerceptualConcept("cool", "cool", PerceptualCategory.TEMPERATURE, "Moderate low temp", ("cool", "chilly", "brisk")),
        PerceptualConcept("freezing", "freezing", PerceptualCategory.TEMPERATURE, "Below zero", ("freezing", "frozen", "frigid")),
        PerceptualConcept("burning", "burning", PerceptualCategory.TEMPERATURE, "Extreme heat", ("burning", "scalding", "searing")),
    )
    
    # Visual Properties - light/surface
    VISUAL_PROPERTIES = (
        PerceptualConcept("bright", "bright", PerceptualCategory.VISUAL_PROPERTY, "High luminance", ("bright", "brilliant", "radiant")),
        PerceptualConcept("dark", "dark", PerceptualCategory.VISUAL_PROPERTY, "Low luminance", ("dark", "dim", "shadowy")),
        PerceptualConcept("shiny", "shiny", PerceptualCategory.VISUAL_PROPERTY, "Reflective surface", ("shiny", "glossy", "lustrous")),
        PerceptualConcept("dull", "dull", PerceptualCategory.VISUAL_PROPERTY, "Non-reflective", ("dull", "matte", "flat")),
        PerceptualConcept("transparent", "transparent", PerceptualCategory.VISUAL_PROPERTY, "See-through", ("transparent", "clear", "crystal")),
        PerceptualConcept("opaque", "opaque", PerceptualCategory.VISUAL_PROPERTY, "Not see-through", ("opaque", "solid", "dense")),
    )
    
    # Body Sensations - interoception
    BODY_SENSATIONS = (
        PerceptualConcept("pain", "pain", PerceptualCategory.BODY_SENSATION, "Nociception", ("pain", "hurt", "ache")),
        PerceptualConcept("pleasure", "pleasure", PerceptualCategory.BODY_SENSATION, "Positive sensation", ("pleasure", "enjoyment", "delight")),
        PerceptualConcept("hunger", "hunger", PerceptualCategory.BODY_SENSATION, "Need for food", ("hunger", "hungry", "starving")),
        PerceptualConcept("thirst", "thirst", PerceptualCategory.BODY_SENSATION, "Need for water", ("thirst", "thirsty", "parched")),
        PerceptualConcept("tired", "tired", PerceptualCategory.BODY_SENSATION, "Need for rest", ("tired", "exhausted", "weary")),
        PerceptualConcept("alert", "alert", PerceptualCategory.BODY_SENSATION, "State of wakefulness", ("alert", "awake", "vigilant")),
    )
    
    @classmethod
    def all_concepts(cls) -> list[PerceptualConcept]:
        """Get all perceptual concepts."""
        concepts: list[PerceptualConcept] = []
        concepts.extend(cls.COLORS)
        concepts.extend(cls.SOUNDS)
        concepts.extend(cls.TEXTURES)
        concepts.extend(cls.TASTES)
        concepts.extend(cls.SMELLS)
        concepts.extend(cls.TEMPERATURE)
        concepts.extend(cls.VISUAL_PROPERTIES)
        concepts.extend(cls.BODY_SENSATIONS)
        return concepts
    
    @classmethod
    def concepts_by_category(cls, category: PerceptualCategory) -> list[PerceptualConcept]:
        """Get concepts for a specific category."""
        return [c for c in cls.all_concepts() if c.category == category]
