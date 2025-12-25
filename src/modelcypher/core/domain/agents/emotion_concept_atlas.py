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
Emotion Concept Atlas.

Embedding-based emotion concept analyzer for model representation geometry.
Based on Plutchik's wheel of emotions with VAD (Valence-Arousal-Dominance)
dimensional coordinates and opposition structure preservation.

Key features:
- 8 primary emotions with 3 intensity levels (24 base concepts)
- 8 primary dyads (blended emotions from adjacent primaries)
- Antipodal opposition pairs (joy↔sadness, trust↔disgust, etc.)
- VAD projection for continuous emotion space
- Opposition preservation scoring for merge validation

References:
- Plutchik, R. (1980). Emotion: A Psychoevolutionary Synthesis
- Russell, J.A. (1980). A circumplex model of affect (VAD)
- arxiv.org/html/2510.22042 - Emotional Latent Space in LLMs
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from modelcypher.core.domain.geometry.signature_base import LabeledSignatureMixin
from modelcypher.core.domain.geometry.vector_math import VectorMath
from modelcypher.ports.embedding import EmbeddingProvider

# Optional: Riemannian density for volume-based emotion representation (CABE-4)
try:
    import numpy as np

    from modelcypher.core.domain.geometry.riemannian_density import (
        ConceptVolume,
        RiemannianDensityEstimator,
    )

    HAS_RIEMANNIAN = True
except ImportError:
    HAS_RIEMANNIAN = False
    np = None
    ConceptVolume = None


class EmotionCategory(str, Enum):
    """Primary emotion categories based on Plutchik's wheel."""

    JOY = "joy"
    TRUST = "trust"
    FEAR = "fear"
    SURPRISE = "surprise"
    SADNESS = "sadness"
    DISGUST = "disgust"
    ANGER = "anger"
    ANTICIPATION = "anticipation"


class EmotionIntensity(str, Enum):
    """Intensity levels from Plutchik's concentric rings."""

    MILD = "mild"  # Outer ring (e.g., serenity, acceptance)
    PRIMARY = "primary"  # Middle ring (e.g., joy, trust)
    INTENSE = "intense"  # Inner ring (e.g., ecstasy, admiration)


@dataclass(frozen=True)
class EmotionConcept:
    """A single emotion concept with VAD coordinates and opposition link."""

    id: str
    category: EmotionCategory
    intensity: EmotionIntensity
    name: str
    description: str
    support_texts: tuple[str, ...]
    valence: float  # -1 to +1 (negative to positive hedonic tone)
    arousal: float  # 0 to 1 (calm to excited/activated)
    dominance: float  # 0 to 1 (submissive to dominant/in-control)
    opposite_id: str | None = None

    @property
    def canonical_name(self) -> str:
        return self.name

    @property
    def vad(self) -> tuple[float, float, float]:
        """VAD coordinates as a tuple."""
        return (self.valence, self.arousal, self.dominance)


@dataclass(frozen=True)
class EmotionDyad:
    """Blended emotion from two adjacent primary emotions on Plutchik's wheel."""

    id: str
    name: str
    description: str
    primary_ids: tuple[str, str]
    support_texts: tuple[str, ...]
    valence: float
    arousal: float
    dominance: float

    @property
    def canonical_name(self) -> str:
        return self.name

    @property
    def vad(self) -> tuple[float, float, float]:
        return (self.valence, self.arousal, self.dominance)


# Opposition pairs: each primary emotion has an antipodal opposite
OPPOSITION_PAIRS: tuple[tuple[EmotionCategory, EmotionCategory], ...] = (
    (EmotionCategory.JOY, EmotionCategory.SADNESS),
    (EmotionCategory.TRUST, EmotionCategory.DISGUST),
    (EmotionCategory.FEAR, EmotionCategory.ANGER),
    (EmotionCategory.SURPRISE, EmotionCategory.ANTICIPATION),
)


class EmotionConceptInventory:
    """
    Complete inventory of emotion concepts based on Plutchik's wheel.

    Structure:
    - 8 primary emotions x 3 intensities = 24 basic emotions
    - 8 primary dyads (adjacent blends)
    - Total: 32 emotion concepts
    """

    @staticmethod
    def primary_emotions() -> list[EmotionConcept]:
        """The 8 primary emotions at PRIMARY intensity."""
        return [
            # JOY family
            EmotionConcept(
                id="joy",
                category=EmotionCategory.JOY,
                intensity=EmotionIntensity.PRIMARY,
                name="Joy",
                description="A feeling of great pleasure and happiness.",
                support_texts=(
                    "Joy is an emotion of happiness and delight.",
                    "I feel so happy right now!",
                    "Laughing, smiling, and celebrating.",
                    "A warm, pleasant feeling spreading through me.",
                    "This brings me great satisfaction.",
                ),
                valence=0.9,
                arousal=0.7,
                dominance=0.7,
                opposite_id="sadness",
            ),
            # TRUST family
            EmotionConcept(
                id="trust",
                category=EmotionCategory.TRUST,
                intensity=EmotionIntensity.PRIMARY,
                name="Trust",
                description="Firm belief in the reliability and truth of someone or something.",
                support_texts=(
                    "Trust is confidence in others' honesty and ability.",
                    "I believe in you completely.",
                    "Feeling safe and secure with someone.",
                    "Relying on another person without worry.",
                    "Having faith in the process.",
                ),
                valence=0.7,
                arousal=0.3,
                dominance=0.5,
                opposite_id="disgust",
            ),
            # FEAR family
            EmotionConcept(
                id="fear",
                category=EmotionCategory.FEAR,
                intensity=EmotionIntensity.PRIMARY,
                name="Fear",
                description="An unpleasant emotion caused by threat of danger or harm.",
                support_texts=(
                    "Fear is a response to perceived danger.",
                    "I'm scared of what might happen.",
                    "Heart pounding, muscles tense, ready to flee.",
                    "A sense of dread and vulnerability.",
                    "Something bad is about to happen.",
                ),
                valence=-0.8,
                arousal=0.8,
                dominance=0.2,
                opposite_id="anger",
            ),
            # SURPRISE family
            EmotionConcept(
                id="surprise",
                category=EmotionCategory.SURPRISE,
                intensity=EmotionIntensity.PRIMARY,
                name="Surprise",
                description="The feeling caused by something unexpected or unusual.",
                support_texts=(
                    "Surprise is a reaction to the unexpected.",
                    "I didn't expect that at all!",
                    "Eyes widening, jaw dropping, gasping.",
                    "A sudden jolt of awareness.",
                    "Something caught me completely off guard.",
                ),
                valence=0.1,
                arousal=0.9,
                dominance=0.3,
                opposite_id="anticipation",
            ),
            # SADNESS family
            EmotionConcept(
                id="sadness",
                category=EmotionCategory.SADNESS,
                intensity=EmotionIntensity.PRIMARY,
                name="Sadness",
                description="A feeling of unhappiness, sorrow, or grief.",
                support_texts=(
                    "Sadness is a response to loss or disappointment.",
                    "I feel so down and blue.",
                    "Tears falling, heavy heart, withdrawn.",
                    "A sense of emptiness and loss.",
                    "Nothing seems to matter anymore.",
                ),
                valence=-0.8,
                arousal=0.3,
                dominance=0.2,
                opposite_id="joy",
            ),
            # DISGUST family
            EmotionConcept(
                id="disgust",
                category=EmotionCategory.DISGUST,
                intensity=EmotionIntensity.PRIMARY,
                name="Disgust",
                description="A strong feeling of revulsion or disapproval.",
                support_texts=(
                    "Disgust is revulsion toward something offensive.",
                    "That's absolutely revolting.",
                    "Nose wrinkling, turning away, nausea.",
                    "A strong urge to avoid and reject.",
                    "I can't stand to look at this.",
                ),
                valence=-0.7,
                arousal=0.5,
                dominance=0.5,
                opposite_id="trust",
            ),
            # ANGER family
            EmotionConcept(
                id="anger",
                category=EmotionCategory.ANGER,
                intensity=EmotionIntensity.PRIMARY,
                name="Anger",
                description="A strong feeling of annoyance, displeasure, or hostility.",
                support_texts=(
                    "Anger is a response to perceived injustice or threat.",
                    "I'm so frustrated and upset!",
                    "Face flushing, fists clenching, voice rising.",
                    "A burning desire to confront or fight back.",
                    "This is completely unacceptable.",
                ),
                valence=-0.6,
                arousal=0.9,
                dominance=0.8,
                opposite_id="fear",
            ),
            # ANTICIPATION family
            EmotionConcept(
                id="anticipation",
                category=EmotionCategory.ANTICIPATION,
                intensity=EmotionIntensity.PRIMARY,
                name="Anticipation",
                description="Expectation or prediction about something in the future.",
                support_texts=(
                    "Anticipation is forward-looking expectation.",
                    "I can't wait to see what happens next.",
                    "Leaning forward, alert, scanning for signals.",
                    "A sense of readiness for what's coming.",
                    "Something interesting is about to happen.",
                ),
                valence=0.3,
                arousal=0.6,
                dominance=0.5,
                opposite_id="surprise",
            ),
        ]

    @staticmethod
    def mild_emotions() -> list[EmotionConcept]:
        """Mild intensity variants (outer ring of Plutchik's wheel)."""
        return [
            # Serenity (mild joy)
            EmotionConcept(
                id="serenity",
                category=EmotionCategory.JOY,
                intensity=EmotionIntensity.MILD,
                name="Serenity",
                description="A state of calm and peaceful happiness.",
                support_texts=(
                    "Serenity is tranquil contentment.",
                    "I feel at peace with everything.",
                    "Relaxed smile, slow breathing, at ease.",
                    "A gentle sense of wellbeing.",
                ),
                valence=0.6,
                arousal=0.2,
                dominance=0.6,
                opposite_id="pensiveness",
            ),
            # Acceptance (mild trust)
            EmotionConcept(
                id="acceptance",
                category=EmotionCategory.TRUST,
                intensity=EmotionIntensity.MILD,
                name="Acceptance",
                description="Willingness to receive or tolerate someone or something.",
                support_texts=(
                    "Acceptance is openness without resistance.",
                    "I can live with this situation.",
                    "Nodding along, going with the flow.",
                    "Making peace with what is.",
                ),
                valence=0.4,
                arousal=0.2,
                dominance=0.4,
                opposite_id="boredom",
            ),
            # Apprehension (mild fear)
            EmotionConcept(
                id="apprehension",
                category=EmotionCategory.FEAR,
                intensity=EmotionIntensity.MILD,
                name="Apprehension",
                description="Anxiety or worry about a future event.",
                support_texts=(
                    "Apprehension is mild worry about what's ahead.",
                    "I'm a bit nervous about this.",
                    "Slight tension, watchful, cautious.",
                    "A nagging sense that something might go wrong.",
                ),
                valence=-0.4,
                arousal=0.4,
                dominance=0.3,
                opposite_id="annoyance",
            ),
            # Distraction (mild surprise)
            EmotionConcept(
                id="distraction",
                category=EmotionCategory.SURPRISE,
                intensity=EmotionIntensity.MILD,
                name="Distraction",
                description="Having attention diverted from the main focus.",
                support_texts=(
                    "Distraction is mild disruption of attention.",
                    "Something caught my attention for a moment.",
                    "Looking up, briefly confused, refocusing.",
                    "A minor interruption in my thoughts.",
                ),
                valence=0.0,
                arousal=0.5,
                dominance=0.3,
                opposite_id="interest",
            ),
            # Pensiveness (mild sadness)
            EmotionConcept(
                id="pensiveness",
                category=EmotionCategory.SADNESS,
                intensity=EmotionIntensity.MILD,
                name="Pensiveness",
                description="Deep thoughtfulness, often tinged with melancholy.",
                support_texts=(
                    "Pensiveness is reflective sadness.",
                    "I'm feeling a bit wistful today.",
                    "Gazing into distance, quiet sighs, reflective.",
                    "Thinking about what might have been.",
                ),
                valence=-0.4,
                arousal=0.2,
                dominance=0.3,
                opposite_id="serenity",
            ),
            # Boredom (mild disgust)
            EmotionConcept(
                id="boredom",
                category=EmotionCategory.DISGUST,
                intensity=EmotionIntensity.MILD,
                name="Boredom",
                description="Weariness from lack of interest or occupation.",
                support_texts=(
                    "Boredom is disengagement from uninteresting stimuli.",
                    "This is so dull and uninteresting.",
                    "Yawning, looking away, fidgeting.",
                    "I wish I were doing something else.",
                ),
                valence=-0.3,
                arousal=0.1,
                dominance=0.4,
                opposite_id="acceptance",
            ),
            # Annoyance (mild anger)
            EmotionConcept(
                id="annoyance",
                category=EmotionCategory.ANGER,
                intensity=EmotionIntensity.MILD,
                name="Annoyance",
                description="Slight irritation or displeasure.",
                support_texts=(
                    "Annoyance is mild frustration.",
                    "That's a bit irritating.",
                    "Slight frown, brief eye roll, sighing.",
                    "A minor inconvenience that bugs me.",
                ),
                valence=-0.3,
                arousal=0.4,
                dominance=0.5,
                opposite_id="apprehension",
            ),
            # Interest (mild anticipation)
            EmotionConcept(
                id="interest",
                category=EmotionCategory.ANTICIPATION,
                intensity=EmotionIntensity.MILD,
                name="Interest",
                description="Curiosity or attention toward something.",
                support_texts=(
                    "Interest is engaged curiosity.",
                    "That's quite intriguing, tell me more.",
                    "Leaning in, focused attention, asking questions.",
                    "I want to learn more about this.",
                ),
                valence=0.3,
                arousal=0.4,
                dominance=0.5,
                opposite_id="distraction",
            ),
        ]

    @staticmethod
    def intense_emotions() -> list[EmotionConcept]:
        """Intense variants (inner ring of Plutchik's wheel)."""
        return [
            # Ecstasy (intense joy)
            EmotionConcept(
                id="ecstasy",
                category=EmotionCategory.JOY,
                intensity=EmotionIntensity.INTENSE,
                name="Ecstasy",
                description="Overwhelming feeling of joy and rapture.",
                support_texts=(
                    "Ecstasy is peak happiness and bliss.",
                    "I've never felt this happy in my life!",
                    "Euphoric laughter, jumping, tears of joy.",
                    "An overwhelming rush of pure happiness.",
                ),
                valence=1.0,
                arousal=0.95,
                dominance=0.8,
                opposite_id="grief",
            ),
            # Admiration (intense trust)
            EmotionConcept(
                id="admiration",
                category=EmotionCategory.TRUST,
                intensity=EmotionIntensity.INTENSE,
                name="Admiration",
                description="Deep respect and warm approval.",
                support_texts=(
                    "Admiration is profound respect and awe.",
                    "I deeply respect and look up to you.",
                    "Eyes shining, nodding appreciatively, praising.",
                    "Feeling inspired by someone's excellence.",
                ),
                valence=0.8,
                arousal=0.5,
                dominance=0.4,
                opposite_id="loathing",
            ),
            # Terror (intense fear)
            EmotionConcept(
                id="terror",
                category=EmotionCategory.FEAR,
                intensity=EmotionIntensity.INTENSE,
                name="Terror",
                description="Extreme fear and panic.",
                support_texts=(
                    "Terror is overwhelming, paralyzing fear.",
                    "I'm absolutely terrified, I can't move!",
                    "Screaming, frozen in place, hyperventilating.",
                    "Complete panic, fight or flight in overdrive.",
                ),
                valence=-1.0,
                arousal=1.0,
                dominance=0.0,
                opposite_id="rage",
            ),
            # Amazement (intense surprise)
            EmotionConcept(
                id="amazement",
                category=EmotionCategory.SURPRISE,
                intensity=EmotionIntensity.INTENSE,
                name="Amazement",
                description="Great wonder and astonishment.",
                support_texts=(
                    "Amazement is overwhelming wonder.",
                    "I can't believe what I'm seeing!",
                    "Jaw dropped, speechless, eyes wide.",
                    "Completely astounded and awestruck.",
                ),
                valence=0.4,
                arousal=1.0,
                dominance=0.3,
                opposite_id="vigilance",
            ),
            # Grief (intense sadness)
            EmotionConcept(
                id="grief",
                category=EmotionCategory.SADNESS,
                intensity=EmotionIntensity.INTENSE,
                name="Grief",
                description="Deep sorrow, especially from loss.",
                support_texts=(
                    "Grief is profound, overwhelming sadness.",
                    "My heart is completely broken.",
                    "Sobbing uncontrollably, chest aching, inconsolable.",
                    "A crushing weight of loss and despair.",
                ),
                valence=-1.0,
                arousal=0.6,
                dominance=0.0,
                opposite_id="ecstasy",
            ),
            # Loathing (intense disgust)
            EmotionConcept(
                id="loathing",
                category=EmotionCategory.DISGUST,
                intensity=EmotionIntensity.INTENSE,
                name="Loathing",
                description="Intense hatred and disgust.",
                support_texts=(
                    "Loathing is extreme revulsion and hatred.",
                    "I absolutely despise this with every fiber.",
                    "Gagging, turning away in disgust, repulsed.",
                    "Deep hatred and complete rejection.",
                ),
                valence=-0.9,
                arousal=0.7,
                dominance=0.6,
                opposite_id="admiration",
            ),
            # Rage (intense anger)
            EmotionConcept(
                id="rage",
                category=EmotionCategory.ANGER,
                intensity=EmotionIntensity.INTENSE,
                name="Rage",
                description="Violent, uncontrollable anger.",
                support_texts=(
                    "Rage is explosive, overwhelming fury.",
                    "I'm absolutely furious beyond words!",
                    "Shouting, shaking, seeing red.",
                    "Uncontrollable desire to destroy.",
                ),
                valence=-0.8,
                arousal=1.0,
                dominance=1.0,
                opposite_id="terror",
            ),
            # Vigilance (intense anticipation)
            EmotionConcept(
                id="vigilance",
                category=EmotionCategory.ANTICIPATION,
                intensity=EmotionIntensity.INTENSE,
                name="Vigilance",
                description="Watchful alertness, ready for action.",
                support_texts=(
                    "Vigilance is hyper-alert readiness.",
                    "I'm watching every move, ready for anything.",
                    "Scanning constantly, muscles tensed, on guard.",
                    "Complete focus on what's coming next.",
                ),
                valence=0.2,
                arousal=0.8,
                dominance=0.7,
                opposite_id="amazement",
            ),
        ]

    @staticmethod
    def primary_dyads() -> list[EmotionDyad]:
        """Primary dyads: blends of adjacent emotions on the wheel."""
        return [
            # Joy + Trust = Love
            EmotionDyad(
                id="love",
                name="Love",
                description="Deep affection combining joy and trust.",
                primary_ids=("joy", "trust"),
                support_texts=(
                    "Love combines happiness with deep trust.",
                    "I care about you so much.",
                    "Warmth, devotion, wanting to be close.",
                    "Feeling safe and happy with someone.",
                ),
                valence=0.9,
                arousal=0.5,
                dominance=0.6,
            ),
            # Trust + Fear = Submission
            EmotionDyad(
                id="submission",
                name="Submission",
                description="Yielding combined with trust and wariness.",
                primary_ids=("trust", "fear"),
                support_texts=(
                    "Submission is deferring to another's authority.",
                    "I'll follow your lead.",
                    "Head bowed, compliant, yielding.",
                    "Trusting but also apprehensive.",
                ),
                valence=0.0,
                arousal=0.4,
                dominance=0.1,
            ),
            # Fear + Surprise = Awe
            EmotionDyad(
                id="awe",
                name="Awe",
                description="Wonder mixed with fearful respect.",
                primary_ids=("fear", "surprise"),
                support_texts=(
                    "Awe is amazement tinged with fear.",
                    "I'm overwhelmed by the magnitude of this.",
                    "Standing speechless, humbled, reverent.",
                    "Both amazed and a little frightened.",
                ),
                valence=0.2,
                arousal=0.8,
                dominance=0.2,
            ),
            # Surprise + Sadness = Disapproval
            EmotionDyad(
                id="disapproval",
                name="Disapproval",
                description="Negative judgment combined with surprise.",
                primary_ids=("surprise", "sadness"),
                support_texts=(
                    "Disapproval is shocked disappointment.",
                    "I can't believe you did that.",
                    "Shaking head, frowning, sighing.",
                    "Surprised and let down at the same time.",
                ),
                valence=-0.5,
                arousal=0.5,
                dominance=0.4,
            ),
            # Sadness + Disgust = Remorse
            EmotionDyad(
                id="remorse",
                name="Remorse",
                description="Deep regret combining sadness with self-disgust.",
                primary_ids=("sadness", "disgust"),
                support_texts=(
                    "Remorse is painful guilt and regret.",
                    "I wish I had never done that.",
                    "Head in hands, self-reproach, apologizing.",
                    "Disgusted with myself for what I did.",
                ),
                valence=-0.8,
                arousal=0.4,
                dominance=0.2,
            ),
            # Disgust + Anger = Contempt
            EmotionDyad(
                id="contempt",
                name="Contempt",
                description="Scornful disdain combining disgust and anger.",
                primary_ids=("disgust", "anger"),
                support_texts=(
                    "Contempt is hostile dismissiveness.",
                    "You're beneath my notice.",
                    "Sneering, looking down, dismissive.",
                    "Feeling superior and disgusted.",
                ),
                valence=-0.6,
                arousal=0.5,
                dominance=0.8,
            ),
            # Anger + Anticipation = Aggressiveness
            EmotionDyad(
                id="aggressiveness",
                name="Aggressiveness",
                description="Hostile readiness combining anger and anticipation.",
                primary_ids=("anger", "anticipation"),
                support_texts=(
                    "Aggressiveness is hostile forward energy.",
                    "I'm ready to take them on.",
                    "Advancing, challenging, confrontational.",
                    "Angry and eager to act on it.",
                ),
                valence=-0.3,
                arousal=0.85,
                dominance=0.9,
            ),
            # Anticipation + Joy = Optimism
            EmotionDyad(
                id="optimism",
                name="Optimism",
                description="Positive expectation combining anticipation and joy.",
                primary_ids=("anticipation", "joy"),
                support_texts=(
                    "Optimism is hopeful positive expectation.",
                    "I have a good feeling about this.",
                    "Smiling, planning, looking forward.",
                    "Confident that good things are coming.",
                ),
                valence=0.7,
                arousal=0.6,
                dominance=0.6,
            ),
        ]

    @staticmethod
    def all_emotions() -> list[EmotionConcept]:
        """All emotion concepts (primary + mild + intense)."""
        return (
            EmotionConceptInventory.primary_emotions()
            + EmotionConceptInventory.mild_emotions()
            + EmotionConceptInventory.intense_emotions()
        )

    @staticmethod
    def all_dyads() -> list[EmotionDyad]:
        """All emotion dyads."""
        return EmotionConceptInventory.primary_dyads()

    @staticmethod
    def all_concepts() -> tuple[list[EmotionConcept], list[EmotionDyad]]:
        """All emotion concepts and dyads."""
        return (
            EmotionConceptInventory.all_emotions(),
            EmotionConceptInventory.all_dyads(),
        )

    @staticmethod
    def by_category(category: EmotionCategory) -> list[EmotionConcept]:
        """Get all emotions in a category (all intensities)."""
        return [e for e in EmotionConceptInventory.all_emotions() if e.category == category]

    @staticmethod
    def get_opposite(emotion_id: str) -> str | None:
        """Get the opposite emotion ID for a given emotion."""
        all_emotions = EmotionConceptInventory.all_emotions()
        for emotion in all_emotions:
            if emotion.id == emotion_id:
                return emotion.opposite_id
        return None


@dataclass
class EmotionConceptSignature(LabeledSignatureMixin):
    """
    Activation signature across emotion concepts.

    Inherits l2_normalized() and cosine_similarity() from LabeledSignatureMixin.
    Has VAD projection and opposition analysis methods.
    """

    emotion_ids: list[str]
    values: list[float]
    _inventory: list[EmotionConcept] | None = field(default=None, repr=False)

    def _with_values(self, new_values: list[float]) -> "EmotionConceptSignature":
        """Create a copy with new values, preserving _inventory."""
        return EmotionConceptSignature(self.emotion_ids, new_values, self._inventory)

    def vad_projection(self) -> tuple[float, float, float]:
        """
        Project signature onto VAD space using weighted average.

        Returns (valence, arousal, dominance) coordinates.
        """
        if not self._inventory or len(self._inventory) != len(self.values):
            return (0.0, 0.0, 0.0)

        total_weight = sum(max(0.0, v) for v in self.values)
        if total_weight <= 0:
            return (0.0, 0.0, 0.0)

        weighted_v = 0.0
        weighted_a = 0.0
        weighted_d = 0.0

        for emotion, activation in zip(self._inventory, self.values):
            w = max(0.0, activation)
            weighted_v += w * emotion.valence
            weighted_a += w * emotion.arousal
            weighted_d += w * emotion.dominance

        return (
            weighted_v / total_weight,
            weighted_a / total_weight,
            weighted_d / total_weight,
        )

    def dominant_emotion(self) -> tuple[str, float]:
        """Return the emotion with highest activation."""
        if not self.values or not self.emotion_ids:
            return ("", 0.0)
        max_idx = max(range(len(self.values)), key=lambda i: self.values[i])
        return (self.emotion_ids[max_idx], self.values[max_idx])

    def opposition_balance(self) -> dict[str, float]:
        """
        Compute balance scores for each opposition pair.

        Balance = activation(A) - activation(opposite(A))
        Positive = leans toward first emotion in pair
        Negative = leans toward opposite
        Zero = balanced
        """
        id_to_value = dict(zip(self.emotion_ids, self.values))
        balances = {}

        for cat_a, cat_b in OPPOSITION_PAIRS:
            val_a = id_to_value.get(cat_a.value, 0.0)
            val_b = id_to_value.get(cat_b.value, 0.0)
            balances[f"{cat_a.value}_vs_{cat_b.value}"] = val_a - val_b

        return balances

    def top_emotions(self, k: int = 5) -> list[tuple[str, float]]:
        """Return top k emotions by activation."""
        paired = list(zip(self.emotion_ids, self.values))
        paired.sort(key=lambda x: x[1], reverse=True)
        return paired[:k]


@dataclass
class EmotionActivationSummary:
    """Summary of emotion activation analysis."""

    class Method(str, Enum):
        EMBEDDINGS = "embeddings"
        SKIPPED = "skipped"

    @dataclass
    class EmotionScore:
        emotion_id: str
        name: str
        similarity: float
        category: str

    method: Method
    top_emotions: list[EmotionScore]
    vad_projection: tuple[float, float, float] | None
    dominant_emotion: tuple[str, float] | None
    opposition_balances: dict[str, float] | None
    normalized_activation_entropy: float | None
    note: str | None


@dataclass
class EmotionAtlasConfiguration:
    """Configuration for EmotionConceptAtlas."""

    enabled: bool = True
    max_characters_per_text: int = 4096
    top_k: int = 8
    include_dyads: bool = True
    include_mild: bool = True
    include_intense: bool = True
    # Volume-based representation (CABE-4: Riemannian density)
    use_volume_representation: bool = False
    # Include support_texts in volume estimation (more accurate but slower)
    include_support_texts_in_volume: bool = True


@dataclass(frozen=True)
class OppositionScore:
    """Result of opposition preservation analysis."""

    mean_preservation: float  # 0-1, how well oppositions are preserved
    pair_scores: dict[str, float]  # Per-pair preservation scores
    violated_pairs: list[str]  # Pairs where opposition is violated


class OppositionPreservationScorer:
    """
    Scores how well a representation preserves emotion opposition structure.

    Key insight: In a well-structured emotion space, opposite emotions should
    have low co-activation (they're mutually exclusive) and opposite positions
    in the embedding space.
    """

    @staticmethod
    def compute_score(
        sig_a: EmotionConceptSignature,
        sig_b: EmotionConceptSignature,
        threshold: float = 0.1,
    ) -> OppositionScore:
        """
        Compare opposition structure between two signatures.

        Measures whether the relative positions of opposite emotions are preserved.

        Args:
            sig_a: First emotion signature
            sig_b: Second emotion signature
            threshold: Minimum difference to consider opposition preserved

        Returns:
            OppositionScore with preservation metrics
        """
        if sig_a.emotion_ids != sig_b.emotion_ids:
            return OppositionScore(
                mean_preservation=0.0,
                pair_scores={},
                violated_pairs=[],
            )

        id_to_idx = {eid: i for i, eid in enumerate(sig_a.emotion_ids)}
        pair_scores = {}
        violated_pairs = []

        for cat_a, cat_b in OPPOSITION_PAIRS:
            pair_name = f"{cat_a.value}_vs_{cat_b.value}"

            idx_a = id_to_idx.get(cat_a.value)
            idx_b = id_to_idx.get(cat_b.value)

            if idx_a is None or idx_b is None:
                continue

            # In sig_a, compute which emotion is dominant in the pair
            diff_a = sig_a.values[idx_a] - sig_a.values[idx_b]
            # In sig_b, compute the same
            diff_b = sig_b.values[idx_a] - sig_b.values[idx_b]

            # Preservation: same sign of difference (same emotion dominant in pair)
            # or both near zero
            if abs(diff_a) < threshold and abs(diff_b) < threshold:
                # Both balanced, consider preserved
                pair_scores[pair_name] = 1.0
            elif (diff_a > 0) == (diff_b > 0):
                # Same emotion is dominant in both, preserved
                # Score by similarity of the ratio
                pair_scores[pair_name] = 1.0 - min(1.0, abs(diff_a - diff_b))
            else:
                # Opposition flipped - violated
                pair_scores[pair_name] = 0.0
                violated_pairs.append(pair_name)

        mean_preservation = sum(pair_scores.values()) / len(pair_scores) if pair_scores else 0.0

        return OppositionScore(
            mean_preservation=mean_preservation,
            pair_scores=pair_scores,
            violated_pairs=violated_pairs,
        )

    @staticmethod
    def co_activation_penalty(signature: EmotionConceptSignature) -> float:
        """
        Compute penalty for co-activating opposite emotions.

        Returns 0-1 where 0 is no penalty (clean opposition) and 1 is maximum
        penalty (all opposites highly co-activated).
        """
        id_to_value = dict(zip(signature.emotion_ids, signature.values))
        penalties = []

        for cat_a, cat_b in OPPOSITION_PAIRS:
            val_a = max(0.0, id_to_value.get(cat_a.value, 0.0))
            val_b = max(0.0, id_to_value.get(cat_b.value, 0.0))
            # Penalty is geometric mean of both activations
            # High penalty only if BOTH are active
            penalty = math.sqrt(val_a * val_b)
            penalties.append(penalty)

        return sum(penalties) / len(penalties) if penalties else 0.0


class EmotionConceptAtlas:
    """
    Embedding-based emotion concept analyzer.

    Maps text to emotion activation signatures using embedding similarity,
    following the same patterns as SemanticPrimeAtlas but with emotion-specific
    features like VAD projection and opposition analysis.

    Supports two representation modes:
    1. Centroid-based (default): Each emotion is a single embedding vector
    2. Volume-based (CABE-4): Each emotion is a ConceptVolume with centroid + covariance

    Volume-based representation enables:
    - More robust similarity via Mahalanobis distance
    - Interference prediction between emotions
    - Better handling of emotion concept variance
    """

    def __init__(
        self,
        embedder: EmbeddingProvider | None = None,
        configuration: EmotionAtlasConfiguration = EmotionAtlasConfiguration(),
        inventory: list[EmotionConcept] | None = None,
    ):
        self.config = configuration
        self.embedder = embedder
        self._cached_emotion_embeddings: list[list[float]] | None = None
        # Volume-based representation (CABE-4)
        self._cached_emotion_volumes: dict[str, "ConceptVolume"] | None = None
        self._density_estimator: "RiemannianDensityEstimator" | None = None
        if configuration.use_volume_representation and HAS_RIEMANNIAN:
            self._density_estimator = RiemannianDensityEstimator()

        # Build inventory based on configuration
        if inventory is not None:
            self.inventory = inventory
        else:
            emotions = EmotionConceptInventory.primary_emotions()
            if configuration.include_mild:
                emotions.extend(EmotionConceptInventory.mild_emotions())
            if configuration.include_intense:
                emotions.extend(EmotionConceptInventory.intense_emotions())
            self.inventory = emotions

        self.dyads = EmotionConceptInventory.primary_dyads() if configuration.include_dyads else []

    @property
    def emotions(self) -> list[EmotionConcept]:
        """Current emotion inventory."""
        return self.inventory

    async def signature(self, text: str) -> EmotionConceptSignature | None:
        """
        Compute emotion activation signature for text.

        Returns None if disabled or embedding fails.
        """
        if not self.config.enabled:
            return None

        trimmed = text.strip()
        if not trimmed:
            return None
        if self.embedder is None:
            return None

        try:
            emotion_embeddings = await self._get_or_create_emotion_embeddings()
            if len(emotion_embeddings) != len(self.inventory):
                return None

            capped = trimmed[: self.config.max_characters_per_text]
            embeddings = await self.embedder.embed([capped])
            if not embeddings:
                return None

            text_vec = VectorMath.l2_normalized(embeddings[0])

            # Compute similarities to each emotion
            similarities = []
            for emotion_vec in emotion_embeddings:
                dot = VectorMath.dot(emotion_vec, text_vec)
                similarities.append(max(0.0, dot))

            return EmotionConceptSignature(
                emotion_ids=[e.id for e in self.inventory],
                values=similarities,
                _inventory=self.inventory,
            )
        except Exception:
            return None

    async def analyze(
        self, text: str
    ) -> tuple[EmotionConceptSignature | None, EmotionActivationSummary]:
        """
        Full analysis of text emotion content.

        Returns signature and summary with top emotions, VAD projection,
        and opposition analysis.
        """
        sig = await self.signature(text)
        if not sig:
            return None, EmotionActivationSummary(
                method=EmotionActivationSummary.Method.SKIPPED,
                top_emotions=[],
                vad_projection=None,
                dominant_emotion=None,
                opposition_balances=None,
                normalized_activation_entropy=None,
                note="no_signature" if self.config.enabled else "disabled",
            )

        # Build summary
        scored = []
        for i, emotion in enumerate(self.inventory):
            scored.append(
                EmotionActivationSummary.EmotionScore(
                    emotion_id=emotion.id,
                    name=emotion.name,
                    similarity=sig.values[i],
                    category=emotion.category.value,
                )
            )

        scored.sort(key=lambda x: x.similarity, reverse=True)
        top_k = scored[: self.config.top_k]

        vad = sig.vad_projection()
        dominant = sig.dominant_emotion()
        opposition = sig.opposition_balance()
        entropy = self._normalized_entropy(sig.values)

        return sig, EmotionActivationSummary(
            method=EmotionActivationSummary.Method.EMBEDDINGS,
            top_emotions=top_k,
            vad_projection=vad,
            dominant_emotion=dominant,
            opposition_balances=opposition,
            normalized_activation_entropy=entropy,
            note=None,
        )

    def opposition_preservation_score(
        self,
        sig_a: EmotionConceptSignature,
        sig_b: EmotionConceptSignature,
    ) -> OppositionScore:
        """Compare opposition structure between two signatures."""
        return OppositionPreservationScorer.compute_score(sig_a, sig_b)

    def vad_distance(
        self,
        sig_a: EmotionConceptSignature,
        sig_b: EmotionConceptSignature,
    ) -> float:
        """Compute Euclidean distance between VAD projections."""
        vad_a = sig_a.vad_projection()
        vad_b = sig_b.vad_projection()
        return math.sqrt(
            (vad_a[0] - vad_b[0]) ** 2 + (vad_a[1] - vad_b[1]) ** 2 + (vad_a[2] - vad_b[2]) ** 2
        )

    async def _get_or_create_emotion_embeddings(self) -> list[list[float]]:
        """Get or create cached emotion embeddings using triangulation."""
        if self._cached_emotion_embeddings:
            return self._cached_emotion_embeddings
        if self.embedder is None:
            return []

        # Triangulate: embed all support texts and average
        # For now, simplified: embed "name: description" as centroid
        texts = [f"{e.name}: {e.description}" for e in self.inventory]
        embeddings = await self.embedder.embed(texts)

        normalized = [VectorMath.l2_normalized(vec) for vec in embeddings]
        self._cached_emotion_embeddings = normalized
        return normalized

    @staticmethod
    def _normalized_entropy(values: list[float]) -> float | None:
        """Compute normalized entropy of activation distribution."""
        clamped = [max(0.0, v) for v in values]
        total = sum(clamped)
        if total <= 0:
            return None

        probs = [v / total for v in clamped]
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log(p)

        n = max(1, len(probs))
        max_entropy = math.log(n)
        return entropy / max_entropy if max_entropy > 0 else None

    # =========================================================================
    # CABE-4: Volume-Based Emotion Representation
    # =========================================================================

    async def _get_or_create_emotion_volumes(self) -> dict[str, "ConceptVolume"]:
        """Create ConceptVolume representations for each emotion.

        Uses triangulated embeddings (name, description, support_texts)
        to estimate the volume each emotion occupies in embedding space.

        This addresses the simplified centroid logic by treating emotions
        as probability distributions rather than points.
        """
        if self._cached_emotion_volumes is not None:
            return self._cached_emotion_volumes

        if not HAS_RIEMANNIAN or self._density_estimator is None:
            return {}

        if self.embedder is None:
            return {}

        volumes: dict[str, ConceptVolume] = {}

        for emotion in self.inventory:
            # Collect all text representations of this emotion
            texts_for_emotion: list[str] = []

            # Core representation: Name + Description
            texts_for_emotion.append(f"{emotion.name}: {emotion.description}")

            # Add support texts if configured
            if self.config.include_support_texts_in_volume:
                for support in emotion.support_texts[:4]:  # Limit to 4 support texts
                    texts_for_emotion.append(f"{emotion.name}: {support}")

            # Need at least 2 samples for covariance estimation
            if len(texts_for_emotion) < 2:
                texts_for_emotion.append(f"The emotion of {emotion.name.lower()}")

            try:
                # Embed all texts for this emotion
                embeddings = await self.embedder.embed(texts_for_emotion)

                if len(embeddings) >= 2:
                    # Convert to numpy array
                    activations = np.array(embeddings)

                    # Estimate ConceptVolume
                    volume = self._density_estimator.estimate_concept_volume(
                        concept_id=emotion.id,
                        activations=activations,
                    )
                    volumes[emotion.id] = volume

            except Exception:
                # Fall back to centroid if volume estimation fails
                pass

        self._cached_emotion_volumes = volumes
        return volumes

    async def volume_similarity(
        self,
        text: str,
        use_mahalanobis: bool = True,
    ) -> EmotionConceptSignature | None:
        """Compute emotion signature using volume-aware similarity.

        Instead of simple cosine similarity to centroids, this uses:
        - Mahalanobis distance for probability-aware similarity
        - ConceptVolume membership testing

        Args:
            text: Text to analyze
            use_mahalanobis: Use Mahalanobis distance (True) or just centroid density (False)

        Returns:
            EmotionConceptSignature with volume-aware similarities
        """
        if not self.config.enabled or not HAS_RIEMANNIAN:
            return await self.signature(text)  # Fall back to centroid

        trimmed = text.strip()
        if not trimmed:
            return None

        if self.embedder is None:
            return None

        try:
            volumes = await self._get_or_create_emotion_volumes()
            if not volumes:
                return await self.signature(text)  # Fall back

            # Embed the input text
            capped = trimmed[: self.config.max_characters_per_text]
            embeddings = await self.embedder.embed([capped])
            if not embeddings:
                return None

            text_vec = np.array(embeddings[0])

            # Compute similarities using volume-aware metrics
            similarities = []
            for emotion in self.inventory:
                if emotion.id not in volumes:
                    similarities.append(0.0)
                    continue

                volume = volumes[emotion.id]

                if use_mahalanobis:
                    # Convert Mahalanobis distance to similarity
                    # Higher distance = lower similarity
                    mahal_dist = volume.mahalanobis_distance(text_vec)
                    # Use exponential decay: sim = exp(-dist/scale)
                    similarity = float(np.exp(-mahal_dist / 3.0))
                else:
                    # Use density at point as similarity
                    density = volume.density_at(text_vec)
                    # Normalize by density at centroid
                    max_density = volume.density_at(volume.centroid)
                    if max_density > 0:
                        similarity = float(density / max_density)
                    else:
                        similarity = 0.0

                similarities.append(max(0.0, similarity))

            return EmotionConceptSignature(
                emotion_ids=[e.id for e in self.inventory],
                values=similarities,
                _inventory=self.inventory,
            )

        except Exception:
            return await self.signature(text)  # Fall back on error

    def get_emotion_volumes(self) -> dict[str, "ConceptVolume"]:
        """Get cached emotion volumes (must call volume_similarity first to populate)."""
        return self._cached_emotion_volumes or {}

    async def compute_emotion_interference(
        self,
        emotion_id_a: str,
        emotion_id_b: str,
    ) -> dict | None:
        """Compute interference between two emotions using ConceptVolume analysis.

        Args:
            emotion_id_a: First emotion ID
            emotion_id_b: Second emotion ID

        Returns:
            Interference analysis dict or None if volumes not available
        """
        if not HAS_RIEMANNIAN or self._density_estimator is None:
            return None

        volumes = await self._get_or_create_emotion_volumes()
        if emotion_id_a not in volumes or emotion_id_b not in volumes:
            return None

        vol_a = volumes[emotion_id_a]
        vol_b = volumes[emotion_id_b]

        # Compute relation
        relation = self._density_estimator.compute_relation(vol_a, vol_b)

        # Get emotion names for reporting
        name_a = next((e.name for e in self.inventory if e.id == emotion_id_a), emotion_id_a)
        name_b = next((e.name for e in self.inventory if e.id == emotion_id_b), emotion_id_b)

        # Check if these are opposites
        is_opposite = any(
            (cat_a.value == emotion_id_a and cat_b.value == emotion_id_b)
            or (cat_a.value == emotion_id_b and cat_b.value == emotion_id_a)
            for cat_a, cat_b in OPPOSITION_PAIRS
        )

        return {
            "emotionA": {"id": emotion_id_a, "name": name_a},
            "emotionB": {"id": emotion_id_b, "name": name_b},
            "isOpposition": is_opposite,
            "bhattacharyyaCoefficient": float(relation.bhattacharyya_coefficient),
            "centroidDistance": float(relation.centroid_distance),
            "geodesicDistance": float(relation.geodesic_centroid_distance),
            "subspaceAlignment": float(relation.subspace_alignment),
            "overlapCoefficient": float(relation.overlap_coefficient),
            "interpretation": (
                f"Emotions {name_a} and {name_b}: "
                f"{'high' if relation.bhattacharyya_coefficient > 0.5 else 'low'} overlap, "
                f"{'aligned' if relation.subspace_alignment > 0.7 else 'divergent'} subspaces"
                f"{', opposite pair' if is_opposite else ''}"
            ),
        }
