"""Cross-cultural semantic anchors that triangulate meaning through diverse idioms.

While SequenceInvariant aligns the mathematical/logical backbone,
MetaphorInvariant aligns the cultural/semantic soul.

The Hypothesis: If a model understands the *concept* of "Futility", the vectors for
"Carrying coals to Newcastle" (English) and "Taking owls to Athens" (Greek)
must collapse to a single point in deep semantic layers, despite orthogonal surface forms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MetaphorFamily(str, Enum):
    """Family of metaphor invariants."""

    FUTILITY = "futility"
    """Redundancy, useless effort."""

    IMPOSSIBILITY = "impossibility"
    """Things that cannot happen."""

    OBVIOUSNESS = "obviousness"
    """Missing what is right in front of you."""

    CONSEQUENCE = "consequence"
    """Cause and effect, reaping what you sow."""

    FRAGILITY = "fragility"
    """Weakness, instability."""

    DECEPTION = "deception"
    """Things are not what they seem."""

    RESILIENCE = "resilience"
    """Strength through adversity."""


@dataclass(frozen=True)
class CulturalExpression:
    """A cultural expression of a universal concept."""

    language: str
    """Language of the expression."""

    phrase: str
    """The idiomatic phrase."""

    literal_meaning: str
    """Literal translation/meaning."""

    cultural_context: str
    """Cultural context explaining the origin."""


@dataclass(frozen=True)
class MetaphorInvariant:
    """Cross-cultural semantic anchor that triangulates meaning through diverse idioms."""

    id: str
    """Unique identifier."""

    family: MetaphorFamily
    """Metaphor family."""

    universal_concept: str
    """The universal concept being expressed."""

    variations: list[CulturalExpression] = field(default_factory=list)
    """Cultural variations of the concept."""


class MetaphorInvariantInventory:
    """Inventory of metaphor invariant probes."""

    # MARK: - Futility / Redundancy

    FUTILITY_PROBES = MetaphorInvariant(
        id="meta_futility",
        family=MetaphorFamily.FUTILITY,
        universal_concept="Performing a useless action by supplying something already abundant.",
        variations=[
            CulturalExpression(
                language="English",
                phrase="Carrying coals to Newcastle",
                literal_meaning="Transporting coal to a major coal-mining city",
                cultural_context="Newcastle was the UK's primary coal exporter.",
            ),
            CulturalExpression(
                language="Greek (Ancient)",
                phrase="Taking owls to Athens",
                literal_meaning="Bringing an owl to a city full of them",
                cultural_context="The owl was the symbol of Athena; Athens minted coins with owls.",
            ),
            CulturalExpression(
                language="Russian",
                phrase="Going to Tula with your own samovar",
                literal_meaning="Bringing a tea urn to the city that manufactures them",
                cultural_context="Tula is the historic center of samovar production.",
            ),
            CulturalExpression(
                language="German",
                phrase="Carrying water to the Rhine",
                literal_meaning="Adding water to a massive river",
                cultural_context="The Rhine is one of Europe's largest rivers.",
            ),
            CulturalExpression(
                language="Spanish",
                phrase="Selling honey to the beekeeper",
                literal_meaning="Offering honey to someone who produces it",
                cultural_context="Agricultural redundancy.",
            ),
            CulturalExpression(
                language="Chinese",
                phrase="多此一举",
                literal_meaning="An unnecessary extra action",
                cultural_context="Used to call out redundancy.",
            ),
            CulturalExpression(
                language="Symbolic",
                phrase="A + 0 = A",
                literal_meaning="Adding zero changes nothing",
                cultural_context="Algebraic redundancy.",
            ),
        ],
    )

    FUTILITY_SCALE_PROBES = MetaphorInvariant(
        id="meta_futility_thimble",
        family=MetaphorFamily.FUTILITY,
        universal_concept="A futile effort when the task dwarfs the available tools.",
        variations=[
            CulturalExpression(
                language="English",
                phrase="Bailing out the ocean with a thimble",
                literal_meaning="Trying to empty the sea with a tiny cup",
                cultural_context="Scale mismatch makes the effort useless.",
            ),
            CulturalExpression(
                language="Symbolic",
                phrase="tiny_tool << infinite_task",
                literal_meaning="Tool scale far below task scale",
                cultural_context="Symbolic mismatch.",
            ),
        ],
    )

    # MARK: - Impossibility

    IMPOSSIBILITY_PROBES = MetaphorInvariant(
        id="meta_impossibility",
        family=MetaphorFamily.IMPOSSIBILITY,
        universal_concept="An event that will never occur.",
        variations=[
            CulturalExpression(
                language="English",
                phrase="When pigs fly",
                literal_meaning="At the moment pigs grow wings",
                cultural_context="Absurd biological impossibility.",
            ),
            CulturalExpression(
                language="Spanish",
                phrase="Cuando las ranas críen pelo",
                literal_meaning="When frogs grow hair",
                cultural_context="Biological absurdity used to denote 'never'.",
            ),
            CulturalExpression(
                language="Russian",
                phrase="When the crayfish whistles on the mountain",
                literal_meaning="A water creature whistling on a peak",
                cultural_context="Double impossibility (location + action).",
            ),
            CulturalExpression(
                language="French",
                phrase="Quand les poules auront des dents",
                literal_meaning="When hens have teeth",
                cultural_context="Absurd biological impossibility.",
            ),
            CulturalExpression(
                language="Chinese",
                phrase="太阳从西边出来",
                literal_meaning="The sun rises from the west",
                cultural_context="Cosmic impossibility.",
            ),
            CulturalExpression(
                language="Symbolic",
                phrase="0 = 1",
                literal_meaning="A contradiction that can never be true",
                cultural_context="Formal impossibility.",
            ),
        ],
    )

    IMPOSSIBILITY_CIRCLE_PROBES = MetaphorInvariant(
        id="meta_impossibility_square_circle",
        family=MetaphorFamily.IMPOSSIBILITY,
        universal_concept="An unsolvable or contradictory task.",
        variations=[
            CulturalExpression(
                language="English",
                phrase="Squaring the circle",
                literal_meaning="Making a circle and a square have equal area using only a compass",
                cultural_context="Classic geometric impossibility.",
            ),
            CulturalExpression(
                language="Symbolic",
                phrase="circle -> square",
                literal_meaning="Transforming incompatible shapes",
                cultural_context="Symbolic impossibility.",
            ),
        ],
    )

    # MARK: - Obviousness / Perspective

    OBVIOUSNESS_PROBES = MetaphorInvariant(
        id="meta_obviousness",
        family=MetaphorFamily.OBVIOUSNESS,
        universal_concept="Failing to see the whole situation because of focus on details.",
        variations=[
            CulturalExpression(
                language="English",
                phrase="Can't see the forest for the trees",
                literal_meaning="Trees obstructing the view of the forest",
                cultural_context="Focus on detail obscuring the gestalt.",
            ),
            CulturalExpression(
                language="French",
                phrase="L'arbre qui cache la forêt",
                literal_meaning="The tree that hides the forest",
                cultural_context="A single problem obscuring the larger reality.",
            ),
            CulturalExpression(
                language="Japanese",
                phrase="Ki o mite mori o mizu",
                literal_meaning="Seeing the trees but not the forest",
                cultural_context="Direct translation of the cognitive failure.",
            ),
            CulturalExpression(
                language="Chinese",
                phrase="坐井观天",
                literal_meaning="Viewing the sky from the bottom of a well",
                cultural_context="Limited perspective hides the whole.",
            ),
            CulturalExpression(
                language="Symbolic",
                phrase="detail >> context",
                literal_meaning="Local focus dominates global view",
                cultural_context="Symbolic imbalance of scale.",
            ),
        ],
    )

    OBVIOUSNESS_ELEPHANT_PROBES = MetaphorInvariant(
        id="meta_obviousness_elephant",
        family=MetaphorFamily.OBVIOUSNESS,
        universal_concept="Ignoring a glaring issue that everyone can see.",
        variations=[
            CulturalExpression(
                language="English",
                phrase="The elephant in the room",
                literal_meaning="A large animal everyone pretends not to notice",
                cultural_context="Shared avoidance of an obvious issue.",
            ),
            CulturalExpression(
                language="Symbolic",
                phrase="visible_problem != discussed",
                literal_meaning="Something obvious is ignored",
                cultural_context="Symbolic mismatch.",
            ),
        ],
    )

    # MARK: - Consequence / Karma

    CONSEQUENCE_PROBES = MetaphorInvariant(
        id="meta_consequence",
        family=MetaphorFamily.CONSEQUENCE,
        universal_concept="You create your own future conditions through present actions.",
        variations=[
            CulturalExpression(
                language="English",
                phrase="You reap what you sow",
                literal_meaning="Harvest matches the seed",
                cultural_context="Agricultural causality.",
            ),
            CulturalExpression(
                language="Italian",
                phrase="Chi semina vento raccoglie tempesta",
                literal_meaning="He who sows wind harvests a storm",
                cultural_context="Escalation of bad actions.",
            ),
            CulturalExpression(
                language="Japanese",
                phrase="Jigō jitoku",
                literal_meaning="Self-act, self-profit/loss",
                cultural_context="Buddhist concept of karma.",
            ),
            CulturalExpression(
                language="Chinese",
                phrase="种瓜得瓜，种豆得豆",
                literal_meaning="Plant melons, get melons; plant beans, get beans",
                cultural_context="Agricultural causality.",
            ),
            CulturalExpression(
                language="Symbolic",
                phrase="cause -> effect",
                literal_meaning="Inputs determine outputs",
                cultural_context="Formal causality mapping.",
            ),
        ],
    )

    CONSEQUENCE_ROOST_PROBES = MetaphorInvariant(
        id="meta_consequence_roost",
        family=MetaphorFamily.CONSEQUENCE,
        universal_concept="Past actions return as consequences over time.",
        variations=[
            CulturalExpression(
                language="English",
                phrase="Chickens come home to roost",
                literal_meaning="Birds return to their roost at night",
                cultural_context="Consequences return to the actor.",
            ),
            CulturalExpression(
                language="Symbolic",
                phrase="action -> delayed return",
                literal_meaning="Consequences loop back later",
                cultural_context="Symbolic causality.",
            ),
        ],
    )

    # MARK: - Fragility / Instability

    FRAGILITY_CARDS_PROBES = MetaphorInvariant(
        id="meta_fragility_cards",
        family=MetaphorFamily.FRAGILITY,
        universal_concept="A fragile structure that collapses from small disturbances.",
        variations=[
            CulturalExpression(
                language="English",
                phrase="A house of cards",
                literal_meaning="A structure made of stacked playing cards",
                cultural_context="Minor disruption causes collapse.",
            ),
            CulturalExpression(
                language="Symbolic",
                phrase="small_shock -> collapse",
                literal_meaning="Low disturbance triggers failure",
                cultural_context="Symbolic fragility.",
            ),
        ],
    )

    FRAGILITY_SAND_PROBES = MetaphorInvariant(
        id="meta_fragility_sand",
        family=MetaphorFamily.FRAGILITY,
        universal_concept="An unstable foundation that cannot support weight.",
        variations=[
            CulturalExpression(
                language="English",
                phrase="Built on sand",
                literal_meaning="A foundation resting on loose sand",
                cultural_context="Structures fail without a stable base.",
            ),
            CulturalExpression(
                language="Symbolic",
                phrase="weak_base -> failure",
                literal_meaning="Unstable foundation causes collapse",
                cultural_context="Symbolic fragility.",
            ),
        ],
    )

    # MARK: - Deception / Appearance

    DECEPTION_WOLF_PROBES = MetaphorInvariant(
        id="meta_deception_wolf",
        family=MetaphorFamily.DECEPTION,
        universal_concept="A threat disguised as something harmless.",
        variations=[
            CulturalExpression(
                language="English",
                phrase="A wolf in sheep's clothing",
                literal_meaning="A predator disguised as prey",
                cultural_context="Deception through appearance.",
            ),
            CulturalExpression(
                language="Chinese",
                phrase="披着羊皮的狼",
                literal_meaning="Wolf wearing sheep skin",
                cultural_context="Literal gloss.",
            ),
            CulturalExpression(
                language="Symbolic",
                phrase="appearance != intent",
                literal_meaning="Surface form hides true motive",
                cultural_context="Symbolic deception.",
            ),
        ],
    )

    DECEPTION_TROJAN_PROBES = MetaphorInvariant(
        id="meta_deception_trojan",
        family=MetaphorFamily.DECEPTION,
        universal_concept="A gift or offer that hides a threat.",
        variations=[
            CulturalExpression(
                language="English",
                phrase="A Trojan horse",
                literal_meaning="A gift that hides attackers inside",
                cultural_context="Hidden danger within an attractive offer.",
            ),
            CulturalExpression(
                language="Symbolic",
                phrase="gift contains threat",
                literal_meaning="Hidden payload inside a wrapper",
                cultural_context="Symbolic deception.",
            ),
        ],
    )

    # MARK: - Resilience / Recovery

    RESILIENCE_BAMBOO_PROBES = MetaphorInvariant(
        id="meta_resilience_bamboo",
        family=MetaphorFamily.RESILIENCE,
        universal_concept="Strength through flexibility under pressure.",
        variations=[
            CulturalExpression(
                language="English",
                phrase="Bamboo bends in the storm",
                literal_meaning="Flexible stalks survive strong winds",
                cultural_context="Flexibility prevents breaking.",
            ),
            CulturalExpression(
                language="Chinese",
                phrase="风中竹子弯而不断",
                literal_meaning="Bamboo bends in the wind without breaking",
                cultural_context="Literal gloss.",
            ),
            CulturalExpression(
                language="Symbolic",
                phrase="flexibility -> survival",
                literal_meaning="Adaptive strength resists failure",
                cultural_context="Symbolic resilience.",
            ),
        ],
    )

    RESILIENCE_PHOENIX_PROBES = MetaphorInvariant(
        id="meta_resilience_phoenix",
        family=MetaphorFamily.RESILIENCE,
        universal_concept="Rebirth or recovery after destruction.",
        variations=[
            CulturalExpression(
                language="English",
                phrase="Rise like a phoenix from the ashes",
                literal_meaning="A bird reborn after burning",
                cultural_context="Recovery and renewal after loss.",
            ),
            CulturalExpression(
                language="Chinese",
                phrase="凤凰涅槃",
                literal_meaning="Phoenix reborn through fire",
                cultural_context="Literal gloss.",
            ),
            CulturalExpression(
                language="Symbolic",
                phrase="loss -> rebirth",
                literal_meaning="Failure precedes renewal",
                cultural_context="Symbolic resilience.",
            ),
        ],
    )

    # All probes
    ALL_PROBES: list[MetaphorInvariant] = [
        FUTILITY_PROBES,
        FUTILITY_SCALE_PROBES,
        IMPOSSIBILITY_PROBES,
        IMPOSSIBILITY_CIRCLE_PROBES,
        OBVIOUSNESS_PROBES,
        OBVIOUSNESS_ELEPHANT_PROBES,
        CONSEQUENCE_PROBES,
        CONSEQUENCE_ROOST_PROBES,
        FRAGILITY_CARDS_PROBES,
        FRAGILITY_SAND_PROBES,
        DECEPTION_WOLF_PROBES,
        DECEPTION_TROJAN_PROBES,
        RESILIENCE_BAMBOO_PROBES,
        RESILIENCE_PHOENIX_PROBES,
    ]

    @classmethod
    def probes_by_family(cls, family: MetaphorFamily) -> list[MetaphorInvariant]:
        """Get all probes for a given family."""
        return [p for p in cls.ALL_PROBES if p.family == family]
