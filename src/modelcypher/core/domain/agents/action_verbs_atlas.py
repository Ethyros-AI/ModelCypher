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
Action Verbs Atlas - Movement and interaction verbs for cross-dimensional alignment.

Covers action verbs that express physical and mental activities:
- Movement (walk, run, jump)
- Manipulation (grab, hold, push)
- Creation (make, build, create)
- Destruction (break, destroy, tear)
- Transformation (change, grow, shrink)
- Communication (speak, listen, ask)
- Cognition (think, remember, learn)

Total: ~70 probes for action grounding.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionCategory(str, Enum):
    """Category of action verb."""
    
    MOVEMENT = "movement"
    MANIPULATION = "manipulation"
    CREATION = "creation"
    DESTRUCTION = "destruction"
    TRANSFORMATION = "transformation"
    COMMUNICATION = "communication"
    COGNITION = "cognition"


@dataclass(frozen=True)
class ActionConcept:
    """An action verb concept."""
    
    id: str
    name: str
    category: ActionCategory
    description: str
    support_texts: tuple[str, ...]
    cross_domain_weight: float = 1.0


class ActionVerbInventory:
    """Inventory of action verb concepts."""
    
    # Movement verbs
    MOVEMENT = (
        ActionConcept("walk", "walk", ActionCategory.MOVEMENT, "Move on foot", ("walk", "walking", "stroll")),
        ActionConcept("run", "run", ActionCategory.MOVEMENT, "Move quickly on foot", ("run", "running", "sprint")),
        ActionConcept("jump", "jump", ActionCategory.MOVEMENT, "Move up from ground", ("jump", "jumping", "leap")),
        ActionConcept("climb", "climb", ActionCategory.MOVEMENT, "Move upward", ("climb", "climbing", "ascend")),
        ActionConcept("fall", "fall", ActionCategory.MOVEMENT, "Move downward uncontrolled", ("fall", "falling", "drop")),
        ActionConcept("fly", "fly", ActionCategory.MOVEMENT, "Move through air", ("fly", "flying", "soar")),
        ActionConcept("swim", "swim", ActionCategory.MOVEMENT, "Move through water", ("swim", "swimming", "paddle")),
        ActionConcept("crawl", "crawl", ActionCategory.MOVEMENT, "Move on hands/knees", ("crawl", "crawling", "creep")),
        ActionConcept("stand", "stand", ActionCategory.MOVEMENT, "Upright position", ("stand", "standing", "upright")),
        ActionConcept("sit", "sit", ActionCategory.MOVEMENT, "Seated position", ("sit", "sitting", "seated")),
    )
    
    # Manipulation verbs
    MANIPULATION = (
        ActionConcept("grab", "grab", ActionCategory.MANIPULATION, "Take quickly", ("grab", "grabbing", "seize")),
        ActionConcept("hold", "hold", ActionCategory.MANIPULATION, "Keep in hand", ("hold", "holding", "grasp")),
        ActionConcept("push", "push", ActionCategory.MANIPULATION, "Apply force away", ("push", "pushing", "shove")),
        ActionConcept("pull", "pull", ActionCategory.MANIPULATION, "Apply force toward", ("pull", "pulling", "tug")),
        ActionConcept("lift", "lift", ActionCategory.MANIPULATION, "Raise upward", ("lift", "lifting", "raise")),
        ActionConcept("drop", "drop", ActionCategory.MANIPULATION, "Release downward", ("drop", "dropping", "release")),
        ActionConcept("throw", "throw", ActionCategory.MANIPULATION, "Propel through air", ("throw", "throwing", "toss")),
        ActionConcept("catch", "catch", ActionCategory.MANIPULATION, "Receive in motion", ("catch", "catching", "capture")),
        ActionConcept("touch", "touch", ActionCategory.MANIPULATION, "Contact lightly", ("touch", "touching", "feel")),
        ActionConcept("squeeze", "squeeze", ActionCategory.MANIPULATION, "Apply pressure", ("squeeze", "squeezing", "compress")),
    )
    
    # Creation verbs
    CREATION = (
        ActionConcept("make", "make", ActionCategory.CREATION, "Produce something", ("make", "making", "create")),
        ActionConcept("build", "build", ActionCategory.CREATION, "Construct structure", ("build", "building", "construct")),
        ActionConcept("create", "create", ActionCategory.CREATION, "Bring into existence", ("create", "creating", "originate")),
        ActionConcept("design", "design", ActionCategory.CREATION, "Plan appearance", ("design", "designing", "plan")),
        ActionConcept("draw", "draw", ActionCategory.CREATION, "Create image", ("draw", "drawing", "sketch")),
        ActionConcept("write", "write", ActionCategory.CREATION, "Create text", ("write", "writing", "compose")),
        ActionConcept("cook", "cook", ActionCategory.CREATION, "Prepare food", ("cook", "cooking", "prepare")),
        ActionConcept("grow", "grow", ActionCategory.CREATION, "Cultivate", ("grow", "growing", "cultivate")),
        ActionConcept("invent", "invent", ActionCategory.CREATION, "Create new thing", ("invent", "inventing", "devise")),
        ActionConcept("compose", "compose", ActionCategory.CREATION, "Create music/text", ("compose", "composing", "write")),
    )
    
    # Destruction verbs
    DESTRUCTION = (
        ActionConcept("break", "break", ActionCategory.DESTRUCTION, "Separate into pieces", ("break", "breaking", "shatter")),
        ActionConcept("destroy", "destroy", ActionCategory.DESTRUCTION, "End existence", ("destroy", "destroying", "demolish")),
        ActionConcept("tear", "tear", ActionCategory.DESTRUCTION, "Rip apart", ("tear", "tearing", "rip")),
        ActionConcept("cut", "cut", ActionCategory.DESTRUCTION, "Divide with blade", ("cut", "cutting", "slice")),
        ActionConcept("burn", "burn", ActionCategory.DESTRUCTION, "Consume with fire", ("burn", "burning", "ignite")),
        ActionConcept("crush", "crush", ActionCategory.DESTRUCTION, "Compress destructively", ("crush", "crushing", "smash")),
        ActionConcept("erase", "erase", ActionCategory.DESTRUCTION, "Remove marks", ("erase", "erasing", "delete")),
        ActionConcept("kill", "kill", ActionCategory.DESTRUCTION, "End life", ("kill", "killing", "slay")),
        ActionConcept("remove", "remove", ActionCategory.DESTRUCTION, "Take away", ("remove", "removing", "eliminate")),
        ActionConcept("dissolve", "dissolve", ActionCategory.DESTRUCTION, "Disperse in liquid", ("dissolve", "dissolving", "melt")),
    )
    
    # Transformation verbs
    TRANSFORMATION = (
        ActionConcept("change", "change", ActionCategory.TRANSFORMATION, "Become different", ("change", "changing", "alter")),
        ActionConcept("transform", "transform", ActionCategory.TRANSFORMATION, "Complete change", ("transform", "transforming", "convert")),
        ActionConcept("shrink", "shrink", ActionCategory.TRANSFORMATION, "Become smaller", ("shrink", "shrinking", "reduce")),
        ActionConcept("expand", "expand", ActionCategory.TRANSFORMATION, "Become larger", ("expand", "expanding", "grow")),
        ActionConcept("melt", "melt", ActionCategory.TRANSFORMATION, "Solid to liquid", ("melt", "melting", "liquefy")),
        ActionConcept("freeze", "freeze", ActionCategory.TRANSFORMATION, "Liquid to solid", ("freeze", "freezing", "solidify")),
        ActionConcept("bend", "bend", ActionCategory.TRANSFORMATION, "Change shape", ("bend", "bending", "curve")),
        ActionConcept("stretch", "stretch", ActionCategory.TRANSFORMATION, "Extend length", ("stretch", "stretching", "elongate")),
        ActionConcept("rotate", "rotate", ActionCategory.TRANSFORMATION, "Turn around axis", ("rotate", "rotating", "spin")),
        ActionConcept("mix", "mix", ActionCategory.TRANSFORMATION, "Combine substances", ("mix", "mixing", "blend")),
    )
    
    # Communication verbs
    COMMUNICATION = (
        ActionConcept("speak", "speak", ActionCategory.COMMUNICATION, "Produce speech", ("speak", "speaking", "talk")),
        ActionConcept("listen", "listen", ActionCategory.COMMUNICATION, "Perceive sound", ("listen", "listening", "hear")),
        ActionConcept("ask", "ask", ActionCategory.COMMUNICATION, "Request information", ("ask", "asking", "question")),
        ActionConcept("answer", "answer", ActionCategory.COMMUNICATION, "Provide response", ("answer", "answering", "reply")),
        ActionConcept("explain", "explain", ActionCategory.COMMUNICATION, "Make clear", ("explain", "explaining", "clarify")),
        ActionConcept("describe", "describe", ActionCategory.COMMUNICATION, "Tell appearance", ("describe", "describing", "depict")),
        ActionConcept("read", "read", ActionCategory.COMMUNICATION, "Interpret text", ("read", "reading", "peruse")),
        ActionConcept("shout", "shout", ActionCategory.COMMUNICATION, "Speak loudly", ("shout", "shouting", "yell")),
        ActionConcept("whisper", "whisper", ActionCategory.COMMUNICATION, "Speak quietly", ("whisper", "whispering", "murmur")),
        ActionConcept("sing", "sing", ActionCategory.COMMUNICATION, "Produce music vocally", ("sing", "singing", "chant")),
    )
    
    # Cognition verbs (mental actions)
    COGNITION = (
        ActionConcept("think", "think", ActionCategory.COGNITION, "Mental processing", ("think", "thinking", "ponder")),
        ActionConcept("remember", "remember", ActionCategory.COGNITION, "Recall past", ("remember", "remembering", "recall")),
        ActionConcept("forget", "forget", ActionCategory.COGNITION, "Lose memory", ("forget", "forgetting", "overlook")),
        ActionConcept("learn", "learn", ActionCategory.COGNITION, "Acquire knowledge", ("learn", "learning", "study")),
        ActionConcept("understand", "understand", ActionCategory.COGNITION, "Comprehend meaning", ("understand", "understanding", "grasp")),
        ActionConcept("decide", "decide", ActionCategory.COGNITION, "Make choice", ("decide", "deciding", "choose")),
        ActionConcept("imagine", "imagine", ActionCategory.COGNITION, "Form mental image", ("imagine", "imagining", "visualize")),
        ActionConcept("believe", "believe", ActionCategory.COGNITION, "Accept as true", ("believe", "believing", "trust")),
        ActionConcept("doubt", "doubt", ActionCategory.COGNITION, "Question truth", ("doubt", "doubting", "question")),
        ActionConcept("focus", "focus", ActionCategory.COGNITION, "Concentrate attention", ("focus", "focusing", "concentrate")),
    )
    
    @classmethod
    def all_concepts(cls) -> list[ActionConcept]:
        """Get all action concepts."""
        concepts: list[ActionConcept] = []
        concepts.extend(cls.MOVEMENT)
        concepts.extend(cls.MANIPULATION)
        concepts.extend(cls.CREATION)
        concepts.extend(cls.DESTRUCTION)
        concepts.extend(cls.TRANSFORMATION)
        concepts.extend(cls.COMMUNICATION)
        concepts.extend(cls.COGNITION)
        return concepts
    
    @classmethod
    def concepts_by_category(cls, category: ActionCategory) -> list[ActionConcept]:
        """Get concepts for a specific category."""
        return [c for c in cls.all_concepts() if c.category == category]
