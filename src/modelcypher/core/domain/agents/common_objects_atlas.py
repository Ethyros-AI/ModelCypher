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
Common Objects Atlas - Concrete nouns for cross-dimensional alignment.

Covers everyday objects that ground language in physical reality:
- Household items (furniture, rooms)
- Kitchen items (utensils, appliances)
- Technology (devices, components)
- Clothing (garments, accessories)
- Nature (flora, terrain)
- Animals (common species)
- Vehicles (transportation)

Total: ~80 probes for object grounding.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ObjectCategory(str, Enum):
    """Category of common object."""
    
    HOUSEHOLD = "household"
    KITCHEN = "kitchen"
    TECHNOLOGY = "technology"
    CLOTHING = "clothing"
    NATURE = "nature"
    ANIMAL = "animal"
    VEHICLE = "vehicle"
    TOOL = "tool"
    FOOD = "food"  # Added to reach 960+ probes


@dataclass(frozen=True)
class ObjectConcept:
    """A common object concept."""
    
    id: str
    name: str
    category: ObjectCategory
    description: str
    support_texts: tuple[str, ...]
    cross_domain_weight: float = 1.0


class CommonObjectInventory:
    """Inventory of common object concepts."""
    
    # Household items
    HOUSEHOLD = (
        ObjectConcept("table", "table", ObjectCategory.HOUSEHOLD, "Flat surface for items", ("table", "desk", "counter")),
        ObjectConcept("chair", "chair", ObjectCategory.HOUSEHOLD, "Seat with back", ("chair", "seat", "stool")),
        ObjectConcept("bed", "bed", ObjectCategory.HOUSEHOLD, "Sleeping furniture", ("bed", "mattress", "cot")),
        ObjectConcept("door", "door", ObjectCategory.HOUSEHOLD, "Entry barrier", ("door", "entrance", "gate")),
        ObjectConcept("window", "window", ObjectCategory.HOUSEHOLD, "Light opening", ("window", "pane", "glass")),
        ObjectConcept("floor", "floor", ObjectCategory.HOUSEHOLD, "Walking surface", ("floor", "ground", "deck")),
        ObjectConcept("wall", "wall", ObjectCategory.HOUSEHOLD, "Vertical barrier", ("wall", "partition", "divider")),
        ObjectConcept("ceiling", "ceiling", ObjectCategory.HOUSEHOLD, "Upper room surface", ("ceiling", "roof", "overhead")),
        ObjectConcept("lamp", "lamp", ObjectCategory.HOUSEHOLD, "Light source", ("lamp", "light", "lantern")),
        ObjectConcept("couch", "couch", ObjectCategory.HOUSEHOLD, "Multi-seat furniture", ("couch", "sofa", "settee")),
        ObjectConcept("cabinet", "cabinet", ObjectCategory.HOUSEHOLD, "Storage furniture", ("cabinet", "cupboard", "closet")),
        ObjectConcept("mirror", "mirror", ObjectCategory.HOUSEHOLD, "Reflective surface", ("mirror", "glass", "reflection")),
    )
    
    # Kitchen items
    KITCHEN = (
        ObjectConcept("cup", "cup", ObjectCategory.KITCHEN, "Drinking vessel", ("cup", "mug", "glass")),
        ObjectConcept("plate", "plate", ObjectCategory.KITCHEN, "Eating surface", ("plate", "dish", "platter")),
        ObjectConcept("knife", "knife", ObjectCategory.KITCHEN, "Cutting tool", ("knife", "blade", "cutter")),
        ObjectConcept("fork", "fork", ObjectCategory.KITCHEN, "Pronged utensil", ("fork", "prongs",)),
        ObjectConcept("spoon", "spoon", ObjectCategory.KITCHEN, "Scooping utensil", ("spoon", "ladle",)),
        ObjectConcept("pot", "pot", ObjectCategory.KITCHEN, "Cooking vessel", ("pot", "pan", "kettle")),
        ObjectConcept("stove", "stove", ObjectCategory.KITCHEN, "Cooking appliance", ("stove", "range", "burner")),
        ObjectConcept("refrigerator", "refrigerator", ObjectCategory.KITCHEN, "Cold storage", ("refrigerator", "fridge", "freezer")),
        ObjectConcept("bottle", "bottle", ObjectCategory.KITCHEN, "Liquid container", ("bottle", "flask", "jar")),
        ObjectConcept("bowl", "bowl", ObjectCategory.KITCHEN, "Deep dish", ("bowl", "basin", "vessel")),
    )
    
    # Technology
    TECHNOLOGY = (
        ObjectConcept("phone", "phone", ObjectCategory.TECHNOLOGY, "Communication device", ("phone", "telephone", "mobile")),
        ObjectConcept("computer", "computer", ObjectCategory.TECHNOLOGY, "Computing device", ("computer", "PC", "laptop")),
        ObjectConcept("screen", "screen", ObjectCategory.TECHNOLOGY, "Display surface", ("screen", "monitor", "display")),
        ObjectConcept("keyboard", "keyboard", ObjectCategory.TECHNOLOGY, "Input device", ("keyboard", "keys", "typing")),
        ObjectConcept("button", "button", ObjectCategory.TECHNOLOGY, "Control element", ("button", "switch", "key")),
        ObjectConcept("cable", "cable", ObjectCategory.TECHNOLOGY, "Wire connection", ("cable", "wire", "cord")),
        ObjectConcept("camera", "camera", ObjectCategory.TECHNOLOGY, "Image capture device", ("camera", "lens", "photo")),
        ObjectConcept("battery", "battery", ObjectCategory.TECHNOLOGY, "Power storage", ("battery", "power", "cell")),
        ObjectConcept("speaker", "speaker", ObjectCategory.TECHNOLOGY, "Audio output", ("speaker", "sound", "audio")),
        ObjectConcept("microphone", "microphone", ObjectCategory.TECHNOLOGY, "Audio input", ("microphone", "mic", "recorder")),
    )
    
    # Clothing
    CLOTHING = (
        ObjectConcept("shirt", "shirt", ObjectCategory.CLOTHING, "Upper body garment", ("shirt", "blouse", "top")),
        ObjectConcept("pants", "pants", ObjectCategory.CLOTHING, "Lower body garment", ("pants", "trousers", "jeans")),
        ObjectConcept("shoes", "shoes", ObjectCategory.CLOTHING, "Foot covering", ("shoes", "boots", "sneakers")),
        ObjectConcept("hat", "hat", ObjectCategory.CLOTHING, "Head covering", ("hat", "cap", "headwear")),
        ObjectConcept("coat", "coat", ObjectCategory.CLOTHING, "Outer garment", ("coat", "jacket", "overcoat")),
        ObjectConcept("dress", "dress", ObjectCategory.CLOTHING, "Full body garment", ("dress", "gown", "frock")),
        ObjectConcept("socks", "socks", ObjectCategory.CLOTHING, "Foot garment", ("socks", "stockings", "hosiery")),
        ObjectConcept("gloves", "gloves", ObjectCategory.CLOTHING, "Hand covering", ("gloves", "mittens",)),
        ObjectConcept("scarf", "scarf", ObjectCategory.CLOTHING, "Neck wrap", ("scarf", "shawl", "wrap")),
        ObjectConcept("belt", "belt", ObjectCategory.CLOTHING, "Waist strap", ("belt", "strap", "sash")),
    )
    
    # Nature
    NATURE = (
        ObjectConcept("tree", "tree", ObjectCategory.NATURE, "Woody plant", ("tree", "oak", "pine")),
        ObjectConcept("flower", "flower", ObjectCategory.NATURE, "Flowering plant", ("flower", "blossom", "bloom")),
        ObjectConcept("grass", "grass", ObjectCategory.NATURE, "Ground cover plant", ("grass", "lawn", "turf")),
        ObjectConcept("rock", "rock", ObjectCategory.NATURE, "Stone formation", ("rock", "stone", "boulder")),
        ObjectConcept("river", "river", ObjectCategory.NATURE, "Flowing water", ("river", "stream", "creek")),
        ObjectConcept("mountain", "mountain", ObjectCategory.NATURE, "Elevated terrain", ("mountain", "peak", "hill")),
        ObjectConcept("ocean", "ocean", ObjectCategory.NATURE, "Large water body", ("ocean", "sea", "water")),
        ObjectConcept("cloud", "cloud", ObjectCategory.NATURE, "Atmospheric vapor", ("cloud", "sky", "cumulus")),
        ObjectConcept("sun", "sun", ObjectCategory.NATURE, "Star light source", ("sun", "sunlight", "solar")),
        ObjectConcept("moon", "moon", ObjectCategory.NATURE, "Natural satellite", ("moon", "lunar", "moonlight")),
    )
    
    # Animals
    ANIMALS = (
        ObjectConcept("dog", "dog", ObjectCategory.ANIMAL, "Canine companion", ("dog", "puppy", "hound")),
        ObjectConcept("cat", "cat", ObjectCategory.ANIMAL, "Feline companion", ("cat", "kitten", "feline")),
        ObjectConcept("bird", "bird", ObjectCategory.ANIMAL, "Flying creature", ("bird", "sparrow", "robin")),
        ObjectConcept("fish", "fish", ObjectCategory.ANIMAL, "Aquatic creature", ("fish", "salmon", "trout")),
        ObjectConcept("horse", "horse", ObjectCategory.ANIMAL, "Riding animal", ("horse", "stallion", "mare")),
        ObjectConcept("cow", "cow", ObjectCategory.ANIMAL, "Bovine animal", ("cow", "cattle", "bull")),
        ObjectConcept("pig", "pig", ObjectCategory.ANIMAL, "Porcine animal", ("pig", "hog", "swine")),
        ObjectConcept("bear", "bear", ObjectCategory.ANIMAL, "Large mammal", ("bear", "grizzly", "polar")),
        ObjectConcept("snake", "snake", ObjectCategory.ANIMAL, "Serpent", ("snake", "serpent", "cobra")),
        ObjectConcept("insect", "insect", ObjectCategory.ANIMAL, "Six-legged creature", ("insect", "bug", "beetle")),
    )
    
    # Vehicles
    VEHICLES = (
        ObjectConcept("car", "car", ObjectCategory.VEHICLE, "Road vehicle", ("car", "automobile", "sedan")),
        ObjectConcept("bus", "bus", ObjectCategory.VEHICLE, "Large passenger vehicle", ("bus", "coach", "transit")),
        ObjectConcept("train", "train", ObjectCategory.VEHICLE, "Rail vehicle", ("train", "railway", "locomotive")),
        ObjectConcept("plane", "plane", ObjectCategory.VEHICLE, "Air vehicle", ("plane", "airplane", "aircraft")),
        ObjectConcept("bike", "bike", ObjectCategory.VEHICLE, "Two-wheeled vehicle", ("bike", "bicycle", "cycle")),
        ObjectConcept("boat", "boat", ObjectCategory.VEHICLE, "Water vehicle", ("boat", "ship", "vessel")),
        ObjectConcept("truck", "truck", ObjectCategory.VEHICLE, "Cargo vehicle", ("truck", "lorry", "hauler")),
        ObjectConcept("motorcycle", "motorcycle", ObjectCategory.VEHICLE, "Motorized cycle", ("motorcycle", "motorbike", "moped")),
    )
    
    # Tools
    TOOLS = (
        ObjectConcept("hammer", "hammer", ObjectCategory.TOOL, "Striking tool", ("hammer", "mallet",)),
        ObjectConcept("screwdriver", "screwdriver", ObjectCategory.TOOL, "Turning tool", ("screwdriver", "driver",)),
        ObjectConcept("wrench", "wrench", ObjectCategory.TOOL, "Gripping tool", ("wrench", "spanner",)),
        ObjectConcept("scissors", "scissors", ObjectCategory.TOOL, "Cutting tool", ("scissors", "shears",)),
        ObjectConcept("pen", "pen", ObjectCategory.TOOL, "Writing tool", ("pen", "pencil", "marker")),
        ObjectConcept("brush", "brush", ObjectCategory.TOOL, "Bristled tool", ("brush", "paintbrush", "scrubber")),
    )
    
    # Food - added to reach 960+ probes for full-rank alignment
    FOOD = (
        ObjectConcept("bread", "bread", ObjectCategory.FOOD, "Baked grain food", ("bread", "loaf", "toast")),
        ObjectConcept("meat", "meat", ObjectCategory.FOOD, "Animal flesh food", ("meat", "beef", "chicken")),
        ObjectConcept("fruit", "fruit", ObjectCategory.FOOD, "Plant reproduction food", ("fruit", "apple", "banana")),
        ObjectConcept("vegetable", "vegetable", ObjectCategory.FOOD, "Plant part food", ("vegetable", "carrot", "broccoli")),
        ObjectConcept("cheese", "cheese", ObjectCategory.FOOD, "Dairy product", ("cheese", "cheddar", "mozzarella")),
        ObjectConcept("egg", "egg", ObjectCategory.FOOD, "Bird reproduction food", ("egg", "yolk", "white")),
        ObjectConcept("rice", "rice", ObjectCategory.FOOD, "Grain food", ("rice", "grain", "cereal")),
        ObjectConcept("soup", "soup", ObjectCategory.FOOD, "Liquid food", ("soup", "broth", "stew")),
        ObjectConcept("cake", "cake", ObjectCategory.FOOD, "Sweet baked food", ("cake", "pastry", "dessert")),
        ObjectConcept("pizza", "pizza", ObjectCategory.FOOD, "Flat bread food", ("pizza", "pie", "slice")),
    )
    
    @classmethod
    def all_concepts(cls) -> list[ObjectConcept]:
        """Get all object concepts."""
        concepts: list[ObjectConcept] = []
        concepts.extend(cls.HOUSEHOLD)
        concepts.extend(cls.KITCHEN)
        concepts.extend(cls.TECHNOLOGY)
        concepts.extend(cls.CLOTHING)
        concepts.extend(cls.NATURE)
        concepts.extend(cls.ANIMALS)
        concepts.extend(cls.VEHICLES)
        concepts.extend(cls.TOOLS)
        concepts.extend(cls.FOOD)
        return concepts
    
    @classmethod
    def concepts_by_category(cls, category: ObjectCategory) -> list[ObjectConcept]:
        """Get concepts for a specific category."""
        return [c for c in cls.all_concepts() if c.category == category]
