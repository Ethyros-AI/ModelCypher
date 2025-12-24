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
Computational Gate Atlas.

Embedding-based computational gates analyzer for adapter/model telemetry.
Unlike SemanticPrimeAtlas (linguistic), this probes fundamental programming
operations. More structurally rigid because gates are decomposable.

Ported from the reference Swift implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import asyncio
import random
import re

from modelcypher.core.domain.geometry.vector_math import VectorMath
from modelcypher.data import load_json
from modelcypher.ports.embedding import EmbeddingProvider

# Optional: Riemannian density for volume-based gate representation
try:
    import numpy as np
    from modelcypher.core.domain.geometry.riemannian_density import (
        ConceptVolume,
        RiemannianDensityEstimator,
        RiemannianDensityConfig,
    )
    HAS_RIEMANNIAN = True
except ImportError:
    HAS_RIEMANNIAN = False
    np = None
    ConceptVolume = None


class ComputationalGateCategory(str, Enum):
    core_concepts = "Core Concepts"
    control_flow = "Control Flow"
    functions_scoping = "Functions & Scoping"
    data_types = "Data & Types"
    domain_specific = "Domain-Specific Paradigms"
    concurrency_parallelism = "Concurrency & Parallelism"
    memory_management = "Memory Management"
    system_io = "System & I/O"
    modularity = "Modularity"
    error_handling = "Error Handling"
    object_oriented = "Object-Oriented Programming"
    metaprogramming = "Metaprogramming & Introspection"
    uncategorized = "Uncategorized"
    composite = "Composite"

    CORE_CONCEPTS = core_concepts
    CONTROL_FLOW = control_flow
    FUNCTIONS_SCOPING = functions_scoping
    DATA_TYPES = data_types
    DOMAIN_SPECIFIC = domain_specific
    CONCURRENCY_PARALLELISM = concurrency_parallelism
    MEMORY_MANAGEMENT = memory_management
    SYSTEM_IO = system_io
    MODULARITY = modularity
    ERROR_HANDLING = error_handling
    OBJECT_ORIENTED = object_oriented
    METAPROGRAMMING = metaprogramming
    UNCATEGORIZED = uncategorized
    COMPOSITE = composite


@dataclass(frozen=True)
class ComputationalGate:
    """A computational prime from the CodeCypher Master Gates canonical inventory."""
    id: str
    position: int
    category: ComputationalGateCategory
    name: str
    description: str
    examples: list[str] = field(default_factory=list)
    polyglot_examples: list[str] = field(default_factory=list)
    decomposes_to: list[str] | None = None

    @property
    def canonical_name(self) -> str:
        return self.name


class ComputationalGateInventory:
    """The 66 core computational gates from Master Gates Canonical."""

    @staticmethod
    def core_gates() -> list[ComputationalGate]:
        return [
            # Core Concepts
            ComputationalGate("1", 1, ComputationalGateCategory.CORE_CONCEPTS, "LITERAL",
                              "Direct, hardcoded value in source code",
                              ["42", "'hello'", "true"],
                              ["let x = 42; // Swift/Rust", "const int x = 42; // C++"]),
            ComputationalGate("4", 4, ComputationalGateCategory.CORE_CONCEPTS, "ASSIGNMENT",
                              "Binds a value to a named variable or storage location",
                              ["x = 10", "let y = 'hello'"],
                              ["var x = 10 // Swift", "let mut x = 10; // Rust"]),
            ComputationalGate("18", 18, ComputationalGateCategory.CORE_CONCEPTS, "COMMENT",
                              "Non-executable comment for human readers",
                              ["# This is a comment", "/* comment */"],
                              ["-- Haskell line comment", "; Lisp/Scheme comment"]),
            ComputationalGate("20", 20, ComputationalGateCategory.CORE_CONCEPTS, "MUTATION",
                              "In-place modification of existing data",
                              ["my_list.append(item)", "x += 1"],
                              ["items.append(x) // Swift", "vec.push(x); // Rust"]),
            ComputationalGate("38", 38, ComputationalGateCategory.CORE_CONCEPTS, "RETRIEVE",
                              "Retrieval of value from data structure",
                              ["my_dict['key']", "db.get(key)"],
                              ["dict[key] // Swift/Python", "Map.lookup key m -- Haskell Maybe"]),
            ComputationalGate("55", 55, ComputationalGateCategory.CORE_CONCEPTS, "NULL",
                              "Absence of a value, null/nil",
                              ["null", "nil"],
                              ["nil // Swift/Clojure", "None // Rust Option::None"]),

            # Control Flow
            ComputationalGate("2", 2, ComputationalGateCategory.CONTROL_FLOW, "CONDITIONAL",
                              "Controls execution flow based on boolean condition",
                              ["if (x > 10)", "switch (value)"],
                              ["if x > 10 { } // Swift/Rust", "match value { } // Rust"]),
            ComputationalGate("6", 6, ComputationalGateCategory.CONTROL_FLOW, "ITERATION",
                              "Repeated execution of a block of code",
                              ["for item in collection:", "while (i < 10)"],
                              ["for item in items { } // Swift/Rust", "while condition { } // Swift/Rust"]),
            ComputationalGate("35", 35, ComputationalGateCategory.CONTROL_FLOW, "AWAIT",
                              "Waiting for async operation without blocking",
                              ["await my_async_function()"],
                              ["await asyncFunc() // Swift/JS", "result.await // Rust"]),
            ComputationalGate("46", 46, ComputationalGateCategory.CONTROL_FLOW, "YIELD",
                              "Yielding value from generator, pausing execution",
                              ["yield 42"],
                              ["yield 42 // Python/JS", "yield return item; // C#"]),
            ComputationalGate("52", 52, ComputationalGateCategory.CONTROL_FLOW, "ESCAPE",
                              "Escapes normal control flow (break/continue)",
                              ["break", "continue"],
                              ["break // Swift/C++/Rust", "throw exit % Erlang"]),

            # Functions & Scoping
            ComputationalGate("3", 3, ComputationalGateCategory.FUNCTIONS_SCOPING, "INVOKE",
                              "Triggers execution of callable unit",
                              ["my_function(arg1, arg2)", "object.method()"],
                              ["myFunc(x, y) // Swift/C++", "(my-func x y) ; Lisp/Scheme"]),
            ComputationalGate("5", 5, ComputationalGateCategory.FUNCTIONS_SCOPING, "FUNCTION",
                              "Defines reusable, callable unit of code",
                              ["def my_func():", "function my_func() {}"],
                              ["func myFunc() { } // Swift", "fn my_func() { } // Rust"]),
            ComputationalGate("19", 19, ComputationalGateCategory.FUNCTIONS_SCOPING, "RETURN",
                              "Returns value from function, terminating execution",
                              ["return 42", "return result"],
                              ["return 42 // Swift", "return 42; // Rust/C++"]),
            ComputationalGate("21", 21, ComputationalGateCategory.FUNCTIONS_SCOPING, "BLOCK",
                              "Block of code forming single syntactic unit/scope",
                              ["{ ... }", "begin ... end"],
                              ["{ stmt1; stmt2 } // Swift/C++/Rust", "(progn s1 s2) ; Common Lisp"]),

            # Data & Types (subset for brevity, filling core ones)
            ComputationalGate("7", 7, ComputationalGateCategory.DATA_TYPES, "ARRAY",
                              "Ordered collection accessed by index",
                              ["[1, 2, 3]", "new int[10]"],
                              ["[1, 2, 3] // Swift/Python/Rust"]),
            ComputationalGate("16", 16, ComputationalGateCategory.DATA_TYPES, "TYPE",
                              "Definition of custom data type",
                              ["struct Point { x: int, y: int }", "type UserId = number"],
                              ["struct Point { x: Int, y: Int } // Swift"]),
            ComputationalGate("45", 45, ComputationalGateCategory.DATA_TYPES, "MAP",
                              "Unordered key-value pair collection",
                              ["{'key': 'value'}", "new Map()"],
                              ["[\"key\": \"value\"] // Swift Dictionary"]),
            ComputationalGate("53", 53, ComputationalGateCategory.DATA_TYPES, "ENUM",
                              "Definition of enumeration (named constants)",
                              ["enum Color { RED, GREEN, BLUE }"],
                              ["enum Color { case red, green, blue } // Swift"]),
            ComputationalGate("56", 56, ComputationalGateCategory.DATA_TYPES, "GENERIC",
                              "Generic type or function for multiple data types",
                              ["List<T>", "function my_func<T>(arg: T) {}"],
                              ["func identity<T>(_ x: T) -> T // Swift"]),
            ComputationalGate("62", 62, ComputationalGateCategory.DATA_TYPES, "TYPE_CHECK",
                              "Runtime type verification",
                              ["isinstance(my_object, MyClass)"],
                              ["x is Int // Swift", "isinstance(x, MyClass) // Python"]),
            ComputationalGate("63", 63, ComputationalGateCategory.DATA_TYPES, "CAST",
                              "Type conversion from one type to another",
                              ["(int)my_float", "my_string as number"],
                              ["x as! Int // Swift force cast"]),
            ComputationalGate("64", 64, ComputationalGateCategory.DATA_TYPES, "SERIALIZE",
                              "Convert data structure to storable/transmittable format",
                              ["JSON.stringify(my_object)", "pickle.dumps(my_object)"],
                              ["JSONEncoder().encode(obj) // Swift"]),

            # Domain Specific (subset)
            ComputationalGate("14", 14, ComputationalGateCategory.DOMAIN_SPECIFIC, "MATCH",
                              "Pattern matching against series of patterns",
                              ["match value:", "case (x, y):"],
                              ["switch value { case .a: ... } // Swift", "match value { P => E } // Rust"]),
            
            # Concurrency (subset)
            ComputationalGate("15", 15, ComputationalGateCategory.CONCURRENCY_PARALLELISM, "ASYNC",
                              "Asynchronous operation running in background",
                              ["async def my_func():", "async function my_func() {}"],
                              ["func myFunc() async { } // Swift", "async fn my_func() { } // Rust"]),
             ComputationalGate("65", 65, ComputationalGateCategory.CONCURRENCY_PARALLELISM, "CONCURRENT",
                              "Concurrent execution in overlapping time",
                              ["go my_func()", "new Thread().start()"],
                              ["async let x = f() // Swift"]),
             ComputationalGate("66", 66, ComputationalGateCategory.CONCURRENCY_PARALLELISM, "PARALLEL",
                              "Parallel execution on multiple processors",
                              ["my_array.par_iter()"],
                              ["items.parallelStream() // Java"]),

            # Memory (subset)
            ComputationalGate("39", 39, ComputationalGateCategory.MEMORY_MANAGEMENT, "ALLOCATION",
                              "Allocation of memory block",
                              ["malloc(size)", "new MyObject()"],
                              ["UnsafeMutablePointer<T>.allocate(capacity: n) // Swift"]),
            
            # System (subset)
            ComputationalGate("23", 23, ComputationalGateCategory.SYSTEM_IO, "INPUT",
                              "Reads input from external source",
                              ["read_line()", "input()"],
                              ["readLine() // Swift"]),
            ComputationalGate("24", 24, ComputationalGateCategory.SYSTEM_IO, "OUTPUT",
                              "Writes output to external destination",
                              ["print('hello')", "console.log('hello')"],
                              ["print(\"hello\") // Swift/Python"]),

            # Modularity
            ComputationalGate("22", 22, ComputationalGateCategory.MODULARITY, "MODULE",
                              "Module or namespace for organizing code",
                              ["package my_package", "module MyModule"],
                              ["import Foundation // Swift module"]),
            ComputationalGate("27", 27, ComputationalGateCategory.MODULARITY, "IMPORT",
                              "Importing code from another module",
                              ["import my_module", "require('my-module')"],
                              ["import Foundation // Swift"]),
            ComputationalGate("28", 28, ComputationalGateCategory.MODULARITY, "EXPORT",
                              "Making code available to other modules",
                              ["export my_function", "module.exports = my_function"],
                              ["public func myFunc() // Swift"]),

            # Error Handling
            ComputationalGate("12", 12, ComputationalGateCategory.ERROR_HANDLING, "ERROR",
                              "Explicit signaling of error condition",
                              ["throw new Exception()", "raise ValueError()"],
                              ["throw MyError.invalid // Swift"]),
            ComputationalGate("44", 44, ComputationalGateCategory.ERROR_HANDLING, "ASSERTION",
                              "Check that condition is true, error if false",
                              ["assert(x > 0)"],
                              ["assert(x > 0) // Swift"]),
             ComputationalGate("48", 48, ComputationalGateCategory.ERROR_HANDLING, "TRY",
                              "Start of exception-handling block",
                              ["try {"],
                              ["do { try expr } catch { } // Swift"]),
            ComputationalGate("49", 49, ComputationalGateCategory.ERROR_HANDLING, "CATCH",
                              "Block executed when exception caught",
                              ["} catch (e) {"],
                              ["catch { error in handle(error) } // Swift"]),
            
            # OOP
            ComputationalGate("13", 13, ComputationalGateCategory.OBJECT_ORIENTED, "CLASS",
                              "Defines class blueprint for objects",
                              ["class MyClass:", "public class MyClass {}"],
                              ["class MyClass { } // Swift"]),
            ComputationalGate("57", 57, ComputationalGateCategory.OBJECT_ORIENTED, "ATTRIBUTE",
                              "Attribute/decorator metadata on code",
                              ["@my_decorator", "[MyAttribute]"],
                              ["@MainActor // Swift attribute"]),

            # Metaprogramming (subset)
             ComputationalGate("11", 11, ComputationalGateCategory.METAPROGRAMMING, "MACRO",
                              "Macro or code generation at compile-time",
                              ["defmacro my_macro ...", "#[derive(Debug)]"],
                              ["@freestanding(expression) macro // Swift macro"]),

            # Uncategorized
            ComputationalGate("54", 54, ComputationalGateCategory.UNCATEGORIZED, "UNKNOWN",
                              "Unrecognized or unclassifiable pattern", [])
        ]

    # Added composite gates
    @staticmethod
    def composite_gates() -> list[ComputationalGate]:
        return [
            ComputationalGate("67", 67, ComputationalGateCategory.COMPOSITE, "LOCK_GATE",
                              "Acquire lock / enter critical section", [], [],
                              decomposes_to=["32_SYNCHRONIZATION", "20_MUTATION"]),
            ComputationalGate("68", 68, ComputationalGateCategory.COMPOSITE, "CONTRACT_GATE",
                              "Contract/interface declaration", [], [],
                              decomposes_to=["13_CLASS", "57_ATTRIBUTE", "5_FUNCTION"]),
            ComputationalGate("69", 69, ComputationalGateCategory.COMPOSITE, "INVARIANT_GATE",
                              "Type/value invariants that must always hold", [], [],
                              decomposes_to=["44_ASSERTION", "62_TYPE_CHECK"]),
             ComputationalGate("70", 70, ComputationalGateCategory.COMPOSITE, "ROLLBACK_GATE",
                              "Transactional rollback / state reversal", [], [],
                              decomposes_to=["36_TRANSACTION", "20_MUTATION", "30_RESOURCE"]),
             ComputationalGate("75", 75, ComputationalGateCategory.COMPOSITE, "DECISION_GATE",
                              "Binary decision point", [], [],
                              decomposes_to=["2_CONDITIONAL", "14_MATCH"]),
             ComputationalGate("76", 76, ComputationalGateCategory.COMPOSITE, "CONSENT_GATE",
                              "Explicit user authorization", [], [],
                              decomposes_to=["23_INPUT", "2_CONDITIONAL", "44_ASSERTION"])
        ]

    @staticmethod
    def all_gates() -> list[ComputationalGate]:
        return ComputationalGateInventory.core_gates() + ComputationalGateInventory.composite_gates()

    @staticmethod
    def probe_gates() -> list[ComputationalGate]:
        excluded = ["QUANTUM", "SYMBOLIC", "KNOWLEDGE", "DEPLOY", "SYSCALL"]
        return [g for g in ComputationalGateInventory.core_gates() if g.name not in excluded]

    _core_cache: list[ComputationalGate] | None = None
    _composite_cache: list[ComputationalGate] | None = None

    @staticmethod
    def _normalize_category(value: str) -> str:
        if not value:
            return "uncategorized"
        normalized = re.sub(r"(?<!^)([A-Z])", r"_\\1", value).lower()
        return normalized

    @classmethod
    def _parse_entries(cls, entries: list[dict]) -> list[ComputationalGate]:
        gates: list[ComputationalGate] = []
        for item in entries:
            category_key = cls._normalize_category(str(item.get("category", "")))
            category = ComputationalGateCategory.__members__.get(category_key, ComputationalGateCategory.uncategorized)
            decomposes = item.get("decomposesTo")
            decomposes_to = [str(value) for value in decomposes] if decomposes else None
            gates.append(
                ComputationalGate(
                    id=str(item.get("id", "")),
                    position=int(item.get("position", 0)),
                    category=category,
                    name=str(item.get("name", "")),
                    description=str(item.get("description", "")),
                    examples=[str(value) for value in item.get("examples", [])],
                    polyglot_examples=[str(value) for value in item.get("polyglotExamples", [])],
                    decomposes_to=decomposes_to,
                )
            )
        return gates

    @classmethod
    def _load_inventory(cls) -> tuple[list[ComputationalGate], list[ComputationalGate]]:
        payload = load_json("computational_gates.json")
        core = cls._parse_entries(payload.get("coreGates", []))
        composite = cls._parse_entries(payload.get("compositeGates", []))
        return core, composite

    @classmethod
    def core_gates(cls) -> list[ComputationalGate]:
        if cls._core_cache is None:
            cls._core_cache, cls._composite_cache = cls._load_inventory()
        return list(cls._core_cache)

    @classmethod
    def composite_gates(cls) -> list[ComputationalGate]:
        if cls._composite_cache is None:
            cls._core_cache, cls._composite_cache = cls._load_inventory()
        return list(cls._composite_cache)

    @classmethod
    def all_gates(cls) -> list[ComputationalGate]:
        return cls.core_gates() + cls.composite_gates()

    @classmethod
    def probe_gates(cls) -> list[ComputationalGate]:
        excluded = {"QUANTUM", "SYMBOLIC", "KNOWLEDGE", "DEPLOY", "SYSCALL"}
        return [gate for gate in cls.core_gates() if gate.name not in excluded]


@dataclass
class ComputationalGateSignature:
    """A 66-dimensional 'gate activation' vector aligned to core gate order."""
    gate_ids: list[str]
    values: list[float]

    def cosine_similarity(self, other: "ComputationalGateSignature") -> float | None:
        if self.gate_ids != other.gate_ids or len(self.values) != len(other.values):
            return None
        return VectorMath.cosine_similarity(self.values, other.values)

    def l2_normalized(self) -> "ComputationalGateSignature":
        """Return L2-normalized copy of this signature."""
        normalized = VectorMath.l2_normalized(self.values)
        return ComputationalGateSignature(self.gate_ids, normalized)


@dataclass
class GateAtlasConfiguration:
    enabled: bool = True
    max_characters_per_text: int = 4096
    top_k: int = 8
    use_probe_subset: bool = True
    # Volume-based representation (CABE-4: Riemannian density)
    use_volume_representation: bool = False
    # Include examples in volume estimation (more accurate but slower)
    include_examples_in_volume: bool = True


class ComputationalGateAtlas:
    """Embedding-based computational gates analyzer.

    Supports two representation modes:
    1. Centroid-based (default): Each gate is a single embedding vector
    2. Volume-based (CABE-4): Each gate is a ConceptVolume with centroid + covariance

    Volume-based representation enables:
    - More robust similarity via Mahalanobis distance
    - Interference prediction between gates
    - Better handling of concept variance
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        configuration: GateAtlasConfiguration = GateAtlasConfiguration()
    ):
        self.config = configuration
        self.inventory = (
            ComputationalGateInventory.probe_gates()
            if configuration.use_probe_subset
            else ComputationalGateInventory.core_gates()
        )
        self.embedder = embedder
        self._cached_gate_embeddings: list[list[float]] | None = None
        # Volume-based representation (CABE-4)
        self._cached_gate_volumes: dict[str, "ConceptVolume"] | None = None
        self._density_estimator: "RiemannianDensityEstimator" | None = None
        if configuration.use_volume_representation and HAS_RIEMANNIAN:
            self._density_estimator = RiemannianDensityEstimator()

    @property
    def gates(self) -> list[ComputationalGate]:
        return self.inventory

    async def signature(self, text: str) -> ComputationalGateSignature | None:
        if not self.config.enabled:
            return None

        trimmed = text.strip()
        if not trimmed:
            return None

        try:
            gate_embeddings = await self._get_or_create_gate_embeddings()
            if len(gate_embeddings) != len(self.inventory):
                return None
            
            capped = trimmed[:self.config.max_characters_per_text]
            embeddings = await self.embedder.embed([capped])
            if not embeddings:
                return None

            text_vec = VectorMath.l2_normalized(embeddings[0])
            
            similarities = []
            for gate_vec in gate_embeddings:
                dot = VectorMath.dot(gate_vec, text_vec)
                similarities.append(max(0.0, dot))
            
            return ComputationalGateSignature(
                gate_ids=[g.id for g in self.inventory],
                values=similarities
            )
        except Exception:
            return None

    # MARK: - Prompt Generation

    class PromptStyle(Enum):
        COMPLETION = "completion"
        EXPLANATION = "explanation"
        EXAMPLE = "example"

    @staticmethod
    def generate_probe_prompts(
        style: PromptStyle = PromptStyle.COMPLETION,
        subset_name: str = "probe"
    ) -> list[tuple[ComputationalGate, str]]:
        gates = []
        if subset_name == "probe":
            gates = ComputationalGateInventory.probe_gates()
        elif subset_name == "core":
            gates = ComputationalGateInventory.core_gates()
        else:
            gates = ComputationalGateInventory.all_gates()

        results = []
        for gate in gates:
            prompt = ""
            if style == ComputationalGateAtlas.PromptStyle.COMPLETION:
                prompt = ComputationalGateAtlas._generate_completion_prompt(gate)
            elif style == ComputationalGateAtlas.PromptStyle.EXPLANATION:
                prompt = f"Explain what the following programming concept does: {gate.name}. It is described as: {gate.description}"
            elif style == ComputationalGateAtlas.PromptStyle.EXAMPLE:
                ex = gate.examples[0] if gate.examples else ""
                if ex:
                    prompt = f"Here is an example of {gate.name} in code:\n{ex}\n\nThis demonstrates"
                else:
                    prompt = f"Here is an example of {gate.name}:\n"
            results.append((gate, prompt))
        return results

    @staticmethod
    def _generate_completion_prompt(gate: ComputationalGate) -> str:
        name = gate.name 
        if name == "LITERAL": return "# Define a constant value\nMY_CONSTANT = "
        if name == "CONDITIONAL": return "# Check if value is valid\nif "
        if name == "INVOKE": return "def process(data):\n    return data.transform()\n\nresult = "
        if name == "ASSIGNMENT": return "# Store the computed result\ntotal = "
        if name == "FUNCTION": return "# Define a helper function\ndef "
        if name == "ITERATION": return "# Process each item\nfor item in "
        if name == "ARRAY": return "# Create a list of values\nvalues = "
        if name == "ASYNC": return "# Define async operation\nasync def "
        if name == "RETURN": return "def calculate(x):\n    result = x * 2\n    return "
        if name == "MUTATION": return "items = [1, 2, 3]\nitems."
        if name == "BLOCK": return "with resource as r:\n    "
        if name == "MODULE": return "# Import utility module\nimport "
        if name == "INPUT": return "# Read user input\nuser_input = "
        if name == "OUTPUT": return "# Print result\nprint("
        if name == "CLASS": return "# Define a data class\nclass "
        if name == "TYPE_CHECK": return "# Verify type at runtime\nif isinstance("
        if name == "ERROR": return "# Raise validation error\nif not valid:\n    raise "
        if name == "TRY": return "# Handle potential error\ntry:\n    "
        if name == "CATCH": return "except Exception as e:\n    "
        if name == "MAP": return "# Create key-value mapping\nconfig = "
        
        return f"# Implement {gate.name.lower().replace('_', ' ')}\n"

    async def _get_or_create_gate_embeddings(self) -> list[list[float]]:
        if self._cached_gate_embeddings:
            return self._cached_gate_embeddings

        # Triangulation logic: definition + examples + polyglot
        texts_to_embed = []
        gate_indices = [] # map index in flattened list back to gate

        # Just embed name + description for simplicity in this port first version
        # to ensure 1-1 mapping easily. 
        # Ideally we'd do the full centroid logic from Swift.
        # Let's do a simplified centroid: Embed (Name + Description) as the representational vector.
        
        descriptions = [f"{g.name}: {g.description}" for g in self.inventory]
        embeddings = await self.embedder.embed(descriptions)

        normalized = [VectorMath.l2_normalized(vec) for vec in embeddings]
        self._cached_gate_embeddings = normalized
        return normalized

    # =========================================================================
    # CABE-4: Volume-Based Gate Representation
    # =========================================================================

    async def _get_or_create_gate_volumes(self) -> dict[str, "ConceptVolume"]:
        """Create ConceptVolume representations for each gate.

        Uses triangulated embeddings (name, description, examples, polyglot)
        to estimate the volume each gate occupies in embedding space.

        This addresses the simplified centroid logic by treating gates
        as probability distributions rather than points.
        """
        if self._cached_gate_volumes is not None:
            return self._cached_gate_volumes

        if not HAS_RIEMANNIAN or self._density_estimator is None:
            return {}

        volumes: dict[str, ConceptVolume] = {}

        for gate in self.inventory:
            # Collect all text representations of this gate
            texts_for_gate: list[str] = []

            # Core representation: Name + Description
            texts_for_gate.append(f"{gate.name}: {gate.description}")

            # Add examples if configured
            if self.config.include_examples_in_volume:
                for example in gate.examples[:3]:  # Limit to 3 examples
                    texts_for_gate.append(f"{gate.name} example: {example}")

                for polyglot in gate.polyglot_examples[:2]:  # Limit to 2 polyglot
                    texts_for_gate.append(f"{gate.name} code: {polyglot}")

            # Need at least 2 samples for covariance estimation
            if len(texts_for_gate) < 2:
                texts_for_gate.append(f"The {gate.name} programming concept")

            try:
                # Embed all texts for this gate
                embeddings = await self.embedder.embed(texts_for_gate)

                if len(embeddings) >= 2:
                    # Convert to numpy array
                    activations = np.array(embeddings)

                    # Estimate ConceptVolume
                    volume = self._density_estimator.estimate_concept_volume(
                        concept_id=gate.id,
                        activations=activations,
                    )
                    volumes[gate.id] = volume

            except Exception as e:
                # Fall back to centroid if volume estimation fails
                pass

        self._cached_gate_volumes = volumes
        return volumes

    async def volume_similarity(
        self,
        text: str,
        use_mahalanobis: bool = True,
    ) -> ComputationalGateSignature | None:
        """Compute gate signature using volume-aware similarity.

        Instead of simple cosine similarity to centroids, this uses:
        - Mahalanobis distance for probability-aware similarity
        - ConceptVolume membership testing

        Args:
            text: Text to analyze
            use_mahalanobis: Use Mahalanobis distance (True) or just centroid density (False)

        Returns:
            ComputationalGateSignature with volume-aware similarities
        """
        if not self.config.enabled or not HAS_RIEMANNIAN:
            return await self.signature(text)  # Fall back to centroid

        trimmed = text.strip()
        if not trimmed:
            return None

        try:
            volumes = await self._get_or_create_gate_volumes()
            if not volumes:
                return await self.signature(text)  # Fall back

            # Embed the input text
            capped = trimmed[:self.config.max_characters_per_text]
            embeddings = await self.embedder.embed([capped])
            if not embeddings:
                return None

            text_vec = np.array(embeddings[0])

            # Compute similarities using volume-aware metrics
            similarities = []
            for gate in self.inventory:
                if gate.id not in volumes:
                    similarities.append(0.0)
                    continue

                volume = volumes[gate.id]

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

            return ComputationalGateSignature(
                gate_ids=[g.id for g in self.inventory],
                values=similarities
            )

        except Exception:
            return await self.signature(text)  # Fall back on error

    def get_gate_volumes(self) -> dict[str, "ConceptVolume"]:
        """Get cached gate volumes (must call volume_similarity first to populate)."""
        return self._cached_gate_volumes or {}

    async def compute_gate_interference(
        self,
        gate_id_a: str,
        gate_id_b: str,
    ) -> Dict | None:
        """Compute interference between two gates using ConceptVolume analysis.

        Args:
            gate_id_a: First gate ID
            gate_id_b: Second gate ID

        Returns:
            Interference analysis dict or None if volumes not available
        """
        if not HAS_RIEMANNIAN or self._density_estimator is None:
            return None

        volumes = await self._get_or_create_gate_volumes()
        if gate_id_a not in volumes or gate_id_b not in volumes:
            return None

        vol_a = volumes[gate_id_a]
        vol_b = volumes[gate_id_b]

        # Compute relation
        relation = self._density_estimator.compute_relation(vol_a, vol_b)

        # Get gate names for reporting
        name_a = next((g.name for g in self.inventory if g.id == gate_id_a), gate_id_a)
        name_b = next((g.name for g in self.inventory if g.id == gate_id_b), gate_id_b)

        return {
            "gateA": {"id": gate_id_a, "name": name_a},
            "gateB": {"id": gate_id_b, "name": name_b},
            "bhattacharyyaCoefficient": float(relation.bhattacharyya_coefficient),
            "centroidDistance": float(relation.centroid_distance),
            "geodesicDistance": float(relation.geodesic_centroid_distance),
            "subspaceAlignment": float(relation.subspace_alignment),
            "overlapCoefficient": float(relation.overlap_coefficient),
            "interpretation": (
                f"Gates {name_a} and {name_b}: "
                f"{'high' if relation.bhattacharyya_coefficient > 0.5 else 'low'} overlap, "
                f"{'aligned' if relation.subspace_alignment > 0.7 else 'divergent'} subspaces"
            ),
        }
