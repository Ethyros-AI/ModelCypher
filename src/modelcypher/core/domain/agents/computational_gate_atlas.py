"""
Computational Gate Atlas.

Embedding-based computational gates analyzer for adapter/model telemetry.
Unlike SemanticPrimeAtlas (linguistic), this probes fundamental programming
operations. More structurally rigid because gates are decomposable.

Ported from TrainingCypher/Domain/Agents/ComputationalGateAtlas.swift.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict
import asyncio
import random

from modelcypher.core.domain.geometry.vector_math import VectorMath
from modelcypher.ports.embedding import EmbeddingProvider


class ComputationalGateCategory(str, Enum):
    CORE_CONCEPTS = "Core Concepts"
    CONTROL_FLOW = "Control Flow"
    FUNCTIONS_SCOPING = "Functions & Scoping"
    DATA_TYPES = "Data & Types"
    DOMAIN_SPECIFIC = "Domain-Specific Paradigms"
    CONCURRENCY_PARALLELISM = "Concurrency & Parallelism"
    MEMORY_MANAGEMENT = "Memory Management"
    SYSTEM_IO = "System & I/O"
    MODULARITY = "Modularity"
    ERROR_HANDLING = "Error Handling"
    OBJECT_ORIENTED = "Object-Oriented Programming"
    METAPROGRAMMING = "Metaprogramming & Introspection"
    UNCATEGORIZED = "Uncategorized"
    COMPOSITE = "Composite"


@dataclass(frozen=True)
class ComputationalGate:
    """A computational prime from the CodeCypher Master Gates canonical inventory."""
    id: str
    position: int
    category: ComputationalGateCategory
    name: str
    description: str
    examples: List[str] = field(default_factory=list)
    polyglot_examples: List[str] = field(default_factory=list)
    decomposes_to: Optional[List[str]] = None

    @property
    def canonical_name(self) -> str:
        return self.name


class ComputationalGateInventory:
    """The 66 core computational gates from Master Gates Canonical."""

    @staticmethod
    def core_gates() -> List[ComputationalGate]:
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
    def composite_gates() -> List[ComputationalGate]:
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
    def all_gates() -> List[ComputationalGate]:
        return ComputationalGateInventory.core_gates() + ComputationalGateInventory.composite_gates()

    @staticmethod
    def probe_gates() -> List[ComputationalGate]:
        excluded = ["QUANTUM", "SYMBOLIC", "KNOWLEDGE", "DEPLOY", "SYSCALL"]
        return [g for g in ComputationalGateInventory.core_gates() if g.name not in excluded]


@dataclass
class ComputationalGateSignature:
    """A 66-dimensional 'gate activation' vector aligned to core gate order."""
    gate_ids: List[str]
    values: List[float]

    def cosine_similarity(self, other: "ComputationalGateSignature") -> Optional[float]:
        if self.gate_ids != other.gate_ids or len(self.values) != len(other.values):
            return None
        return VectorMath.cosine_similarity(self.values, other.values)

    def l2_normalized(self) -> "ComputationalGateSignature":
        norm = VectorMath.l2_norm(self.values)
        if norm > 0:
            return ComputationalGateSignature(
                self.gate_ids,
                [v / norm for v in self.values]
            )
        return self


@dataclass
class GateAtlasConfiguration:
    enabled: bool = True
    max_characters_per_text: int = 4096
    top_k: int = 8
    use_probe_subset: bool = True


class ComputationalGateAtlas:
    """Embedding-based computational gates analyzer."""

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
        self._cached_gate_embeddings: Optional[List[List[float]]] = None

    @property
    def gates(self) -> List[ComputationalGate]:
        return self.inventory

    async def signature(self, text: str) -> Optional[ComputationalGateSignature]:
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
    ) -> List[Tuple[ComputationalGate, str]]:
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

    async def _get_or_create_gate_embeddings(self) -> List[List[float]]:
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
