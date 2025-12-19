from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from modelcypher.adapters.embedding_defaults import EmbeddingDefaults
from modelcypher.core.domain.geometry import VectorMath
from modelcypher.data import load_json
from modelcypher.ports.embedding import EmbeddingProvider
from modelcypher.utils.text import truncate


class ComputationalGateCategory(str, Enum):
    core_concepts = "coreConcepts"
    control_flow = "controlFlow"
    functions_scoping = "functionsScoping"
    data_types = "dataTypes"
    domain_specific = "domainSpecific"
    concurrency_parallelism = "concurrencyParallelism"
    memory_management = "memoryManagement"
    system_io = "systemIO"
    modularity = "modularity"
    error_handling = "errorHandling"
    object_oriented = "objectOriented"
    metaprogramming = "metaprogramming"
    uncategorized = "uncategorized"
    composite = "composite"


@dataclass(frozen=True)
class ComputationalGate:
    id: str
    position: int
    category: ComputationalGateCategory
    name: str
    description: str
    examples: list[str]
    polyglot_examples: list[str]
    decomposes_to: list[str] | None = None

    @property
    def canonical_name(self) -> str:
        return self.name


class ComputationalGateInventory:
    _core_gates: list[ComputationalGate] | None = None
    _composite_gates: list[ComputationalGate] | None = None

    @classmethod
    def core_gates(cls) -> list[ComputationalGate]:
        if cls._core_gates is None:
            data = load_json("computational_gates.json")
            cls._core_gates = _load_gates(data.get("coreGates", []))
        return list(cls._core_gates)

    @classmethod
    def composite_gates(cls) -> list[ComputationalGate]:
        if cls._composite_gates is None:
            data = load_json("computational_gates.json")
            cls._composite_gates = _load_gates(data.get("compositeGates", []))
        return list(cls._composite_gates)

    @classmethod
    def all_gates(cls) -> list[ComputationalGate]:
        return cls.core_gates() + cls.composite_gates()

    @classmethod
    def probe_gates(cls) -> list[ComputationalGate]:
        excluded = {"QUANTUM", "SYMBOLIC", "KNOWLEDGE", "DEPLOY", "SYSCALL"}
        return [gate for gate in cls.core_gates() if gate.name not in excluded]


@dataclass(frozen=True)
class ComputationalGateSignature:
    gate_ids: list[str]
    values: list[float]

    def cosine_similarity(self, other: ComputationalGateSignature) -> float | None:
        if self.gate_ids != other.gate_ids or len(self.values) != len(other.values):
            return None
        return VectorMath.cosine_similarity(self.values, other.values)

    @staticmethod
    def mean(signatures: list[ComputationalGateSignature]) -> ComputationalGateSignature | None:
        if not signatures:
            return None
        first = signatures[0]
        if not all(sig.gate_ids == first.gate_ids and len(sig.values) == len(first.values) for sig in signatures):
            return None
        summed = [0.0] * len(first.values)
        for signature in signatures:
            for idx, value in enumerate(signature.values):
                summed[idx] += float(value)
        inv_count = 1.0 / float(len(signatures))
        mean = [value * inv_count for value in summed]
        return ComputationalGateSignature(gate_ids=first.gate_ids, values=mean).l2_normalized()

    def l2_normalized(self) -> ComputationalGateSignature:
        norm = VectorMath.l2_norm(self.values)
        if not norm or norm <= 0:
            return self
        return ComputationalGateSignature(
            gate_ids=self.gate_ids,
            values=[float(value) / norm for value in self.values],
        )


class PromptStyle(str, Enum):
    completion = "completion"
    explanation = "explanation"
    example = "example"


class GateSubset(str, Enum):
    probe = "probe"
    core = "core"
    all = "all"


@dataclass(frozen=True)
class ComputationalGateAtlasConfig:
    enabled: bool = True
    max_characters_per_text: int = 4096
    top_k: int = 8
    use_probe_subset: bool = True


class ComputationalGateAtlas:
    def __init__(
        self,
        configuration: ComputationalGateAtlasConfig | None = None,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self._config = configuration or ComputationalGateAtlasConfig()
        self._inventory = (
            ComputationalGateInventory.probe_gates()
            if self._config.use_probe_subset
            else ComputationalGateInventory.core_gates()
        )
        self._embedder = embedder if embedder is not None else EmbeddingDefaults.make_default_embedder()
        self._cached_gate_embeddings: list[list[float]] | None = None

    @property
    def gates(self) -> list[ComputationalGate]:
        return list(self._inventory)

    def signature(self, text: str) -> ComputationalGateSignature | None:
        if not self._config.enabled:
            return None
        trimmed = text.strip()
        if not trimmed or self._embedder is None:
            return None
        try:
            gate_embeddings = self._get_or_create_gate_embeddings()
            if len(gate_embeddings) != len(self._inventory):
                return None
            capped = truncate(trimmed, self._config.max_characters_per_text)
            embedded = self._embedder.embed([capped])
            if not embedded:
                return None
            text_embedding = VectorMath.l2_normalized([float(v) for v in embedded[0]])
            similarities = [
                max(0.0, VectorMath.dot(gate_vector, text_embedding) or 0.0)
                for gate_vector in gate_embeddings
            ]
            return ComputationalGateSignature(
                gate_ids=[gate.id for gate in self._inventory],
                values=similarities,
            )
        except Exception:
            return None

    @staticmethod
    def generate_probe_prompts(style: PromptStyle = PromptStyle.completion, subset: GateSubset = GateSubset.probe) -> list[tuple[ComputationalGate, str]]:
        if subset == GateSubset.core:
            gates = ComputationalGateInventory.core_gates()
        elif subset == GateSubset.all:
            gates = ComputationalGateInventory.all_gates()
        else:
            gates = ComputationalGateInventory.probe_gates()
        return [(gate, _prompt_for_gate(gate, style)) for gate in gates]

    def _get_or_create_gate_embeddings(self) -> list[list[float]]:
        if self._cached_gate_embeddings is not None:
            return self._cached_gate_embeddings
        if self._embedder is None:
            return []

        embeddings: list[list[float]] = []
        for gate in self._inventory:
            texts = [f"{gate.name}: {gate.description}"]
            texts.extend(gate.examples)
            texts.extend(gate.polyglot_examples)
            texts = [text for text in (text.strip() for text in texts) if text]

            if texts:
                vecs = self._embedder.embed(texts)
                if vecs:
                    embeddings.append(_centroid(vecs))
                    continue

            if embeddings:
                embeddings.append([0.0 for _ in range(len(embeddings[0]))])
            else:
                embeddings.append([])

        self._cached_gate_embeddings = embeddings
        return embeddings


def _centroid(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        return []
    dimension = len(embeddings[0])
    summed = [0.0] * dimension
    for vec in embeddings:
        for idx, value in enumerate(vec):
            summed[idx] += float(value)
    return VectorMath.l2_normalized(summed)


def _prompt_for_gate(gate: ComputationalGate, style: PromptStyle) -> str:
    if style == PromptStyle.explanation:
        return f"Explain what the following programming concept does: {gate.name}. It is described as: {gate.description}"
    if style == PromptStyle.example:
        if gate.examples:
            return f"Here is an example of {gate.name} in code:\\n{gate.examples[0]}\\n\\nThis demonstrates"
        return f"Here is an example of {gate.name}:\\n"
    return _completion_prompt(gate)


def _completion_prompt(gate: ComputationalGate) -> str:
    if gate.name == "LITERAL":
        return "# Define a constant value\nMY_CONSTANT = "
    if gate.name == "CONDITIONAL":
        return "# Check if value is valid\nif "
    if gate.name == "INVOKE":
        return "def process(data):\n    return data.transform()\n\nresult = "
    if gate.name == "ASSIGNMENT":
        return "# Store the computed result\ntotal = "
    if gate.name == "FUNCTION":
        return "# Define a helper function\ndef "
    if gate.name == "ITERATION":
        return "# Process each item\nfor item in "
    if gate.name == "ARRAY":
        return "# Create a list of values\nvalues = "
    if gate.name == "ASYNC":
        return "# Define async operation\nasync def "
    if gate.name == "RETURN":
        return "def calculate(x):\n    result = x * 2\n    return "
    if gate.name == "MUTATION":
        return "items = [1, 2, 3]\nitems."
    if gate.name == "BLOCK":
        return "with resource as r:\n    "
    if gate.name == "MODULE":
        return "# Import utility module\nimport "
    if gate.name == "INPUT":
        return "# Read user input\nuser_input = "
    if gate.name == "OUTPUT":
        return "# Print result\nprint("
    if gate.name == "CLASS":
        return "# Define a data class\nclass "
    if gate.name == "TYPE_CHECK":
        return "# Verify type at runtime\nif isinstance("
    if gate.name == "ERROR":
        return "# Raise validation error\nif not valid:\n    raise "
    if gate.name == "TRY":
        return "# Handle potential error\ntry:\n    "
    if gate.name == "CATCH":
        return "except Exception as e:\n    "
    if gate.name == "MAP":
        return "# Create key-value mapping\nconfig = "
    return f"# Implement {gate.name.lower().replace('_', ' ')}\n"


def _load_gates(items: list[dict]) -> list[ComputationalGate]:
    gates: list[ComputationalGate] = []
    for item in items:
        gates.append(
            ComputationalGate(
                id=str(item["id"]),
                position=int(item["position"]),
                category=ComputationalGateCategory(str(item["category"])),
                name=str(item["name"]),
                description=str(item["description"]),
                examples=[str(value) for value in item.get("examples", [])],
                polyglot_examples=[str(value) for value in item.get("polyglotExamples", [])],
                decomposes_to=[str(value) for value in item.get("decomposesTo", [])]
                if item.get("decomposesTo") is not None
                else None,
            )
        )
    return gates
