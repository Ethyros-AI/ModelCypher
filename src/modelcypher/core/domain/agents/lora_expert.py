"""LoRA Expert types for Agent Cypher skill-based routing.

A LoRA adapter exposed as a skill expert for Agent Cypher workflows.
LoRAExpert bridges the LAP (LoRA Adapter Protocol) manifest system with
the agent orchestration layer.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol, runtime_checkable
from uuid import UUID


class SkillCategory(str, Enum):
    """Category of skill an adapter provides."""

    DOMAIN = "domain"
    """Domain-specific knowledge (e.g., medical, legal)."""

    TOOL = "tool"
    """Tool usage skills (e.g., API calling, code execution)."""

    REASONING = "reasoning"
    """Advanced reasoning capabilities."""

    KNOWLEDGE = "knowledge"
    """General knowledge enhancement."""

    STYLE = "style"
    """Output style and formatting."""

    BEHAVIOR = "behavior"
    """Behavioral modifications."""


class SkillComplexity(str, Enum):
    """Complexity level of a skill."""

    ATOMIC = "atomic"
    """Simple, single-step skills."""

    COMPOSITE = "composite"
    """Multi-step skills requiring coordination."""

    COMPREHENSIVE = "comprehensive"
    """Complex skills requiring deep integration."""


class AgentIntent(str, Enum):
    """Classified intent of an agent query."""

    CODE = "code"
    """Code-related tasks."""

    QUESTION = "question"
    """Question answering."""

    GENERATION = "generation"
    """Content generation."""

    ANALYSIS = "analysis"
    """Analysis tasks."""

    TOOL_USE = "tool_use"
    """Tool usage tasks."""

    CONVERSATION = "conversation"
    """Conversational tasks."""


@dataclass(frozen=True)
class AgentQuery:
    """Query for agent routing decisions."""

    text: str
    """The query text."""

    required_skill_tags: set[str] = field(default_factory=set)
    """Skill tags that must all match."""

    preferred_skill_tags: set[str] = field(default_factory=set)
    """Skill tags that are weighted higher."""

    required_category: Optional[SkillCategory] = None
    """Required skill category."""

    minimum_complexity: Optional[SkillComplexity] = None
    """Minimum skill complexity level."""

    intent: Optional[AgentIntent] = None
    """Classified intent of the query."""

    embedding: Optional[list[float]] = None
    """Geometric embedding for semantic routing."""


@runtime_checkable
class AdapterActivator(Protocol):
    """Protocol for activating/deactivating LoRA adapters on the inference layer."""

    async def activate_adapter(self, adapter_id: UUID, path: str) -> None:
        """Load and activate an adapter from the given path."""
        ...

    async def deactivate_adapter(self, adapter_id: UUID) -> None:
        """Deactivate an adapter by ID."""
        ...

    async def active_adapter_id(self) -> Optional[UUID]:
        """Return the ID of the currently active adapter, if any."""
        ...

    async def is_adapter_active(self, adapter_id: UUID) -> bool:
        """Check if a specific adapter is currently active."""
        ...


@runtime_checkable
class BlendedAdapterActivator(AdapterActivator, Protocol):
    """Extended protocol for activators that support geometric weight blending."""

    async def activate_blended_adapters(
        self,
        adapters: list[tuple[str, float]],
        base_model_id: str,
    ) -> UUID:
        """Load blended adapter weights from multiple sources."""
        ...

    async def deactivate_blended_adapter(self, ephemeral_id: UUID) -> None:
        """Deactivate an ephemeral blended adapter."""
        ...


class LoRAExpert(ABC):
    """A LoRA adapter exposed as a skill expert for Agent Cypher workflows.

    LoRAExpert bridges the LAP (LoRA Adapter Protocol) manifest system with
    the agent orchestration layer. Each expert wraps a trained adapter and provides:
    - Skill-based routing confidence (prior_confidence)
    - Adapter lifecycle management (activate/deactivate)
    - Metadata for composition and scheduling
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for this expert."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable display name."""
        ...

    @property
    @abstractmethod
    def skill_tags(self) -> set[str]:
        """Skill tags from the adapter manifest (for routing)."""
        ...

    @property
    @abstractmethod
    def skill_category(self) -> SkillCategory:
        """Skill category from the adapter manifest."""
        ...

    @property
    @abstractmethod
    def skill_complexity(self) -> SkillComplexity:
        """Skill complexity level."""
        ...

    @property
    @abstractmethod
    def adapter_id(self) -> UUID:
        """Reference to the underlying adapter manifest."""
        ...

    @property
    @abstractmethod
    def adapter_path(self) -> str:
        """Path to the adapter directory (PEFT checkpoint)."""
        ...

    @property
    @abstractmethod
    def base_model_id(self) -> str:
        """Base model this adapter was trained on."""
        ...

    @property
    @abstractmethod
    def embedding(self) -> Optional[list[float]]:
        """Semantic embedding of the expert's skills (for geometric routing)."""
        ...

    @abstractmethod
    def prior_confidence(self, query: AgentQuery) -> float:
        """Compute a cheap prior confidence (0.0-1.0) for routing.

        This is NOT a full inference - just heuristic matching based on:
        - Geometric alignment (embedding similarity)
        - Skill tag overlap with query requirements
        - Category alignment
        - Complexity appropriateness
        """
        ...

    @abstractmethod
    async def activate(self, activator: AdapterActivator) -> None:
        """Activate this adapter on the inference context."""
        ...

    @abstractmethod
    async def deactivate(self, activator: AdapterActivator) -> None:
        """Deactivate this adapter from the inference context."""
        ...

    @abstractmethod
    async def is_active(self, activator: AdapterActivator) -> bool:
        """Check if this adapter is currently active."""
        ...


@dataclass
class AdapterBackedLoRAExpert(LoRAExpert):
    """Concrete LoRAExpert implementation backed by adapter metadata.

    This is the primary implementation for production use, wrapping adapter
    metadata and providing skill-based routing.
    """

    _id: str
    _display_name: str
    _skill_tags: set[str]
    _skill_category: SkillCategory
    _skill_complexity: SkillComplexity
    _adapter_id: UUID
    _adapter_path: str
    _base_model_id: str
    _embedding: Optional[list[float]] = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def skill_tags(self) -> set[str]:
        return self._skill_tags

    @property
    def skill_category(self) -> SkillCategory:
        return self._skill_category

    @property
    def skill_complexity(self) -> SkillComplexity:
        return self._skill_complexity

    @property
    def adapter_id(self) -> UUID:
        return self._adapter_id

    @property
    def adapter_path(self) -> str:
        return self._adapter_path

    @property
    def base_model_id(self) -> str:
        return self._base_model_id

    @property
    def embedding(self) -> Optional[list[float]]:
        return self._embedding

    def prior_confidence(self, query: AgentQuery) -> float:
        """Compute prior confidence for routing."""
        score = 0.0
        components = 0.0

        # 1. Required skill tag matching (must match ALL)
        if query.required_skill_tags:
            matched_required = query.required_skill_tags.intersection(self._skill_tags)
            if len(matched_required) < len(query.required_skill_tags):
                return 0.0
            score += 0.4
            components += 1.0

        # 2. Preferred skill tag matching (weighted by overlap)
        if query.preferred_skill_tags:
            matched_preferred = query.preferred_skill_tags.intersection(self._skill_tags)
            overlap_ratio = len(matched_preferred) / len(query.preferred_skill_tags)
            score += 0.3 * overlap_ratio
            components += 1.0

        # 3. Category alignment
        if query.required_category is not None:
            if query.required_category == self._skill_category:
                score += 0.2
            else:
                score -= 0.3
            components += 1.0

        # 4. Complexity appropriateness
        if query.minimum_complexity is not None:
            if self._complexity_meets_minimum(self._skill_complexity, query.minimum_complexity):
                score += 0.1
            else:
                score -= 0.1
            components += 1.0

        # 5. Intent-based matching
        if query.intent is not None:
            intent_score = self._intent_category_alignment(query.intent, self._skill_category)
            score += 0.2 * intent_score
            components += 1.0

        # Heuristic Score (normalized)
        heuristic_score = max(0.0, min(1.0, score)) if components > 0 else 0.3

        # Geometric Routing
        if query.embedding is not None and self._embedding is not None:
            geometric_score = max(0.0, self._cosine_similarity(query.embedding, self._embedding))
            # Blend: 40% Geometry, 60% Metadata
            return (geometric_score * 0.4) + (heuristic_score * 0.6)

        return heuristic_score

    async def activate(self, activator: AdapterActivator) -> None:
        await activator.activate_adapter(self._adapter_id, self._adapter_path)

    async def deactivate(self, activator: AdapterActivator) -> None:
        await activator.deactivate_adapter(self._adapter_id)

    async def is_active(self, activator: AdapterActivator) -> bool:
        return await activator.is_adapter_active(self._adapter_id)

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b) or not a:
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _complexity_meets_minimum(
        self,
        complexity: SkillComplexity,
        minimum: SkillComplexity,
    ) -> bool:
        """Check if complexity meets minimum requirement."""
        order = [SkillComplexity.ATOMIC, SkillComplexity.COMPOSITE, SkillComplexity.COMPREHENSIVE]
        try:
            current_index = order.index(complexity)
            minimum_index = order.index(minimum)
            return current_index >= minimum_index
        except ValueError:
            return False

    def _intent_category_alignment(self, intent: AgentIntent, category: SkillCategory) -> float:
        """Compute alignment score between intent and category."""
        alignments: dict[AgentIntent, set[SkillCategory]] = {
            AgentIntent.CODE: {SkillCategory.DOMAIN, SkillCategory.TOOL, SkillCategory.REASONING},
            AgentIntent.QUESTION: {SkillCategory.KNOWLEDGE, SkillCategory.REASONING},
            AgentIntent.GENERATION: {SkillCategory.STYLE, SkillCategory.KNOWLEDGE},
            AgentIntent.ANALYSIS: {SkillCategory.REASONING, SkillCategory.KNOWLEDGE},
            AgentIntent.TOOL_USE: {SkillCategory.TOOL, SkillCategory.BEHAVIOR},
            AgentIntent.CONVERSATION: {SkillCategory.STYLE, SkillCategory.BEHAVIOR},
        }

        aligned = alignments.get(intent, set())
        return 1.0 if category in aligned else 0.0


@dataclass(frozen=True)
class LoRAExpertInfo:
    """Summary information about a registered LoRA expert."""

    id: str
    display_name: str
    adapter_id: UUID
    skill_tags: set[str]
    skill_category: SkillCategory
    skill_complexity: SkillComplexity
    base_model_id: str

    @classmethod
    def from_expert(cls, expert: LoRAExpert) -> LoRAExpertInfo:
        """Create info from a LoRA expert."""
        return cls(
            id=expert.id,
            display_name=expert.display_name,
            adapter_id=expert.adapter_id,
            skill_tags=expert.skill_tags,
            skill_category=expert.skill_category,
            skill_complexity=expert.skill_complexity,
            base_model_id=expert.base_model_id,
        )


@dataclass(frozen=True)
class LoRAExpertSelection:
    """Result of selecting a LoRA expert for a query."""

    expert_id: str
    """The selected expert's ID."""

    adapter_id: UUID
    """The adapter UUID."""

    confidence: float
    """Confidence score for this selection (0.0-1.0)."""

    weight: float
    """Routing weight for blending (for ensemble strategies)."""

    activation_reason: str
    """Human-readable reason for selection."""

    matched_tags: int
    """Number of skill tags that matched the query."""


@runtime_checkable
class LoRAExpertRegistry(Protocol):
    """Registry for managing LoRA experts available for agent workflows."""

    async def register(self, expert: LoRAExpert) -> None:
        """Register a LoRA expert for agent routing."""
        ...

    async def unregister(self, expert_id: str) -> None:
        """Unregister a LoRA expert by ID."""
        ...

    async def list_experts(self) -> list[LoRAExpertInfo]:
        """Return all registered experts."""
        ...

    async def find_experts(
        self,
        query: AgentQuery,
        limit: int,
    ) -> list[tuple[LoRAExpert, float]]:
        """Find experts matching a query, sorted by confidence (highest first)."""
        ...

    async def expert(self, adapter_id: UUID) -> Optional[LoRAExpert]:
        """Return a specific expert by adapter ID."""
        ...
