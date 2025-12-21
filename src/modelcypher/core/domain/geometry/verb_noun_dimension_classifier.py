"""
Verb/Noun Dimension Classifier.

Classifies embedding dimensions as Verb (skill/trajectory) or Noun (knowledge/position).
Implements the geometric equivalent of the "Verb vs Noun" training data philosophy.

Ported 1:1 from TrainingCypher/Domain/Geometry/VerbNounDimensionClassifier.swift.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional

import mlx.core as mx

# =============================================================================
# Inventories (Minimal Shim)
# =============================================================================

@dataclass
class SemanticPrime:
    id: str
    canonical_english: str

class SemanticPrimeInventory:
    @staticmethod
    def english_2014() -> List[SemanticPrime]:
        # Sample of Wierzbicka's semantic primes
        return [
            SemanticPrime("I", "I"),
            SemanticPrime("YOU", "YOU"),
            SemanticPrime("SOMEONE", "SOMEONE"),
            SemanticPrime("PEOPLE", "PEOPLE"),
            SemanticPrime("SOMETHING", "SOMETHING"),
            SemanticPrime("BODY", "BODY"),
            SemanticPrime("KIND", "KIND"),
            SemanticPrime("PART", "PART"),
        ]

@dataclass
class ComputationalGate:
    id: str
    name: str
    description: str
    examples: List[str]

class ComputationalGateInventory:
    @staticmethod
    def core_gates() -> List[ComputationalGate]:
        return [
            ComputationalGate("gate:if", "Conditional", "Branching logic", ["if x > 0:", "if condition:"]),
            ComputationalGate("gate:loop", "Loop", "Iterative logic", ["for i in range(10):", "while True:"]),
        ]

# =============================================================================
# VerbNounDimensionClassifier
# =============================================================================

class DimensionClass(str, Enum):
    VERB = "verb"   # Skill dimension
    NOUN = "noun"   # Knowledge dimension
    MIXED = "mixed" # Mixed

@dataclass(frozen=True)
class VerbNounConfig:
    verb_threshold: float = 2.0
    noun_threshold: float = 0.5
    epsilon: float = 1e-6
    verb_blend_weight: float = 0.2
    noun_blend_weight: float = 0.8
    mixed_blend_weight: float = 0.5
    modulation_strength: float = 0.7
    min_activation_variance: float = 1e-8

    @classmethod
    def default(cls) -> "VerbNounConfig":
        return cls()

@dataclass(frozen=True)
class DimensionResult:
    dimension: int
    classification: DimensionClass
    noun_stability: float
    verb_variance: float
    ratio: float
    blend_weight: float

@dataclass(frozen=True)
class Classification:
    dimensions: List[DimensionResult]
    blend_weights: List[float]
    verb_count: int
    noun_count: int
    mixed_count: int
    mean_noun_stability: float
    mean_verb_variance: float
    overall_ratio: float

    @property
    def total_dimensions(self) -> int:
        return len(self.dimensions)

    @property
    def verb_fraction(self) -> float:
        return self.verb_count / max(1, self.total_dimensions)

    @property
    def noun_fraction(self) -> float:
        return self.noun_count / max(1, self.total_dimensions)


class VerbNounDimensionClassifier:
    """Classifies embedding dimensions based on semantic prime and computational gate activations."""

    @staticmethod
    def classify(
        prime_activations: mx.array,
        gate_activations: mx.array,
        config: VerbNounConfig = VerbNounConfig.default(),
    ) -> Classification:
        """
        Classify dimensions.
        prime_activations: [num_primes, hidden_dim]
        gate_activations: [num_gates, hidden_dim]
        """
        hidden_dim = prime_activations.shape[1]

        # Compute stats
        noun_stabilities = VerbNounDimensionClassifier.compute_noun_stability(
            prime_activations, config.epsilon
        )
        verb_variances = VerbNounDimensionClassifier.compute_verb_variance(
            gate_activations
        )
        
        mx.eval(noun_stabilities, verb_variances)
        
        # Convert to python list for iteration (or vectorize if preferred, strictly iterating for parity structure)
        noun_stab_list = noun_stabilities.tolist()
        verb_var_list = verb_variances.tolist()
        
        dimension_results = []
        blend_weights = []
        verb_count = 0
        noun_count = 0
        mixed_count = 0
        
        for dim in range(hidden_dim):
            noun_stab = float(noun_stab_list[dim])
            verb_var = float(verb_var_list[dim])
            ratio = verb_var / (noun_stab + config.epsilon)
            
            classification: DimensionClass
            base_weight: float
            
            if ratio > config.verb_threshold:
                classification = DimensionClass.VERB
                base_weight = config.verb_blend_weight
                verb_count += 1
            elif ratio < config.noun_threshold:
                classification = DimensionClass.NOUN
                base_weight = config.noun_blend_weight
                noun_count += 1
            else:
                classification = DimensionClass.MIXED
                base_weight = config.mixed_blend_weight
                mixed_count += 1
                
            res = DimensionResult(
                dimension=dim,
                classification=classification,
                noun_stability=noun_stab,
                verb_variance=verb_var,
                ratio=ratio,
                blend_weight=base_weight
            )
            dimension_results.append(res)
            blend_weights.append(base_weight)
            
        mean_noun_stability = float(mx.mean(noun_stabilities).item())
        mean_verb_variance = float(mx.mean(verb_variances).item())
        overall_ratio = mean_verb_variance / (mean_noun_stability + config.epsilon)
        
        return Classification(
            dimensions=dimension_results,
            blend_weights=blend_weights,
            verb_count=verb_count,
            noun_count=noun_count,
            mixed_count=mixed_count,
            mean_noun_stability=mean_noun_stability,
            mean_verb_variance=mean_verb_variance,
            overall_ratio=overall_ratio
        )

    @staticmethod
    def compute_noun_stability(
        prime_activations: mx.array,
        epsilon: float = 1e-6
    ) -> mx.array:
        """Noun stability = 1 - coefficient of variation of prime activations."""
        mean = mx.mean(prime_activations, axis=0)
        variance = mx.var(prime_activations, axis=0)
        
        std = mx.sqrt(variance + epsilon)
        abs_mean = mx.abs(mean) + epsilon
        coeff_var = std / abs_mean
        
        normalized_coeff_var = mx.minimum(coeff_var / 2.0, mx.array(1.0))
        stability = 1.0 - normalized_coeff_var
        return stability

    @staticmethod
    def compute_verb_variance(
        gate_activations: mx.array
    ) -> mx.array:
        """Verb variance = variance of gate activations."""
        variance = mx.var(gate_activations, axis=0)
        return variance

    @staticmethod
    def modulate_weights(
        correlation_weights: List[float],
        classification: Classification,
        strength: float
    ) -> List[float]:
        if len(correlation_weights) != len(classification.blend_weights):
            return correlation_weights
            
        clamped_strength = max(0.0, min(1.0, strength))
        
        result = []
        for c_w, vn_w in zip(correlation_weights, classification.blend_weights):
            val = (1.0 - clamped_strength) * c_w + clamped_strength * vn_w
            result.append(val)
        return result

    @staticmethod
    def generate_prime_prompts() -> List[Tuple[str, str]]:
        return [
            (p.id, f"The concept of '{p.canonical_english}' means")
            for p in SemanticPrimeInventory.english_2014()
        ]

    @staticmethod
    def generate_gate_prompts() -> List[Tuple[str, str]]:
        return [
            (g.id, f"// Code pattern: {g.name}\n{g.examples[0]}")
            for g in ComputationalGateInventory.core_gates()
        ]
