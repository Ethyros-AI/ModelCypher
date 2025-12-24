
from typing import Protocol, Any, runtime_checkable
from modelcypher.core.domain.geometry.types import (
    ConceptConfiguration, DetectionResult, ConceptComparisonResult, DetectedConcept
)

@runtime_checkable
class ConceptDiscoveryPort(Protocol):
    """
    Interface for semantic concept detection in generated text.
    """
    
    async def detect_concepts(
        self,
        response: str,
        model_id: str,
        prompt_id: str,
        config: ConceptConfiguration
    ) -> DetectionResult:
        ...
        
    async def compare_results(
        self,
        result_a: DetectionResult,
        result_b: DetectionResult
    ) -> ConceptComparisonResult:
        ...
