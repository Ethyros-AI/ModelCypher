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
Verification tests for Phase 8: Semantic Agents.
Tests SemanticPrimeAtlas, ComputationalGateAtlas, and TaskDiversionDetector.
"""
import unittest
import asyncio
from typing import List
from dataclasses import dataclass

# Mock Embedder
from modelcypher.ports.embedding import EmbeddingProvider
from modelcypher.core.domain.geometry.vector_math import VectorMath
from modelcypher.core.domain.agents.semantic_prime_atlas import (
    SemanticPrimeAtlas, AtlasConfiguration, SemanticPrimeInventory
)
from modelcypher.core.domain.agents.computational_gate_atlas import (
    ComputationalGateAtlas, GateAtlasConfiguration, ComputationalGateInventory
)
from modelcypher.core.domain.agents.task_diversion_detector import (
    TaskDiversionDetector, DiversionDetectorConfiguration, TaskDiversionAssessment
)

class MockEmbedder:
    def __init__(self, expected_dim=4):
        self._dimension = expected_dim
        # Simple deterministic embedding for testing:
        # Vector depends on hash of text
    
    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            # Deterministic pseudo-random vector
            h = hash(text)
            vec = []
            for i in range(self._dimension):
                # Use different salt for each dim
                val = float((h + i) % 100) / 100.0
                vec.append(val)
            results.append(vec)
        return results

class MockSemanticAwareEmbedder(MockEmbedder):
    """
    Slightly smarter mock that returns similar vectors for known related concepts
    to test similarity thresholds.
    """
    async def embed(self, texts: List[str]) -> List[List[float]]:
        # Hardcoded concepts for testing
        vectors = {
            "task_a": [1.0, 0.0, 0.0, 0.0],
            "task_a_rephrased": [0.9, 0.1, 0.0, 0.0], # High similarity
            "task_b": [0.0, 1.0, 0.0, 0.0], # Orthogonal
        }
        
        results = []
        for text in texts:
            if text in vectors:
                results.append(vectors[text])
            else:
                # Fallback to random consistent baseline
                results.append([0.5, 0.5, 0.5, 0.5])
        return results

class TestPhase8Agents(unittest.TestCase):
    def setUp(self):
        self.mock_embedder = MockEmbedder()

    def test_semantic_prime_atlas_signature(self):
        atlas = SemanticPrimeAtlas(embedder=self.mock_embedder)
        
        # Test signature generation
        sig = asyncio.run(atlas.signature("hello world"))
        
        self.assertIsNotNone(sig)
        self.assertEqual(len(sig.prime_ids), len(SemanticPrimeInventory.english_2014()))
        self.assertEqual(len(sig.values), len(SemanticPrimeInventory.english_2014()))
        
        # Verify normalization
        norm = VectorMath.l2_norm(sig.values)
        # Note: Signature itself might not be unit norm, but the embedding inside is.
        # Actually, signature values are dot products (similarities), so they are just scalars.
        # The logic: similarities = dot(prime_vec, text_vec). If both unit, values are cos sims.
        # They don't necessarily sum to 1 or have norm 1.
        
        # Test summary
        _, summary = asyncio.run(atlas.analyze("test text"))
        self.assertTrue(len(summary.top_primes) > 0)
        # Verify top prime is a valid prime from the inventory
        valid_prime_ids = {p.id for p in SemanticPrimeInventory.english_2014()}
        self.assertIn(summary.top_primes[0].prime_id, valid_prime_ids) 

    def test_computational_gate_atlas_signature(self):
        config = GateAtlasConfiguration(use_probe_subset=True)
        atlas = ComputationalGateAtlas(embedder=self.mock_embedder, configuration=config)
        
        sig = asyncio.run(atlas.signature("def my_func(): return 1"))
        
        self.assertIsNotNone(sig)
        self.assertEqual(len(sig.gate_ids), len(ComputationalGateInventory.probe_gates()))

    def test_computational_gate_prompt_generation(self):
        prompts = ComputationalGateAtlas.generate_probe_prompts(
            style=ComputationalGateAtlas.PromptStyle.COMPLETION,
            subset_name="probe"
        )
        self.assertEqual(len(prompts), len(ComputationalGateInventory.probe_gates()))
        self.assertTrue(prompts[0][1].startswith("#") or prompts[0][1].startswith("def") or "import" in prompts[0][1])

    def test_task_diversion_detector_embeddings(self):
        embedder = MockSemanticAwareEmbedder()
        detector = TaskDiversionDetector(embedder=embedder)
        
        # Aligned
        assessment = asyncio.run(detector.assess("task_a", "task_a_rephrased"))
        self.assertEqual(assessment.method, TaskDiversionAssessment.Method.EMBEDDINGS)
        self.assertEqual(assessment.verdict, TaskDiversionAssessment.Verdict.ALIGNED)
        self.assertGreaterEqual(assessment.embedding_cosine_similarity, 0.8) # 0.9 dot 1.0 approx
        
        # Diverged
        assessment = asyncio.run(detector.assess("task_a", "task_b"))
        self.assertEqual(assessment.method, TaskDiversionAssessment.Method.EMBEDDINGS)
        self.assertEqual(assessment.verdict, TaskDiversionAssessment.Verdict.DIVERGED)
        self.assertLess(assessment.embedding_cosine_similarity, 0.1)

    def test_task_diversion_detector_lexical_fallback(self):
        # Embedder fails or returns empty -> fallback
        class BrokenEmbedder:
            @property
            def dimension(self): return 0
            async def embed(self, texts): return []
            
        detector = TaskDiversionDetector(
            embedder=BrokenEmbedder(),
            configuration=DiversionDetectorConfiguration(enable_lexical_fallback=True)
        )
        
        # Aligned (lexically similar)
        # Stop words: "this", "is" might be stripped.
        # "calculate" "fibonacci" vs "calculate" "fibonacci" "sequence"
        t1 = "calculate fibonacci"
        t2 = "calculate fibonacci sequence"
        assessment = asyncio.run(detector.assess(t1, t2))
        
        self.assertEqual(assessment.method, TaskDiversionAssessment.Method.LEXICAL_FALLBACK)
        self.assertEqual(assessment.verdict, TaskDiversionAssessment.Verdict.ALIGNED)
        
        # Diverged
        t3 = "make sandwich"
        assessment = asyncio.run(detector.assess(t1, t3))
        self.assertEqual(assessment.method, TaskDiversionAssessment.Method.LEXICAL_FALLBACK)
        self.assertEqual(assessment.verdict, TaskDiversionAssessment.Verdict.DIVERGED)

if __name__ == "__main__":
    unittest.main()
