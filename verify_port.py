
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

try:
    import sys
    from unittest.mock import MagicMock
    sys.modules["mlx_lm"] = MagicMock()
    print("Mocked mlx_lm for verification.")
except ImportError:
    pass

try:
    print("Importing geometry...")
    import modelcypher.core.domain.geometry as geom
    print("Geometry imported.")
    
    print("Importing permutation_aligner...")
    from modelcypher.core.domain.geometry.permutation_aligner import PermutationAligner
    print("PermutationAligner imported.")
    
    print("Importing entropy_dynamics...")
    from modelcypher.core.domain.inference.entropy_dynamics import EntropyDeltaTracker
    print("EntropyDynamics imported.")
    
    print("Importing advanced geometry...")
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimensionEstimator
    from modelcypher.core.domain.geometry.manifold_clusterer import ManifoldClusterer, ManifoldPoint
    from modelcypher.core.domain.geometry.fingerprints import ModelFingerprintsProjection
    from modelcypher.core.domain.geometry.probes import CompositionalProbes, CompositionProbe
    from modelcypher.core.domain.generalized_procrustes import GeneralizedProcrustes
    print("Advanced geometry imported.")

    print("Importing inference...")
    from modelcypher.core.domain.inference.dual_path import DualPathGenerator
    print("DualPathGenerator imported.")
    
    from modelcypher.core.domain.inference.adapter_pool import MLXAdapterPool
    print("MLXAdapterPool imported.")
    
    from modelcypher.core.domain.inference.comparison import CheckpointComparisonCoordinator
    print("CheckpointComparisonCoordinator imported.")
    
    # Architecture Verification
    print("Verifying Ports and Adapters...")
    from modelcypher.core.ports.geometry import GeometryPort
    from modelcypher.infrastructure.adapters.mlx.geometry import MLXGeometryAdapter
    
    assert issubclass(MLXGeometryAdapter, GeometryPort), "Adapter must implement Port"
    print("Architecture Verified: MLXGeometryAdapter implements GeometryPort.")
    
    from modelcypher.core.ports.inference import InferenceEnginePort
    from modelcypher.infrastructure.adapters.mlx.inference import MLXInferenceAdapter
    
    assert issubclass(MLXInferenceAdapter, InferenceEnginePort), "Adapter must implement Port"
    print("Architecture Verified: MLXInferenceAdapter implements InferenceEnginePort.")
    
    assert issubclass(MLXInferenceAdapter, InferenceEnginePort), "Adapter must implement Port"
    print("Architecture Verified: MLXInferenceAdapter implements InferenceEnginePort.")
    
    # Verify Use Case Injection
    print("Verifying Use Case Injection...")
    from modelcypher.core.use_cases.permutation_aligner import PermutationAligner
    from modelcypher.ports.backend import Backend
    
    # Mock Backend
    class MockBackend(Backend):
        pass
        
    adapter = MLXGeometryAdapter()
    backend = MockBackend()
    use_case = PermutationAligner(backend, adapter)
    print("Use Case Injection Verified: PermutationAligner accepted MLXGeometryAdapter.")
    
    adapter = MLXGeometryAdapter()
    backend = MockBackend()
    use_case = PermutationAligner(backend, adapter)
    print("Use Case Injection Verified: PermutationAligner accepted MLXGeometryAdapter.")
    
    # Verify Concept Adapter
    print("Verifying Concept Adapter...")
    from modelcypher.core.ports.concepts import ConceptDiscoveryPort
    from modelcypher.core.ports.embeddings import EmbedderPort
    from modelcypher.infrastructure.adapters.mlx.embeddings import MockMLXEmbedder
    from modelcypher.infrastructure.adapters.mlx.concepts import MLXConceptAdapter
    
    assert issubclass(MockMLXEmbedder, EmbedderPort), "Embedder Adapter must implement Port"
    assert issubclass(MLXConceptAdapter, ConceptDiscoveryPort), "Concept Adapter must implement Port"
    
    embedder = MockMLXEmbedder()
    concept_adapter = MLXConceptAdapter(embedder)
    print("Concept Adapter Verified: Instantiated with Embedder.")

    print("Verification Successful: All modules imported and architecture validated.")
    
except Exception as e:
    print(f"Verification Failed: {e}")
    sys.exit(1)
