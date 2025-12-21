
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

    embedder = MockMLXEmbedder()
    concept_adapter = MLXConceptAdapter(embedder)
    print("Concept Adapter Verified: Instantiated with Embedder.")
    
    # Run async verification
    import asyncio
    
    async def run_async_checks():
        # Verify Refusal Logic
        print("Verifying Refusal Logic (Async)...")
        from modelcypher.core.domain.geometry.types import RefusalConfig
        import mlx.core as mx
        
        # Dummy activations
        harmful = mx.random.normal((10, 64)) + 1.0 # Shift mean
        harmless = mx.random.normal((10, 64)) - 1.0 # Shift mean
        
        ref_conf = RefusalConfig()
        rd = await adapter.compute_refusal_direction(harmful, harmless, ref_conf, {"model_id": "test"})
        
        if rd:
            print(f"Refusal Direction Computed. Strength: {rd.strength:.4f}")
            metrics = await adapter.measure_refusal_distance(mx.random.normal((64,)), rd, 0)
            print(f"Refusal Metrics: Proj={metrics.projection_magnitude:.4f}")
        else:
            print("Warning: Refusal Direction not computed (threshold?)")
            
        # Verify Transport Merger
        print("Verifying Transport Guided Merger...")
        from modelcypher.core.domain.geometry.types import MergerConfig
        
        # Dimensions: 10 neurons, 4 samples. Emb dim 8.
        # W: [10, 8]
        # Acts: [4, 10] (Samples, Neurons)
        
        sw = mx.random.normal((10, 8))
        tw = mx.random.normal((10, 8))
        sa = mx.random.normal((4, 10))
        ta = mx.random.normal((4, 10))
        
        m_conf = MergerConfig(min_samples=2)
        
        mr = await adapter.merge_models_transport(sw, tw, sa, ta, m_conf)
        print(f"Merger Result: GW Dist={mr.gw_distance:.4f}, Converged={mr.converged}")
        assert mr.merged_weights.shape == (10, 8), "Merged weights shape mismatch"

    asyncio.run(run_async_checks())

    print("Verification Successful: All modules imported and architecture validated.")
    
except Exception as e:
    print(f"Verification Failed: {e}")
    sys.exit(1)
