
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
    
    print("Verification Successful: All modules imported.")
    
except Exception as e:
    print(f"Verification Failed: {e}")
    sys.exit(1)
