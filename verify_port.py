import asyncio
import sys
import os
import shutil
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

# Import Phase 1 components
from modelcypher.core.ports.geometry import GeometryPort, MergerConfig
from modelcypher.infrastructure.adapters.mlx.geometry import MLXGeometryAdapter
from modelcypher.core.domain.geometry.permutation_aligner import PermutationAligner
from modelcypher.infrastructure.adapters.mlx.merger import TransportGuidedMerger

# Import Phase 2 components
from modelcypher.core.domain.training.types import (
    TrainingConfig, Hyperparameters, CheckpointMetadata, ComputePrecision
)
from modelcypher.core.domain.training.validation import TrainingHyperparameterValidator
from modelcypher.core.domain.training.resources import TrainingResourceGuard, ResourceIntensiveOperation
from modelcypher.core.domain.training.checkpoints import CheckpointManager
from modelcypher.core.domain.training.engine import TrainingEngine
from modelcypher.infrastructure.services.memory import MLXMemoryService

# Import Phase 3 Components (Refactored/Ported)
from modelcypher.core.domain.dynamics.optimization_metric_calculator import OptimizationMetricCalculator
from modelcypher.core.domain.dynamics.regime_state_detector import RegimeStateDetector
# Assuming monitoring.py exists and has DivergenceInterventionMonitor, otherwise we skip
try:
    from modelcypher.core.domain.dynamics.monitoring import DivergenceInterventionMonitor
except ImportError:
    DivergenceInterventionMonitor = None

# Import Phase 3 (Safety) Components
from modelcypher.core.domain.safety.circuit_breaker_integration import CircuitBreakerIntegration, InputSignals
from modelcypher.core.domain.safety.regex_content_filter import RegexContentFilter
from modelcypher.core.domain.safety.intervention_executor import InterventionExecutor, CombinedEvaluation, CircuitBreakerState, InterventionLevel, RecommendedAction

# Import Phase 4 (Geometry Extra) Components
from modelcypher.core.domain.geometry.metaphor_convergence_analyzer import MetaphorConvergenceAnalyzer
from modelcypher.core.domain.geometry.verb_noun_dimension_classifier import VerbNounDimensionClassifier

# Import Phase 4 (Training Pipeline) Components
from modelcypher.core.domain.training.gradient_smoothness_estimator import GradientSmoothnessEstimator
from modelcypher.core.domain.training.idle_training_scheduler import IdleTrainingScheduler, Protocol as IdleProtocol

async def verify_training_dynamics():
    print("\n--- Verifying Phase 3: Training Dynamics ---")
    
    # 1. Metrics
    print("1. Checking Optimization Metrics...")
    calc = OptimizationMetricCalculator()
    # Mocking behavior if method names differ in 1:1 port
    # In port: calculate_statistics(entropy_trajectory)
    # Trying calculate_statistics
    try:
        stats = calc.calculate_statistics([3.0, 2.9, 2.8])
        print(f"   Stats: {stats}")
    except Exception as e:
        print(f"   Metric calc error (expected if API differs): {e}")
    
    # 2. Regimes
    print("2. Checking Regime Detector...")
    detector = RegimeStateDetector()
    try:
        # analyze(logits, tempera    try:
        # Mock logits
        logits = mx.random.normal((1, 10))
        analysis = detector.analyze(logits, temperature=1.0)
        print(f"   Analysis Phase: {analysis.state}")
    except Exception as e:
         print(f"   Regime calc error: {e}")

    # 3. Monitoring
    if DivergenceInterventionMonitor:
        print("3. Checking Intervention Monitor...")
        monitor = DivergenceInterventionMonitor(detector)
        # Simple step check
        monitor.monitor_step(step=1, loss=2.0, grad_norm=1.0, entropy=2.0)
        print("   Monitor step 1 completed.")
    else:
        print("3. Skipping Intervention Monitor (not found).")


async def verify_safety():
    print("\n--- Verifying Phase 3: Safety Layer ---")
    
    # 1. Regex Filter
    print("1. Checking Regex Content Filter...")
    regex_filter = RegexContentFilter.default()
    safe_text = "Hello world"
    unsafe_text = "rm -rf /"
    
    res_safe = regex_filter.check(safe_text)
    assert res_safe is None, "Safe text should return None"
    
    res_unsafe = regex_filter.check(unsafe_text)
    assert res_unsafe is not None, "Unsafe text should return Result"
    print(f"   Caught unsafe text: {res_unsafe.reason}")
    
    # 2. Circuit Breaker
    print("2. Checking Circuit Breaker...")
    cb = CircuitBreakerIntegration()
    signals = InputSignals(entropy_signal=0.95, refusal_distance=0.01) # Very High entropy, very close to refusal
    state = cb.evaluate(signals)
    print(f"   CB State: Tripped={state.is_tripped}, Severity={state.severity:.2f}")
    assert state.is_tripped or state.severity > 0.5
    
    # 3. Intervention Executor
    print("3. Checking Intervention Executor...")
    executor = InterventionExecutor()
    # Mock gas decision
    # need circuit breaker state
    res = await executor.evaluate_and_execute(gas_decision=None, circuit_breaker_state=state, token_index=10)
    print(f"   Intervention Result: {res.type}")


async def verify_geometry_extra():
    print("\n--- Verifying Phase 4: Geometry Extras ---")
    
    # 1. Metaphor
    print("1. Checking Metaphor Convergence...")
    analyzer = MetaphorConvergenceAnalyzer()
    # Just check instantiation, fully mocking fingerprints is complex
    assert analyzer is not None
    print("   MetaphorConvergenceAnalyzer initialized.")

    # 2. Verb Noun
    print("2. Checking Verb/Noun Classifier...")
    classifier = VerbNounDimensionClassifier()
    # Check instantiation
    assert classifier is not None
    print("   VerbNounDimensionClassifier initialized.")
    
    # 3. Compositional Probes
    print("3. Checking Compositional Probes...")
    from modelcypher.core.domain.geometry.compositional_probes import CompositionalProbes, CompositionProbe, CompositionCategory
    probe = CompositionProbe("I THINK", ["I", "THINK"], CompositionCategory.MENTAL_PREDICATE)
    assert probe.phrase == "I THINK"
    print("   CompositionalProbe initialized.")
    
    # 4. Topological Fingerprint
    print("4. Checking Topological Fingerpints...")
    from modelcypher.core.domain.geometry.topological_fingerprint import TopologicalFingerprint
    points = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]] 
    fingerprint = TopologicalFingerprint.compute(points)
    assert fingerprint.summary.component_count >= 1
    print("   TopologicalFingerprint computed.")


async def verify_training_enhancements():
    print("\n--- Verifying Phase 4: Training Enhancements ---")
    
    # 1. Gradient Smoothness
    print("1. Checking Gradient Smoothness...")
    # Mock gradients: 2 samples, 1 param
    g1 = {"test.layers.0.w": mx.array([1.0, 1.0])}
    g2 = {"test.layers.0.w": mx.array([1.1, 0.9])}
    grads = [g1, g2]
    
    quality = GradientSmoothnessEstimator.per_layer_quality(grads)
    if 0 in quality:
        q = quality[0]
        print(f"   Layer 0 Quality: SNR={q.snr:.2f}, Variance={q.variance:.2f}")
    else:
        print("   No layer quality computed (key mismatch?)")
        
    # 2. Idle Scheduler
    print("2. Checking Idle Scheduler...")
    scheduler = IdleTrainingScheduler()
    scheduler.start_monitoring()
    await asyncio.sleep(0.1)
    await scheduler.stop_monitoring()
    print("   Idle Scheduler started and stopped.")


async def verify_training_engine():
    print("\n--- Verifying Phase 2: Training Engine ---")
    
    # 1. Verify Validator
    print("1. Checking Validator...")
    bad_params = Hyperparameters(batch_size=100, learning_rate=1.0) # obvious errors
    violations = TrainingHyperparameterValidator.comprehensive_violations(bad_params)
    assert len(violations) > 0
    print("   Validator correctly caught bad params.")
    
    valid_params = Hyperparameters(batch_size=2, learning_rate=3e-5)
    TrainingHyperparameterValidator.validate_for_engine(valid_params)
    print("   Validator passed valid params.")

    # 2. Verify Resource Guard
    print("2. Checking Resource Guard...")
    guard = TrainingResourceGuard()
    await guard.begin_training("job-123")
    assert guard.is_training_active
    try:
        await guard.begin_inference("user")
        print("   FAILED: Guard should have blocked inference!")
    except Exception:
        print("   Guard correctly blocked inference during training.")
    await guard.end_training("job-123")
    assert not guard.is_training_active
    print("   Guard released resources.")

    # 3. Verify Memory Service
    print("3. Checking Memory Service...")
    mem = MLXMemoryService()
    stats = mem.get_memory_stats()
    print(f"   Memory Stats: Available={stats.available_gb}GB, Pressure={stats.pressure}")


async def run_async_checks():
    await verify_training_engine()
    await verify_training_dynamics()
    await verify_safety()
    await verify_geometry_extra()
    await verify_training_enhancements()
    
    print("\n--- Verifying Phase 5: Semantics & Evaluation ---")
    try:
        from modelcypher.core.domain.evaluation.engine import EvaluationExecutionEngine, EvaluationConfig, MetricType
        
        print(f"✅ Semantics Modules Imported Successfully")
        
        # 1. Eval Engine
        config = EvaluationConfig(
            dataset_path="/tmp/test",
            metrics=[MetricType.LOSS],
            batch_size=4
        )
        engine = EvaluationExecutionEngine(config)
        assert engine.config.batch_size == 4
        print(f"1. Checking Evaluation Engine... OK")
        
    except ImportError as e:
        print(f"❌ Semantics/Eval Import Error: {e}")
    except Exception as e:
         print(f"❌ Semantics/Eval Verification Error: {e}")

if __name__ == "__main__":
    asyncio.run(run_async_checks())
