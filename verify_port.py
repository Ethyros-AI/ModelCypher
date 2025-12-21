import asyncio
import sys
import os
import shutil
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

# Import Phase 1 components (to keep previous checks valid)
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

# Import Phase 3 Components
from modelcypher.core.domain.thermodynamics.calorimetry import LinguisticCalorimeter
from modelcypher.core.domain.thermodynamics.phase_transition import PhaseTransitionTheory, MatterState
from modelcypher.core.domain.thermodynamics.defense import EntropyDefenseMonitor

async def verify_thermodynamics():
    print("\n--- Verifying Phase 3: Thermodynamics ---")
    
    # 1. Calorimetry
    print("1. Checking Calorimetry...")
    cal = LinguisticCalorimeter()
    # Simulate normal state
    state = cal.measure_state(loss=2.5, gradient_norm=1.0, entropy=3.0) # Temp ~20
    print(f"   State 1: Temp={state.temperature:.2f}, Analysis={cal.analyze_stability(state)}")
    
    # Simulate plasma
    plasma_state = cal.measure_state(loss=10.0, gradient_norm=5.0, entropy=10.0) # Temp High
    print(f"   State 2: Temp={plasma_state.temperature:.2f}, Analysis={cal.analyze_stability(plasma_state)}")
    
    # 2. Phase Transition & Defense
    print("2. Checking Defense Monitor...")
    theory = PhaseTransitionTheory()
    defense = EntropyDefenseMonitor(theory)
    
    triggered = False
    def on_defense(reason):
        nonlocal triggered
        triggered = True
        print(f"   Defense Callback Triggered: {reason}")
        
    defense.set_defense_protocol(on_defense)
    
    # Step 1: Liquid
    defense.monitor_step(step=1, loss=2.0, grad_norm=1.0, entropy=2.0)
    assert not triggered
    
    # Step 2: Sudden Spike (Plasma)
    defense.monitor_step(step=2, loss=10.0, grad_norm=10.0, entropy=200.0) # Huge entropy
    assert triggered
    print("   Defense mechanism successfully verified.")


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

    # 4. Verify Engine & Checkpoints (Mock Run)
    print("4. Running Mock Training Job...")
    engine = TrainingEngine()
    
    # Simple Linear Model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(10, 10)
        def __call__(self, x):
            return self.l1(x)
            
    model = SimpleModel()
    optimizer = optim.AdamW(learning_rate=1e-3)
    
    # Dummy Data: List of (input, target) tuples
    data = [ (mx.random.normal((2, 10)), mx.random.normal((2, 10))) for _ in range(5) ]
    
    output_path = "/tmp/modelcypher-test-output"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    config = TrainingConfig(
        model_id="test-model",
        dataset_path="/tmp",
        output_path=output_path,
        hyperparameters=valid_params
    )
    
    def on_progress(p):
        print(f"   Step {p.step}: Loss={p.loss:.4f}")
        
    await engine.train(
        job_id="test-job",
        config=config,
        model=model,
        optimizer=optimizer,
        data_provider=data,
        progress_callback=on_progress
    )
    print("   Training job completed.")
    
    # Check if checkpoint exists
    ckpt_dir = os.path.join(config.output_path, "checkpoints")
    assert os.path.exists(ckpt_dir)
    print("   Checkpoint directory created.")
    
    # List files
    files = os.listdir(ckpt_dir)
    safetensors = [f for f in files if f.endswith(".safetensors")]
    metadata = [f for f in files if f.endswith(".json")]
    
    print(f"   Found {len(safetensors)} checkpoint files and {len(metadata)} metadata files.")
    assert len(safetensors) > 0
    assert len(metadata) > 0

async def run_async_checks():
    await verify_training_engine()
    await verify_thermodynamics()

if __name__ == "__main__":
    asyncio.run(run_async_checks())
