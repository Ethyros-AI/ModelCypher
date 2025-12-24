
import argparse
import os
from pathlib import Path

DEFAULT_PYTHON_ROOT = Path(__file__).resolve().parent / "src/modelcypher/core"

# Mapping: Swift Filename -> Python Filename (relative to strict domain roots if possible, or just base name)
# We will define a strict map for the components we ported.
MAPPINGS = {
    # Semantics & Geometry
    "MetaphorConvergenceAnalyzer": ("MetaphorConvergenceAnalyzer.swift", "metaphor_convergence_analyzer.py"),
    "VerbNounDimensionClassifier": ("VerbNounDimensionClassifier.swift", "verb_noun_dimension_classifier.py"),
    "ManifoldStitcher": ("ManifoldStitcher.swift", "manifold_stitcher.py"),
    "CompositionalProbes": ("CompositionalProbes.swift", "compositional_probes.py"),
    "TopologicalFingerprint": ("TopologicalFingerprint.swift", "topological_fingerprint.py"),
    
    # Safety
    "CircuitBreakerIntegration": ("CircuitBreakerIntegration.swift", "circuit_breaker_integration.py"),
    "RegexContentFilter": ("RegexContentFilter.swift", "regex_content_filter.py"),
    "InterventionExecutor": ("InterventionExecutor.swift", "intervention_executor.py"),
    
    # Dynamics (Thermodynamics)
    "LinguisticCalorimeter": ("LinguisticCalorimeter.swift", "optimization_metric_calculator.py"),
    "PhaseTransitionTheory": ("PhaseTransitionTheory.swift", "regime_state_detector.py"),
    "BehavioralOutcomeClassifier": ("BehavioralOutcomeClassifier.swift", "behavioral_outcome_classifier.py"),
    
    # Training Loop Integration
    "GradientSmoothnessEstimator": ("GradientSmoothnessEstimator.swift", "gradient_smoothness_estimator.py"),
    "IdleTrainingScheduler": ("IdleTrainingScheduler.swift", "idle_training_scheduler.py"),
    
    # Evaluation
    "EvaluationEngine": ("MLXTrainingEngine+Evaluation.swift", "engine.py"),
}

# Also defining directory mapping for broader search
DIR_MAP = {
    "Domain/Geometry": "domain/geometry",
    "Domain/Safety": "domain/safety",
    "Domain/Thermodynamics": "domain/dynamics",
    "Domain/Training": "domain/training",
    "Domain/Semantics": "domain/semantics",
    "Domain/Evaluation": "domain/evaluation" # Python has it in evaluation/engine.py?
}

def count_lines(path: str) -> int:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0

def find_file(root: str, filename: str) -> str | None:
    for dirpath, _, filenames in os.walk(root):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--swift-root", required=True, help="Root directory for the reference Swift source tree")
    parser.add_argument(
        "--python-root",
        default=str(DEFAULT_PYTHON_ROOT),
        help="Root directory for the ModelCypher Python source tree",
    )
    args = parser.parse_args()

    swift_root = args.swift_root
    python_root = args.python_root

    print(f"| Component | Swift File | Swift Lines | Python File | Python Lines | Ratio (Py/Swift) |")
    print(f"|---|---|---|---|---|---|")
    
    total_swift = 0
    total_python = 0
    
    # 1. Check Explicit Mapping
    for component, (swift_name, py_name) in MAPPINGS.items():
        swift_path = find_file(swift_root, swift_name)
        
        # Try to find python file
        py_path = find_file(python_root, py_name)

        s_lines = count_lines(swift_path) if swift_path else 0
        p_lines = count_lines(py_path) if py_path else 0
        
        ratio = f"{p_lines/s_lines:.2f}" if s_lines > 0 else "N/A"
        
        total_swift += s_lines
        total_python += p_lines
        
        print(f"| {component} | {swift_name} | {s_lines} | {py_name} | {p_lines} | {ratio} |")

    print(f"| **TOTAL** | | **{total_swift}** | | **{total_python}** | **{total_python/total_swift:.2f}** |")

if __name__ == "__main__":
    main()
