
import os
import glob
from typing import Dict, Tuple

SWIFT_ROOT = "/Users/jasonkempf/TrainingCypher/app/TrainingCypherPackage/Sources/TrainingCypherCore"
PYTHON_ROOT = "/Users/jasonkempf/ModelCypher/src/modelcypher/core"

# Mapping: Swift Filename -> Python Filename (relative to strict domain roots if possible, or just base name)
# We will define a strict map for the components we ported.
MAPPING = {
    # Geometry
    "MetaphorConvergenceAnalyzer.swift": "metaphor_convergence_analyzer.py",
    "VerbNounDimensionClassifier.swift": "verb_noun_dimension_classifier.py",
    "ManifoldStitcher.swift": "manifold_stitcher.py",
    
    # Safety
    "CircuitBreakerIntegration.swift": "circuit_breaker.py", # note: we kept it as circuit_breaker.py in previous steps or renamed? Let's check.
    "RegexContentFilter.swift": "regex_content_filter.py",
    "InterventionExecutor.swift": "intervention_executor.py",
    
    # Dynamics (Renamed)
    "LinguisticCalorimeter.swift": "optimization_metric_calculator.py",
    "PhaseTransitionTheory.swift": "regime_state_detector.py",
    "BehavioralOutcomeClassifier.swift": "behavioral_outcome_classifier.py",
    
    # Training
    "GradientSmoothnessEstimator.swift": "gradient_smoothness_estimator.py",
    "IdleTrainingScheduler.swift": "idle_training_scheduler.py",
    
    # Semantics
    "ConceptVectorSpace.swift": "vector_space.py", 
    "EvaluationExecutionEngine.swift": "engine.py" 
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

def find_file(root: str, filename: str) -> str:
    for dirpath, _, filenames in os.walk(root):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None

def main():
    print(f"| Component | Swift File | Swift Lines | Python File | Python Lines | Ratio (Py/Swift) |")
    print(f"|---|---|---|---|---|---|")
    
    total_swift = 0
    total_python = 0
    
    # 1. Check Explicit Mapping
    for swift_name, py_name in MAPPING.items():
        swift_path = find_file(SWIFT_ROOT, swift_name)
        
        # Try to find python file - we might need to be smart about renamed circuit_breaker
        py_path = find_file(PYTHON_ROOT, py_name)
        if not py_path and py_name == "circuit_breaker.py":
             py_path = find_file(PYTHON_ROOT, "circuit_breaker_integration.py")
             if py_path: py_name = "circuit_breaker_integration.py"

        s_lines = count_lines(swift_path) if swift_path else 0
        p_lines = count_lines(py_path) if py_path else 0
        
        ratio = f"{p_lines/s_lines:.2f}" if s_lines > 0 else "N/A"
        
        total_swift += s_lines
        total_python += p_lines
        
        print(f"| {swift_name.replace('.swift','')} | {swift_name} | {s_lines} | {py_name} | {p_lines} | {ratio} |")

    print(f"| **TOTAL** | | **{total_swift}** | | **{total_python}** | **{total_python/total_swift:.2f}** |")

if __name__ == "__main__":
    main()
