"""Curriculum module for systematic capability training.

This module provides tools for:
1. Loading standard benchmarks (GSM8K, ARC, MMLU, etc.)
2. Converting benchmarks to text continuation training format
3. Sequencing capabilities by formal dependency (skill DAG)
4. Tracking mastery progress via auto-regime detection
"""

from modelcypher.core.use_cases.curriculum.benchmark_loader import (
    Benchmark,
    BenchmarkLoader,
    BenchmarkSample,
    BenchmarkTier,
    save_for_training,
)
from modelcypher.core.use_cases.curriculum.phase_scheduler import (
    MasteryRecord,
    PhaseScheduler,
)
from modelcypher.core.use_cases.curriculum.skill_dag import (
    CURRICULUM_DAG,
    SkillDAG,
    SkillNode,
    build_curriculum_dag,
)

__all__ = [
    # Benchmark loading
    "BenchmarkLoader",
    "Benchmark",
    "BenchmarkSample",
    "BenchmarkTier",
    "save_for_training",
    # Skill DAG
    "SkillDAG",
    "SkillNode",
    "build_curriculum_dag",
    "CURRICULUM_DAG",
    # Phase scheduling
    "PhaseScheduler",
    "MasteryRecord",
    # evaluate_skill_mastery is in modelcypher.adapters.curriculum_eval_adapter
]
