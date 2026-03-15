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

"""Domain to industry benchmark mapping.

Maps ModelCypher's 12 knowledge domains to industry-standard benchmarks
from lm-eval-harness for validation of transplant effectiveness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EvalDomain(str, Enum):
    """Knowledge domains with corresponding industry benchmarks."""

    COMPUTATIONAL = "computational"
    MATHEMATICAL = "mathematical"
    LOGICAL = "logical"
    LINGUISTIC = "linguistic"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    RELATIONAL = "relational"
    AFFECTIVE = "affective"
    MORAL = "moral"
    PHILOSOPHICAL = "philosophical"
    MENTAL = "mental"
    STRUCTURAL = "structural"


@dataclass
class BenchmarkMapping:
    """Mapping of a domain to its industry benchmarks."""

    domain: EvalDomain
    primary_benchmarks: list[str]
    secondary_benchmarks: list[str] = field(default_factory=list)
    description: str = ""

    @property
    def all_benchmarks(self) -> list[str]:
        """All benchmarks for this domain."""
        return self.primary_benchmarks + self.secondary_benchmarks


# Domain to benchmark mappings based on what concepts each benchmark tests
DOMAIN_BENCHMARK_MAP: dict[EvalDomain, BenchmarkMapping] = {
    EvalDomain.COMPUTATIONAL: BenchmarkMapping(
        domain=EvalDomain.COMPUTATIONAL,
        primary_benchmarks=["humaneval", "mbpp"],
        secondary_benchmarks=["mbpp_plus", "humaneval_plus"],
        description="Code generation and programming ability",
    ),
    EvalDomain.MATHEMATICAL: BenchmarkMapping(
        domain=EvalDomain.MATHEMATICAL,
        primary_benchmarks=["gsm8k", "minerva_math"],
        secondary_benchmarks=["gsm8k_cot", "mathqa"],
        description="Mathematical reasoning and calculation",
    ),
    EvalDomain.LOGICAL: BenchmarkMapping(
        domain=EvalDomain.LOGICAL,
        primary_benchmarks=["arc_challenge", "logiqa2"],
        secondary_benchmarks=["boolq", "logiqa"],
        description="Logical reasoning and deduction",
    ),
    EvalDomain.LINGUISTIC: BenchmarkMapping(
        domain=EvalDomain.LINGUISTIC,
        primary_benchmarks=["hellaswag", "lambada_openai"],
        secondary_benchmarks=["lambada_standard"],
        description="Language understanding and prediction",
    ),
    EvalDomain.SPATIAL: BenchmarkMapping(
        domain=EvalDomain.SPATIAL,
        primary_benchmarks=["piqa"],
        secondary_benchmarks=["arc_easy"],
        description="Physical and spatial reasoning",
    ),
    EvalDomain.TEMPORAL: BenchmarkMapping(
        domain=EvalDomain.TEMPORAL,
        primary_benchmarks=["hellaswag"],  # Tests temporal coherence
        secondary_benchmarks=["lambada_openai"],
        description="Temporal reasoning and sequence understanding",
    ),
    EvalDomain.RELATIONAL: BenchmarkMapping(
        domain=EvalDomain.RELATIONAL,
        primary_benchmarks=["winogrande"],
        secondary_benchmarks=["arc_easy"],
        description="Understanding relationships and references",
    ),
    EvalDomain.AFFECTIVE: BenchmarkMapping(
        domain=EvalDomain.AFFECTIVE,
        primary_benchmarks=["hellaswag"],  # Social situations
        secondary_benchmarks=[],
        description="Emotional and social understanding",
    ),
    EvalDomain.MORAL: BenchmarkMapping(
        domain=EvalDomain.MORAL,
        primary_benchmarks=["truthfulqa_mc1", "truthfulqa_mc2"],
        secondary_benchmarks=["truthfulqa_gen"],
        description="Ethical reasoning and truthfulness",
    ),
    EvalDomain.PHILOSOPHICAL: BenchmarkMapping(
        domain=EvalDomain.PHILOSOPHICAL,
        primary_benchmarks=["truthfulqa_mc1", "arc_challenge"],
        secondary_benchmarks=["logiqa2"],
        description="Abstract reasoning and philosophy",
    ),
    EvalDomain.MENTAL: BenchmarkMapping(
        domain=EvalDomain.MENTAL,
        primary_benchmarks=["winogrande"],  # Theory of mind via coreference
        secondary_benchmarks=["hellaswag"],
        description="Theory of mind and mental state reasoning",
    ),
    EvalDomain.STRUCTURAL: BenchmarkMapping(
        domain=EvalDomain.STRUCTURAL,
        primary_benchmarks=["lambada_openai"],  # Syntax-dependent
        secondary_benchmarks=["hellaswag"],
        description="Syntactic structure understanding",
    ),
}


# Benchmark suites for common evaluation scenarios
BENCHMARK_SUITES: dict[str, list[str]] = {
    "leaderboard_v2": [
        "leaderboard_ifeval",
        "leaderboard_bbh",
        "leaderboard_math_hard",
        "leaderboard_gpqa",
        "leaderboard_musr",
        "leaderboard_mmlu_pro",
    ],
    "quick": [
        "arc_easy",
        "hellaswag",
        "boolq",
    ],
    "standard": [
        "arc_easy",
        "arc_challenge",
        "hellaswag",
        "winogrande",
        "boolq",
        "piqa",
    ],
    "comprehensive": [
        "arc_easy",
        "arc_challenge",
        "hellaswag",
        "winogrande",
        "boolq",
        "piqa",
        "logiqa2",
        "truthfulqa_mc1",
        "truthfulqa_mc2",
        "lambada_openai",
        "sciq",
    ],
    "code": [
        "humaneval",
        "mbpp",
    ],
    "math": [
        "gsm8k",
        "minerva_math",
    ],
    "reasoning": [
        "arc_challenge",
        "logiqa2",
        "boolq",
        "winogrande",
    ],
}


def get_benchmarks_for_domain(domain: str | EvalDomain) -> list[str]:
    """Get benchmark tasks for a specific domain.

    Args:
        domain: Domain name or EvalDomain enum.

    Returns:
        List of benchmark task names.
    """
    if isinstance(domain, str):
        domain = EvalDomain(domain)

    mapping = DOMAIN_BENCHMARK_MAP.get(domain)
    if not mapping:
        return []
    return mapping.all_benchmarks


def get_benchmarks_for_domains(domains: list[str]) -> list[str]:
    """Get unique benchmark tasks for multiple domains.

    Args:
        domains: List of domain names.

    Returns:
        Deduplicated list of benchmark task names.
    """
    benchmarks = set()
    for domain in domains:
        benchmarks.update(get_benchmarks_for_domain(domain))
    return sorted(benchmarks)


def get_suite(suite_name: str) -> list[str]:
    """Get benchmark tasks for a named suite.

    Args:
        suite_name: Name of the suite (quick, standard, comprehensive, etc.)

    Returns:
        List of benchmark task names.
    """
    return BENCHMARK_SUITES.get(suite_name, [])


def domain_from_benchmark(benchmark: str) -> list[EvalDomain]:
    """Find which domains a benchmark tests.

    Args:
        benchmark: Benchmark task name.

    Returns:
        List of domains this benchmark tests.
    """
    domains = []
    for domain, mapping in DOMAIN_BENCHMARK_MAP.items():
        if benchmark in mapping.all_benchmarks:
            domains.append(domain)
    return domains
