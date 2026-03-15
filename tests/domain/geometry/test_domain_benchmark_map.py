# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from modelcypher.core.domain.geometry.domain_benchmark_map import (
    BENCHMARK_SUITES,
    BenchmarkMapping,
    EvalDomain,
    domain_from_benchmark,
    get_benchmarks_for_domain,
    get_benchmarks_for_domains,
    get_suite,
)


def test_get_benchmarks_for_domain_accepts_str_and_enum() -> None:
    by_str = get_benchmarks_for_domain("mathematical")
    by_enum = get_benchmarks_for_domain(EvalDomain.MATHEMATICAL)

    assert by_str == by_enum
    assert "gsm8k" in by_str
    assert "minerva_math" in by_str


def test_get_benchmarks_for_domains_is_sorted_and_deduplicated() -> None:
    benchmarks = get_benchmarks_for_domains(["logical", "philosophical"])

    assert benchmarks == sorted(set(benchmarks))
    assert "arc_challenge" in benchmarks
    assert "logiqa2" in benchmarks


def test_get_suite_known_and_unknown() -> None:
    assert get_suite("quick") == BENCHMARK_SUITES["quick"]
    assert get_suite("leaderboard_v2") == [
        "leaderboard_ifeval",
        "leaderboard_bbh",
        "leaderboard_math_hard",
        "leaderboard_gpqa",
        "leaderboard_musr",
        "leaderboard_mmlu_pro",
    ]
    assert get_suite("does-not-exist") == []


def test_domain_from_benchmark_returns_matching_domains() -> None:
    domains = domain_from_benchmark("truthfulqa_mc1")

    assert EvalDomain.MORAL in domains
    assert EvalDomain.PHILOSOPHICAL in domains


def test_benchmark_mapping_all_benchmarks_property() -> None:
    mapping = BenchmarkMapping(
        domain=EvalDomain.COMPUTATIONAL,
        primary_benchmarks=["humaneval"],
        secondary_benchmarks=["mbpp"],
        description="test",
    )

    assert mapping.all_benchmarks == ["humaneval", "mbpp"]
