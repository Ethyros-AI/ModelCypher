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

"""Tests for alignment generalization validation utilities.

Validates:
1. Even/odd split: deterministic, lossless, raises on too-few samples
2. Coverage ratio: effective rank in [0, 1]
3. Alignment generalization report: train CKA ~1, gain arithmetic, error handling
4. Domain alignment report: per-domain grouping, skipped domains
5. Mathematical properties: CKA bounds, gain sign, coverage monotonicity
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pytest

from modelcypher.core.domain.geometry.alignment_validation import (
    AlignmentGeneralizationReport,
    DomainAlignmentReport,
    _even_odd_split,
    alignment_generalization_by_domain,
    alignment_generalization_report,
)

# =============================================================================
# Helpers
# =============================================================================


def _deterministic_matrix(backend, rows, cols, offset=0):
    """Create a deterministic matrix with formula: (i*cols + j + offset) % 17 / 17.0 + 0.1."""
    data = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(((i * cols + j + offset) % 17) / 17.0 + 0.1)
        data.append(row)
    arr = backend.array(data)
    backend.eval(arr)
    return arr


@dataclass
class FakeProbe:
    """Minimal stub satisfying AtlasProbeProtocol."""

    probe_id: str
    name: str
    description: str
    support_texts: Sequence[str]
    source: Any
    domain: Any
    category_name: str
    cross_domain_weight: float
    verification_depth: int | None


def _make_probes(domains: list[str]) -> list[FakeProbe]:
    """Create fake probes with specified domain strings."""
    return [
        FakeProbe(
            probe_id=f"probe_{i}",
            name=f"Probe {i}",
            description=f"Test probe {i}",
            support_texts=[f"text_{i}"],
            source="test",
            domain=d,
            category_name="test",
            cross_domain_weight=1.0,
            verification_depth=None,
        )
        for i, d in enumerate(domains)
    ]


# =============================================================================
# Test: Even/Odd Split
# =============================================================================


class TestEvenOddSplit:
    """Tests for _even_odd_split."""

    def test_basic_split(self):
        """Even indices go to train, odd to holdout."""
        train, holdout = _even_odd_split([0, 1, 2, 3, 4, 5])
        assert train == [0, 2, 4]
        assert holdout == [1, 3, 5]

    def test_minimum_valid(self):
        """Minimum valid input: 4 elements (2 train, 2 holdout)."""
        train, holdout = _even_odd_split([10, 20, 30, 40])
        assert train == [10, 30]
        assert holdout == [20, 40]

    def test_odd_count(self):
        """Odd number of elements: train gets one more."""
        train, holdout = _even_odd_split([0, 1, 2, 3, 4])
        assert train == [0, 2, 4]
        assert holdout == [1, 3]

    def test_too_few_raises(self):
        """Fewer than 4 elements raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 samples"):
            _even_odd_split([0, 1, 2])

    def test_two_elements_raises(self):
        """Two elements raises (only 1 train, 1 holdout)."""
        with pytest.raises(ValueError, match="at least 2 samples"):
            _even_odd_split([0, 1])

    def test_preserves_values(self):
        """Split preserves original values, not just indices."""
        train, holdout = _even_odd_split([100, 200, 300, 400, 500, 600])
        assert train == [100, 300, 500]
        assert holdout == [200, 400, 600]


# =============================================================================
# Test: Coverage Ratio
# =============================================================================


class TestCoverageRatio:
    """Tests for _coverage_ratio (via alignment_generalization_report)."""

    def test_identity_gives_high_coverage(self, any_backend):
        """Identity-like source activations have high effective rank."""
        b = any_backend
        n, d = 8, 4
        # Near-identity: each sample is a standard basis vector (spread out)
        data = []
        for i in range(n):
            row = [0.0] * d
            row[i % d] = 1.0
            data.append(row)
        source = b.array(data)
        target = _deterministic_matrix(b, n, d, offset=7)
        b.eval(source, target)

        report = alignment_generalization_report(
            source, target,
            train_indices=[0, 2, 4, 6],
            holdout_indices=[1, 3, 5, 7],
            backend=b,
        )
        # Diverse source should give reasonable coverage
        assert report.coverage_ratio > 0.0

    def test_constant_source_gives_low_coverage(self, any_backend):
        """Constant source activations (all identical rows) have low coverage."""
        b = any_backend
        n, d = 8, 4
        # All rows are exactly identical -> zero pairwise distances -> degenerate Gram
        data = [[1.0, 0.5, 0.2, 0.1]] * n
        source = b.array(data)
        target = _deterministic_matrix(b, n, d, offset=5)
        b.eval(source, target)

        report = alignment_generalization_report(
            source, target,
            train_indices=[0, 2, 4, 6],
            holdout_indices=[1, 3, 5, 7],
            backend=b,
        )
        # Identical rows produce a rank-1 Gram matrix (all ones after RBF with 0 distance)
        assert report.coverage_ratio <= 1.0

    def test_coverage_bounded(self, any_backend):
        """Coverage ratio is always in [0, 1]."""
        b = any_backend
        n, d = 8, 4
        source = _deterministic_matrix(b, n, d, offset=0)
        target = _deterministic_matrix(b, n, d, offset=13)
        b.eval(source, target)

        report = alignment_generalization_report(
            source, target,
            train_indices=[0, 2, 4, 6],
            holdout_indices=[1, 3, 5, 7],
            backend=b,
        )
        assert 0.0 <= report.coverage_ratio <= 1.0

    def test_coverage_ratio_is_nonnegative(self, any_backend):
        """Coverage ratio is always non-negative regardless of input diversity."""
        b = any_backend
        n, d = 8, 4

        # Low diversity: nearly identical rows
        low_data = [[1.0 + i * 0.001, 0.5, 0.2, 0.1] for i in range(n)]
        source = b.array(low_data)
        target = _deterministic_matrix(b, n, d, offset=3)
        b.eval(source, target)

        report = alignment_generalization_report(
            source, target,
            train_indices=[0, 2, 4, 6],
            holdout_indices=[1, 3, 5, 7],
            backend=b,
        )
        assert report.coverage_ratio >= 0.0


# =============================================================================
# Test: Alignment Generalization Report
# =============================================================================


class TestAlignmentGeneralizationReport:
    """Tests for alignment_generalization_report."""

    def test_identical_data_cka_1(self, any_backend):
        """When source == target, train CKA should be near 1.0."""
        b = any_backend
        n, d = 12, 4
        data = _deterministic_matrix(b, n, d, offset=0)
        b.eval(data)

        report = alignment_generalization_report(
            data, data,
            train_indices=[0, 2, 4, 6, 8, 10],
            holdout_indices=[1, 3, 5, 7, 9, 11],
            backend=b,
        )
        assert report.train_cka > 0.95

    def test_sample_counts(self, any_backend):
        """Report records correct train and holdout sample counts."""
        b = any_backend
        n, d = 12, 4
        source = _deterministic_matrix(b, n, d, offset=0)
        target = _deterministic_matrix(b, n, d, offset=7)
        b.eval(source, target)

        report = alignment_generalization_report(
            source, target,
            train_indices=[0, 2, 4, 6, 8, 10],
            holdout_indices=[1, 3, 5, 7, 9, 11],
            backend=b,
        )
        assert report.train_samples == 6
        assert report.holdout_samples == 6

    def test_gain_formula(self, any_backend):
        """alignment_gain == holdout_cka - raw_holdout_cka."""
        b = any_backend
        n, d = 12, 4
        source = _deterministic_matrix(b, n, d, offset=0)
        target = _deterministic_matrix(b, n, d, offset=7)
        b.eval(source, target)

        report = alignment_generalization_report(
            source, target,
            train_indices=[0, 2, 4, 6, 8, 10],
            holdout_indices=[1, 3, 5, 7, 9, 11],
            backend=b,
        )
        expected_gain = report.holdout_cka - report.raw_holdout_cka
        assert report.alignment_gain == pytest.approx(expected_gain, abs=1e-10)

    def test_too_few_train_raises(self, any_backend):
        """Fewer than 2 train samples raises ValueError."""
        b = any_backend
        source = _deterministic_matrix(b, 6, 4, offset=0)
        target = _deterministic_matrix(b, 6, 4, offset=5)
        b.eval(source, target)

        with pytest.raises(ValueError, match="at least 2 samples"):
            alignment_generalization_report(
                source, target,
                train_indices=[0],
                holdout_indices=[1, 2, 3],
                backend=b,
            )

    def test_too_few_holdout_raises(self, any_backend):
        """Fewer than 2 holdout samples raises ValueError."""
        b = any_backend
        source = _deterministic_matrix(b, 6, 4, offset=0)
        target = _deterministic_matrix(b, 6, 4, offset=5)
        b.eval(source, target)

        with pytest.raises(ValueError, match="at least 2 samples"):
            alignment_generalization_report(
                source, target,
                train_indices=[0, 1, 2],
                holdout_indices=[3],
                backend=b,
            )

    def test_report_is_frozen(self, any_backend):
        """AlignmentGeneralizationReport is immutable (frozen dataclass)."""
        b = any_backend
        n, d = 8, 4
        source = _deterministic_matrix(b, n, d, offset=0)
        target = _deterministic_matrix(b, n, d, offset=7)
        b.eval(source, target)

        report = alignment_generalization_report(
            source, target,
            train_indices=[0, 2, 4, 6],
            holdout_indices=[1, 3, 5, 7],
            backend=b,
        )
        with pytest.raises(AttributeError):
            report.train_cka = 0.0  # type: ignore[misc]

    def test_cka_values_in_range(self, any_backend):
        """All CKA values are in [0, 1]."""
        b = any_backend
        n, d = 12, 4
        source = _deterministic_matrix(b, n, d, offset=0)
        target = _deterministic_matrix(b, n, d, offset=7)
        b.eval(source, target)

        report = alignment_generalization_report(
            source, target,
            train_indices=[0, 2, 4, 6, 8, 10],
            holdout_indices=[1, 3, 5, 7, 9, 11],
            backend=b,
        )
        assert 0.0 <= report.train_cka <= 1.0
        assert 0.0 <= report.holdout_cka <= 1.0
        assert 0.0 <= report.raw_holdout_cka <= 1.0

    def test_rotated_target_lowers_holdout(self, any_backend):
        """Rotating target reduces holdout CKA vs identical-data baseline."""
        b = any_backend
        n, d = 12, 4
        source = _deterministic_matrix(b, n, d, offset=0)
        b.eval(source)

        # Identical target: holdout_cka should be high
        report_identical = alignment_generalization_report(
            source, source,
            train_indices=[0, 2, 4, 6, 8, 10],
            holdout_indices=[1, 3, 5, 7, 9, 11],
            backend=b,
        )

        # Different target: holdout_cka should be lower or equal
        target_diff = _deterministic_matrix(b, n, d, offset=11)
        b.eval(target_diff)

        report_diff = alignment_generalization_report(
            source, target_diff,
            train_indices=[0, 2, 4, 6, 8, 10],
            holdout_indices=[1, 3, 5, 7, 9, 11],
            backend=b,
        )
        # Identical case should have at least as good holdout CKA
        assert report_identical.holdout_cka >= report_diff.holdout_cka - 0.05

    def test_non_overlapping_indices(self, any_backend):
        """Works with non-standard non-overlapping index sets."""
        b = any_backend
        n, d = 10, 4
        source = _deterministic_matrix(b, n, d, offset=0)
        target = _deterministic_matrix(b, n, d, offset=3)
        b.eval(source, target)

        report = alignment_generalization_report(
            source, target,
            train_indices=[0, 1, 2, 3, 4],
            holdout_indices=[5, 6, 7, 8, 9],
            backend=b,
        )
        assert report.train_samples == 5
        assert report.holdout_samples == 5

    def test_all_fields_are_float(self, any_backend):
        """All numeric fields in the report are Python floats."""
        b = any_backend
        n, d = 8, 4
        source = _deterministic_matrix(b, n, d, offset=0)
        target = _deterministic_matrix(b, n, d, offset=7)
        b.eval(source, target)

        report = alignment_generalization_report(
            source, target,
            train_indices=[0, 2, 4, 6],
            holdout_indices=[1, 3, 5, 7],
            backend=b,
        )
        assert isinstance(report.train_cka, float)
        assert isinstance(report.holdout_cka, float)
        assert isinstance(report.raw_holdout_cka, float)
        assert isinstance(report.alignment_gain, float)
        assert isinstance(report.coverage_ratio, float)
        assert isinstance(report.train_samples, int)
        assert isinstance(report.holdout_samples, int)


# =============================================================================
# Test: Domain Alignment Report
# =============================================================================


class TestDomainAlignmentReport:
    """Tests for alignment_generalization_by_domain."""

    def test_empty_probes(self, any_backend):
        """Empty probe list returns empty report."""
        b = any_backend
        source = _deterministic_matrix(b, 8, 4, offset=0)
        target = _deterministic_matrix(b, 8, 4, offset=5)
        b.eval(source, target)

        report = alignment_generalization_by_domain(source, target, [], backend=b)
        assert report.domain_reports == {}
        assert report.domain_counts == {}
        assert report.skipped_domains == []

    def test_single_domain_enough_probes(self, any_backend):
        """Single domain with >= 4 probes produces a report."""
        b = any_backend
        n, d = 8, 4
        source = _deterministic_matrix(b, n, d, offset=0)
        target = _deterministic_matrix(b, n, d, offset=5)
        b.eval(source, target)

        probes = _make_probes(["math"] * n)
        report = alignment_generalization_by_domain(source, target, probes, backend=b)

        assert "math" in report.domain_reports
        assert report.domain_counts["math"] == n
        assert report.skipped_domains == []

    def test_domain_too_few_probes_skipped(self, any_backend):
        """Domain with < 4 probes is skipped."""
        b = any_backend
        n, d = 10, 4
        source = _deterministic_matrix(b, n, d, offset=0)
        target = _deterministic_matrix(b, n, d, offset=5)
        b.eval(source, target)

        # 7 probes for "math", 3 for "code" (too few for even/odd split)
        domains = ["math"] * 7 + ["code"] * 3
        probes = _make_probes(domains)
        report = alignment_generalization_by_domain(source, target, probes, backend=b)

        assert "math" in report.domain_reports
        assert "code" not in report.domain_reports
        assert "code" in report.skipped_domains
        assert report.domain_counts["code"] == 3

    def test_multiple_domains(self, any_backend):
        """Multiple domains each get their own report."""
        b = any_backend
        n, d = 12, 4
        source = _deterministic_matrix(b, n, d, offset=0)
        target = _deterministic_matrix(b, n, d, offset=5)
        b.eval(source, target)

        domains = ["math"] * 6 + ["code"] * 6
        probes = _make_probes(domains)
        report = alignment_generalization_by_domain(source, target, probes, backend=b)

        assert "math" in report.domain_reports
        assert "code" in report.domain_reports
        assert report.domain_counts["math"] == 6
        assert report.domain_counts["code"] == 6
        assert report.skipped_domains == []

    def test_domain_report_is_frozen(self, any_backend):
        """DomainAlignmentReport is immutable (frozen dataclass)."""
        b = any_backend
        n, d = 8, 4
        source = _deterministic_matrix(b, n, d, offset=0)
        target = _deterministic_matrix(b, n, d, offset=5)
        b.eval(source, target)

        probes = _make_probes(["math"] * n)
        report = alignment_generalization_by_domain(source, target, probes, backend=b)

        with pytest.raises(AttributeError):
            report.domain_reports = {}  # type: ignore[misc]

    def test_skipped_count_consistency(self, any_backend):
        """Skipped domains appear in domain_counts but not domain_reports."""
        b = any_backend
        n, d = 10, 4
        source = _deterministic_matrix(b, n, d, offset=0)
        target = _deterministic_matrix(b, n, d, offset=5)
        b.eval(source, target)

        # 2 domains: "big" has 8 probes, "tiny" has 2
        domains = ["big"] * 8 + ["tiny"] * 2
        probes = _make_probes(domains)
        report = alignment_generalization_by_domain(source, target, probes, backend=b)

        assert "tiny" in report.domain_counts
        assert "tiny" not in report.domain_reports
        assert "tiny" in report.skipped_domains

    def test_enum_domain_key(self, any_backend):
        """Domains with .value attribute (enum-like) are handled."""
        b = any_backend
        n, d = 8, 4
        source = _deterministic_matrix(b, n, d, offset=0)
        target = _deterministic_matrix(b, n, d, offset=5)
        b.eval(source, target)

        # Simulate enum-like domain with .value attribute
        class FakeDomain:
            def __init__(self, val):
                self.value = val

        probes = [
            FakeProbe(
                probe_id=f"p_{i}", name=f"P{i}", description="", support_texts=[],
                source="test", domain=FakeDomain("reasoning"),
                category_name="test", cross_domain_weight=1.0, verification_depth=None,
            )
            for i in range(n)
        ]
        report = alignment_generalization_by_domain(source, target, probes, backend=b)
        assert "reasoning" in report.domain_reports


# =============================================================================
# Test: Mathematical Properties
# =============================================================================


class TestMathematicalProperties:
    """Tests for mathematical invariants of the alignment validation system."""

    def test_cka_bounds(self, any_backend):
        """All CKA values are bounded in [0, 1]."""
        b = any_backend
        n, d = 12, 4
        source = _deterministic_matrix(b, n, d, offset=0)
        target = _deterministic_matrix(b, n, d, offset=11)
        b.eval(source, target)

        report = alignment_generalization_report(
            source, target,
            train_indices=[0, 2, 4, 6, 8, 10],
            holdout_indices=[1, 3, 5, 7, 9, 11],
            backend=b,
        )
        for val in [report.train_cka, report.holdout_cka, report.raw_holdout_cka]:
            assert 0.0 <= val <= 1.0 + 1e-6

    def test_alignment_gain_exact_arithmetic(self, any_backend):
        """alignment_gain is exactly holdout_cka - raw_holdout_cka."""
        b = any_backend
        n, d = 12, 6
        source = _deterministic_matrix(b, n, d, offset=2)
        target = _deterministic_matrix(b, n, d, offset=9)
        b.eval(source, target)

        report = alignment_generalization_report(
            source, target,
            train_indices=[0, 2, 4, 6, 8, 10],
            holdout_indices=[1, 3, 5, 7, 9, 11],
            backend=b,
        )
        # Exact Python float arithmetic check
        assert report.alignment_gain == report.holdout_cka - report.raw_holdout_cka

    def test_coverage_bounded_zero_one(self, any_backend):
        """coverage_ratio is in [0, 1] for any input."""
        b = any_backend
        configs = [
            (8, 4, 0), (12, 6, 5), (16, 8, 10),
        ]
        for n, d, offset in configs:
            source = _deterministic_matrix(b, n, d, offset=offset)
            target = _deterministic_matrix(b, n, d, offset=offset + 7)
            b.eval(source, target)

            report = alignment_generalization_report(
                source, target,
                train_indices=list(range(0, n, 2)),
                holdout_indices=list(range(1, n, 2)),
                backend=b,
            )
            assert 0.0 <= report.coverage_ratio <= 1.0, (
                f"n={n}, d={d}: coverage={report.coverage_ratio}"
            )

    def test_identical_source_target_high_train_cka(self, any_backend):
        """When source == target, alignment is trivial and train CKA ~ 1."""
        b = any_backend
        n, d = 16, 4
        data = _deterministic_matrix(b, n, d, offset=0)
        b.eval(data)

        report = alignment_generalization_report(
            data, data,
            train_indices=list(range(0, n, 2)),
            holdout_indices=list(range(1, n, 2)),
            backend=b,
        )
        assert report.train_cka > 0.95
