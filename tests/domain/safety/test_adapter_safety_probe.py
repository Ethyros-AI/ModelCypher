"""Tests for adapter safety probe types and ABC.

Covers constants, dataclass instantiation, computed properties on
SafetyProbeResult, CompositeProbeResult, ProbeContext, and the
AdapterSafetyProbe abstract base class.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from modelcypher.core.domain.safety.adapter_safety_models import (
    AdapterSafetyTier,
    AdapterSafetyTrigger,
)
from modelcypher.core.domain.safety.adapter_safety_probe import (
    ALL_TIERS,
    AdapterSafetyProbe,
    CompositeProbeResult,
    ProbeContext,
    SafetyProbeResult,
)


# ---------------------------------------------------------------------------
# ALL_TIERS constant
# ---------------------------------------------------------------------------


class TestAllTiers:
    def test_contains_all_tier_values(self) -> None:
        for tier in AdapterSafetyTier:
            assert tier in ALL_TIERS

    def test_is_frozenset(self) -> None:
        assert isinstance(ALL_TIERS, frozenset)

    def test_length_matches_enum(self) -> None:
        assert len(ALL_TIERS) == len(AdapterSafetyTier)


# ---------------------------------------------------------------------------
# SafetyProbeResult tests
# ---------------------------------------------------------------------------


class TestSafetyProbeResult:
    def test_instantiation_minimal(self) -> None:
        result = SafetyProbeResult(probe_name="delta", probe_version="1.0")
        assert result.probe_name == "delta"
        assert result.probe_version == "1.0"
        assert result.details is None
        assert result.findings == ()
        assert result.finding_counts is None

    def test_has_findings_false_when_empty(self) -> None:
        result = SafetyProbeResult(probe_name="probe", probe_version="1.0")
        assert result.has_findings is False

    def test_has_findings_true_when_present(self) -> None:
        result = SafetyProbeResult(
            probe_name="probe",
            probe_version="1.0",
            findings=("high delta in layer 3",),
        )
        assert result.has_findings is True

    def test_frozen(self) -> None:
        result = SafetyProbeResult(probe_name="probe", probe_version="1.0")
        with pytest.raises(AttributeError):
            result.probe_name = "changed"  # type: ignore[misc]

    def test_with_all_fields(self) -> None:
        result = SafetyProbeResult(
            probe_name="canary",
            probe_version="2.1",
            details="Found known-answer deviation",
            findings=("canary_failed_q1", "canary_failed_q2"),
            finding_counts={"canary_failure": 2},
        )
        assert result.details == "Found known-answer deviation"
        assert len(result.findings) == 2
        assert result.finding_counts == {"canary_failure": 2}


# ---------------------------------------------------------------------------
# CompositeProbeResult tests
# ---------------------------------------------------------------------------


class TestCompositeProbeResult:
    @pytest.fixture()
    def probe_with_findings(self) -> SafetyProbeResult:
        return SafetyProbeResult(
            probe_name="delta",
            probe_version="1.0",
            findings=("high_delta",),
            finding_counts={"delta": 1, "drift": 2},
        )

    @pytest.fixture()
    def probe_no_findings(self) -> SafetyProbeResult:
        return SafetyProbeResult(
            probe_name="canary",
            probe_version="2.0",
        )

    @pytest.fixture()
    def probe_with_overlap_counts(self) -> SafetyProbeResult:
        return SafetyProbeResult(
            probe_name="redteam",
            probe_version="3.0",
            findings=("injection_detected",),
            finding_counts={"delta": 3, "injection": 1},
        )

    def test_empty_probe_results(self) -> None:
        composite = CompositeProbeResult(probe_results=())
        assert composite.any_findings is False
        assert composite.all_findings == ()
        assert composite.aggregate_finding_counts == {}
        assert composite.combined_probe_version == ""
        assert composite.findings_probe_count == 0
        assert composite.total_probes == 0
        assert composite.findings_ratio == 0.0

    def test_mixed_probes_any_findings(
        self, probe_with_findings: SafetyProbeResult, probe_no_findings: SafetyProbeResult
    ) -> None:
        composite = CompositeProbeResult(
            probe_results=(probe_with_findings, probe_no_findings)
        )
        assert composite.any_findings is True

    def test_no_findings_across_all_probes(
        self, probe_no_findings: SafetyProbeResult
    ) -> None:
        composite = CompositeProbeResult(
            probe_results=(probe_no_findings, probe_no_findings)
        )
        assert composite.any_findings is False

    def test_all_findings_collects_from_all_probes(
        self,
        probe_with_findings: SafetyProbeResult,
        probe_with_overlap_counts: SafetyProbeResult,
    ) -> None:
        composite = CompositeProbeResult(
            probe_results=(probe_with_findings, probe_with_overlap_counts)
        )
        assert composite.all_findings == ("high_delta", "injection_detected")

    def test_aggregate_finding_counts_merges(
        self,
        probe_with_findings: SafetyProbeResult,
        probe_with_overlap_counts: SafetyProbeResult,
    ) -> None:
        composite = CompositeProbeResult(
            probe_results=(probe_with_findings, probe_with_overlap_counts)
        )
        counts = composite.aggregate_finding_counts
        assert counts["delta"] == 4  # 1 + 3
        assert counts["drift"] == 2
        assert counts["injection"] == 1

    def test_aggregate_finding_counts_skips_none(
        self, probe_no_findings: SafetyProbeResult
    ) -> None:
        composite = CompositeProbeResult(probe_results=(probe_no_findings,))
        assert composite.aggregate_finding_counts == {}

    def test_combined_probe_version(
        self, probe_with_findings: SafetyProbeResult, probe_no_findings: SafetyProbeResult
    ) -> None:
        composite = CompositeProbeResult(
            probe_results=(probe_with_findings, probe_no_findings)
        )
        assert composite.combined_probe_version == "1.0+2.0"

    def test_findings_probe_count(
        self,
        probe_with_findings: SafetyProbeResult,
        probe_no_findings: SafetyProbeResult,
        probe_with_overlap_counts: SafetyProbeResult,
    ) -> None:
        composite = CompositeProbeResult(
            probe_results=(probe_with_findings, probe_no_findings, probe_with_overlap_counts)
        )
        assert composite.findings_probe_count == 2

    def test_total_probes(
        self, probe_with_findings: SafetyProbeResult, probe_no_findings: SafetyProbeResult
    ) -> None:
        composite = CompositeProbeResult(
            probe_results=(probe_with_findings, probe_no_findings)
        )
        assert composite.total_probes == 2

    def test_findings_ratio(
        self,
        probe_with_findings: SafetyProbeResult,
        probe_no_findings: SafetyProbeResult,
        probe_with_overlap_counts: SafetyProbeResult,
    ) -> None:
        composite = CompositeProbeResult(
            probe_results=(probe_with_findings, probe_no_findings, probe_with_overlap_counts)
        )
        assert composite.findings_ratio == pytest.approx(2 / 3)

    def test_findings_ratio_all_have_findings(
        self,
        probe_with_findings: SafetyProbeResult,
        probe_with_overlap_counts: SafetyProbeResult,
    ) -> None:
        composite = CompositeProbeResult(
            probe_results=(probe_with_findings, probe_with_overlap_counts)
        )
        assert composite.findings_ratio == pytest.approx(1.0)

    def test_findings_ratio_none_have_findings(
        self, probe_no_findings: SafetyProbeResult
    ) -> None:
        composite = CompositeProbeResult(probe_results=(probe_no_findings,))
        assert composite.findings_ratio == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ProbeContext tests
# ---------------------------------------------------------------------------


class TestProbeContext:
    def test_instantiation_required_fields(self) -> None:
        ctx = ProbeContext(
            adapter_path=Path("/tmp/adapter"),
            tier=AdapterSafetyTier.STANDARD,
            trigger=AdapterSafetyTrigger.IMPORT_ADAPTER,
        )
        assert ctx.adapter_path == Path("/tmp/adapter")
        assert ctx.tier == AdapterSafetyTier.STANDARD
        assert ctx.trigger == AdapterSafetyTrigger.IMPORT_ADAPTER

    def test_default_optional_fields(self) -> None:
        ctx = ProbeContext(
            adapter_path=Path("/tmp/adapter"),
            tier=AdapterSafetyTier.QUICK,
            trigger=AdapterSafetyTrigger.MERGE,
        )
        assert ctx.adapter_id is None
        assert ctx.adapter_name is None
        assert ctx.inference_hook is None
        assert ctx.adapter_description is None
        assert ctx.skill_tags == ()
        assert ctx.creator is None
        assert ctx.base_model_id is None
        assert ctx.target_modules == ()
        assert ctx.training_datasets == ()
        assert ctx.embedder is None

    def test_with_metadata_fields(self) -> None:
        ctx = ProbeContext(
            adapter_path=Path("/tmp/adapter"),
            tier=AdapterSafetyTier.FULL,
            trigger=AdapterSafetyTrigger.POST_TRAINING,
            adapter_id="adapter-001",
            adapter_name="My Adapter",
            adapter_description="A fine-tuned adapter",
            skill_tags=("math", "reasoning"),
            creator="user@example.com",
            base_model_id="LFM2-350M",
            target_modules=("attention.wq", "attention.wk"),
            training_datasets=("dataset-a", "dataset-b"),
        )
        assert ctx.adapter_id == "adapter-001"
        assert ctx.skill_tags == ("math", "reasoning")
        assert ctx.target_modules == ("attention.wq", "attention.wk")

    def test_frozen(self) -> None:
        ctx = ProbeContext(
            adapter_path=Path("/tmp/adapter"),
            tier=AdapterSafetyTier.QUICK,
            trigger=AdapterSafetyTrigger.MERGE,
        )
        with pytest.raises(AttributeError):
            ctx.tier = AdapterSafetyTier.FULL  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AdapterSafetyProbe ABC tests
# ---------------------------------------------------------------------------


class _StubProbe(AdapterSafetyProbe):
    """Concrete subclass for testing the ABC."""

    def __init__(self, tiers: frozenset[AdapterSafetyTier]) -> None:
        self._tiers = tiers

    @property
    def name(self) -> str:
        return "stub_probe"

    @property
    def version(self) -> str:
        return "0.1"

    @property
    def supported_tiers(self) -> frozenset[AdapterSafetyTier]:
        return self._tiers

    async def evaluate(self, context: ProbeContext) -> SafetyProbeResult:
        return SafetyProbeResult(probe_name=self.name, probe_version=self.version)


class TestAdapterSafetyProbe:
    def test_should_run_true_when_tier_supported(self) -> None:
        probe = _StubProbe(tiers=frozenset({AdapterSafetyTier.QUICK, AdapterSafetyTier.FULL}))
        assert probe.should_run(AdapterSafetyTier.QUICK) is True
        assert probe.should_run(AdapterSafetyTier.FULL) is True

    def test_should_run_false_when_tier_not_supported(self) -> None:
        probe = _StubProbe(tiers=frozenset({AdapterSafetyTier.QUICK}))
        assert probe.should_run(AdapterSafetyTier.STANDARD) is False
        assert probe.should_run(AdapterSafetyTier.FULL) is False

    def test_should_run_all_tiers(self) -> None:
        probe = _StubProbe(tiers=ALL_TIERS)
        for tier in AdapterSafetyTier:
            assert probe.should_run(tier) is True

    def test_abstract_properties(self) -> None:
        probe = _StubProbe(tiers=frozenset())
        assert probe.name == "stub_probe"
        assert probe.version == "0.1"

    def test_cannot_instantiate_abc_directly(self) -> None:
        with pytest.raises(TypeError):
            AdapterSafetyProbe()  # type: ignore[abstract]
