from __future__ import annotations

from modelcypher.core.domain.sparse_region_validator import (
    BaselineMetrics,
    SparseRegionValidator,
)


def test_sparse_region_validator_analyze_results() -> None:
    validator = SparseRegionValidator()
    baseline = BaselineMetrics(
        mean_entropy=1.0,
        entropy_std_dev=0.1,
        refusal_rate=0.05,
        coherence_score=0.8,
        per_prompt_entropy=[1.0, 1.1],
        duration=0.1,
    )
    post = BaselineMetrics(
        mean_entropy=1.02,
        entropy_std_dev=0.1,
        refusal_rate=0.06,
        coherence_score=0.81,
        per_prompt_entropy=[1.02, 1.0],
        duration=0.1,
    )
    result = validator.analyze_results(baseline=baseline, post_perturbation=post, perturbed_layers=[1, 2])
    assert result.capabilities_preserved is True
    assert result.assessment.entropy_ok is True


def test_sparse_region_validator_helpers() -> None:
    coherence = SparseRegionValidator.compute_coherence([1.0, 1.0, 1.0])
    assert coherence == 1.0
    assert SparseRegionValidator.detect_refusal("I cannot comply with that request.")


def test_validation_report_contains_fields() -> None:
    baseline = BaselineMetrics(
        mean_entropy=1.0,
        entropy_std_dev=0.1,
        refusal_rate=0.05,
        coherence_score=0.8,
        per_prompt_entropy=[1.0, 1.1],
        duration=0.1,
    )
    validator = SparseRegionValidator()
    result = validator.analyze_results(baseline=baseline, post_perturbation=baseline, perturbed_layers=[0])
    report = result.generate_report()
    assert "Capability Preservation Validation Report" in report
