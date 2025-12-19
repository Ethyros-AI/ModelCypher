from __future__ import annotations

from modelcypher.core.domain.geometry_validation_suite import GeometryValidationSuite


def test_geometry_validation_suite_runs() -> None:
    report = GeometryValidationSuite.run()
    assert report.suite_version == "1.0"
    assert report.gromov_wasserstein is not None
    assert report.traversal_coherence is not None
    assert report.path_signature is not None
