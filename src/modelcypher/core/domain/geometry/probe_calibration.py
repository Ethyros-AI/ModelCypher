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

"""Probe Calibration.

Measures per-probe CKA across model pairs to determine empirical alignment quality.
Probes with low CKA values indicate measurement issues requiring investigation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeCalibrationResult:
    """Result of calibrating a single probe across model pairs.

    Attributes
    ----------
    probe_id : str
        Identifier for the probe.
    measured_cka : float
        Mean CKA across model pairs.
    cka_std : float
        Standard deviation of CKA values.
    n_model_pairs : int
        Number of model pairs used for calibration.
    min_cka : float
        Minimum CKA observed.
    max_cka : float
        Maximum CKA observed.
    """

    probe_id: str
    measured_cka: float
    cka_std: float
    n_model_pairs: int
    min_cka: float
    max_cka: float

    @property
    def is_well_calibrated(self) -> bool:
        """A probe is well-calibrated if CKA > 0.9 consistently."""
        return self.measured_cka > 0.9 and self.cka_std < 0.1

    @property
    def needs_investigation(self) -> bool:
        """Low CKA means the measurement is wrong, not the concept."""
        return self.measured_cka < 0.8


@dataclass(frozen=True)
class CalibrationReport:
    """Full calibration report for all probes."""

    per_probe_results: dict[str, ProbeCalibrationResult]
    model_pairs_used: list[tuple[str, str]]  # (model_a, model_b) pairs
    mean_cka: float  # Overall mean across all probes
    well_calibrated_count: int
    needs_investigation_count: int

    def probes_needing_investigation(self) -> list[ProbeCalibrationResult]:
        """Probes with low CKA need their measurement method investigated."""
        return [r for r in self.per_probe_results.values() if r.needs_investigation]

    def well_calibrated_probes(self) -> list[ProbeCalibrationResult]:
        """Probes that consistently achieve high CKA."""
        return [r for r in self.per_probe_results.values() if r.is_well_calibrated]


class ActivationProvider(Protocol):
    """Protocol for getting activations from a model."""

    def get_activations(self, texts: list[str], layer: int) -> list[list[float]]:
        """Get activation vectors for texts at specified layer."""
        ...


class ProbeCalibrator:
    """Calibrates probes by measuring their CKA across model pairs.

    Low CKA values indicate potential issues with probe text, layer selection,
    or embedding extraction methods.
    """

    def __init__(self, backend: Backend | None = None):
        self.backend = backend or get_default_backend()

    def compute_cka(
        self,
        activations_a: list[list[float]],
        activations_b: list[list[float]],
    ) -> float:
        """
        Compute Centered Kernel Alignment between two activation matrices.

        CKA measures similarity of representational geometry - it's invariant
        to orthogonal transformations and isotropic scaling.

        Args:
            activations_a: [n_samples, dim_a] activations from model A
            activations_b: [n_samples, dim_b] activations from model B

        Returns:
            CKA score in [0, 1]. 1.0 = identical geometry.
        """
        if len(activations_a) != len(activations_b):
            raise ValueError("Activation matrices must have same number of samples")

        if len(activations_a) < 2:
            return 0.0

        # Convert to backend arrays
        X = self.backend.array(activations_a, dtype="float32")
        Y = self.backend.array(activations_b, dtype="float32")

        # Center the data
        X = X - self.backend.mean(X, axis=0, keepdims=True)
        Y = Y - self.backend.mean(Y, axis=0, keepdims=True)

        # Compute Gram matrices (linear kernel)
        K = self.backend.matmul(X, self.backend.transpose(X))
        L = self.backend.matmul(Y, self.backend.transpose(Y))

        # Center the Gram matrices (HSIC centering)
        n = K.shape[0]
        H = self.backend.eye(n) - self.backend.ones((n, n)) / n
        K_centered = self.backend.matmul(self.backend.matmul(H, K), H)
        L_centered = self.backend.matmul(self.backend.matmul(H, L), H)

        # CKA = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
        hsic_kl = self.backend.sum(K_centered * L_centered)
        hsic_kk = self.backend.sum(K_centered * K_centered)
        hsic_ll = self.backend.sum(L_centered * L_centered)

        denom = self.backend.sqrt(hsic_kk * hsic_ll)
        if float(denom) < 1e-10:
            return 0.0

        cka = float(hsic_kl / denom)
        return max(0.0, min(1.0, cka))

    def calibrate_probe(
        self,
        probe_id: str,
        probe_texts: list[str],
        model_activations: list[tuple[str, list[list[float]]]],
    ) -> ProbeCalibrationResult:
        """
        Calibrate a single probe by measuring CKA across all model pairs.

        Args:
            probe_id: Unique identifier for the probe
            probe_texts: Text samples for this probe
            model_activations: List of (model_name, activations) for each model
                              where activations is [n_texts, dim] matrix

        Returns:
            CalibrationResult with measured CKA as the weight.
        """
        if len(model_activations) < 2:
            return ProbeCalibrationResult(
                probe_id=probe_id,
                measured_cka=1.0,  # Can't compute with < 2 models
                cka_std=0.0,
                n_model_pairs=0,
                min_cka=1.0,
                max_cka=1.0,
            )

        # Compute CKA for all model pairs
        cka_values: list[float] = []
        for i in range(len(model_activations)):
            for j in range(i + 1, len(model_activations)):
                _, acts_a = model_activations[i]
                _, acts_b = model_activations[j]
                cka = self.compute_cka(acts_a, acts_b)
                cka_values.append(cka)

        if not cka_values:
            return ProbeCalibrationResult(
                probe_id=probe_id,
                measured_cka=1.0,
                cka_std=0.0,
                n_model_pairs=0,
                min_cka=1.0,
                max_cka=1.0,
            )

        mean_cka = sum(cka_values) / len(cka_values)
        variance = sum((x - mean_cka) ** 2 for x in cka_values) / len(cka_values)
        std_cka = variance ** 0.5

        return ProbeCalibrationResult(
            probe_id=probe_id,
            measured_cka=mean_cka,
            cka_std=std_cka,
            n_model_pairs=len(cka_values),
            min_cka=min(cka_values),
            max_cka=max(cka_values),
        )

    def generate_calibration_report(
        self,
        results: list[ProbeCalibrationResult],
        model_pairs: list[tuple[str, str]],
    ) -> CalibrationReport:
        """Generate a full calibration report from individual probe results."""
        per_probe = {r.probe_id: r for r in results}

        if not results:
            return CalibrationReport(
                per_probe_results={},
                model_pairs_used=model_pairs,
                mean_cka=0.0,
                well_calibrated_count=0,
                needs_investigation_count=0,
            )

        mean_cka = sum(r.measured_cka for r in results) / len(results)
        well_calibrated = sum(1 for r in results if r.is_well_calibrated)
        needs_investigation = sum(1 for r in results if r.needs_investigation)

        return CalibrationReport(
            per_probe_results=per_probe,
            model_pairs_used=model_pairs,
            mean_cka=mean_cka,
            well_calibrated_count=well_calibrated,
            needs_investigation_count=needs_investigation,
        )


def load_calibration_weights(calibration_path: str) -> dict[str, float]:
    """Load empirically measured probe weights from calibration file.

    Returns
    -------
    dict
        Mapping probe_id -> measured_cka value.
    """
    import json
    from pathlib import Path

    path = Path(calibration_path)
    if not path.exists():
        logger.warning(f"No calibration file at {calibration_path}, using uniform weights")
        return {}

    with open(path) as f:
        data = json.load(f)

    # Extract measured CKA values
    return {k: v["measured_cka"] for k, v in data.get("per_probe_results", {}).items()}


def save_calibration_weights(report: CalibrationReport, calibration_path: str) -> None:
    """Save calibration results for future use."""
    import json
    from pathlib import Path

    path = Path(calibration_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "model_pairs_used": report.model_pairs_used,
        "mean_cka": report.mean_cka,
        "well_calibrated_count": report.well_calibrated_count,
        "needs_investigation_count": report.needs_investigation_count,
        "per_probe_results": {
            probe_id: {
                "measured_cka": result.measured_cka,
                "cka_std": result.cka_std,
                "n_model_pairs": result.n_model_pairs,
                "min_cka": result.min_cka,
                "max_cka": result.max_cka,
                "is_well_calibrated": result.is_well_calibrated,
                "needs_investigation": result.needs_investigation,
            }
            for probe_id, result in report.per_probe_results.items()
        },
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved calibration to {calibration_path}")
    logger.info(f"  Mean CKA: {report.mean_cka:.3f}")
    logger.info(f"  Well calibrated: {report.well_calibrated_count}")
    logger.info(f"  Needs investigation: {report.needs_investigation_count}")
