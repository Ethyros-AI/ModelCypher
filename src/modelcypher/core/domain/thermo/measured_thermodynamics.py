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

"""Measured thermodynamics - real physics from real observations.

This module replaces hardcoded energy levels and intensity scores with
values derived from empirical measurement. The thermodynamics of language
models is not metaphorical - it IS thermodynamics. The math is identical.

Key insight: If we observe a probability distribution p(x) at temperature T,
the energy of state x is:

    E(x) = -T * log(p(x)) + constant

This is exact. The constant drops out when computing relative energies:

    E(x) - E(y) = -T * log(p(x)/p(y))

We measure p(x) from observed behavioral outcomes. We derive E(x) from the
measurement. No guessing. No vibes. Just physics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    exp_scalar,
    log_scalar,
    sqrt_scalar,
)
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _log_safe_min(backend) -> float:
    info = backend.finfo()
    return float(info.tiny)


@dataclass(frozen=True)
class MeasuredEnergy:
    """Energy level derived from measured probability.

    Attributes
    ----------
    value : float
        The energy value (relative to reference state).
    probability : float
        The measured probability that yielded this energy.
    sample_count : int
        Number of observations used to compute probability.
    temperature : float
        Temperature at which measurement was taken.
    """

    value: float
    probability: float
    sample_count: int
    temperature: float

    @classmethod
    def from_probability(
        cls,
        probability: float,
        reference_probability: float,
        temperature: float,
        sample_count: int,
    ) -> MeasuredEnergy:
        """Derive energy from observed probability relative to reference.

        E(x) - E(ref) = -T * log(p(x)/p(ref))

        Parameters
        ----------
        probability : float
            Observed probability of this state.
        reference_probability : float
            Probability of the reference state (E=0).
        temperature : float
            Temperature at which measurement was taken.
        sample_count : int
            Number of observations.

        Returns
        -------
        MeasuredEnergy
            Energy derived from the measurement.
        """
        # Numerical stability - use dtype-derived tiny for log safety
        _b = get_default_backend()
        safe_min = _log_safe_min(_b)
        p = max(probability, safe_min)
        p_ref = max(reference_probability, safe_min)

        # E(x) - E(ref) = -T * log(p(x)/p(ref))
        energy = -temperature * log_scalar(p / p_ref, _b)

        return cls(
            value=energy,
            probability=probability,
            sample_count=sample_count,
            temperature=temperature,
        )

    @property
    def confidence(self) -> float:
        """Confidence in this measurement based on sample count.

        Returns
        -------
        float
            Confidence score [0, 1]. Higher sample counts yield higher confidence.
        """
        # Simple model: confidence approaches 1 as samples increase
        # 100 samples -> 0.9, 1000 samples -> 0.99
        return 1.0 - 1.0 / (1.0 + self.sample_count / 10.0)


@dataclass
class MeasuredBasinTopology:
    """Basin topology derived from empirical measurement.

    Instead of hardcoding energy levels, we measure them from observed
    behavioral outcome distributions. The energy of each basin is derived
    from the probability of that outcome.

    Attributes
    ----------
    refusal_energy : MeasuredEnergy
        Energy of refusal basin (reference, always 0).
    caution_energy : MeasuredEnergy
        Energy of caution/hedging basin.
    solution_energy : MeasuredEnergy
        Energy of solution basin.
    transition_ridge : MeasuredEnergy
        Energy of the transition ridge (barrier).
    temperature : float
        Temperature at which measurements were taken.
    model_id : str
        Identifier for the model that was measured.
    calibration_timestamp : datetime
        When this calibration was performed.
    """

    refusal_energy: MeasuredEnergy
    caution_energy: MeasuredEnergy
    solution_energy: MeasuredEnergy
    transition_ridge: MeasuredEnergy
    temperature: float
    model_id: str
    calibration_timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_outcome_counts(
        cls,
        refused_count: int,
        hedged_count: int,
        attempted_count: int,
        solved_count: int,
        temperature: float,
        model_id: str,
        escape_rate: float | None = None,
    ) -> MeasuredBasinTopology:
        """Derive basin topology from observed outcome counts.

        Parameters
        ----------
        refused_count : int
            Number of refusal outcomes observed.
        hedged_count : int
            Number of hedged/caution outcomes observed.
        attempted_count : int
            Number of attempted outcomes observed.
        solved_count : int
            Number of solved outcomes observed.
        temperature : float
            Temperature at which measurements were taken.
        model_id : str
            Identifier for the model.
        escape_rate : float | None
            Observed rate of escaping caution basin. If None, derived from
            attempted + solved counts.

        Returns
        -------
        MeasuredBasinTopology
            Calibrated topology from measurements.
        """
        total = refused_count + hedged_count + attempted_count + solved_count
        if total == 0:
            raise ValueError("Cannot derive topology from zero observations")

        # Compute probabilities
        p_refused = refused_count / total
        p_hedged = hedged_count / total
        p_attempted = attempted_count / total
        p_solved = solved_count / total

        _b = get_default_backend()
        # Reference state: refusal (E=0)
        safe_min = _log_safe_min(_b)
        p_ref = max(p_refused, safe_min)

        # Derive energies: E(x) = -T * log(p(x)/p(ref))
        refusal = MeasuredEnergy(
            value=0.0,  # Reference
            probability=p_refused,
            sample_count=refused_count,
            temperature=temperature,
        )

        caution = MeasuredEnergy.from_probability(
            probability=p_hedged,
            reference_probability=p_ref,
            temperature=temperature,
            sample_count=hedged_count,
        )

        solution = MeasuredEnergy.from_probability(
            probability=p_solved,
            reference_probability=p_ref,
            temperature=temperature,
            sample_count=solved_count,
        )

        # Transition ridge: derive from escape rate
        # P_escape = exp(-ΔE/T) where ΔE = E_ridge - E_caution
        # ΔE = -T * log(P_escape)
        # E_ridge = E_caution + ΔE = E_caution - T * log(P_escape)
        if escape_rate is None:
            # Escape rate = P(attempted or solved | started in caution)
            # Approximate as (attempted + solved) / total
            escape_rate = (p_attempted + p_solved)

        escape_rate = max(escape_rate, safe_min)  # Log safety
        barrier_height = -temperature * log_scalar(escape_rate, _b)
        ridge_energy = caution.value + barrier_height

        ridge = MeasuredEnergy(
            value=ridge_energy,
            probability=escape_rate,  # Store escape rate as "probability"
            sample_count=attempted_count + solved_count,
            temperature=temperature,
        )

        return cls(
            refusal_energy=refusal,
            caution_energy=caution,
            solution_energy=solution,
            transition_ridge=ridge,
            temperature=temperature,
            model_id=model_id,
        )

    def escape_probability(self, temperature: float) -> float:
        """Escape probability from caution basin to solution basin.

        P_escape = exp(-(E_ridge - E_caution) / T)

        Parameters
        ----------
        temperature : float
            Temperature for prediction.

        Returns
        -------
        float
            Predicted escape probability.
        """
        if temperature <= 0:
            return 0.0
        barrier = self.transition_ridge.value - self.caution_energy.value
        return exp_scalar(-barrier / temperature, get_default_backend())

    def basin_weights(self, temperature: float) -> list[float]:
        """Boltzmann weights for each basin at given temperature.

        w_i = exp(-E_i / T)
        p_i = w_i / Z

        Parameters
        ----------
        temperature : float
            Temperature for prediction.

        Returns
        -------
        list[float]
            Normalized weights for each basin [basin_0, basin_1, basin_2].
            Indices correspond to energy levels in order of increasing energy.
        """
        if temperature <= 0:
            return [1.0, 0.0, 0.0]

        _b = get_default_backend()
        w_0 = exp_scalar(-self.refusal_energy.value / temperature, _b)
        w_1 = exp_scalar(-self.caution_energy.value / temperature, _b)
        w_2 = exp_scalar(-self.solution_energy.value / temperature, _b)
        partition = w_0 + w_1 + w_2

        if partition <= 0:
            return [0.33, 0.33, 0.33]

        return [w_0 / partition, w_1 / partition, w_2 / partition]

    def to_dict(self) -> dict:
        """Serialize to dictionary for storage."""
        return {
            "refusal_energy": {
                "value": self.refusal_energy.value,
                "probability": self.refusal_energy.probability,
                "sample_count": self.refusal_energy.sample_count,
                "temperature": self.refusal_energy.temperature,
            },
            "caution_energy": {
                "value": self.caution_energy.value,
                "probability": self.caution_energy.probability,
                "sample_count": self.caution_energy.sample_count,
                "temperature": self.caution_energy.temperature,
            },
            "solution_energy": {
                "value": self.solution_energy.value,
                "probability": self.solution_energy.probability,
                "sample_count": self.solution_energy.sample_count,
                "temperature": self.solution_energy.temperature,
            },
            "transition_ridge": {
                "value": self.transition_ridge.value,
                "probability": self.transition_ridge.probability,
                "sample_count": self.transition_ridge.sample_count,
                "temperature": self.transition_ridge.temperature,
            },
            "temperature": self.temperature,
            "model_id": self.model_id,
            "calibration_timestamp": self.calibration_timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> MeasuredBasinTopology:
        """Deserialize from dictionary."""
        return cls(
            refusal_energy=MeasuredEnergy(**data["refusal_energy"]),
            caution_energy=MeasuredEnergy(**data["caution_energy"]),
            solution_energy=MeasuredEnergy(**data["solution_energy"]),
            transition_ridge=MeasuredEnergy(**data["transition_ridge"]),
            temperature=data["temperature"],
            model_id=data["model_id"],
            calibration_timestamp=datetime.fromisoformat(data["calibration_timestamp"]),
        )


@dataclass(frozen=True)
class MeasuredModifierEffect:
    """Modifier effect derived from empirical measurement.

    Instead of hardcoding intensity scores, we measure the actual entropy
    change (delta_H) caused by each modifier.

    Attributes
    ----------
    modifier : str
        Name of the linguistic modifier.
    mean_delta_h : float
        Mean entropy change observed.
    std_delta_h : float
        Standard deviation of entropy change.
    sample_count : int
        Number of measurements.
    temperature : float
        Temperature at which measurements were taken.
    """

    modifier: str
    mean_delta_h: float
    std_delta_h: float
    sample_count: int
    temperature: float

    @property
    def intensity_score(self) -> float:
        """Raw intensity magnitude from measured effect.

        Returns the absolute value of mean_delta_h directly. No arbitrary
        normalization - the raw measurement is the measurement.

        Returns
        -------
        float
            Absolute value of mean entropy change.
        """
        return abs(self.mean_delta_h)


@dataclass
class MeasuredModifierProfile:
    """Complete profile of modifier effects for a model.

    Attributes
    ----------
    effects : dict[str, MeasuredModifierEffect]
        Measured effects for each modifier.
    model_id : str
        Identifier for the model.
    calibration_timestamp : datetime
        When this calibration was performed.
    """

    effects: dict[str, MeasuredModifierEffect]
    model_id: str
    calibration_timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_measurements(
        cls,
        measurements: list[tuple[str, float]],
        temperature: float,
        model_id: str,
    ) -> MeasuredModifierProfile:
        """Build profile from list of (modifier, delta_h) measurements.

        Parameters
        ----------
        measurements : list[tuple[str, float]]
            List of (modifier_name, delta_h) pairs.
        temperature : float
            Temperature at which measurements were taken.
        model_id : str
            Identifier for the model.

        Returns
        -------
        MeasuredModifierProfile
            Calibrated modifier profile.
        """
        # Group by modifier
        by_modifier: dict[str, list[float]] = {}
        for modifier, delta_h in measurements:
            if modifier not in by_modifier:
                by_modifier[modifier] = []
            by_modifier[modifier].append(delta_h)

        # Compute statistics for each
        effects = {}
        for modifier, deltas in by_modifier.items():
            n = len(deltas)
            mean = sum(deltas) / n
            variance = sum((d - mean) ** 2 for d in deltas) / n if n > 1 else 0.0
            std = sqrt_scalar(variance, get_default_backend())

            effects[modifier] = MeasuredModifierEffect(
                modifier=modifier,
                mean_delta_h=mean,
                std_delta_h=std,
                sample_count=n,
                temperature=temperature,
            )

        return cls(
            effects=effects,
            model_id=model_id,
        )

    def get_intensity(self, modifier: str) -> float:
        """Get measured intensity score for a modifier.

        Parameters
        ----------
        modifier : str
            Modifier name.

        Returns
        -------
        float
            Measured intensity score, or 0.0 if not calibrated.
        """
        if modifier in self.effects:
            return self.effects[modifier].intensity_score
        return 0.0

    def to_dict(self) -> dict:
        """Serialize to dictionary for storage."""
        return {
            "effects": {
                name: {
                    "modifier": effect.modifier,
                    "mean_delta_h": effect.mean_delta_h,
                    "std_delta_h": effect.std_delta_h,
                    "sample_count": effect.sample_count,
                    "temperature": effect.temperature,
                }
                for name, effect in self.effects.items()
            },
            "model_id": self.model_id,
            "calibration_timestamp": self.calibration_timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> MeasuredModifierProfile:
        """Deserialize from dictionary."""
        effects = {
            name: MeasuredModifierEffect(**effect_data)
            for name, effect_data in data["effects"].items()
        }
        return cls(
            effects=effects,
            model_id=data["model_id"],
            calibration_timestamp=datetime.fromisoformat(data["calibration_timestamp"]),
        )


@dataclass
class MeasuredThresholds:
    """Outcome classification thresholds derived from baseline measurements.

    Instead of hardcoding entropy thresholds (4.0, 3.0, etc.), we derive them
    from the baseline distribution of entropy measurements.

    Attributes
    ----------
    refused_threshold : float
        Entropy above which outcome is classified as REFUSED.
    hedged_threshold : float
        Entropy above which outcome is classified as HEDGED.
    attempted_threshold : float
        Entropy above which outcome is classified as ATTEMPTED.
    percentiles : dict[int, float]
        Baseline entropy percentiles.
    model_id : str
        Identifier for the model.
    variance_threshold : float | None
        Variance threshold for distinguishing refused vs hedged.
        If None, high-entropy outcomes are classified as "hedged" (no refused/hedged distinction).
        Should be derived from baseline variance measurements.
    calibration_timestamp : datetime
        When this calibration was performed.
    """

    refused_threshold: float
    hedged_threshold: float
    attempted_threshold: float
    percentiles: dict[int, float]
    model_id: str
    variance_threshold: float | None = None
    calibration_timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_baseline_entropies(
        cls,
        entropies: list[float],
        model_id: str,
        refused_percentile: float = 95.0,
        hedged_percentile: float = 75.0,
        attempted_percentile: float = 50.0,
    ) -> MeasuredThresholds:
        """Derive thresholds from baseline entropy distribution.

        Parameters
        ----------
        entropies : list[float]
            Baseline entropy measurements.
        model_id : str
            Identifier for the model.
        refused_percentile : float
            Percentile for REFUSED threshold (default 95th).
        hedged_percentile : float
            Percentile for HEDGED threshold (default 75th).
        attempted_percentile : float
            Percentile for ATTEMPTED threshold (default 50th).

        Returns
        -------
        MeasuredThresholds
            Calibrated thresholds from baseline.
        """
        if not entropies:
            raise ValueError("Cannot derive thresholds from empty baseline")

        sorted_e = sorted(entropies)
        n = len(sorted_e)

        def percentile(p: float) -> float:
            idx = int(n * p / 100.0)
            return sorted_e[min(idx, n - 1)]

        percentiles = {
            5: percentile(5),
            25: percentile(25),
            50: percentile(50),
            75: percentile(75),
            95: percentile(95),
            99: percentile(99),
        }

        return cls(
            refused_threshold=percentile(refused_percentile),
            hedged_threshold=percentile(hedged_percentile),
            attempted_threshold=percentile(attempted_percentile),
            percentiles=percentiles,
            model_id=model_id,
        )

    def classify_outcome(self, entropy: float, variance: float) -> str:
        """Classify behavioral outcome from entropy using measured thresholds.

        Parameters
        ----------
        entropy : float
            Measured entropy.
        variance : float
            Entropy variance.

        Returns
        -------
        str
            Outcome classification: "refused", "hedged", "attempted", or "solved".
        """
        if entropy >= self.refused_threshold:
            # Use calibrated variance threshold if available
            if self.variance_threshold is not None and variance < self.variance_threshold:
                # High entropy + low variance = stuck on one token/pattern
                return "refused"
            return "hedged"
        elif entropy >= self.hedged_threshold:
            return "hedged"
        elif entropy >= self.attempted_threshold:
            return "attempted"
        return "solved"

    def to_dict(self) -> dict:
        """Serialize to dictionary for storage."""
        return {
            "refused_threshold": self.refused_threshold,
            "hedged_threshold": self.hedged_threshold,
            "attempted_threshold": self.attempted_threshold,
            "percentiles": self.percentiles,
            "model_id": self.model_id,
            "variance_threshold": self.variance_threshold,
            "calibration_timestamp": self.calibration_timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> MeasuredThresholds:
        """Deserialize from dictionary."""
        return cls(
            refused_threshold=data["refused_threshold"],
            hedged_threshold=data["hedged_threshold"],
            attempted_threshold=data["attempted_threshold"],
            percentiles={int(k): v for k, v in data["percentiles"].items()},
            model_id=data["model_id"],
            variance_threshold=data.get("variance_threshold"),
            calibration_timestamp=datetime.fromisoformat(data["calibration_timestamp"]),
        )


@dataclass
class ThermoCalibration:
    """Complete thermodynamic calibration for a model.

    Combines basin topology, modifier profile, and classification thresholds
    into a single calibration artifact that can be stored and loaded.

    Attributes
    ----------
    basin_topology : MeasuredBasinTopology | None
        Calibrated basin energy levels.
    modifier_profile : MeasuredModifierProfile | None
        Calibrated modifier effects.
    thresholds : MeasuredThresholds | None
        Calibrated classification thresholds.
    model_id : str
        Identifier for the model.
    calibration_timestamp : datetime
        When this calibration was performed.
    """

    basin_topology: MeasuredBasinTopology | None
    modifier_profile: MeasuredModifierProfile | None
    thresholds: MeasuredThresholds | None
    model_id: str
    calibration_timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_complete(self) -> bool:
        """Whether all calibration components are present."""
        return all([
            self.basin_topology is not None,
            self.modifier_profile is not None,
            self.thresholds is not None,
        ])

    def save(self, path: Path) -> None:
        """Save calibration to JSON file.

        Parameters
        ----------
        path : Path
            Output file path.
        """
        data = {
            "model_id": self.model_id,
            "calibration_timestamp": self.calibration_timestamp.isoformat(),
            "basin_topology": self.basin_topology.to_dict() if self.basin_topology else None,
            "modifier_profile": self.modifier_profile.to_dict() if self.modifier_profile else None,
            "thresholds": self.thresholds.to_dict() if self.thresholds else None,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved thermodynamic calibration to {path}")

    @classmethod
    def load(cls, path: Path) -> ThermoCalibration:
        """Load calibration from JSON file.

        Parameters
        ----------
        path : Path
            Input file path.

        Returns
        -------
        ThermoCalibration
            Loaded calibration.
        """
        with open(path) as f:
            data = json.load(f)

        return cls(
            basin_topology=(
                MeasuredBasinTopology.from_dict(data["basin_topology"])
                if data.get("basin_topology")
                else None
            ),
            modifier_profile=(
                MeasuredModifierProfile.from_dict(data["modifier_profile"])
                if data.get("modifier_profile")
                else None
            ),
            thresholds=(
                MeasuredThresholds.from_dict(data["thresholds"])
                if data.get("thresholds")
                else None
            ),
            model_id=data["model_id"],
            calibration_timestamp=datetime.fromisoformat(data["calibration_timestamp"]),
        )


class UncalibratedError(Exception):
    """Raised when thermodynamic operation requires calibration that doesn't exist."""

    def __init__(self, operation: str, model_id: str | None = None):
        self.operation = operation
        self.model_id = model_id
        msg = f"Operation '{operation}' requires thermodynamic calibration"
        if model_id:
            msg += f" for model '{model_id}'"
        msg += ". Run 'mc thermo calibrate' first."
        super().__init__(msg)
