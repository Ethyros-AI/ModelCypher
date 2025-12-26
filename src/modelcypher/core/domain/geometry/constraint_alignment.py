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

"""Constraint-Based Alignment.

Probes serve as calibration tools for determining concept locations in hyperspace.
The alignment finds where concepts live in both models and aligns those locations.

Notes
-----
Each probe provides a constraint on where a concept lives. Probe disagreement
indicates measurement error to investigate, not uncertainty to average.

The approach is analogous to GPS triangulation: each satellite (probe) provides
a constraint, and the intersection of constraints determines the position
(layer mapping).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from modelcypher.core.domain._backend import Backend, get_default_backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeConstraint:
    """A constraint from a single probe about layer correspondence.

    Each probe tells us: "This concept peaks at layer X in model A
    and layer Y in model B."

    This is NOT a vote with weight. It's a calibration measurement.
    """

    probe_id: str
    source_peak_layer: int  # Layer in model A where this concept peaks
    target_peak_layer: int  # Layer in model B where this concept peaks
    source_activation: float  # How strongly concept activates at peak
    target_activation: float  # How strongly concept activates at peak
    agreement_score: float  # How well the activations match (CKA at this layer pair)


@dataclass(frozen=True)
class LayerCorrespondence:
    """A potential layer correspondence with supporting/conflicting evidence."""

    source_layer: int
    target_layer: int
    supporting_probes: tuple[str, ...]  # Probes that agree on this mapping
    conflicting_probes: tuple[str, ...]  # Probes that disagree
    consensus_ratio: float  # supporting / (supporting + conflicting)

    @property
    def is_unanimous(self) -> bool:
        """True if ALL probes agree on this mapping."""
        return len(self.conflicting_probes) == 0 and len(self.supporting_probes) > 0

    @property
    def has_conflicts(self) -> bool:
        """True if any probes disagree."""
        return len(self.conflicting_probes) > 0


@dataclass(frozen=True)
class ConstraintAlignmentResult:
    """Result of constraint-based alignment between two models.

    Success = all probes agree on layer mappings (CKA = 1.0 achievable)
    Failure = probes conflict (measurement errors to investigate)
    """

    layer_mappings: tuple[LayerCorrespondence, ...]
    unanimous_mappings: int  # How many mappings have full consensus
    conflicting_mappings: int  # How many mappings have disagreement
    probes_needing_investigation: tuple[str, ...]  # Probes that conflict

    @property
    def is_fully_aligned(self) -> bool:
        """True if all layer mappings have unanimous probe agreement."""
        return self.conflicting_mappings == 0 and self.unanimous_mappings > 0

    @property
    def alignment_quality(self) -> str:
        """Human-readable assessment."""
        if self.is_fully_aligned:
            return "ALIGNED: All probes agree. CKA = 1.0 achievable."
        elif self.conflicting_mappings > 0:
            return (
                f"CONFLICTS: {len(self.probes_needing_investigation)} probes disagree. "
                "These probe measurements need investigation - the concepts ARE invariant, "
                "the measurement is wrong."
            )
        else:
            return "INCOMPLETE: Need more probes for alignment."


class ConstraintAligner:
    """
    Aligns models using probe constraints, not weighted voting.

    Each probe is a calibration tool that tells us where a concept lives.
    We find consensus across probes. Disagreement = measurement error.
    """

    def __init__(self, backend: Backend | None = None, cka_threshold: float = 0.95):
        """
        Initialize the constraint aligner.

        Args:
            backend: Compute backend
            cka_threshold: CKA above this means "agreement" (default 0.95)
        """
        self.backend = backend or get_default_backend()
        self.cka_threshold = cka_threshold

    def find_peak_layer(
        self,
        activations_by_layer: dict[int, list[float]],
    ) -> tuple[int, float]:
        """
        Find which layer a concept peaks at.

        The peak is where the concept lives. Not weighted. Not averaged.
        Just: where is activation highest?

        Returns:
            (peak_layer, peak_activation)
        """
        if not activations_by_layer:
            return (-1, 0.0)

        peak_layer = -1
        peak_activation = -float("inf")

        for layer, activations in activations_by_layer.items():
            # Use L2 norm as activation strength
            strength = sum(a * a for a in activations) ** 0.5
            if strength > peak_activation:
                peak_activation = strength
                peak_layer = layer

        return (peak_layer, peak_activation)

    def compute_per_layer_cka(
        self,
        source_activations: dict[int, list[float]],  # layer -> activation vector
        target_activations: dict[int, list[float]],
    ) -> dict[tuple[int, int], float]:
        """
        Compute CKA for each (source_layer, target_layer) pair.

        This tells us which layer pairs have matching conceptual geometry.
        A pair with CKA ≈ 1.0 means the concepts are identically positioned.
        """
        from modelcypher.core.domain.geometry.cka import compute_cka
        cka_matrix: dict[tuple[int, int], float] = {}

        for source_layer, source_acts in source_activations.items():
            for target_layer, target_acts in target_activations.items():
                min_len = min(len(source_acts), len(target_acts))
                if min_len < 2:
                    cka_matrix[(source_layer, target_layer)] = 0.0
                    continue

                # Treat dimensions as samples to avoid the n=1 CKA degeneracy.
                source_vec = self.backend.array(source_acts[:min_len])
                target_vec = self.backend.array(target_acts[:min_len])
                source_mat = self.backend.reshape(source_vec, (-1, 1))
                target_mat = self.backend.reshape(target_vec, (-1, 1))
                result = compute_cka(source_mat, target_mat, backend=self.backend)
                cka_matrix[(source_layer, target_layer)] = result.cka if result.is_valid else 0.0

        return cka_matrix

    def extract_constraints(
        self,
        probe_id: str,
        source_activations: dict[int, list[float]],
        target_activations: dict[int, list[float]],
    ) -> ProbeConstraint:
        """
        Extract the constraint from a single probe.

        The probe tells us: "This concept is at layer X in source, layer Y in target."
        That's a constraint, not a vote.
        """
        source_peak, source_strength = self.find_peak_layer(source_activations)
        target_peak, target_strength = self.find_peak_layer(target_activations)

        # Compute CKA at the peak layers
        cka_matrix = self.compute_per_layer_cka(source_activations, target_activations)
        agreement = cka_matrix.get((source_peak, target_peak), 0.0)

        return ProbeConstraint(
            probe_id=probe_id,
            source_peak_layer=source_peak,
            target_peak_layer=target_peak,
            source_activation=source_strength,
            target_activation=target_strength,
            agreement_score=agreement,
        )

    def align_from_constraints(
        self,
        constraints: list[ProbeConstraint],
    ) -> ConstraintAlignmentResult:
        """
        Build layer mapping from probe constraints.

        Each constraint says "source layer X corresponds to target layer Y."
        We find which mappings have consensus and which have conflicts.

        Conflicts = measurement errors to investigate.
        """
        # Group constraints by source layer
        constraints_by_source: dict[int, list[ProbeConstraint]] = {}
        for c in constraints:
            if c.source_peak_layer not in constraints_by_source:
                constraints_by_source[c.source_peak_layer] = []
            constraints_by_source[c.source_peak_layer].append(c)

        mappings: list[LayerCorrespondence] = []
        all_conflicting_probes: set[str] = set()

        for source_layer, layer_constraints in constraints_by_source.items():
            # Find which target layer has most support
            target_votes: dict[int, list[str]] = {}  # target_layer -> [probe_ids]
            for c in layer_constraints:
                if c.target_peak_layer not in target_votes:
                    target_votes[c.target_peak_layer] = []
                target_votes[c.target_peak_layer].append(c.probe_id)

            if not target_votes:
                continue

            # Find majority target
            best_target = max(target_votes.keys(), key=lambda t: len(target_votes[t]))
            supporting = target_votes[best_target]
            conflicting = [
                pid
                for t, pids in target_votes.items()
                if t != best_target
                for pid in pids
            ]

            all_conflicting_probes.update(conflicting)

            total = len(supporting) + len(conflicting)
            consensus = len(supporting) / total if total > 0 else 0.0

            mappings.append(
                LayerCorrespondence(
                    source_layer=source_layer,
                    target_layer=best_target,
                    supporting_probes=tuple(supporting),
                    conflicting_probes=tuple(conflicting),
                    consensus_ratio=consensus,
                )
            )

        unanimous = sum(1 for m in mappings if m.is_unanimous)
        conflicting = sum(1 for m in mappings if m.has_conflicts)

        return ConstraintAlignmentResult(
            layer_mappings=tuple(mappings),
            unanimous_mappings=unanimous,
            conflicting_mappings=conflicting,
            probes_needing_investigation=tuple(sorted(all_conflicting_probes)),
        )


def diagnose_probe_conflict(
    probe_id: str,
    source_activations: dict[int, list[float]],
    target_activations: dict[int, list[float]],
    backend: Backend | None = None,
) -> dict:
    """
    Diagnose why a probe is giving conflicting constraints.

    When a probe disagrees with others about layer mapping, the MEASUREMENT
    is wrong, not the concept. This function helps identify what's wrong:
    - Weak activation (probe text doesn't capture concept well)
    - Multiple peaks (concept splits across layers - maybe probe is ambiguous)
    - Low CKA at peak (layer extraction or embedding has issues)

    Returns diagnostic info to guide probe improvement.
    """
    backend = backend or get_default_backend()
    aligner = ConstraintAligner(backend)

    source_peak, source_strength = aligner.find_peak_layer(source_activations)
    target_peak, target_strength = aligner.find_peak_layer(target_activations)

    # Check for secondary peaks (suggests ambiguous probe)
    def find_secondary_peak(activations: dict[int, list[float]], primary: int) -> tuple[int, float]:
        secondary_layer = -1
        secondary_strength = 0.0
        primary_strength = sum(a * a for a in activations.get(primary, [])) ** 0.5

        for layer, acts in activations.items():
            if layer == primary:
                continue
            strength = sum(a * a for a in acts) ** 0.5
            # Secondary peak if within 80% of primary
            if strength > secondary_strength and strength > 0.8 * primary_strength:
                secondary_strength = strength
                secondary_layer = layer

        return (secondary_layer, secondary_strength)

    source_secondary, source_secondary_strength = find_secondary_peak(
        source_activations, source_peak
    )
    target_secondary, target_secondary_strength = find_secondary_peak(
        target_activations, target_peak
    )

    # Compute CKA at peak
    cka_matrix = aligner.compute_per_layer_cka(source_activations, target_activations)
    peak_cka = cka_matrix.get((source_peak, target_peak), 0.0)

    diagnosis = {
        "probe_id": probe_id,
        "source_peak_layer": source_peak,
        "target_peak_layer": target_peak,
        "source_activation_strength": source_strength,
        "target_activation_strength": target_strength,
        "peak_cka": peak_cka,
        "issues": [],
    }

    # Diagnose issues
    if source_strength < 0.1:
        diagnosis["issues"].append("WEAK_SOURCE_ACTIVATION: Probe text may not capture concept well")
    if target_strength < 0.1:
        diagnosis["issues"].append("WEAK_TARGET_ACTIVATION: Probe text may not capture concept well")
    if source_secondary >= 0:
        diagnosis["issues"].append(
            f"AMBIGUOUS_SOURCE: Secondary peak at layer {source_secondary} "
            f"({source_secondary_strength:.2f} vs primary {source_strength:.2f})"
        )
    if target_secondary >= 0:
        diagnosis["issues"].append(
            f"AMBIGUOUS_TARGET: Secondary peak at layer {target_secondary} "
            f"({target_secondary_strength:.2f} vs primary {target_strength:.2f})"
        )
    if peak_cka < 0.9:
        diagnosis["issues"].append(
            f"LOW_PEAK_CKA ({peak_cka:.3f}): Activation patterns don't match at peaks. "
            "Check embedding extraction or layer selection."
        )

    if not diagnosis["issues"]:
        diagnosis["issues"].append("NO_OBVIOUS_ISSUES: Measurement looks good")

    return diagnosis
