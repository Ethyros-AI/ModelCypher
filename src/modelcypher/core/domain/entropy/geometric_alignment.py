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

"""
Geometric Alignment System (GAS) — entropy geometry measurements.

This module computes raw geometric measurements from entropy dynamics:
- Entropy deltas, spikes, dips (SentinelSample)
- Oscillation patterns, severity (OscillationPattern)
- Consecutive instability counts

Notes
-----
Consumers interpret raw measurements for their context.
Ported from the reference Swift implementation, then cleaned to pure geometry.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class SentinelConfiguration:
    """Entropy measurement thresholds."""

    entropy_ceiling: float = 4.0
    """Entropy value considered 'high'. Raw measurement threshold."""

    spike_threshold: float = 1.0
    """Delta magnitude that qualifies as a spike."""

    minimum_delta_for_signal: float = 0.3
    """Minimum delta magnitude to register as directional signal."""

    @classmethod
    def default(cls) -> SentinelConfiguration:
        return cls()


@dataclass
class SeverityDenominators:
    """Normalization denominators for severity calculation."""

    sign_changes: float = 5.0
    spike_count: float = 4.0
    failed_deflections: float = 2.0

    @classmethod
    def default(cls) -> SeverityDenominators:
        return cls()


@dataclass
class OscillatorConfiguration:
    """Oscillation pattern detection parameters."""

    window_size_tokens: int = 20
    """Rolling window size for pattern detection."""

    sign_change_threshold: int = 3
    """Sign changes to consider unstable."""

    spike_count_threshold: int = 3
    """Spikes to consider unstable."""

    rebound_window_tokens: int = 5
    """Window for W-shape detection."""

    pseudo_dip_escalation_weight: float = 1.5
    """Weight for pseudo-dips in severity calculation."""

    severity_denominators: SeverityDenominators = field(
        default_factory=SeverityDenominators.default
    )

    @classmethod
    def default(cls) -> OscillatorConfiguration:
        return cls()


@dataclass
class GASConfig:
    """Geometric Alignment System Configuration."""

    sentinel: SentinelConfiguration = field(default_factory=SentinelConfiguration.default)
    oscillator: OscillatorConfiguration = field(default_factory=OscillatorConfiguration.default)

    @classmethod
    def default(cls) -> GASConfig:
        return cls()


@dataclass
class SentinelSample:
    """Raw entropy measurements for a single token.

    All fields are direct measurements, not classifications.
    """

    token_index: int
    """Position in the generation sequence."""

    entropy: float
    """Raw entropy value."""

    delta_h: float
    """Change from previous entropy."""

    is_spike: bool
    """True if |delta_h| >= spike_threshold."""

    is_negative_delta: bool
    """True if delta_h <= -minimum_delta. The dip signal."""

    is_below_ceiling: bool
    """True if entropy < ceiling."""

    @property
    def is_true_dip(self) -> bool:
        """True dip: negative delta AND below ceiling."""
        return self.is_negative_delta and self.is_below_ceiling

    @property
    def is_pseudo_dip(self) -> bool:
        """Pseudo dip: negative delta but above ceiling."""
        return self.is_negative_delta and not self.is_below_ceiling

    @property
    def is_any_dip(self) -> bool:
        """Any dip (true or pseudo): negative delta."""
        return self.is_negative_delta


@dataclass
class OscillationPattern:
    """Raw oscillation pattern measurements.

    All fields are direct measurements. severity is normalized to [0,1]
    but is still a continuous measurement, not a classification.
    """

    window_sign_changes: int
    """Number of direction reversals in the window."""

    window_spike_count: int
    """Number of spikes in the window."""

    w_shape_count: int
    """Number of spike-dip-spike patterns detected."""

    failed_deflections: int
    """Pseudo-dips (dips above ceiling)."""

    severity: float
    """Normalized instability measure [0,1]. The geometry, not a verdict."""

    @property
    def is_unstable(self) -> bool:
        """Whether any instability signal is present."""
        return self.window_sign_changes > 0 or self.window_spike_count > 0 or self.w_shape_count > 0


@dataclass
class Decision:
    """Raw geometric measurements from entropy observation.

    Contains only measurements. No classification. No policy.
    Consumers interpret these signals for their context.
    """

    sentinel: SentinelSample
    """Current token's entropy measurements."""

    pattern: OscillationPattern
    """Rolling window oscillation measurements."""

    consecutive_oscillations: int
    """Count of consecutive unstable windows."""

    above_ceiling: bool
    """Whether current entropy exceeds ceiling."""


class GeometricAlignmentSystem:
    """Geometric Alignment System (GAS).

    Computes raw entropy geometry measurements. No classification. No policy.
    """

    class Session:
        """
        Per-generation session that computes entropy geometry.
        Thread-safe.
        """

        @dataclass
        class _Sample:
            token_index: int
            entropy: float
            delta_h: float
            is_spike: bool
            is_negative_delta: bool
            is_below_ceiling: bool
            delta_sign: int | None

            @property
            def is_any_dip(self) -> bool:
                return self.is_negative_delta

            @property
            def is_pseudo_dip(self) -> bool:
                return self.is_negative_delta and not self.is_below_ceiling

        @dataclass
        class _State:
            last_entropy: float | None = None
            samples: list["GeometricAlignmentSystem.Session._Sample"] = field(default_factory=list)
            consecutive_oscillations: int = 0
            last_decision: Decision | None = None

            # Telemetry (raw counts)
            tokens_observed: int = 0
            tokens_above_ceiling: int = 0
            spike_count: int = 0
            max_severity: float = 0.0

        def __init__(self, config: GASConfig = GASConfig.default()):
            self.config = config
            self._lock = threading.Lock()
            self._state = self._State()

        def reset(self):
            """Reset session state."""
            with self._lock:
                self._state = self._State()

        def observe(self, entropy: float, token_index: int) -> Decision:
            """Observe entropy and return raw geometric measurements."""
            with self._lock:
                state = self._state

                sentinel = self._compute_sentinel_sample(
                    entropy=entropy,
                    token_index=token_index,
                    previous_entropy=state.last_entropy,
                    config=self.config.sentinel,
                )
                state.last_entropy = entropy

                state.samples.append(
                    self._Sample(
                        token_index=token_index,
                        entropy=entropy,
                        delta_h=sentinel.delta_h,
                        is_spike=sentinel.is_spike,
                        is_negative_delta=sentinel.is_negative_delta,
                        is_below_ceiling=sentinel.is_below_ceiling,
                        delta_sign=self._delta_sign(
                            sentinel.delta_h, self.config.sentinel.minimum_delta_for_signal
                        ),
                    )
                )

                if len(state.samples) > self.config.oscillator.window_size_tokens:
                    state.samples.pop(0)

                pattern = self._compute_oscillation_pattern(state.samples, self.config)

                above_ceiling = entropy >= self.config.sentinel.entropy_ceiling
                unstable_now = self._is_unstable(pattern, self.config.oscillator)

                if unstable_now and above_ceiling:
                    state.consecutive_oscillations += 1
                elif not unstable_now:
                    state.consecutive_oscillations = 0

                # Telemetry updates
                state.tokens_observed += 1
                if above_ceiling:
                    state.tokens_above_ceiling += 1
                if sentinel.is_spike:
                    state.spike_count += 1
                state.max_severity = max(state.max_severity, pattern.severity)

                decision = Decision(
                    sentinel=sentinel,
                    pattern=pattern,
                    consecutive_oscillations=state.consecutive_oscillations,
                    above_ceiling=above_ceiling,
                )

                state.last_decision = decision
                return decision

        @property
        def last_decision(self) -> Decision | None:
            with self._lock:
                return self._state.last_decision

        def telemetry_snapshot(self) -> SessionTelemetry:
            """Returns raw telemetry measurements."""
            with self._lock:
                state = self._state
                return SessionTelemetry(
                    tokens_observed=state.tokens_observed,
                    tokens_above_ceiling=state.tokens_above_ceiling,
                    spike_count=state.spike_count,
                    max_severity=state.max_severity,
                    consecutive_oscillations=state.consecutive_oscillations,
                )

        # --- Computation Methods ---

        @staticmethod
        def _compute_sentinel_sample(
            entropy: float,
            token_index: int,
            previous_entropy: float | None,
            config: SentinelConfiguration,
        ) -> SentinelSample:
            delta_h = entropy - previous_entropy if previous_entropy is not None else 0.0
            return SentinelSample(
                token_index=token_index,
                entropy=entropy,
                delta_h=delta_h,
                is_spike=abs(delta_h) >= config.spike_threshold,
                is_negative_delta=delta_h <= -config.minimum_delta_for_signal,
                is_below_ceiling=entropy < config.entropy_ceiling,
            )

        @staticmethod
        def _delta_sign(delta_h: float, minimum_delta: float) -> int | None:
            if abs(delta_h) < minimum_delta:
                return None
            return 1 if delta_h >= 0 else -1

        @staticmethod
        def _compute_oscillation_pattern(
            samples: list[_Sample], config: GASConfig
        ) -> OscillationPattern:
            sign_changes = GeometricAlignmentSystem.Session._count_sign_changes(samples)
            spike_count = sum(1 for s in samples if s.is_spike)
            w_shape_count = GeometricAlignmentSystem.Session._count_w_shapes(
                samples, config.oscillator.rebound_window_tokens
            )
            pseudo_dip_count = sum(1 for s in samples if s.is_pseudo_dip)

            denom = config.oscillator.severity_denominators
            severity = max(
                min(1.0, float(sign_changes) / max(denom.sign_changes, 1.0)),
                min(1.0, float(spike_count) / max(denom.spike_count, 1.0)),
                min(
                    1.0,
                    (float(pseudo_dip_count) * config.oscillator.pseudo_dip_escalation_weight)
                    / max(denom.failed_deflections, 1.0),
                ),
            )

            return OscillationPattern(
                window_sign_changes=sign_changes,
                window_spike_count=spike_count,
                w_shape_count=w_shape_count,
                failed_deflections=pseudo_dip_count,
                severity=severity,
            )

        @staticmethod
        def _count_sign_changes(samples: list[_Sample]) -> int:
            previous_sign = None
            changes = 0
            for sample in samples:
                if sample.delta_sign is None:
                    continue
                if previous_sign is not None and sample.delta_sign != previous_sign:
                    changes += 1
                previous_sign = sample.delta_sign
            return changes

        @staticmethod
        def _count_w_shapes(samples: list[_Sample], rebound_window: int) -> int:
            class EventKind(Enum):
                SPIKE = 1
                DIP = 2

            @dataclass
            class Event:
                token_index: int
                kind: EventKind

            events = []
            for s in samples:
                if s.is_spike:
                    events.append(Event(s.token_index, EventKind.SPIKE))
                if s.is_any_dip:
                    events.append(Event(s.token_index, EventKind.DIP))

            if len(events) < 3:
                return 0

            count = 0
            for i in range(2, len(events)):
                a, b, c = events[i - 2], events[i - 1], events[i]
                if (
                    a.kind == EventKind.SPIKE
                    and b.kind == EventKind.DIP
                    and c.kind == EventKind.SPIKE
                ):
                    span = c.token_index - a.token_index
                    if span <= rebound_window:
                        count += 1
            return count

        @staticmethod
        def _is_unstable(pattern: OscillationPattern, config: OscillatorConfiguration) -> bool:
            return (
                pattern.window_sign_changes >= config.sign_change_threshold
                or pattern.window_spike_count >= config.spike_count_threshold
                or pattern.w_shape_count > 0
            )


@dataclass(frozen=True)
class SessionTelemetry:
    """Raw telemetry measurements from a GAS session."""

    tokens_observed: int
    """Total tokens observed."""

    tokens_above_ceiling: int
    """Tokens with entropy above ceiling."""

    spike_count: int
    """Number of entropy spikes detected."""

    max_severity: float
    """Maximum severity reached [0,1]."""

    consecutive_oscillations: int
    """Current consecutive oscillation count."""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "tokens_observed": self.tokens_observed,
            "tokens_above_ceiling": self.tokens_above_ceiling,
            "spike_count": self.spike_count,
            "max_severity": self.max_severity,
            "consecutive_oscillations": self.consecutive_oscillations,
        }
