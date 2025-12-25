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
Geometric Alignment System (GAS) — pre-emission safety via entropy geometry.

This module implements the core concepts from `GEOMETRIC_ALIGNMENT_SYSTEM_PRD.md`:
- Knowledge as shape (valley → ridge → cliff)
- Consequential / physics-based detection (entropy instability, oscillation)
- Pre-emission intervention (stop/deflect before emitting unsafe tokens)

This module is intentionally content-agnostic: it never inspects token text.
Ported from the reference Swift implementation.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum


class DipClassification(str, Enum):
    NONE = "none"
    TRUE_DIP = "true_dip"
    PSEUDO_DIP = "pseudo_dip"


@dataclass
class SentinelConfiguration:
    """Physics layer (Sentinel) configuration."""

    entropy_ceiling: float = 4.0
    spike_threshold: float = 1.0
    minimum_delta_for_signal: float = 0.3

    @classmethod
    def default(cls) -> "SentinelConfiguration":
        return cls()


@dataclass
class SeverityDenominators:
    sign_changes: float = 5.0
    spike_count: float = 4.0
    failed_deflections: float = 2.0

    @classmethod
    def default(cls) -> "SeverityDenominators":
        return cls()


@dataclass
class OscillatorConfiguration:
    """Pattern layer (Oscillator) configuration."""

    window_size_tokens: int = 20
    sign_change_threshold: int = 3
    spike_count_threshold: int = 3
    failed_deflection_threshold: int = 2
    consecutive_oscillations_for_termination: int = 3
    rebound_window_tokens: int = 5
    max_w_shapes_before_escalation: int = 2
    pseudo_dip_escalation_weight: float = 1.5
    severity_denominators: SeverityDenominators = field(
        default_factory=SeverityDenominators.default
    )

    @classmethod
    def default(cls) -> "OscillatorConfiguration":
        return cls()


@dataclass
class MonitorConfiguration:
    """Decision layer (Monitor) configuration."""

    require_above_ceiling_for_hard_levels: bool = True

    @classmethod
    def default(cls) -> "MonitorConfiguration":
        return cls()


class InterventionLevel(int, Enum):
    LEVEL_0_CONTINUE = 0
    LEVEL_1_GENTLE = 1
    LEVEL_2_CLARIFY = 2
    LEVEL_3_HARD = 3
    LEVEL_4_TERMINATE = 4

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented


@dataclass
class DirectorConfiguration:
    """Execution layer (Director) configuration."""

    cooldown_tokens_by_level: dict[InterventionLevel, int] = field(
        default_factory=lambda: {
            InterventionLevel.LEVEL_1_GENTLE: 50,
            InterventionLevel.LEVEL_2_CLARIFY: 100,
            InterventionLevel.LEVEL_3_HARD: 0,
            InterventionLevel.LEVEL_4_TERMINATE: 0,
            InterventionLevel.LEVEL_0_CONTINUE: 0,
        }
    )
    max_count_by_level: dict[InterventionLevel, int] = field(
        default_factory=lambda: {
            InterventionLevel.LEVEL_1_GENTLE: 5,
            InterventionLevel.LEVEL_2_CLARIFY: 3,
            InterventionLevel.LEVEL_3_HARD: 2,
            InterventionLevel.LEVEL_4_TERMINATE: 1,
            InterventionLevel.LEVEL_0_CONTINUE: 1000000,
        }
    )
    gentle_logit_scale: float = 1.25
    terminate_generation_on_level_2_plus: bool = True

    @classmethod
    def default(cls) -> "DirectorConfiguration":
        return cls()


@dataclass
class GASConfig:
    """Geometric Alignment System Configuration."""

    sentinel: SentinelConfiguration = field(default_factory=SentinelConfiguration.default)
    oscillator: OscillatorConfiguration = field(default_factory=OscillatorConfiguration.default)
    monitor: MonitorConfiguration = field(default_factory=MonitorConfiguration.default)
    director: DirectorConfiguration = field(default_factory=DirectorConfiguration.default)

    @classmethod
    def default(cls) -> "GASConfig":
        return cls()


@dataclass
class SentinelSample:
    token_index: int
    entropy: float
    delta_h: float
    is_spike: bool
    dip_classification: DipClassification


@dataclass
class OscillationPattern:
    window_sign_changes: int
    window_spike_count: int
    w_shape_count: int
    failed_deflections: int
    severity: float  # 0...1

    @property
    def is_unstable(self) -> bool:
        return self.window_sign_changes > 0 or self.window_spike_count > 0 or self.w_shape_count > 0


@dataclass
class Intervention:
    level: InterventionLevel
    message: str
    reason: str
    token_index: int
    severity: float


@dataclass
class Decision:
    sentinel: SentinelSample
    pattern: OscillationPattern
    level: InterventionLevel
    cooldown_remaining: int
    logit_scale: float


class GeometricAlignmentSystem:
    """
    Geometric Alignment System (GAS).
    Observes entropy dynamics and orchestrates pre-emission interventions.
    """

    class Session:
        """
        A per-generation session that observes entropy and produces pre-emission interventions.
        Thread-safe.
        """

        @dataclass
        class _Sample:
            token_index: int
            entropy: float
            delta_h: float
            is_spike: bool
            dip: DipClassification
            delta_sign: int | None  # -1 or +1

        @dataclass
        class _State:
            last_entropy: float | None = None
            samples: list["GeometricAlignmentSystem.Session._Sample"] = field(default_factory=list)
            consecutive_oscillations: int = 0

            current_level: InterventionLevel = InterventionLevel.LEVEL_0_CONTINUE
            cooldown_remaining: int = 0
            level_counts: dict[InterventionLevel, int] = field(default_factory=dict)
            token_counts_by_level: dict[InterventionLevel, int] = field(default_factory=dict)

            pending_intervention: Intervention | None = None
            last_decision: Decision | None = None

            tokens_observed: int = 0
            tokens_above_ceiling: int = 0
            spike_count: int = 0
            max_severity: float = 0.0
            max_level_reached: InterventionLevel = InterventionLevel.LEVEL_0_CONTINUE

        def __init__(self, config: GASConfig = GASConfig.default()):
            self.config = config
            self._lock = threading.Lock()
            self._state = self._State()

        def reset(self):
            with self._lock:
                self._state = self._State()

        def observe(self, entropy: float, token_index: int) -> Decision:
            """Observes entropy for the next token and returns the current decision."""
            with self._lock:
                state = self._state

                # Cooldown bookkeeping
                if state.cooldown_remaining > 0:
                    state.cooldown_remaining = max(0, state.cooldown_remaining - 1)

                sentinel = self._compute_sentinel_sample(
                    entropy=entropy,
                    token_index=token_index,
                    previous_entropy=state.last_entropy,
                    config=self.config.sentinel,
                )
                state.last_entropy = entropy

                # Append sample to window
                state.samples.append(
                    self._Sample(
                        token_index=token_index,
                        entropy=entropy,
                        delta_h=sentinel.delta_h,
                        is_spike=sentinel.is_spike,
                        dip=sentinel.dip_classification,
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

                if unstable_now and (
                    not self.config.monitor.require_above_ceiling_for_hard_levels or above_ceiling
                ):
                    state.consecutive_oscillations += 1
                else:
                    state.consecutive_oscillations = 0

                recommended = self._recommended_level(
                    sentinel=sentinel,
                    pattern=pattern,
                    consecutive_oscillations=state.consecutive_oscillations,
                    config=self.config,
                    above_ceiling=above_ceiling,
                )

                applied = self._apply_director_policy(
                    recommended=recommended, state=state, config=self.config.director
                )

                # Telemetry updates
                state.tokens_observed += 1
                if above_ceiling:
                    state.tokens_above_ceiling += 1
                if sentinel.is_spike:
                    state.spike_count += 1
                state.max_severity = max(state.max_severity, pattern.severity)

                if applied > state.max_level_reached:
                    state.max_level_reached = applied

                state.token_counts_by_level[applied] = (
                    state.token_counts_by_level.get(applied, 0) + 1
                )

                logit_scale = 1.0
                if applied == InterventionLevel.LEVEL_1_GENTLE:
                    logit_scale = self.config.director.gentle_logit_scale

                decision = Decision(
                    sentinel=sentinel,
                    pattern=pattern,
                    level=applied,
                    cooldown_remaining=state.cooldown_remaining,
                    logit_scale=logit_scale,
                )

                state.last_decision = decision

                # Pre-emission intervention
                if (
                    self.config.director.terminate_generation_on_level_2_plus
                    and applied >= InterventionLevel.LEVEL_2_CLARIFY
                    and state.pending_intervention is None
                ):
                    state.pending_intervention = self._make_intervention(
                        level=applied,
                        token_index=token_index,
                        severity=pattern.severity,
                        reason=self._default_reason(
                            applied, sentinel, pattern, state.consecutive_oscillations
                        ),
                    )

                return decision

        @property
        def last_decision(self) -> Decision | None:
            with self._lock:
                return self._state.last_decision

        def consume_pending_intervention(self) -> Intervention | None:
            with self._lock:
                pending = self._state.pending_intervention
                self._state.pending_intervention = None
                return pending

        # --- Internal Computation ---

        @staticmethod
        def _compute_sentinel_sample(
            entropy: float,
            token_index: int,
            previous_entropy: float | None,
            config: SentinelConfiguration,
        ) -> SentinelSample:
            if previous_entropy is not None:
                delta_h = entropy - previous_entropy
            else:
                delta_h = 0.0

            is_spike = abs(delta_h) >= config.spike_threshold

            dip = DipClassification.NONE
            if delta_h <= -config.minimum_delta_for_signal:
                if entropy < config.entropy_ceiling:
                    dip = DipClassification.TRUE_DIP
                else:
                    dip = DipClassification.PSEUDO_DIP

            return SentinelSample(
                token_index=token_index,
                entropy=entropy,
                delta_h=delta_h,
                is_spike=is_spike,
                dip_classification=dip,
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
            pseudo_dip_count = sum(1 for s in samples if s.dip == DipClassification.PSEUDO_DIP)

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
            # Simple W-shape detection: Spike -> Dip -> Spike
            # Implementation simplified for parity
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
                if s.dip != DipClassification.NONE:
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

        @staticmethod
        def _recommended_level(
            sentinel: SentinelSample,
            pattern: OscillationPattern,
            consecutive_oscillations: int,
            config: GASConfig,
            above_ceiling: bool,
        ) -> InterventionLevel:
            # Level 4
            if (
                consecutive_oscillations
                >= config.oscillator.consecutive_oscillations_for_termination
            ):
                return InterventionLevel.LEVEL_4_TERMINATE

            # Level 3
            if (
                pattern.failed_deflections >= config.oscillator.failed_deflection_threshold
                or pattern.severity > 0.8
                or pattern.w_shape_count >= config.oscillator.max_w_shapes_before_escalation
            ):
                if not config.monitor.require_above_ceiling_for_hard_levels or above_ceiling:
                    return InterventionLevel.LEVEL_3_HARD

            # Level 2
            if GeometricAlignmentSystem.Session._is_unstable(pattern, config.oscillator):
                if not config.monitor.require_above_ceiling_for_hard_levels or above_ceiling:
                    return InterventionLevel.LEVEL_2_CLARIFY

            # Level 1
            half_sign = max(1, config.oscillator.sign_change_threshold // 2)
            half_spike = max(1, config.oscillator.spike_count_threshold // 2)

            if (
                pattern.window_sign_changes >= half_sign
                or pattern.window_spike_count >= half_spike
                or (sentinel.is_spike and above_ceiling)
            ):
                return InterventionLevel.LEVEL_1_GENTLE

            return InterventionLevel.LEVEL_0_CONTINUE

        @staticmethod
        def _apply_director_policy(
            recommended: InterventionLevel, state: _State, config: DirectorConfiguration
        ) -> InterventionLevel:
            # Escalation bypass
            if recommended > state.current_level:
                state.current_level = recommended
                state.cooldown_remaining = max(
                    0, config.cooldown_tokens_by_level.get(recommended, 0)
                )

                next_count = state.level_counts.get(recommended, 0) + 1
                state.level_counts[recommended] = next_count

                # Check limits
                if next_count > config.max_count_by_level.get(recommended, float("inf")):
                    escalated = GeometricAlignmentSystem.Session._next_level(recommended)
                    state.current_level = escalated
                    state.cooldown_remaining = max(
                        0, config.cooldown_tokens_by_level.get(escalated, 0)
                    )
                    state.level_counts[escalated] = state.level_counts.get(escalated, 0) + 1

                return state.current_level

            # Respect cooldown
            if state.cooldown_remaining > 0:
                return state.current_level

            # Cooldown expired
            state.current_level = recommended
            return state.current_level

        @staticmethod
        def _next_level(level: InterventionLevel) -> InterventionLevel:
            if level == InterventionLevel.LEVEL_0_CONTINUE:
                return InterventionLevel.LEVEL_1_GENTLE
            if level == InterventionLevel.LEVEL_1_GENTLE:
                return InterventionLevel.LEVEL_2_CLARIFY
            if level == InterventionLevel.LEVEL_2_CLARIFY:
                return InterventionLevel.LEVEL_3_HARD
            if level == InterventionLevel.LEVEL_3_HARD:
                return InterventionLevel.LEVEL_4_TERMINATE
            return InterventionLevel.LEVEL_4_TERMINATE

        @staticmethod
        def _make_intervention(
            level: InterventionLevel, token_index: int, severity: float, reason: str
        ) -> Intervention:
            message = ""
            if level == InterventionLevel.LEVEL_2_CLARIFY:
                message = "[Stability warning: please clarify the task scope.]"
            elif level == InterventionLevel.LEVEL_3_HARD:
                message = "[Request appears out of scope for the current workflow; narrow scope or select a different tool.]"
            elif level == InterventionLevel.LEVEL_4_TERMINATE:
                message = "[Generation stopped for safety: instability detected (pre-emission).]"

            return Intervention(
                level=level,
                message=message,
                reason=reason,
                token_index=token_index,
                severity=severity,
            )

        @staticmethod
        def _default_reason(
            level: InterventionLevel,
            sentinel: SentinelSample,
            pattern: OscillationPattern,
            consecutive_oscillations: int,
        ) -> str:
            if level == InterventionLevel.LEVEL_0_CONTINUE:
                return "stable"
            if level == InterventionLevel.LEVEL_1_GENTLE:
                return f"elevated volatility (ΔH={sentinel.delta_h:.3f}, spikes={pattern.window_spike_count})"
            if level == InterventionLevel.LEVEL_2_CLARIFY:
                return f"oscillation detected (severity={pattern.severity:.2f})"
            if level == InterventionLevel.LEVEL_3_HARD:
                return f"repeated instability (wShapes={pattern.w_shape_count}, severity={pattern.severity:.2f})"
            if level == InterventionLevel.LEVEL_4_TERMINATE:
                return f"sustained oscillation ({consecutive_oscillations} windows)"
            return "unknown"

        def telemetry_snapshot(self) -> "SessionTelemetry":
            """Returns a snapshot of session telemetry."""
            with self._lock:
                state = self._state
                by_level = [
                    LevelCounts(
                        level=level,
                        tokens=state.token_counts_by_level.get(level, 0),
                        escalations=state.level_counts.get(level, 0),
                    )
                    for level in InterventionLevel
                ]

                return SessionTelemetry(
                    tokens_observed=state.tokens_observed,
                    tokens_above_ceiling=state.tokens_above_ceiling,
                    spike_count=state.spike_count,
                    max_severity=state.max_severity,
                    max_level_reached=state.max_level_reached,
                    by_level=tuple(by_level),
                )


@dataclass(frozen=True)
class LevelCounts:
    """Per-level token and escalation counts."""

    level: InterventionLevel
    tokens: int
    escalations: int


@dataclass(frozen=True)
class SessionTelemetry:
    """Telemetry snapshot from a GAS session."""

    tokens_observed: int
    """Total tokens observed in this session."""

    tokens_above_ceiling: int
    """Tokens with entropy above ceiling."""

    spike_count: int
    """Number of entropy spikes detected."""

    max_severity: float
    """Maximum severity reached (0-1)."""

    max_level_reached: InterventionLevel
    """Maximum intervention level reached."""

    by_level: tuple[LevelCounts, ...]
    """Per-level statistics."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "tokens_observed": self.tokens_observed,
            "tokens_above_ceiling": self.tokens_above_ceiling,
            "spike_count": self.spike_count,
            "max_severity": self.max_severity,
            "max_level_reached": self.max_level_reached.value,
            "by_level": [
                {"level": lc.level.value, "tokens": lc.tokens, "escalations": lc.escalations}
                for lc in self.by_level
            ],
        }
