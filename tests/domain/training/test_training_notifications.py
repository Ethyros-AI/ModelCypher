# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import pytest

from modelcypher.core.domain.training.training_notifications import (
    TrainingEvent,
    TrainingEventBus,
    TrainingEventKind,
    TrainingNotificationProgress,
    get_training_event_bus,
    reset_training_event_bus,
)


class TestTrainingNotificationProgress:
    """Tests for progress fraction/percent computation."""

    def test_fraction_midpoint(self):
        """step=50, total=100 -> 0.5."""
        p = TrainingNotificationProgress(job_id="t", step=50, total_steps=100)
        assert p.progress_fraction == pytest.approx(0.5)

    def test_fraction_zero_total(self):
        """total_steps=0 -> fraction=0.0 (not division by zero)."""
        p = TrainingNotificationProgress(job_id="t", step=10, total_steps=0)
        assert p.progress_fraction == 0.0

    def test_fraction_clamped_to_one(self):
        """step > total -> fraction clamped to 1.0."""
        p = TrainingNotificationProgress(job_id="t", step=200, total_steps=100)
        assert p.progress_fraction == 1.0

    def test_percent_at_three_quarters(self):
        """step=75, total=100 -> 75%."""
        p = TrainingNotificationProgress(job_id="t", step=75, total_steps=100)
        assert p.progress_percent == 75

    def test_percent_truncates_not_rounds(self):
        """progress_percent uses int() truncation, not rounding."""
        # step=1, total=3 -> fraction=0.333 -> percent=33 (not 34)
        p = TrainingNotificationProgress(job_id="t", step=1, total_steps=3)
        assert p.progress_percent == 33


class TestTrainingEventBus:
    """Tests for subscribe/emit/unsubscribe event delivery."""

    def test_subscribe_and_emit(self):
        """Subscribed handler receives emitted events."""
        bus = TrainingEventBus()
        received = []
        bus.subscribe(lambda evt: received.append(evt))

        event = TrainingEvent(kind=TrainingEventKind.STARTED, job_id="j1")
        bus.emit(event)

        assert len(received) == 1
        assert received[0].kind == TrainingEventKind.STARTED
        assert received[0].job_id == "j1"

    def test_unsubscribe_stops_delivery(self):
        """After unsubscribe, handler no longer receives events."""
        bus = TrainingEventBus()
        received = []
        sub_id = bus.subscribe(lambda evt: received.append(evt))

        bus.emit(TrainingEvent(kind=TrainingEventKind.STARTED, job_id="j1"))
        assert len(received) == 1

        bus.unsubscribe(sub_id)
        bus.emit(TrainingEvent(kind=TrainingEventKind.COMPLETED, job_id="j1"))
        assert len(received) == 1  # Still 1, not 2

    def test_multiple_subscribers(self):
        """All subscribers receive the same event."""
        bus = TrainingEventBus()
        received_a = []
        received_b = []
        bus.subscribe(lambda evt: received_a.append(evt))
        bus.subscribe(lambda evt: received_b.append(evt))

        bus.emit(TrainingEvent(kind=TrainingEventKind.FAILED, job_id="j2"))
        assert len(received_a) == 1
        assert len(received_b) == 1

    def test_handler_exception_swallowed(self):
        """Raising handler doesn't prevent other handlers from receiving events.

        This tests the INTENDED behavior: exceptions in one handler should be
        caught and logged, allowing remaining handlers to fire.
        """
        bus = TrainingEventBus()
        received = []

        def bad_handler(evt):
            raise ValueError("handler crashed")

        bus.subscribe(bad_handler)
        bus.subscribe(lambda evt: received.append(evt))

        event = TrainingEvent(kind=TrainingEventKind.PROGRESS, job_id="j3")
        bus.emit(event)
        assert len(received) == 1

    def test_emit_progress_convenience(self):
        """emit_progress() constructs and emits a PROGRESS event."""
        bus = TrainingEventBus()
        received = []
        bus.subscribe(lambda evt: received.append(evt))

        bus.emit_progress(job_id="j4", step=10, total_steps=100, loss=0.5)

        assert len(received) == 1
        assert received[0].kind == TrainingEventKind.PROGRESS
        assert received[0].job_id == "j4"


class TestGetTrainingEventBus:
    """Tests for global event bus singleton."""

    def test_returns_same_instance(self):
        """Singleton returns the same bus each time."""
        reset_training_event_bus()
        bus1 = get_training_event_bus()
        bus2 = get_training_event_bus()
        assert bus1 is bus2

    def test_reset_creates_new_instance(self):
        """reset_training_event_bus() creates a fresh bus."""
        reset_training_event_bus()
        bus1 = get_training_event_bus()
        reset_training_event_bus()
        bus2 = get_training_event_bus()
        assert bus1 is not bus2
