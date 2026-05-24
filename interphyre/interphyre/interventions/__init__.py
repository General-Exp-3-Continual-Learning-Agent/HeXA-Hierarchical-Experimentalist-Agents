"""
Interventions API for Interphyre.

This module provides triggers and state management for multi-turn simulation control.

## Quick Start

Use triggers with InterphyreEnv:

    from interphyre import InterphyreEnv
    from interphyre.interventions import on_contact, on_success, at_step

    env = InterphyreEnv("two_body_problem", seed=42, enable_interventions=True)

    # Run until contact event
    snapshot, step = env.run_until(
        on_contact("green_ball", "blue_ball"),
        action=(0.5, 3.0, 0.6),
        max_steps=500
    )

    if snapshot:
        env.restore(snapshot)
        env.apply_impulse("green_ball", impulse=(5.0, 0.0))
        obs, reward, term, trunc, info = env.step_until(on_success())

## Available Triggers

- `at_step(n)` - Fire at specific simulation step
- `on_contact(a, b)` - Fire when two objects touch
- `on_contact_with(obj)` - Fire when object touches anything
- `on_contact_duration(a, b, seconds)` - Fire after sustained contact for min_seconds
- `on_success()` - Fire when level's success condition is met
- `on_velocity_threshold(obj, speed, above=True)` - Fire on speed threshold
- `on_position_threshold(obj, axis, threshold, direction)` - Fire on position threshold
- `when(condition)` - Fire when custom condition is True
- `on_sequence([triggers])` - Fire when triggers fire in order
- `on_any([triggers])` - Fire when any trigger fires

## State Management

- `StateSnapshot` - Captured simulation state (returned by run_until)
"""

from interphyre.interventions.state import StateSnapshot
from interphyre.interventions.triggers import (
    Trigger,
    TimeBasedTrigger,
    EventBasedTrigger,
    ConditionBasedTrigger,
    SequenceTrigger,
    AnyTrigger,
    at_step,
    on_contact,
    on_contact_with,
    on_contact_duration,
    on_success,
    when,
    on_position_threshold,
    on_velocity_threshold,
    on_sequence,
    on_any,
)

__all__ = [
    "StateSnapshot",
    "Trigger",
    "TimeBasedTrigger",
    "EventBasedTrigger",
    "ConditionBasedTrigger",
    "SequenceTrigger",
    "AnyTrigger",
    "at_step",
    "on_contact",
    "on_contact_with",
    "on_contact_duration",
    "on_success",
    "when",
    "on_position_threshold",
    "on_velocity_threshold",
    "on_sequence",
    "on_any",
]
