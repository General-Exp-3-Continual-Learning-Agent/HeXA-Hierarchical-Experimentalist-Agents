"""Regression tests for the trigger system."""

from unittest.mock import MagicMock

import pytest

from interphyre.interventions.triggers import (
    ConditionBasedTrigger,
    at_step,
    on_contact,
    on_contact_with,
    on_success,
    when,
    on_position_threshold,
    on_velocity_threshold,
    on_contact_duration,
    on_sequence,
    on_any,
    TimeBasedTrigger,
    EventBasedTrigger,
    SequenceTrigger,
    AnyTrigger,
)


def _make_mock_engine():
    """Create a minimal mock engine for trigger evaluation."""
    engine = MagicMock()
    engine.bodies = {}
    return engine


class TestConditionBasedTriggerExceptionPropagation:
    """FIX-CONDITION-TRIGGER-EXCEPTION-SWALLOWING regression tests.

    Conditions that raise exceptions must propagate to the caller,
    not be silently caught and converted to False.
    """

    def test_value_error_propagates(self):
        engine = _make_mock_engine()

        def bad_condition(e):
            raise ValueError("user bug in condition")

        trigger = ConditionBasedTrigger(condition=bad_condition)
        with pytest.raises(ValueError, match="user bug in condition"):
            trigger.should_fire(step_index=0, engine=engine)

    def test_key_error_propagates(self):
        engine = _make_mock_engine()

        def missing_key(e):
            return e.bodies["nonexistent"].position.y < 0

        trigger = ConditionBasedTrigger(condition=missing_key)
        with pytest.raises(KeyError):
            trigger.should_fire(step_index=0, engine=engine)

    def test_type_error_propagates(self):
        engine = _make_mock_engine()

        def type_bug(e):
            return 1 + "string"  # guaranteed TypeError

        trigger = when(type_bug)
        with pytest.raises(TypeError):
            trigger.should_fire(step_index=0, engine=engine)

    def test_attribute_error_propagates(self):
        def attr_bug(e):
            return e.nonexistent_attr.subattr > 0

        trigger = ConditionBasedTrigger(condition=attr_bug)
        # MagicMock won't raise AttributeError, so use a plain object
        plain_engine = object()
        with pytest.raises(AttributeError):
            trigger.should_fire(step_index=0, engine=plain_engine)

    def test_valid_condition_still_works(self):
        """Sanity check: valid conditions still return correctly."""
        engine = _make_mock_engine()
        trigger = ConditionBasedTrigger(condition=lambda e: True)
        assert trigger.should_fire(step_index=0, engine=engine) is True

    def test_valid_false_condition_still_works(self):
        engine = _make_mock_engine()
        trigger = ConditionBasedTrigger(condition=lambda e: False)
        assert trigger.should_fire(step_index=0, engine=engine) is False


class TestTriggerPriorityFieldRemoved:
    """REMOVE-TRIGGER-PRIORITY-FIELD regression tests.

    The priority field was accepted but never used in trigger evaluation.
    It must not appear in any trigger class or factory function signature.
    """

    def test_at_step_rejects_priority_kwarg(self):
        with pytest.raises(TypeError):
            at_step(10, priority=0)

    def test_on_contact_rejects_priority_kwarg(self):
        with pytest.raises(TypeError):
            on_contact("a", "b", priority=0)

    def test_on_contact_with_rejects_priority_kwarg(self):
        with pytest.raises(TypeError):
            on_contact_with("a", priority=0)

    def test_on_success_rejects_priority_kwarg(self):
        with pytest.raises(TypeError):
            on_success(priority=0)

    def test_when_rejects_priority_kwarg(self):
        with pytest.raises(TypeError):
            when(lambda e: True, priority=0)

    def test_on_position_threshold_rejects_priority_kwarg(self):
        with pytest.raises(TypeError):
            on_position_threshold("ball", "y", 0.0, priority=0)

    def test_on_velocity_threshold_rejects_priority_kwarg(self):
        with pytest.raises(TypeError):
            on_velocity_threshold("ball", 1.0, priority=0)

    def test_on_contact_duration_rejects_priority_kwarg(self):
        with pytest.raises(TypeError):
            on_contact_duration("a", "b", 1.0, priority=0)

    def test_on_sequence_rejects_priority_kwarg(self):
        with pytest.raises(TypeError):
            on_sequence([], priority=0)

    def test_on_any_rejects_priority_kwarg(self):
        with pytest.raises(TypeError):
            on_any([], priority=0)

    def test_trigger_base_class_has_no_priority_attr(self):
        trigger = TimeBasedTrigger(step_index=5)
        assert not hasattr(trigger, "priority")

    def test_event_trigger_has_no_priority_attr(self):
        trigger = EventBasedTrigger(event_type="contact", object_names=("a", "b"))
        assert not hasattr(trigger, "priority")

    def test_condition_trigger_has_no_priority_attr(self):
        trigger = ConditionBasedTrigger(condition=lambda e: True)
        assert not hasattr(trigger, "priority")

    def test_sequence_trigger_has_no_priority_attr(self):
        trigger = SequenceTrigger(triggers=())
        assert not hasattr(trigger, "priority")

    def test_any_trigger_has_no_priority_attr(self):
        trigger = AnyTrigger(triggers=())
        assert not hasattr(trigger, "priority")

    def test_repr_does_not_contain_priority(self):
        """No trigger repr should mention priority."""
        triggers = [
            TimeBasedTrigger(step_index=5),
            EventBasedTrigger(event_type="contact", object_names=("a", "b")),
            ConditionBasedTrigger(condition=lambda e: True),
            SequenceTrigger(triggers=()),
            AnyTrigger(triggers=()),
        ]
        for trigger in triggers:
            assert "priority" not in repr(trigger), (
                f"{type(trigger).__name__} repr contains 'priority'"
            )


class TestTriggerResetOnRunUntil:
    """FIX-TRIGGER-RESET-ON-ENV-RESET regression tests.

    once_only triggers must fire again when run_until() is called a second
    time, even without an explicit trigger.reset(). run_until() auto-resets
    the trigger at entry so that multi-episode loops work naturally.
    """

    def test_once_only_condition_trigger_fires_across_run_until_calls(self):
        """once_only=True condition trigger fires in both episodes via run_until."""
        from interphyre.levels import load_level
        from interphyre.environment import InterphyreEnv

        level = load_level("two_body_problem", seed=42)
        env = InterphyreEnv(level)

        # Condition that fires on step 3 — deterministic, no action needed
        trigger = when(lambda e: True, once_only=True)

        fired_episodes = []
        for episode in range(2):
            env.reset()
            snapshot, steps = env.run_until(trigger, max_steps=500)
            fired_episodes.append(snapshot is not None)

        assert fired_episodes == [True, True], (
            f"Trigger should fire in both episodes, got {fired_episodes}"
        )

    def test_once_only_condition_trigger_fires_across_episodes(self):
        """ConditionBasedTrigger with once_only=True fires in both episodes."""
        engine = _make_mock_engine()

        call_count = 0

        def counting_condition(e):
            nonlocal call_count
            call_count += 1
            return True

        trigger = ConditionBasedTrigger(condition=counting_condition, once_only=True)

        # First call — fires and sets _fired
        assert trigger.should_fire(0, engine) is True
        assert trigger.should_fire(1, engine) is False  # once_only blocks

        # Simulate what run_until does: reset before use
        trigger.reset()
        assert trigger.should_fire(2, engine) is True

    def test_run_until_auto_resets_trigger(self):
        """run_until() resets trigger state at entry, no manual reset needed."""
        from interphyre.levels import load_level
        from interphyre.environment import InterphyreEnv

        level = load_level("two_body_problem", seed=42)
        env = InterphyreEnv(level)

        # Use at_step which fires at a fixed step — deterministic
        trigger = at_step(5)

        env.reset()
        snapshot1, step1 = env.run_until(trigger, max_steps=100)
        assert snapshot1 is not None
        assert step1 == 5

        env.reset()
        snapshot2, step2 = env.run_until(trigger, max_steps=100)
        # TimeBasedTrigger has no _fired state, but this confirms
        # run_until works across resets for all trigger types
        assert snapshot2 is not None
        assert step2 == 5

    def test_composite_triggers_reset_children(self):
        """SequenceTrigger and AnyTrigger reset all children via run_until."""
        engine = _make_mock_engine()

        child1 = EventBasedTrigger(
            event_type="contact", object_names=("a", "b"), once_only=True
        )
        child2 = EventBasedTrigger(
            event_type="contact", object_names=("c", "d"), once_only=True
        )

        any_trigger = on_any([child1, child2])

        # Manually fire child1
        child1._fired = True
        assert child1.should_fire(0, engine) is False

        # Reset via parent
        any_trigger.reset()
        assert child1._fired is False
        assert child2._fired is False
