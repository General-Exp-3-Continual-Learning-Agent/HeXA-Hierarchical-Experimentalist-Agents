"""
Tests for core engine behaviors that need validation coverage.
"""

import pytest

from interphyre.engine import Box2DEngine
from interphyre.level import Level
from interphyre.objects import Ball, Bar, Basket, PhyreObject


def _make_level(objects):
    def success_condition(engine):
        return False

    return Level(
        name="engine_contact_test",
        objects=objects,
        action_objects=[],
        success_condition=success_condition,
        metadata={},
    )


@pytest.mark.fast
def test_is_in_basket_sensor_contact_true():
    """Ball overlapping basket sensor should register as in-basket."""
    objects = {
        "basket": Basket(x=0.0, y=0.0, bottom_width=2.0, enable_sensor=True),
        "ball": Ball(x=0.0, y=0.7, radius=0.1, dynamic=True),
    }
    engine = Box2DEngine(level=_make_level(objects))

    engine.world.Step(
        engine.config.time_step,
        engine.config.velocity_iters,
        engine.config.position_iters,
    )
    engine.time_update(engine.config.time_step)

    assert engine.is_in_basket("basket", "ball") is True


@pytest.mark.fast
def test_is_in_basket_no_contact_false():
    """Ball outside basket sensor should not register as in-basket."""
    objects = {
        "basket": Basket(x=0.0, y=0.0, bottom_width=2.0, enable_sensor=True),
        "ball": Ball(x=0.0, y=3.0, radius=0.1, dynamic=True),
    }
    engine = Box2DEngine(level=_make_level(objects))

    engine.world.Step(
        engine.config.time_step,
        engine.config.velocity_iters,
        engine.config.position_iters,
    )
    engine.time_update(engine.config.time_step)

    assert engine.is_in_basket("basket", "ball") is False


@pytest.mark.fast
def test_goal_contact_listener_duration_paths():
    """Exercise contact duration bookkeeping for missing and present contacts."""
    engine = Box2DEngine(level=_make_level({"ball": Ball(x=0.0, y=0.0, radius=0.5)}))
    listener = engine.contact_listener

    assert listener.GetContactDuration("a", "b") == 0
    assert listener.IsInContactForDuration("a", "b", required_duration=0.1) is False

    pair = frozenset(("a", "b"))
    listener.contacts.add(pair)
    listener.contact_start_time[pair] = 0.0
    listener.current_time = 1.0

    assert listener.GetContactDuration("a", "b") == pytest.approx(1.0)
    assert listener.IsInContactForDuration("a", "b", required_duration=0.5) is True


@pytest.mark.fast
def test_place_action_objects_bar_and_basket():
    """Place action objects for bar and basket paths."""
    objects = {
        "action_bar": Bar(x=0.0, y=0.0, length=2.0, angle=0.0, dynamic=True),
        "action_basket": Basket(x=0.0, y=0.0, bottom_width=2.0, dynamic=True),
    }
    level = Level(
        name="action_place",
        objects=objects,
        action_objects=["action_bar", "action_basket"],
        success_condition=lambda e: False,
        metadata={},
    )
    engine = Box2DEngine(level=level)
    engine.place_action_objects([(1.0, 1.0, 0.5), (-1.0, -1.0, 0.5)])

    assert "action_bar" in engine.bodies
    assert "action_basket" in engine.bodies


@pytest.mark.fast
def test_place_action_objects_requires_level():
    """place_action_objects should reject calls without a level."""
    engine = Box2DEngine(level=None)
    with pytest.raises(ValueError, match="level is not set"):
        engine.place_action_objects([(0.0, 0.0, 0.5)])


@pytest.mark.fast
def test_place_action_objects_unknown_type_raises():
    """Unknown action object types should raise ValueError."""

    class DummyObject(PhyreObject):
        def __init__(self):
            super().__init__(x=0.0, y=0.0)

    objects = {"dummy": DummyObject()}
    level = Level(
        name="unknown_action",
        objects=objects,
        action_objects=["dummy"],
        success_condition=lambda e: False,
        metadata={},
    )
    engine = Box2DEngine(level=level)
    with pytest.raises(ValueError, match="Unknown object type"):
        engine.place_action_objects([(0.0, 0.0, 0.5)])


@pytest.mark.fast
def test_create_world_unknown_object_type_raises():
    """Unknown non-action object types should fail world creation."""

    class DummyObject(PhyreObject):
        def __init__(self):
            super().__init__(x=0.0, y=0.0)

    objects = {"dummy": DummyObject()}
    level = Level(
        name="unknown_create",
        objects=objects,
        action_objects=[],
        success_condition=lambda e: False,
        metadata={},
    )
    with pytest.raises(ValueError, match="Unknown object type"):
        Box2DEngine(level=level)


@pytest.mark.fast
def test_update_relevant_contacts_no_level_noop():
    engine = Box2DEngine(level=None)
    engine.level = None
    assert engine._update_relevant_contacts() is None


@pytest.mark.fast
def test_get_state_empty_when_missing_level():
    """get_state returns empty dict when world or level missing."""
    engine = Box2DEngine(level=None)
    engine.level = None
    assert engine.get_state() == {}


@pytest.mark.fast
def test_get_state_includes_contact_entries():
    objects = {
        "ball_a": Ball(x=0.0, y=0.0, radius=0.5, dynamic=False),
        "ball_b": Ball(x=0.0, y=0.0, radius=0.5, dynamic=False),
    }
    engine = Box2DEngine(level=_make_level(objects))
    pair = frozenset(("ball_a", "ball_b"))
    engine.contact_listener.contacts.add(pair)
    engine.contact_listener.contact_start_time[pair] = 0.0
    engine.contact_listener.current_time = 1.5

    state = engine.get_state()
    assert len(state["contacts"]) == 1
    contact = next(iter(state["contacts"].values()))
    assert contact["duration"] == pytest.approx(1.5)


@pytest.mark.fast
def test_contact_key_ordering_is_alphabetical():
    """Contact keys in get_state() are alphabetically sorted regardless of insertion order."""
    objects = {
        "zebra": Ball(x=0.0, y=0.0, radius=0.5, dynamic=False),
        "alpha": Ball(x=1.0, y=0.0, radius=0.5, dynamic=False),
    }
    engine = Box2DEngine(level=_make_level(objects))
    # Insert with reverse-alphabetical order in the frozenset
    pair = frozenset(("zebra", "alpha"))
    engine.contact_listener.contacts.add(pair)
    engine.contact_listener.contact_start_time[pair] = 0.0
    engine.contact_listener.current_time = 1.0

    state = engine.get_state()
    keys = list(state["contacts"].keys())
    assert keys == ["alpha_zebra"]
    contact = state["contacts"]["alpha_zebra"]
    assert contact["objects"] == ("alpha", "zebra")


@pytest.mark.fast
def test_objects_requires_level():
    engine = Box2DEngine(level=None)
    engine.level = None
    with pytest.raises(ValueError, match="level is not set"):
        engine.objects()


@pytest.mark.fast
def test_objects_returns_level_objects():
    objects = {"ball": Ball(x=0.0, y=0.0, radius=0.5, dynamic=False)}
    engine = Box2DEngine(level=_make_level(objects))
    assert engine.objects()["ball"] is objects["ball"]


@pytest.mark.fast
def test_world_is_stationary_requires_world_and_level():
    """world_is_stationary should guard for missing world/level."""
    engine = Box2DEngine(level=None)
    engine.level = None
    with pytest.raises(ValueError, match="Level is not set"):
        engine.world_is_stationary()

    engine = Box2DEngine(level=None)
    engine.world = None
    with pytest.raises(ValueError, match="World is not initialized"):
        engine.world_is_stationary()


@pytest.mark.fast
def test_point_inside_polygon():
    engine = Box2DEngine(level=None)
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert engine._is_point_inside_polygon(0.5, 0.5, square) is True
    assert engine._is_point_inside_polygon(2.0, 2.0, square) is False


@pytest.mark.fast
def test_is_in_basket_error_paths():
    """is_in_basket should enforce type checks."""
    objects = {
        "not_basket": Ball(x=0.0, y=0.0, radius=0.5),
        "not_ball": Bar(x=0.0, y=0.0, length=2.0, angle=0.0),
    }
    engine = Box2DEngine(level=_make_level(objects))

    with pytest.raises(ValueError, match="not a basket"):
        engine.is_in_basket("not_basket", "not_ball")

    objects = {
        "basket": Basket(x=0.0, y=0.0, bottom_width=2.0, enable_sensor=True),
        "not_ball": Bar(x=0.0, y=0.0, length=2.0, angle=0.0),
    }
    engine = Box2DEngine(level=_make_level(objects))
    with pytest.raises(ValueError, match="only works with Balls"):
        engine.is_in_basket("basket", "not_ball")


@pytest.mark.fast
def test_is_in_basket_requires_level_or_world():
    engine = Box2DEngine(level=None)
    engine.level = None
    with pytest.raises(ValueError, match="Level or world not initialized"):
        engine.is_in_basket("basket", "ball")


@pytest.mark.fast
def test_is_in_basket_missing_objects_returns_false():
    objects = {
        "basket": Basket(x=0.0, y=0.0, bottom_width=2.0, enable_sensor=True),
        "ball": Ball(x=0.0, y=0.0, radius=0.1, dynamic=True),
    }
    engine = Box2DEngine(level=_make_level(objects))
    assert engine.is_in_basket("missing_basket", "missing_ball") is False


@pytest.mark.fast
def test_is_in_basket_missing_body_returns_false():
    objects = {
        "basket": Basket(x=0.0, y=0.0, bottom_width=2.0, enable_sensor=True),
        "ball": Ball(x=0.0, y=0.0, radius=0.1, dynamic=True),
    }
    level = Level(
        name="basket_missing_body",
        objects=objects,
        action_objects=["ball"],
        success_condition=lambda e: False,
        metadata={},
    )
    engine = Box2DEngine(level=level)
    assert engine.is_in_basket("basket", "ball") is False


@pytest.mark.fast
def test_contact_duration_default_success_time():
    """Default success time should be used when omitted."""
    objects = {
        "ball_a": Ball(x=0.0, y=0.0, radius=0.5, dynamic=False),
        "ball_b": Ball(x=0.9, y=0.0, radius=0.5, dynamic=False),
    }
    engine = Box2DEngine(level=_make_level(objects))
    contact_pair = frozenset(("ball_a", "ball_b"))
    engine.contact_listener.contacts.add(contact_pair)
    engine.contact_listener.contact_start_time[contact_pair] = 0.0
    engine.contact_listener.current_time = engine.config.default_success_time + 0.1

    assert engine.is_in_contact_for_duration("ball_a", "ball_b") is True


@pytest.mark.fast
def test_contact_duration_requires_level():
    engine = Box2DEngine(level=None)
    engine.level = None
    with pytest.raises(ValueError, match="Level is not set"):
        engine.is_in_contact_for_duration("a", "b")


@pytest.mark.fast
def test_contact_duration_missing_objects_returns_false():
    objects = {"ball": Ball(x=0.0, y=0.0, radius=0.5, dynamic=False)}
    engine = Box2DEngine(level=_make_level(objects))
    assert engine.is_in_contact_for_duration("missing", "ball") is False


@pytest.mark.fast
def test_get_contact_duration_error_and_missing_objects():
    engine = Box2DEngine(level=None)
    engine.level = None
    with pytest.raises(ValueError, match="Level is not set"):
        engine.get_contact_duration("a", "b")

    objects = {"ball": Ball(x=0.0, y=0.0, radius=0.5, dynamic=False)}
    engine = Box2DEngine(level=_make_level(objects))
    assert engine.get_contact_duration("missing", "ball") == 0


@pytest.mark.fast
def test_get_contact_duration_returns_value():
    objects = {
        "ball_a": Ball(x=0.0, y=0.0, radius=0.5, dynamic=False),
        "ball_b": Ball(x=0.0, y=0.0, radius=0.5, dynamic=False),
    }
    engine = Box2DEngine(level=_make_level(objects))
    pair = frozenset(("ball_a", "ball_b"))
    engine.contact_listener.contacts.add(pair)
    engine.contact_listener.contact_start_time[pair] = 0.0
    engine.contact_listener.current_time = 2.0
    assert engine.get_contact_duration("ball_a", "ball_b") == pytest.approx(2.0)
