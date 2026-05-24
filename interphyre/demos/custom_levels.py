#!/usr/bin/env python3
"""
Custom Levels: Build your own physics puzzles.

This demo shows how to create custom levels from scratch:
1. Define objects (balls, bars, baskets)
2. Set action objects (user-placeable)
3. Write success conditions
4. Use InterphyreEnv() to run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interphyre import InterphyreEnv
from interphyre.level import Level
from interphyre.objects import Ball, Bar


def simple_contact_level():
    """Create a level where green ball must touch blue ball."""
    print("\n1. Simple contact level")

    objects = {
        "green_ball": Ball(x=-3.0, y=2.0, radius=0.5, color="green", dynamic=True),
        "blue_ball": Ball(x=3.0, y=2.0, radius=0.5, color="blue", dynamic=True),
        "red_ball": Ball(x=0.0, y=4.0, radius=0.5, color="red", dynamic=True),
    }

    def success_condition(engine):
        success_time = engine.config.default_success_time
        return engine.is_in_contact_for_duration(
            "green_ball", "blue_ball", success_time
        )

    level = Level(
        name="simple_contact",
        objects=objects,
        action_objects=["red_ball"],
        success_condition=success_condition,
    )

    env = InterphyreEnv(level)
    env.reset()
    obs, reward, term, trunc, info = env.step([(-1.0, 3.0, 0.5)])

    print(f"   Action objects: {level.action_objects}")
    print(f"   Success: {info['success']}")
    env.close()


def ramp_level():
    """Create a level with a ramp."""
    print("\n2. Ramp level")

    objects = {
        "ball": Ball(x=-3.0, y=3.0, radius=0.3, color="green", dynamic=True),
        "ramp": Bar(x=-1.5, y=1.5, length=4.0, thickness=0.2, angle=-20, dynamic=False),
        "target": Ball(x=2.0, y=-2.0, radius=0.4, color="blue", dynamic=False),
        "action_ball": Ball(x=0.0, y=4.0, radius=0.5, color="red", dynamic=True),
    }

    def success_condition(engine):
        return engine.has_contact("ball", "target")

    level = Level(
        name="ramp_puzzle",
        objects=objects,
        action_objects=["action_ball"],
        success_condition=success_condition,
    )

    env = InterphyreEnv(level)
    env.reset()
    obs, reward, term, trunc, info = env.step([(-3.0, 4.0, 0.5)])

    print("   Objects: ball, ramp (static), target (static), action_ball")
    print(f"   Success: {info['success']}")
    env.close()


def platform_level():
    """Create a level with platforms."""
    print("\n3. Platform level")

    objects = {
        "ball": Ball(x=-4.0, y=4.0, radius=0.3, color="green", dynamic=True),
        "platform1": Bar(
            x=-2.5, y=2.5, length=2.0, thickness=0.2, angle=0, dynamic=False
        ),
        "platform2": Bar(
            x=0.0, y=1.0, length=2.0, thickness=0.2, angle=0, dynamic=False
        ),
        "platform3": Bar(
            x=2.5, y=-0.5, length=2.0, thickness=0.2, angle=0, dynamic=False
        ),
        "goal": Ball(x=3.5, y=-3.0, radius=0.5, color="blue", dynamic=False),
        "pusher": Ball(x=0.0, y=5.0, radius=0.6, color="red", dynamic=True),
    }

    def success_condition(engine):
        return engine.has_contact("ball", "goal")

    level = Level(
        name="platformer",
        objects=objects,
        action_objects=["pusher"],
        success_condition=success_condition,
    )

    env = InterphyreEnv(level)
    env.reset()
    obs, reward, term, trunc, info = env.step([(-4.0, 4.5, 0.6)])

    print("   3 platforms, 1 goal, 1 pusher (action)")
    print(f"   Success: {info['success']}")
    env.close()


def custom_success_level():
    """Create a level with a complex success condition."""
    print("\n4. Custom success condition")

    objects = {
        "ball_a": Ball(x=-2.0, y=3.0, radius=0.4, color="green", dynamic=True),
        "ball_b": Ball(x=2.0, y=3.0, radius=0.4, color="blue", dynamic=True),
        "pusher": Ball(x=0.0, y=5.0, radius=0.5, color="red", dynamic=True),
    }

    def success_condition(engine):
        # Both balls must be below y=0 AND touching
        a = engine.bodies.get("ball_a")
        b = engine.bodies.get("ball_b")
        if not a or not b:
            return False
        both_low = a.position.y < 0 and b.position.y < 0
        touching = engine.has_contact("ball_a", "ball_b")
        return both_low and touching

    level = Level(
        name="double_condition",
        objects=objects,
        action_objects=["pusher"],
        success_condition=success_condition,
    )

    env = InterphyreEnv(level)
    env.reset()
    obs, reward, term, trunc, info = env.step([(0.0, 4.0, 0.5)])

    print("   Condition: balls below y=0 AND touching")
    print(f"   Success: {info['success']}")
    env.close()


def main():
    print("Custom Levels Demo")

    simple_contact_level()
    ramp_level()
    platform_level()
    custom_success_level()


if __name__ == "__main__":
    main()
