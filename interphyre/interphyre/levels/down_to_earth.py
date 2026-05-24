import numpy as np

from interphyre.level import Level
from interphyre.levels import register_level
from interphyre.objects import Ball, Bar


def success_condition(engine):
    success_time = engine.config.default_success_time
    return engine.is_in_contact_for_duration(
        "green_ball", "purple_ground", success_time
    )


@register_level
def build_level(seed=None, variant=0, scene=None) -> Level:
    rng = np.random.default_rng(seed if variant == 0 else (seed, variant))

    # Ground plane
    purple_ground = Bar(
        left=-5,
        right=5,
        y=-4.9,
        thickness=0.2,
        color="purple",
        dynamic=False,
    )

    # Draw all RNG values in fixed order
    platform_width = rng.uniform(1, 7)
    platform_x = rng.uniform(-5, 5 - platform_width)
    platform_y = rng.uniform(-2, 2)
    red_ball_radius = rng.uniform(0.3, 0.6)

    # Platform
    platform = Bar(
        left=platform_x,
        right=platform_x + platform_width,
        y=platform_y,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    # Green ball centered above platform, near the top of the screen
    green_ball_radius = 0.5
    green_ball_x = platform_x + platform_width / 2
    green_ball_y = 4.5 - green_ball_radius

    green_ball = Ball(
        x=green_ball_x,
        y=green_ball_y,
        radius=green_ball_radius,
        color="green",
        dynamic=True,
    )

    # Red action ball
    red_ball = Ball(
        x=0,
        y=0,
        radius=red_ball_radius,
        color="red",
        dynamic=True,
    )

    objects = {
        "green_ball": green_ball,
        "red_ball": red_ball,
        "purple_ground": purple_ground,
        "platform": platform,
    }

    return Level(
        name="down_to_earth",
        objects=objects,
        action_objects=["red_ball"],
        success_condition=success_condition,
        metadata={"description": "Make the green ball hit the ground"},
    )
