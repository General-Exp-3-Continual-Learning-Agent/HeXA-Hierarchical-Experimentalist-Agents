import numpy as np
from interphyre.objects import Ball, Bar
from interphyre.level import Level
from interphyre.levels import register_level
from interphyre.config import MIN_X, MAX_X, MIN_Y


def success_condition(engine):
    success_time = engine.config.default_success_time
    return engine.is_in_contact_for_duration(
        "green_bar",
        "purple_ground",
        success_time,
    )


@register_level
def build_level(seed=None, variant=0, scene=None) -> Level:
    rng = np.random.default_rng(seed if variant == 0 else (seed, variant))

    platform_length = rng.uniform(4.0, 6.0)
    clearance = 0.5
    platform_left = rng.uniform(MIN_X + clearance, MAX_X - platform_length - clearance)
    platform_y = rng.uniform(-3, 0)

    black_platform = Bar(
        left=platform_left,
        right=platform_left + platform_length,
        y=platform_y,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    green_bar_length = rng.uniform(2.0, 3.0)
    ceiling_thickness = 0.2
    ceiling_y = platform_y + 0.1 + green_bar_length + 2.0

    center_offset = 0.1
    dist_to_left = platform_left - MIN_X
    dist_to_right = MAX_X - (platform_left + platform_length)
    if dist_to_left > dist_to_right:
        green_bar_x = platform_left + center_offset
    else:
        green_bar_x = platform_left + platform_length - center_offset
    green_bar_y = platform_y + 0.1 + green_bar_length / 2

    green_bar = Bar.from_point_and_angle(
        x=green_bar_x,
        y=green_bar_y,
        angle=90.0,
        length=green_bar_length,
        thickness=0.2,
        color="green",
        dynamic=True,
    )

    ceiling = Bar(
        left=MIN_X,
        right=MAX_X,
        y=ceiling_y,
        thickness=ceiling_thickness,
        color="black",
        dynamic=False,
    )

    red_ball_radius = rng.uniform(0.3, 0.6)
    red_ball = Ball(
        x=0.0,
        y=0.0,
        radius=red_ball_radius,
        color="red",
        dynamic=True,
    )

    purple_ground = Bar(
        left=MIN_X,
        right=MAX_X,
        y=MIN_Y + 0.2 / 2,
        color="purple",
        dynamic=False,
    )

    objects = {
        "black_platform": black_platform,
        "green_bar": green_bar,
        "ceiling": ceiling,
        "red_ball": red_ball,
        "purple_ground": purple_ground,
    }

    return Level(
        name="cliffhanger",
        objects=objects,
        action_objects=["red_ball"],
        success_condition=success_condition,
        metadata={
            "description": "Tip over the green bar so it hits the purple ground.",
        },
    )
