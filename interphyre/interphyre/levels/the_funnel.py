import numpy as np
from interphyre.objects import Ball, Bar
from interphyre.level import Level
from interphyre.levels import register_level
from interphyre.config import MIN_X, MAX_X, MAX_Y


def success_condition(engine):
    success_time = engine.config.default_success_time
    return engine.is_in_contact_for_duration(
        "green_ball", "purple_target", success_time
    )


@register_level
def build_level(seed=None, variant=0, scene=None) -> Level:
    rng = np.random.default_rng(seed if variant == 0 else (seed, variant))

    green_ball_radius = 0.3
    green_ball = Ball(
        x=rng.uniform(-1.0, 1.0),
        y=MAX_Y - green_ball_radius,
        radius=green_ball_radius,
        color="green",
        dynamic=True,
    )

    red_ball_radius = rng.uniform(0.3, 0.6)
    red_ball = Ball(
        x=0.0,
        y=0.0,
        radius=red_ball_radius,
        color="red",
        dynamic=True,
    )

    funnel_angle = rng.uniform(20.0, 30.0)
    funnel_length = 4.95
    funnel_top_y = 4.0

    left_funnel = Bar.from_corner(
        corner_x=MIN_X,
        corner_y=funnel_top_y,
        angle=-funnel_angle,
        length=funnel_length,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    right_funnel = Bar.from_corner(
        corner_x=MAX_X,
        corner_y=funnel_top_y,
        angle=180 + funnel_angle,
        length=funnel_length,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    target_side = rng.choice(["left", "right"])

    if target_side == "left":
        purple_target = Bar(
            left=MIN_X,
            right=-3,
            y=-4.9,
            thickness=0.2,
            color="purple",
            dynamic=False,
        )
        floor = Bar(
            left=-3,
            right=MAX_X,
            y=-4.9,
            thickness=0.2,
            color="black",
            dynamic=False,
        )
        blocker = Bar(
            left=-3,
            right=-1,
            y=-4.7,
            thickness=0.2,
            color="black",
            dynamic=False,
        )
    else:
        purple_target = Bar(
            left=3,
            right=MAX_X,
            y=-4.9,
            thickness=0.2,
            color="purple",
            dynamic=False,
        )
        floor = Bar(
            left=MIN_X,
            right=3,
            y=-4.9,
            thickness=0.2,
            color="black",
            dynamic=False,
        )
        blocker = Bar(
            left=1,
            right=3,
            y=-4.7,
            thickness=0.2,
            color="black",
            dynamic=False,
        )

    objects = {
        "green_ball": green_ball,
        "red_ball": red_ball,
        "left_funnel": left_funnel,
        "right_funnel": right_funnel,
        "blocker": blocker,
        "purple_target": purple_target,
        "floor": floor,
    }

    return Level(
        name="the_funnel",
        objects=objects,
        action_objects=["red_ball"],
        success_condition=success_condition,
        metadata={
            "description": "Make the green ball go through the funnel and touch the purple target"
        },
    )
