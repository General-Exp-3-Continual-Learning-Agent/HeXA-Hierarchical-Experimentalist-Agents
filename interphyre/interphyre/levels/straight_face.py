import numpy as np
from interphyre.objects import Ball, Bar
from interphyre.level import Level
from interphyre.levels import register_level


def success_condition(engine):
    success_time = engine.config.default_success_time
    return engine.is_in_contact_for_duration("green_ball", "purple_pad", success_time)


@register_level
def build_level(seed=None, variant=0, scene=None) -> Level:
    rng = np.random.default_rng(seed if variant == 0 else (seed, variant))

    floor = Bar.from_point_and_angle(
        x=0.0,
        y=-4.9,
        length=10.0,
        angle=0,
        color="black",
        dynamic=False,
    )

    # Two balls stacked vertically
    ball_radius = 0.5
    ball_x = rng.uniform(-4, 4)

    # Ensure ball not too far right
    if ball_x - ball_radius >= 2.0:
        ball_x = 1.0

    # Upper ball near top
    green_ball_y = 4.5
    green_ball = Ball(
        x=ball_x,
        y=green_ball_y,
        radius=ball_radius,
        color="green",
        dynamic=True,
    )

    # Lower ball with variable spacing
    gray_ball_y = rng.choice([0.5, 2.5])
    gray_ball = Ball(
        x=ball_x,
        y=gray_ball_y,
        radius=ball_radius,
        color="gray",
        dynamic=True,
    )

    # Target pad on floor with vertical posts on sides
    target_length = rng.uniform(1.0, 2.0)
    target_x = rng.uniform(-4, 4)
    pad_thickness = 0.2
    purple_pad = Bar.from_point_and_angle(
        x=target_x,
        y=-4.7,
        length=target_length,
        angle=0,
        thickness=pad_thickness,
        color="purple",
        dynamic=False,
    )

    # Vertical posts on target edges
    post_length = 0.2
    post_y = -4.7 + pad_thickness / 2 + post_length / 2

    pad_left_rim = Bar.from_point_and_angle(
        x=purple_pad.left - pad_thickness / 2,
        y=post_y,
        length=post_length,
        angle=90,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    pad_right_rim = Bar.from_point_and_angle(
        x=purple_pad.right + pad_thickness / 2,
        y=post_y,
        length=post_length,
        angle=90,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    # Randomize red ball position
    red_ball_radius = rng.uniform(0.6, 1.2)
    red_ball = Ball(
        x=rng.uniform(-4.5, 4.5),
        y=rng.uniform(2.0, 4.5),
        radius=red_ball_radius,
        color="red",
        dynamic=True,
    )

    objects = {
        "green_ball": green_ball,
        "red_ball": red_ball,
        "gray_ball": gray_ball,
        "purple_pad": purple_pad,
        "pad_left_rim": pad_left_rim,
        "pad_right_rim": pad_right_rim,
        "floor": floor,
    }

    return Level(
        name="straight_face",
        objects=objects,
        action_objects=["red_ball"],
        success_condition=success_condition,
        metadata={"description": "Knock the green ball onto the purple pad"},
    )
