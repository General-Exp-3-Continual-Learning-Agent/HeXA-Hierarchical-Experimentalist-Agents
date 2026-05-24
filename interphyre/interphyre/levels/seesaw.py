import numpy as np
from interphyre.objects import Ball, Bar
from interphyre.level import Level
from interphyre.levels import register_level


def success_condition(engine):
    success_time = engine.config.default_success_time
    return engine.is_in_contact_for_duration("green_ball", "blue_beam", success_time)


@register_level
def build_level(seed=None, variant=0, scene=None) -> Level:
    rng = np.random.default_rng(seed if variant == 0 else (seed, variant))

    floor_y = rng.uniform(-4, 1)
    fulcrum_x = rng.uniform(-2.5, 1.5)
    fulcrum_radius = 0.5
    beam_length = rng.uniform(3, 4.5)
    ball_radius = 0.5
    offset = rng.uniform(0.1, 0.3)
    beam_left = fulcrum_x - beam_length / 2
    beam_right = fulcrum_x + beam_length / 2
    # Ball at left or right edge of beam
    if rng.random() < 0.5:
        ball_x = beam_left - ball_radius + offset
    else:
        ball_x = beam_right + ball_radius - offset

    floor = Bar(
        left=-5,
        right=5,
        y=floor_y,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    fulcrum = Ball(
        x=fulcrum_x,
        y=floor_y + 0.1 + fulcrum_radius,
        radius=fulcrum_radius,
        color="black",
        dynamic=False,
    )

    beam_y = fulcrum.y + fulcrum_radius + 0.1
    blue_beam = Bar(
        x=fulcrum_x,
        y=beam_y,
        length=beam_length,
        angle=0.0,
        thickness=0.2,
        color="blue",
        dynamic=True,
    )

    green_ball_y = 4.5 - ball_radius
    green_ball = Ball(
        x=ball_x,
        y=green_ball_y,
        radius=ball_radius,
        color="green",
        dynamic=True,
    )

    guard_offset = 1.5
    guard_length = 2.0
    guard_bottom = max(floor_y + 0.1, green_ball_y - ball_radius - 1.5 - guard_length)
    guard_top = guard_bottom + guard_length

    left_guard = Bar(
        x=blue_beam.left - guard_offset,
        top=guard_top,
        bottom=guard_bottom,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    right_guard = Bar(
        x=blue_beam.right + guard_offset,
        top=guard_top,
        bottom=guard_bottom,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    red_ball_radius = rng.uniform(0.3, 0.6)
    red_ball = Ball(
        x=0,
        y=0,
        radius=red_ball_radius,
        color="red",
        dynamic=True,
    )

    objects = {
        "floor": floor,
        "fulcrum": fulcrum,
        "blue_beam": blue_beam,
        "green_ball": green_ball,
        "red_ball": red_ball,
        "left_guard": left_guard,
        "right_guard": right_guard,
    }

    return Level(
        name="seesaw",
        objects=objects,
        action_objects=["red_ball"],
        success_condition=success_condition,
        metadata={"description": "Make the green ball land on the blue beam"},
    )
