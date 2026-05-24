import numpy as np
from interphyre.objects import Ball, Bar
from interphyre.level import Level
from interphyre.levels import register_level
from interphyre.config import MAX_X, MAX_Y, MIN_Y


def success_condition(engine):
    success_time = engine.config.default_success_time
    return engine.is_in_contact_for_duration("green_ball", "purple_pad", success_time)


@register_level
def build_level(seed=None, variant=0, scene=None) -> Level:
    """Build keyhole level.

    The green ball must pass through a narrow gap between two vertical dividers
    to reach the purple target pad on the opposite side.
    """
    rng = np.random.default_rng(seed if variant == 0 else (seed, variant))

    # Randomly place target pad on left or right side
    purple_pad_x = rng.choice([-2.5, 2.5])
    purple_pad = Bar.from_point_and_angle(
        x=purple_pad_x,
        y=-4.9,
        length=5,
        angle=0,
        color="purple",
        dynamic=False,
    )
    black_pad = Bar.from_point_and_angle(
        x=-purple_pad_x,
        y=-4.9,
        length=5,
        angle=0,
        color="black",
        dynamic=False,
    )

    # Create vertical divider at center with gap at bottom
    gap_height = rng.uniform(2.5, 4.5)
    gap_width = rng.uniform(0.5, 1)
    room_height = MAX_Y - MIN_Y
    top_divider = Bar.from_point_and_angle(
        x=0,
        y=MAX_Y - (room_height - gap_height) / 2,
        length=room_height - gap_height,
        angle=90,
        color="black",
        dynamic=False,
    )
    # Size ball to fit through gap
    green_ball_radius = np.clip(
        (
            rng.uniform(
                min(gap_width / 2, gap_height / 2), max(gap_width / 2, gap_height / 2)
            )
            - 0.05
        ),
        None,
        0.7,
    )

    # Create second vertical divider on ball's side to form keyhole
    # Position it opposite to target pad
    bottom_divider_x = rng.uniform(2 * green_ball_radius + 0.1, 3.5) * np.sign(
        -purple_pad_x
    )
    # Size to ensure navigable gap (ball diameter + clearance).
    # Cap at 1.2 ensures bd.top = MIN_Y + length ≤ -3.8, below the -3.5 threshold
    # above which the keyhole gap is too narrow for the green ball to pass through
    # regardless of placement (94 impossible seeds in 10k regen used bd.top > -3.8).
    max_bottom_length = gap_height - 3 * green_ball_radius
    bottom_divider_length = min(max_bottom_length * rng.uniform(0.75, 0.95), 1.2)
    bottom_divider_y = MIN_Y + (bottom_divider_length) / 2
    # Place ball on same side as bottom divider, above the gap
    green_ball_offset = rng.uniform(
        green_ball_radius, (MAX_X - np.abs(bottom_divider_x)) * 0.5
    )
    green_ball_x = (np.abs(bottom_divider_x) + green_ball_offset) * np.sign(
        bottom_divider_x
    )
    green_ball_y = rng.uniform(bottom_divider_y + gap_height / 2, MAX_Y)
    green_ball = Ball(
        x=green_ball_x,
        y=green_ball_y,
        radius=green_ball_radius,
        color="green",
        dynamic=True,
    )

    bottom_divider = Bar.from_point_and_angle(
        x=bottom_divider_x,
        y=bottom_divider_y,
        length=bottom_divider_length,
        angle=90,
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

    objects = {
        "purple_pad": purple_pad,
        "black_pad": black_pad,
        "top_divider": top_divider,
        "bottom_divider": bottom_divider,
        "green_ball": green_ball,
        "red_ball": red_ball,
    }

    return Level(
        name="keyhole",
        objects=objects,
        action_objects=["red_ball"],
        success_condition=success_condition,
        metadata={
            "description": "Get the green ball through the keyhole to hit the purple pad"
        },
    )
