import numpy as np
from interphyre.objects import Ball, Bar
from interphyre.level import Level
from interphyre.levels import register_level
from interphyre.config import MIN_X, MIN_Y, WORLD_WIDTH, WORLD_HEIGHT


def success_condition(engine):
    success_time = engine.config.default_success_time
    return engine.is_in_contact_for_duration("green_ball", "purple_floor", success_time)


@register_level
def build_level(seed=None, variant=0, scene=None) -> Level:
    """Build the_cradle level.

    A green ball rests in a V-shaped cradle formed by two angled bars.
    The player must knock it out of the cradle onto the floor below.
    """
    rng = np.random.default_rng(seed if variant == 0 else (seed, variant))

    purple_floor = Bar.from_point_and_angle(
        x=0.0,
        y=MIN_Y + 0.1,
        length=WORLD_WIDTH,
        angle=0,
        color="purple",
        dynamic=False,
    )

    # Position green ball in cradle
    green_ball_radius = 0.5
    green_ball_x = rng.uniform(0.2, 0.8) * WORLD_WIDTH + MIN_X
    green_ball_y = rng.uniform(0.2, 0.5) * WORLD_HEIGHT + MIN_Y

    # Create V-shaped cradle with slight angle
    holder_length = rng.uniform(0.5, 0.75)
    holder_angle = 5.0
    holder_y = green_ball_y - green_ball_radius

    left_holder = Bar.from_corner(
        corner_x=green_ball_x,
        corner_y=holder_y,
        angle=180 - holder_angle,
        length=holder_length,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    right_holder = Bar.from_corner(
        corner_x=green_ball_x,
        corner_y=holder_y,
        angle=holder_angle,
        length=holder_length,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    green_ball = Ball(
        x=green_ball_x,
        y=green_ball_y,
        radius=green_ball_radius,
        color="green",
        dynamic=True,
    )

    red_ball_radius = rng.uniform(0.45, 0.6)
    red_ball = Ball(
        x=0.0,
        y=0.0,
        radius=red_ball_radius,
        color="red",
        dynamic=True,
    )

    objects = {
        "green_ball": green_ball,
        "red_ball": red_ball,
        "left_holder": left_holder,
        "right_holder": right_holder,
        "purple_floor": purple_floor,
    }

    return Level(
        name="the_cradle",
        objects=objects,
        action_objects=["red_ball"],
        success_condition=success_condition,
        metadata={"description": "Push the green ball onto the purple floor"},
    )
