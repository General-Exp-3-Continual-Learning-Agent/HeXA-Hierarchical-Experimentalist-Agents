import numpy as np
from interphyre.objects import Ball, Bar
from interphyre.level import Level
from interphyre.levels import register_level
from interphyre.config import MIN_X, MAX_X


def success_condition(engine):
    success_time = engine.config.default_success_time
    return engine.is_in_contact_for_duration(
        "green_ball", "purple_ground", success_time
    )


@register_level
def build_level(seed=None, variant=0, scene=None) -> Level:
    rng = np.random.default_rng(seed if variant == 0 else (seed, variant))

    purple_ground = Bar.from_point_and_angle(
        x=0.0,
        y=-4.9,
        length=10.0,
        thickness=0.2,
        angle=0.0,
        color="purple",
        dynamic=False,
    )

    # Ramps form 45° right triangles with wall and floor
    ramp_length = round(rng.uniform(0.5, 2.0), 2)
    ramp_height = ramp_length / np.sqrt(2)
    floor_y = purple_ground.y + purple_ground.thickness / 2

    left_ramp = Bar.from_corner(
        corner_x=MIN_X,
        corner_y=floor_y + ramp_height,
        angle=315,
        length=ramp_length,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    right_ramp = Bar.from_corner(
        corner_x=MAX_X,
        corner_y=floor_y + ramp_height,
        angle=225,
        length=ramp_length,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    flagpole_x = round(rng.uniform(-3.0, 3.0), 2)
    flagpole_length = round(rng.uniform(3.0, 5.0), 2)
    flagpole_y = round(
        purple_ground.y + purple_ground.thickness / 2 + flagpole_length / 2, 2
    )

    flagpole = Bar.from_point_and_angle(
        x=flagpole_x,
        y=flagpole_y,
        angle=90.0,
        length=flagpole_length,
        thickness=0.2,
        color="gray",
        dynamic=True,
    )

    green_ball_radius = round(rng.uniform(0.25, 1.25), 2)
    green_ball_y = flagpole.y + (flagpole.length / 2) + green_ball_radius

    green_ball = Ball(
        x=flagpole.x,
        y=green_ball_y,
        radius=green_ball_radius,
        color="green",
        dynamic=True,
    )

    red_ball_radius = round(rng.uniform(0.2, 0.7), 2)
    red_ball_offset = round(rng.uniform(1.5, 3.0), 2)
    red_ball_x = flagpole.x + rng.choice([-1, 1]) * red_ball_offset
    red_ball_y = flagpole.y + flagpole.length / 2 + red_ball_radius

    red_ball = Ball(
        x=red_ball_x,
        y=red_ball_y,
        radius=red_ball_radius,
        color="red",
        dynamic=True,
    )

    ceiling_clearance = 0.1
    ceiling_y = green_ball_y + green_ball_radius + ceiling_clearance + 0.1

    ceiling = Bar.from_point_and_angle(
        x=0.0,
        y=ceiling_y,
        length=10.0,
        thickness=0.2,
        angle=0.0,
        color="black",
        dynamic=False,
    )

    objects = {
        "flagpole": flagpole,
        "green_ball": green_ball,
        "red_ball": red_ball,
        "ceiling": ceiling,
        "left_ramp": left_ramp,
        "right_ramp": right_ramp,
        "purple_ground": purple_ground,
    }

    action_bounds = {
        "x": (-4.0, 4.0),
        "y": (-4.0, 3.5),
        "r": (0.2, 1.25),
    }

    return Level(
        name="flagpole_sitta",
        objects=objects,
        action_objects=["red_ball"],
        success_condition=success_condition,
        metadata={
            "description": "Knock the green ball off of the pole and onto the ground",
            "action_bounds": action_bounds,
        },
    )
