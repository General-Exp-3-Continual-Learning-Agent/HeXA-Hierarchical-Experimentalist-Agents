import numpy as np
from interphyre.objects import Ball, Bar
from interphyre.level import Level
from interphyre.levels import register_level


def success_condition(engine):
    success_time = engine.config.default_success_time
    return engine.is_in_contact_for_duration("green_ball", "purple_wall", success_time)


@register_level
def build_level(seed=None, variant=0, scene=None) -> Level:
    rng = np.random.default_rng(seed if variant == 0 else (seed, variant))

    ball_radius = rng.uniform(0.25, 0.6)

    # Target wall on left or right
    left_wall = rng.choice([True, False])
    wall_x = -4.9 if left_wall else 4.9

    purple_wall = Bar(
        top=5.0,
        bottom=-5.0,
        x=wall_x,
        thickness=0.2,
        color="purple",
        dynamic=False,
    )

    # Shelf is wide enough for the green ball to reach the wall while avoiding the ground
    shelf_width = 9.9 - ball_radius * 4
    shelf_top = -4.2

    shelf = Bar(
        left=-shelf_width / 2,
        right=shelf_width / 2,
        y=shelf_top,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    # Legs at 65° angles
    leg_length = 2.0

    left_leg = Bar.from_corner(
        corner_x=shelf.left + 0.05,
        corner_y=shelf_top + 0.05,
        angle=180 + 65,
        length=leg_length,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    right_leg = Bar.from_corner(
        corner_x=shelf.right - 0.05,
        corner_y=shelf_top + 0.05,
        angle=-65,
        length=leg_length,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    # Ball on shelf, constrained to be within reach of target wall
    ball_x_min = -shelf_width / 2 + ball_radius + 0.2
    ball_x_max = shelf_width / 2 - ball_radius - 0.2
    ball_x = rng.uniform(ball_x_min, ball_x_max)

    max_dist_from_wall = 7.0
    if abs(ball_x - wall_x) > max_dist_from_wall:
        if left_wall:
            ball_x = rng.uniform(ball_x_min, -wall_x - max_dist_from_wall)
        else:
            ball_x = rng.uniform(wall_x - max_dist_from_wall, ball_x_max)

    ball_y_offset = rng.uniform(0, 1.5)
    ball_y = shelf_top + ball_radius + 0.1 + ball_y_offset

    green_ball = Ball(
        x=ball_x,
        y=ball_y,
        radius=ball_radius,
        color="green",
        dynamic=True,
    )

    red_ball_radius = rng.uniform(0.25, 0.6)
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
        "purple_wall": purple_wall,
        "shelf": shelf,
        "left_leg": left_leg,
        "right_leg": right_leg,
    }

    return Level(
        name="end_of_line",
        objects=objects,
        action_objects=["red_ball"],
        success_condition=success_condition,
        metadata={
            "description": "Knock the green ball off the table so it hits the purple wall"
        },
    )
