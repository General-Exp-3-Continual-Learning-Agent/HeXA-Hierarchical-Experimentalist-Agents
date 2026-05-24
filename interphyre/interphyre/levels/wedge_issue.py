import math
import numpy as np
from interphyre.objects import Ball, Bar
from interphyre.level import Level
from interphyre.levels import register_level
from interphyre.config import MIN_X, MAX_X, MIN_Y, WORLD_WIDTH, WORLD_HEIGHT


def success_condition(engine):
    success_time = engine.config.default_success_time
    return engine.is_in_contact_for_duration("green_ball", "purple_bar", success_time)


@register_level
def build_level(seed=None, variant=0, scene=None) -> Level:
    rng = np.random.default_rng(seed if variant == 0 else (seed, variant))

    bar_thickness = 0.2
    bar_y = rng.uniform(0.1, 0.5)
    angle_left = rng.uniform(15, 30)
    angle_right = rng.uniform(15, 30)
    length_left = rng.uniform(0.2, 0.8)

    right_edge = MAX_X + 0.1
    left_edge = MIN_X - 0.1
    bar_gap_ratio = 0.15

    def center_for_edges(
        angle_deg, length, thickness, *, right=None, left=None, bottom=None
    ):
        angle_rad = math.radians(angle_deg)
        ux, uy = math.cos(angle_rad), math.sin(angle_rad)
        vx, vy = -uy, ux
        dx = length / 2
        dy = thickness / 2
        corners = (
            (dx * ux + dy * vx, dx * uy + dy * vy),
            (dx * ux - dy * vx, dx * uy - dy * vy),
            (-dx * ux + dy * vx, -dx * uy + dy * vy),
            (-dx * ux - dy * vx, -dx * uy - dy * vy),
        )
        xs = [corner[0] for corner in corners]
        ys = [corner[1] for corner in corners]
        min_x, max_x = min(xs), max(xs)
        min_y = min(ys)
        center_x = 0.0
        center_y = 0.0
        if right is not None:
            center_x = right - max_x
        elif left is not None:
            center_x = left - min_x
        if bottom is not None:
            center_y = bottom - min_y
        return center_x, center_y

    purple_bar_length = (1 - length_left) * WORLD_WIDTH
    purple_bar_bottom = bar_y * WORLD_HEIGHT + MIN_Y
    purple_center_x, purple_center_y = center_for_edges(
        angle_left,
        purple_bar_length,
        bar_thickness,
        right=right_edge,
        bottom=purple_bar_bottom,
    )
    purple_bar = Bar(
        x=purple_center_x,
        y=purple_center_y,
        angle=angle_left,
        length=purple_bar_length,
        thickness=bar_thickness,
        color="purple",
        dynamic=False,
    )

    black_bar_length = length_left * WORLD_WIDTH
    black_bar_bottom = (bar_y + bar_gap_ratio) * WORLD_HEIGHT + MIN_Y
    black_center_x, black_center_y = center_for_edges(
        -angle_right,
        black_bar_length,
        bar_thickness,
        left=left_edge,
        bottom=black_bar_bottom,
    )
    black_bar = Bar(
        x=black_center_x,
        y=black_center_y,
        angle=-angle_right,
        length=black_bar_length,
        thickness=bar_thickness,
        color="black",
        dynamic=False,
    )

    green_ball_radius = 0.5
    green_ball_bottom = 0.9 * WORLD_HEIGHT + MIN_Y
    green_ball_y = green_ball_bottom + green_ball_radius
    ball_x_min = MIN_X + 0.2 * WORLD_WIDTH
    ball_x_max = MIN_X + 0.8 * WORLD_WIDTH
    if black_bar.top > green_ball_bottom:
        ball_x_min = max(ball_x_min, black_bar.right + green_ball_radius + 0.05)
    green_ball_x = rng.uniform(ball_x_min, max(ball_x_min, ball_x_max))

    green_ball = Ball(
        x=green_ball_x,
        y=green_ball_y,
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

    objects = {
        "green_ball": green_ball,
        "red_ball": red_ball,
        "purple_bar": purple_bar,
        "black_bar": black_bar,
    }

    return Level(
        name="wedge_issue",
        objects=objects,
        action_objects=["red_ball"],
        success_condition=success_condition,
        metadata={
            "description": "Get the green ball to stay in contact with the purple bar"
        },
    )
