import numpy as np
from interphyre.objects import Ball, Bar, Basket
from interphyre.level import Level
from interphyre.levels import register_level
from interphyre.config import MAX_X, MAX_Y, MIN_X, MIN_Y


def success_condition(engine):
    success_time = engine.config.default_success_time
    return engine.is_in_contact_for_duration(
        "green_ball", "purple_basket", success_time
    )


@register_level
def build_level(seed=None, variant=0, scene=None) -> Level:
    rng = np.random.default_rng(seed if variant == 0 else (seed, variant))

    # Level parameters
    h0 = rng.uniform(0.2, 0.5)
    w2 = rng.uniform(0.15, 0.3)
    angle = rng.uniform(15, 20)

    # Basket scale: linear mapping from w2 (0.15-0.3) to scale (1.0-1.5)
    basket_scale = 1.0 + (w2 - 0.15) * (10 / 3)

    bar_thickness = 0.2
    basket_x = rng.uniform(-1, 1)
    basket_y_from_h0 = MIN_Y + h0 * 10
    basket_y = rng.uniform(basket_y_from_h0, basket_y_from_h0 + 1.0)

    basket = Basket(
        x=basket_x,
        y=basket_y,
        scale=basket_scale,
        angle=0.0,
        anchor="bottom_center",
        color="purple",
        dynamic=False,
    )

    right_ramp_angle = -angle
    right_ramp_split = rng.uniform(0.2, 0.5)
    basket_height = basket.height
    basket_width = basket.bottom_width + 2 * basket_height * np.tan(np.radians(5))
    basket_right_edge = basket_x + basket_width / 2
    right_space = MAX_X - basket_right_edge

    right_ramp_length = (
        right_space * right_ramp_split / np.cos(np.radians(right_ramp_angle))
    )

    right_ramp_x = (
        basket_x + basket_width / 2 + right_ramp_length / 2 + bar_thickness / 2 + 0.005
    )
    right_ramp_y = (
        basket_y
        + basket.floor_thickness
        + basket_height
        + (right_ramp_length / 2) * np.sin(np.radians(right_ramp_angle))
    )

    right_ramp_corner_x = right_ramp_x - (right_ramp_length / 2) * np.cos(
        np.radians(right_ramp_angle)
    )
    right_ramp_corner_y = right_ramp_y - (right_ramp_length / 2) * np.sin(
        np.radians(right_ramp_angle)
    )

    right_ramp = Bar.from_corner(
        corner_x=right_ramp_corner_x,
        corner_y=right_ramp_corner_y,
        angle=right_ramp_angle,
        length=right_ramp_length,
        thickness=bar_thickness,
        color="black",
        dynamic=False,
    )

    ramp_edge_x = right_ramp.x + (right_ramp_length / 2) * np.cos(
        np.radians(right_ramp_angle)
    )
    ramp_edge_y = right_ramp.y + (right_ramp_length / 2) * np.sin(
        np.radians(right_ramp_angle)
    )

    right_beam_length = MAX_X - ramp_edge_x - bar_thickness
    right_beam_x = ramp_edge_x + right_beam_length / 2 + bar_thickness / 2
    right_beam_y = ramp_edge_y - 0.005

    right_beam = Bar.from_point_and_angle(
        x=right_beam_x,
        y=right_beam_y,
        angle=0,
        length=right_beam_length,
        thickness=bar_thickness,
        color="black",
        dynamic=False,
    )

    left_ramp_angle = -angle
    left_beam_split = rng.uniform(0.3, 0.6)
    basket_left_edge = basket_x - basket_width / 2
    left_space = basket_left_edge - MIN_X

    black_ball_radius = 0.3
    min_required_split = (6 * black_ball_radius + bar_thickness) / left_space
    left_beam_split = max(left_beam_split, min_required_split)

    left_ramp_length = (
        left_space * (1 - left_beam_split) * 0.5 / np.cos(np.radians(left_ramp_angle))
    ) - bar_thickness / 2

    left_ramp_1_x = (
        basket_x - basket_width / 2 - left_ramp_length / 2 - bar_thickness / 2
    )
    left_ramp_1_y = (
        basket_y
        + basket.floor_thickness
        + basket_height
        - (left_ramp_length / 2) * np.sin(np.radians(left_ramp_angle))
    )

    left_ramp_1_corner_x = left_ramp_1_x - (left_ramp_length / 2) * np.cos(
        np.radians(left_ramp_angle)
    )
    left_ramp_1_corner_y = left_ramp_1_y - (left_ramp_length / 2) * np.sin(
        np.radians(left_ramp_angle)
    )

    left_ramp_1 = Bar.from_corner(
        corner_x=left_ramp_1_corner_x,
        corner_y=left_ramp_1_corner_y,
        angle=left_ramp_angle,
        length=left_ramp_length,
        thickness=bar_thickness,
        color="black",
        dynamic=False,
    )

    left_beam_length = left_space * left_beam_split - bar_thickness
    left_ramp_1_right_x = left_ramp_1.x - (left_ramp_length / 2) * np.cos(
        np.radians(left_ramp_angle)
    )
    left_ramp_1_right_y = left_ramp_1.y - (left_ramp_length / 2) * np.sin(
        np.radians(left_ramp_angle)
    )
    left_beam_x = left_ramp_1_right_x - left_beam_length / 2 - bar_thickness / 2
    left_beam_y = left_ramp_1_right_y
    left_beam = Bar.from_point_and_angle(
        x=left_beam_x,
        y=left_beam_y,
        angle=0,
        length=left_beam_length,
        thickness=bar_thickness,
        color="gray",
        dynamic=True,
    )

    left_ramp_2_x = (
        left_beam_x - left_beam_length / 2 - left_ramp_length / 2 - bar_thickness / 2
    )
    left_ramp_2_y = left_beam_y - (left_ramp_length / 2) * np.sin(
        np.radians(left_ramp_angle)
    )

    left_ramp_2_corner_x = left_ramp_2_x - (left_ramp_length / 2) * np.cos(
        np.radians(left_ramp_angle)
    )
    left_ramp_2_corner_y = left_ramp_2_y - (left_ramp_length / 2) * np.sin(
        np.radians(left_ramp_angle)
    )

    left_ramp_2 = Bar.from_corner(
        corner_x=left_ramp_2_corner_x,
        corner_y=left_ramp_2_corner_y,
        angle=left_ramp_angle,
        length=left_ramp_length,
        thickness=bar_thickness,
        color="black",
        dynamic=False,
    )

    black_ball_1_x = left_beam.right - black_ball_radius
    black_ball_1_y = left_beam.y - black_ball_radius - bar_thickness / 2
    black_ball_1 = Ball(
        x=black_ball_1_x,
        y=black_ball_1_y,
        radius=black_ball_radius,
        color="black",
        dynamic=False,
    )

    black_ball_2_x = left_beam.left + 3 * black_ball_radius
    black_ball_2_y = left_beam.y - black_ball_radius - bar_thickness / 2
    black_ball_2 = Ball(
        x=black_ball_2_x,
        y=black_ball_2_y,
        radius=black_ball_radius,
        color="black",
        dynamic=False,
    )

    left_beam_base = Bar.from_point_and_angle(
        x=left_beam_x,
        y=left_beam_y - 2 * black_ball_radius - bar_thickness,
        angle=0,
        length=left_beam_length,
        thickness=bar_thickness,
        color="black",
        dynamic=False,
    )

    green_ball_radius = 0.3
    green_ball_x = left_ramp_2.x + (left_ramp_length / 2) * np.cos(
        np.radians(left_ramp_angle)
    )
    green_ball_y = (
        left_ramp_2.y
        + (left_ramp_length / 2) * np.sin(np.radians(left_ramp_angle))
        + 1.5 * green_ball_radius
    )
    green_ball = Ball(
        x=green_ball_x,
        y=green_ball_y,
        radius=green_ball_radius,
        color="green",
        dynamic=True,
    )

    # Position ceiling with 1.0 unit clearance above ball
    ball_top = green_ball_y + green_ball_radius
    min_ceiling_y = ball_top + 1.0 - bar_thickness / 2
    max_ceiling_y = MAX_Y - bar_thickness / 2

    if min_ceiling_y >= max_ceiling_y:
        ceiling_y = max_ceiling_y
    else:
        ceiling_y = rng.uniform(min_ceiling_y, max_ceiling_y)
    ceiling = Bar.from_point_and_angle(
        x=0,
        y=ceiling_y,
        length=10,
        thickness=bar_thickness,
        angle=0,
        color="black",
        dynamic=False,
    )

    red_ball = Ball(
        x=rng.uniform(MIN_X, MAX_X),
        y=green_ball_y,
        radius=rng.uniform(0.2, 0.4),
        color="red",
        dynamic=True,
    )

    objects = {
        "purple_basket": basket,
        "right_ramp": right_ramp,
        "right_beam": right_beam,
        "left_ramp_1": left_ramp_1,
        "left_beam": left_beam,
        "left_ramp_2": left_ramp_2,
        "left_beam_base": left_beam_base,
        "black_ball_1": black_ball_1,
        "black_ball_2": black_ball_2,
        "green_ball": green_ball,
        "red_ball": red_ball,
        "ceiling": ceiling,
    }

    return Level(
        name="marble_race",
        objects=objects,
        action_objects=["red_ball"],
        success_condition=success_condition,
        metadata={"description": "Get the green ball into the purple basket."},
    )
