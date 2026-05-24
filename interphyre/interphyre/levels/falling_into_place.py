import numpy as np
from interphyre.objects import Ball, Bar, Basket
from interphyre.level import Level
from interphyre.levels import register_level


def success_condition(engine):
    success_time = engine.config.default_success_time
    return engine.is_in_contact_for_duration("green_ball", "blue_basket", success_time)


@register_level
def build_level(seed=None, variant=0, scene=None) -> Level:
    rng = np.random.default_rng(seed if variant == 0 else (seed, variant))

    bar_height = rng.uniform(-2, 1)
    basket_scale = 0.75
    basket_wall_thickness = 0.175
    basket_top_width = 1.083 * basket_scale * 1.25
    # Total half width includes wall thickness
    basket_half_width = (basket_top_width + 2 * basket_wall_thickness) / 2
    hole_width = 2.0

    # Basket is at x=0, hole center varies but must contain the basket
    max_hole_offset = (hole_width / 2) - basket_half_width - 0.1
    hole_center = rng.uniform(-max_hole_offset, max_hole_offset)

    left_bar = Bar(
        left=-5,
        right=hole_center - hole_width / 2,
        y=bar_height,
        thickness=0.2,
        color="black",
        dynamic=False,
    )
    right_bar = Bar(
        left=hole_center + hole_width / 2,
        right=5,
        y=bar_height,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    bottom_ramp = Bar.from_corner(
        corner_x=-5,
        corner_y=-5,
        angle=10,
        length=11,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    blue_basket = Basket(
        x=0,
        y=4.3,
        scale=basket_scale,
        color="blue",
        angle=180,
        dynamic=True,
    )

    ball_radius = 0.5
    hole_left = hole_center - hole_width / 2
    hole_right = hole_center + hole_width / 2
    if rng.random() < 0.5:
        ball_x = rng.uniform(-4.5, hole_left - ball_radius)
    else:
        ball_x = rng.uniform(hole_right + ball_radius, 4.5)

    green_ball = Ball(
        x=ball_x,
        y=bar_height + 0.1 + ball_radius,
        radius=ball_radius,
        color="green",
        dynamic=True,
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
        "green_ball": green_ball,
        "blue_basket": blue_basket,
        "red_ball": red_ball,
        "left_bar": left_bar,
        "right_bar": right_bar,
        "bottom_ramp": bottom_ramp,
    }

    return Level(
        name="falling_into_place",
        objects=objects,
        action_objects=["red_ball"],
        success_condition=success_condition,
        metadata={"description": "Make the green ball touch the blue basket"},
    )
