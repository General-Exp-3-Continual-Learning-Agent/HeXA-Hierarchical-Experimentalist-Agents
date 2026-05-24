import numpy as np
from interphyre.objects import Ball, Basket, Bar
from interphyre.level import Level
from interphyre.levels import register_level


def success_condition(engine):
    success_time = engine.config.default_success_time
    return engine.is_in_contact_for_duration("green_bar", "purple_wall", success_time)


@register_level
def build_level(seed=None, variant=0, scene=None) -> Level:
    """Build the tipping point level.

    A vertical bar rests on a basket. The goal is to tip the bar so it contacts
    the wall and maintains contact for the success duration.

    Geometry constraint: Basket is positioned to ensure the bar can rest at a
    stable angle (30-60° from vertical) against the wall when tipped.
    """
    rng = np.random.default_rng(seed if variant == 0 else (seed, variant))

    bar_length = rng.uniform(2, 5)
    jitter_x = rng.uniform(-0.1, 0.1)

    basket_min_x = -3
    basket_max_x = 3

    # Ensure bar can rest at stable angle (sin(53°) ≈ 0.8) when tipped
    max_stable_distance = bar_length * 0.8

    left_wall_max = min(basket_max_x, -4.9 + max_stable_distance - 0.1)
    right_wall_min = max(basket_min_x, 4.9 - max_stable_distance + 0.1)

    can_target_left = left_wall_max >= basket_min_x
    can_target_right = right_wall_min <= basket_max_x

    # For short bars, relax constraint to just ensure bar can reach wall
    if not can_target_left and not can_target_right:
        left_wall_max = min(basket_max_x, -4.9 + bar_length)
        right_wall_min = max(basket_min_x, 4.9 - bar_length)
        can_target_left = left_wall_max >= basket_min_x
        can_target_right = right_wall_min <= basket_max_x

    # Choose wall and position basket within constraints
    if can_target_left and can_target_right:
        if rng.choice([True, False]):
            basket_x = rng.uniform(basket_min_x, left_wall_max)
            wall_x = -4.9
        else:
            basket_x = rng.uniform(right_wall_min, basket_max_x)
            wall_x = 4.9
    elif can_target_left:
        basket_x = rng.uniform(basket_min_x, left_wall_max)
        wall_x = -4.9
    else:
        basket_x = rng.uniform(right_wall_min, basket_max_x)
        wall_x = 4.9
    basket = Basket(
        x=basket_x,
        y=-4.9,
        scale=0.45,
        wall_thickness=0.13,
        anchor="bottom_center",
        color="gray",
        dynamic=True,
    )

    bar_bottom = -4.9 + basket.floor_thickness + 0.02
    bar_x = basket_x + jitter_x
    green_bar = Bar(
        x=bar_x,
        y=bar_bottom + bar_length / 2,
        length=bar_length,
        angle=90.0,
        thickness=0.2,
        color="green",
        dynamic=True,
    )

    purple_wall = Bar(
        top=5.0,
        bottom=-5.0,
        x=wall_x,
        thickness=0.2,
        color="purple",
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
        "green_bar": green_bar,
        "red_ball": red_ball,
        "purple_wall": purple_wall,
        "basket": basket,
    }

    return Level(
        name="tipping_point",
        objects=objects,
        action_objects=["red_ball"],
        success_condition=success_condition,
        metadata={"description": "Make the green bar tip over and hit the wall"},
    )
