import numpy as np
from interphyre.objects import Ball, Basket, Bar
from interphyre.level import Level
from interphyre.levels import register_level


def success_condition(engine):
    success_time = engine.config.default_success_time
    return engine.is_in_contact_for_duration(
        "green_ball", "purple_ground", success_time
    )


@register_level
def build_level(seed=None, variant=0, scene=None) -> Level:
    rng = np.random.default_rng(seed if variant == 0 else (seed, variant))

    purple_ground = Bar(
        left=-5,
        right=5,
        y=-4.9,
        thickness=0.2,
        color="purple",
        dynamic=False,
    )

    basket_scale = rng.uniform(0.75, 2.0)
    basket_x = rng.uniform(-3.5, 3.5)
    basket_y = -4.8 + rng.uniform(0, 2.0)

    ball_radius = 0.5
    green_ball_x = basket_x
    green_ball_y = rng.uniform(2.0, 4.0)

    red_ball_radius = rng.uniform(0.3, 0.6)

    green_ball = Ball(
        x=green_ball_x,
        y=green_ball_y,
        radius=ball_radius,
        color="green",
        dynamic=True,
    )
    red_ball = Ball(
        x=0,
        y=0,
        radius=red_ball_radius,
        color="red",
        dynamic=True,
    )

    basket = Basket(
        x=basket_x,
        y=basket_y,
        scale=basket_scale,
        anchor="bottom_center",
        color="gray",
        dynamic=True,
    )

    objects = {
        "green_ball": green_ball,
        "red_ball": red_ball,
        "purple_ground": purple_ground,
        "basket": basket,
    }

    return Level(
        name="basket_case",
        objects=objects,
        action_objects=["red_ball"],
        success_condition=success_condition,
        metadata={
            "description": "Make sure the green ball hits the purple ground and is not trapped in the basket."
        },
    )
