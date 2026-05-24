import numpy as np
from interphyre.objects import Ball, Basket, Bar
from interphyre.level import Level
from interphyre.levels import register_level
from interphyre.config import MIN_X, MAX_X, MIN_Y, MAX_Y, WORLD_WIDTH, WORLD_HEIGHT


def success_condition(engine):
    success_time = engine.config.default_success_time
    return engine.is_in_contact_for_duration("green_ball", "purple_wall", success_time)


@register_level
def build_level(seed=None, variant=0, scene=None) -> Level:
    rng = np.random.default_rng(seed if variant == 0 else (seed, variant))

    right_bar_angle = rng.uniform(10, 50)
    bar_angle = rng.uniform(75, 105)
    left_bar_angle_raw = right_bar_angle - bar_angle
    left_bar_angle = 180 - abs(left_bar_angle_raw)

    center_x = rng.uniform(0.2, 0.6) * WORLD_WIDTH + MIN_X

    purple_wall = Bar.from_corner(
        corner_x=center_x,
        corner_y=MIN_Y,
        angle=right_bar_angle,
        length=WORLD_WIDTH,
        thickness=0.2,
        color="purple",
        dynamic=False,
    )

    black_wall = Bar.from_corner(
        corner_x=center_x,
        corner_y=MIN_Y,
        angle=left_bar_angle,
        length=WORLD_WIDTH,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    offset = 0.0
    if purple_wall.right < MAX_X:
        offset = MAX_X - purple_wall.right
    elif black_wall.left > MIN_X:
        offset = MIN_X - black_wall.left

    if offset != 0.0:
        center_x += offset
        purple_wall = Bar.from_corner(
            corner_x=center_x,
            corner_y=MIN_Y,
            angle=right_bar_angle,
            length=WORLD_WIDTH,
            thickness=0.2,
            color="purple",
            dynamic=False,
        )
        black_wall = Bar.from_corner(
            corner_x=center_x,
            corner_y=MIN_Y,
            angle=left_bar_angle,
            length=WORLD_WIDTH,
            thickness=0.2,
            color="black",
            dynamic=False,
        )

    basket_scale = 1.0
    basket_angle = left_bar_angle_raw

    basket_top_offset = rng.uniform(0.06, 0.14) * WORLD_HEIGHT
    basket_top_y = MAX_Y - basket_top_offset

    black_wall_slope = np.tan(np.radians(left_bar_angle))
    basket_x_on_wall = (
        center_x + (basket_top_y - basket_scale * 2 - MIN_Y) / black_wall_slope
    )

    min_basket_x = MIN_X + basket_scale + 0.5
    basket_x = max(basket_x_on_wall, min_basket_x)

    basket = Basket(
        x=basket_x,
        y=basket_top_y - basket_scale,
        scale=basket_scale,
        angle=basket_angle,
        anchor="bottom_center",
        color="gray",
        dynamic=True,
    )

    green_ball_radius = 0.4
    green_ball = Ball(
        x=basket_x,
        y=basket.y + green_ball_radius + 0.1,
        radius=green_ball_radius,
        color="green",
        dynamic=True,
    )

    min_separation = 2.0
    if green_ball.x + green_ball_radius + min_separation > center_x:
        green_ball.x = center_x - min_separation - green_ball_radius
        basket.x = green_ball.x

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
        "purple_wall": purple_wall,
        "black_wall": black_wall,
        "basket": basket,
    }

    return Level(
        name="off_the_rails",
        objects=objects,
        action_objects=["red_ball"],
        success_condition=success_condition,
        metadata={
            "description": "Get the green ball out of the basket and onto the purple wall"
        },
    )
