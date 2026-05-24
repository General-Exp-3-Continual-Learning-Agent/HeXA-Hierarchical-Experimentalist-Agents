import numpy as np
from interphyre.objects import Ball, Bar, Basket
from interphyre.level import Level
from interphyre.levels import register_level
from interphyre.config import MAX_X, MAX_Y


def success_condition(engine):
    success_time = engine.config.default_success_time
    return engine.is_in_contact_for_duration("green_ball", "blue_ball", success_time)


@register_level
def build_level(seed=None, variant=0, scene=None) -> Level:
    """Build pass_the_parcel level.

    The goal is to push the inverted top basket so the green ball (sitting on the
    platform beside it) rolls off and falls into the bottom basket, hitting the blue
    ball. The player places a red ball to knock the top basket.

    Layout:
    - Bottom basket on ground (left side) with blue ball inside
    - Platform bar above it (height varies)
    - Inverted top basket sitting on platform (left edge)
    - Green ball resting ON platform next to basket
    - Ramp extending from platform to upper right
    """
    rng = np.random.default_rng(seed if variant == 0 else (seed, variant))

    # Generate level parameters with PHYRE's constraint to avoid impossible configurations
    bottom_basket_scale = rng.uniform(0.7, 1.0)
    bottom_basket_x = rng.uniform(-2.5, 0.0)
    bar_offset = rng.uniform(1.0, 2.0)

    # Conditional sampling to satisfy constraint: NOT (platform_y >= 1.0 AND bottom_basket_x > -1.0)
    if bottom_basket_x > -1.0:
        platform_y = rng.uniform(-1.0, 1.0)  # Restricted range
    else:
        platform_y = rng.uniform(-1.0, 1.5)  # Full range
    bottom_basket = Basket(
        x=bottom_basket_x,
        y=-4.6,
        scale=bottom_basket_scale,
        wall_thickness=0.15,
        angle=0,
        anchor="bottom_center",
        color="gray",
        dynamic=True,
    )

    # Blue ball sized to fit inside bottom basket
    blue_ball_radius = round(0.25 + bottom_basket_scale / 8, 2)
    blue_ball_x = bottom_basket_x
    blue_ball_y = bottom_basket.y + blue_ball_radius + 0.2
    blue_ball = Ball(
        x=blue_ball_x,
        y=blue_ball_y,
        radius=blue_ball_radius,
        color="blue",
        dynamic=True,
    )

    # Platform bar extends from bottom basket + offset to right edge
    black_platform_y = platform_y
    basket_left = (
        bottom_basket_x - bottom_basket.bottom_width / 2 - bottom_basket.wall_thickness
    )
    platform_left = basket_left + bar_offset
    platform_right = MAX_X
    black_platform_length = platform_right - platform_left
    black_platform_x = (platform_left + platform_right) / 2
    black_platform = Bar.from_point_and_angle(
        x=black_platform_x,
        y=black_platform_y,
        length=black_platform_length,
        angle=0,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    # Inverted top basket sits on platform with opening facing down
    top_basket_scale = 0.6
    top_basket = Basket(
        x=0,
        y=0,
        scale=top_basket_scale,
        wall_thickness=0.15,
        angle=180,
        anchor="top_center",
        color="gray",
        dynamic=True,
    )
    top_basket_x = (
        black_platform.left + top_basket.top_width / 2 + top_basket.wall_thickness
    )
    top_basket_y = black_platform_y + black_platform.thickness
    top_basket.x = top_basket_x
    top_basket.y = top_basket_y

    # Green ball rests on platform next to basket
    green_ball_radius = 0.25
    green_ball_x = top_basket_x
    green_ball_y = black_platform_y + black_platform.thickness + green_ball_radius
    green_ball = Ball(
        x=green_ball_x,
        y=green_ball_y,
        radius=green_ball_radius,
        color="green",
        dynamic=True,
    )

    # Ensure sufficient space left of platform for action objects
    if black_platform.left < -2.0:
        platform_left = -1.5
        platform_right = MAX_X
        black_platform_length = platform_right - platform_left
        black_platform_x = (platform_left + platform_right) / 2
        black_platform.x = black_platform_x
        black_platform.length = black_platform_length
        top_basket_x = (
            black_platform.left + top_basket.top_width / 2 + top_basket.wall_thickness
        )
        top_basket.x = top_basket_x
        green_ball_x = top_basket_x
        green_ball.x = green_ball_x

    # Ramp from platform to upper right
    ramp_angle = rng.uniform(30, 70)

    ramp_start_x = black_platform_x
    ramp_start_y = black_platform_y

    distance_to_right = (MAX_X - black_platform_x) / np.cos(np.radians(ramp_angle))
    distance_to_top = (MAX_Y - black_platform_y) / np.sin(np.radians(ramp_angle))
    ramp_length = min(distance_to_right, distance_to_top)

    ramp_end_x = ramp_start_x + ramp_length * np.cos(np.radians(ramp_angle))
    ramp_end_y = ramp_start_y + ramp_length * np.sin(np.radians(ramp_angle))
    ramp = Bar.from_endpoints(
        x1=ramp_start_x,
        y1=ramp_start_y,
        x2=ramp_end_x,
        y2=ramp_end_y,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    # Randomize red ball position
    red_ball_radius = rng.uniform(0.5, 0.8)
    red_ball = Ball(
        x=rng.uniform(-4.5, 4.5),
        y=rng.uniform(2.0, 4.5),
        radius=red_ball_radius,
        color="red",
        dynamic=True,
    )

    objects = {
        "green_ball": green_ball,
        "red_ball": red_ball,
        "blue_ball": blue_ball,
        "ramp": ramp,
        "top_basket": top_basket,
        "bottom_basket": bottom_basket,
        "black_platform": black_platform,
    }

    return Level(
        name="pass_the_parcel",
        objects=objects,
        action_objects=["red_ball"],
        success_condition=success_condition,
        metadata={
            "description": "Push the basket so the green ball falls in and hits the blue ball"
        },
    )
