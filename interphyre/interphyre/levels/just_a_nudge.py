import numpy as np
from interphyre.objects import Ball, Bar, Basket
from interphyre.level import Level
from interphyre.levels import register_level
from interphyre.config import MIN_X, MAX_X, MIN_Y


def success_condition(engine):
    success_time = engine.config.default_success_time
    return engine.is_in_contact_for_duration("green_ball", "blue_ball", success_time)


@register_level
def build_level(seed=None, variant=0, scene=None) -> Level:
    rng = np.random.default_rng(seed if variant == 0 else (seed, variant))

    # Level parameters
    green_ball_radius = rng.uniform(0.2, 0.35)
    basket_scale = rng.uniform(0.8, 1.2)
    ramp_angle = rng.uniform(45, 60)
    ball_offset = rng.uniform(0.2, 0.5)
    platform_x = rng.uniform(0.6, 1.0)
    platform_angle = rng.uniform(1, 9)

    # Ramps form right triangles with walls and floor
    ramp_height = 2.0
    floor_distance = ramp_height / np.tan(np.radians(ramp_angle))

    # Left ramp
    left_ramp = Bar.from_endpoints(
        x1=MIN_X,
        y1=MIN_Y + ramp_height,
        x2=MIN_X + floor_distance,
        y2=MIN_Y,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    # Right ramp
    right_ramp = Bar.from_endpoints(
        x1=MAX_X,
        y1=MIN_Y + ramp_height,
        x2=MAX_X - floor_distance,
        y2=MIN_Y,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    # Calculate basket dimensions for platform positioning
    basket_dims = Basket.calculate_dimensions(basket_scale)
    basket_top = (MIN_Y + 0.1) + basket_dims["total_height"]

    # platform bar positioned above basket
    min_platform_bottom = max(-2.0, basket_top + green_ball_radius * 4)
    platform_length = 3.5
    platform_center_y = min_platform_bottom + platform_length / 2

    platform = Bar.from_point_and_angle(
        x=platform_x,
        y=platform_center_y,
        angle=platform_angle,
        length=platform_length,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    # Green ball x is determined by platform geometry; must be computed before
    # basket_x so the basket can track the ball's initial position.
    green_ball_x = platform.left + ball_offset + green_ball_radius

    # Basket tracks green_ball: placed 0.3-0.8 units left of green ball x.
    # 99.2% of valid seeds require basket_x < green_ball_x (ball falls leftward
    # off the platform when the positive-angle tilt is engaged by the red ball).
    # Prior logic (basket_x = rng.uniform(-1.0, max_basket_x)) produced dx > 0
    # for 90.7% of seeds — the primary root cause of geometric impossibility.
    basket_offset = rng.uniform(0.3, 0.8)
    basket_x = float(np.clip(green_ball_x - basket_offset, -1.0, 1.0))

    # Basket at bottom center
    basket = Basket(
        x=basket_x,
        y=MIN_Y + 0.1,
        scale=basket_scale,
        angle=0,
        anchor="bottom_center",
        color="gray",
        dynamic=True,
    )

    # Blue ball in basket
    blue_ball_radius = 0.4 + basket_scale * 0.08
    blue_ball = Ball(
        x=basket_x,
        y=MIN_Y + 0.6 + blue_ball_radius,
        radius=blue_ball_radius,
        color="blue",
        dynamic=True,
    )

    # Green ball on platform
    platform_top_surface = platform_center_y + platform_length / 2
    green_ball_y = platform_top_surface + green_ball_radius

    green_ball = Ball(
        x=green_ball_x,
        y=green_ball_y,
        radius=green_ball_radius,
        color="green",
        dynamic=True,
    )

    # User-placeable red ball
    red_ball = Ball(
        x=0.0,
        y=0.0,
        radius=0.5,
        color="red",
        dynamic=True,
    )

    objects = {
        "green_ball": green_ball,
        "red_ball": red_ball,
        "left_ramp": left_ramp,
        "right_ramp": right_ramp,
        "blue_ball": blue_ball,
        "platform": platform,
        "basket": basket,
    }

    return Level(
        name="just_a_nudge",
        objects=objects,
        action_objects=["red_ball"],
        success_condition=success_condition,
        metadata={
            "description": "Nudge the basket so the green ball falls and touches the blue ball.",
        },
    )
