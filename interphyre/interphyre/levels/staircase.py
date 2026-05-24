import numpy as np
from interphyre.objects import Ball, Bar, Basket
from interphyre.level import Level
from interphyre.levels import register_level
from interphyre.config import MIN_X, MAX_X, MIN_Y, MAX_Y, WORLD_WIDTH, WORLD_HEIGHT


def success_condition(engine):
    return engine.is_in_contact_for_duration(
        "basket", "green_ball", engine.config.default_success_time
    )


@register_level
def build_level(seed=None, variant=0, scene=None) -> Level:
    rng = np.random.default_rng(seed if variant == 0 else (seed, variant))

    num_bars = 5
    bar_scale = 0.7 / num_bars
    bar_length = bar_scale * WORLD_WIDTH
    y_span = rng.uniform(0.4, 0.6)

    stairs = []
    for i in range(num_bars):
        bar_top = MIN_Y + 0.8 * WORLD_HEIGHT - i / num_bars * y_span * WORLD_HEIGHT
        bar_left = MIN_X + i / num_bars * WORLD_WIDTH
        stairs.append(
            Bar.from_corner(
                corner_x=bar_left,
                corner_y=bar_top,
                angle=-5.0,
                length=bar_length,
                thickness=0.2,
                color="black",
                dynamic=False,
            )
        )

    lowest_bar_bottom = stairs[-1].bottom
    max_basket_height = lowest_bar_bottom - MIN_Y - 0.1
    max_basket_scale = min(2.0, max_basket_height / 2.0)
    basket_scale = rng.uniform(1.5, max(1.5, max_basket_scale))

    target_index = rng.integers(1, num_bars)

    basket = Basket(
        x=0,
        y=MIN_Y,
        scale=basket_scale,
        angle=0.0,
        anchor="bottom_center",
        color="purple",
        dynamic=False,
    )

    guard_offset = basket.top_width / 2 + 0.305 + 0.1
    min_basket_x = MIN_X + guard_offset
    max_basket_x = MAX_X - guard_offset

    target_bar = stairs[target_index]
    left_align_x = target_bar.left + basket.top_width / 2
    right_align_x = target_bar.right - basket.top_width / 2

    valid_aligns = []
    if min_basket_x <= left_align_x <= max_basket_x:
        valid_aligns.append("left")
    if min_basket_x <= right_align_x <= max_basket_x:
        valid_aligns.append("right")

    if not valid_aligns:
        basket.x = np.clip(left_align_x, min_basket_x, max_basket_x)
    else:
        align = rng.choice(valid_aligns)
        if align == "right":
            basket.x = right_align_x
        else:
            basket.x = left_align_x

    guard_length = basket.height
    left_guard = Bar(
        x=basket.x - basket.top_width / 2 - 0.305,
        top=MIN_Y + guard_length + 0.2,
        bottom=MIN_Y,
        thickness=0.2,
        color="black",
        dynamic=False,
    )
    right_guard = Bar(
        x=basket.x + basket.top_width / 2 + 0.305,
        top=MIN_Y + guard_length + 0.2,
        bottom=MIN_Y,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    green_ball_radius = 0.3
    max_ball_x = basket.x + basket.top_width / 2 + green_ball_radius
    green_ball = Ball(
        x=rng.uniform(
            MIN_X + green_ball_radius, min(max_ball_x, MAX_X - green_ball_radius)
        ),
        y=MAX_Y - green_ball_radius,
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
        "basket": basket,
        "left_guard": left_guard,
        "right_guard": right_guard,
    }
    for i, stair in enumerate(stairs):
        objects[f"stair_{i + 1}"] = stair

    return Level(
        name="staircase",
        objects=objects,
        action_objects=["red_ball"],
        success_condition=success_condition,
        metadata={"description": "Make the green ball fall into the purple basket"},
    )
