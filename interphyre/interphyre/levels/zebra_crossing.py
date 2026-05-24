import numpy as np
from interphyre.objects import Ball, Bar
from interphyre.level import Level
from interphyre.levels import register_level
from interphyre.config import MAX_X, MAX_Y, MIN_Y, WORLD_HEIGHT


def success_condition(engine):
    success_time = engine.config.default_success_time
    return engine.is_in_contact_for_duration(
        "green_ball", "purple_ground", success_time
    )


@register_level
def build_level(seed=None, variant=0, scene=None) -> Level:
    rng = np.random.default_rng(seed if variant == 0 else (seed, variant))

    # Level parameters
    bar_y = rng.integers(0, 10)
    angle = rng.uniform(10, 45)
    spacing = rng.uniform(0.1, 0.5)

    # Vertical separator x position (divides left and right chambers)
    separator_x = rng.uniform(0.0, 1.0)

    # Green ball on the left side
    green_ball_radius = 0.4
    ball_diameter = 0.8
    ball_y_start = 4.4

    # Diagonal bars positioned left of the ball
    diagonal_bar_length = 3.0
    bar_vertical_spacing = 1.5
    bars = []
    bar_positions = []

    # Calculate diagonal bar x position
    bar_x_offset = separator_x - ball_diameter - spacing - 0.12

    for i in range(10):
        bar_bottom_y = MIN_Y + i * bar_vertical_spacing
        bar_center_y = bar_bottom_y + diagonal_bar_length / 2

        if bar_center_y - diagonal_bar_length / 2 < MAX_Y:
            bars.append(i)
            bar_positions.append(bar_center_y)

    if not bars or bar_y >= len(bars):
        bar_y = min(bar_y, len(bars) - 1) if bars else 0

    # Calculate diagonal bar center x accounting for angle
    bar_center_x = bar_x_offset - (diagonal_bar_length / 2) * np.cos(np.radians(-angle))

    # Position green ball relative to first bar
    first_bar_right_edge = bar_center_x + (diagonal_bar_length / 2) * np.cos(
        np.radians(-angle)
    )
    green_ball_x = first_bar_right_edge + 0.04 + green_ball_radius

    green_ball = Ball(
        x=green_ball_x,
        y=ball_y_start,
        radius=green_ball_radius,
        color="green",
        dynamic=True,
    )

    # Create diagonal bars
    slats = {}
    for idx, (i, bar_y_pos) in enumerate(zip(bars, bar_positions), start=1):
        slat = Bar.from_point_and_angle(
            x=bar_center_x,
            y=bar_y_pos,
            length=diagonal_bar_length,
            angle=-angle,
            thickness=0.2,
            color="black",
            dynamic=False,
        )
        slats[f"diagonal_bar_{idx}"] = slat

    # Vertical separator with hole (gate mechanism)
    selected_bar_index = bars[bar_y] if bars and bar_y < len(bars) else 3
    hole_top_ratio = selected_bar_index * 0.15 - spacing / 10
    hole_top_ratio = np.clip(hole_top_ratio, 0.2, 0.9)

    hole_size_ratio = ball_diameter * 0.2

    # Top separator
    top_sep_height_ratio = 1.0 - hole_top_ratio
    top_sep_length = top_sep_height_ratio * WORLD_HEIGHT
    top_sep_y = MAX_Y - top_sep_length / 2

    top_separator = Bar.from_point_and_angle(
        x=separator_x,
        y=top_sep_y,
        length=top_sep_length,
        angle=90,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    # Bottom separator
    bottom_sep_height_ratio = hole_top_ratio - hole_size_ratio
    bottom_sep_length = bottom_sep_height_ratio * WORLD_HEIGHT
    bottom_sep_y = MIN_Y + bottom_sep_length / 2

    bottom_separator = Bar.from_point_and_angle(
        x=separator_x,
        y=bottom_sep_y,
        length=bottom_sep_length,
        angle=90,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    # Gray ball on right side
    gray_ball_radius = 0.5
    gray_ball_y = (4.0 if hole_top_ratio < 0.7 else -2.0) + gray_ball_radius
    gray_ball_x = separator_x + 0.3 + gray_ball_radius

    gray_ball = Ball(
        x=gray_ball_x,
        y=gray_ball_y,
        radius=gray_ball_radius,
        color="gray",
        dynamic=True,
    )

    # Purple ground (target) - extends from separator to right edge
    purple_ground_length = MAX_X - separator_x
    purple_ground_x = separator_x + purple_ground_length / 2
    purple_ground_y = MIN_Y + 0.1

    purple_ground = Bar.from_point_and_angle(
        x=purple_ground_x,
        y=purple_ground_y,
        length=purple_ground_length,
        angle=0,
        thickness=0.2,
        color="purple",
        dynamic=False,
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
        "gray_ball": gray_ball,
        "red_ball": red_ball,
        "top_separator": top_separator,
        "bottom_separator": bottom_separator,
        "purple_ground": purple_ground,
    }
    objects.update(slats)

    return Level(
        name="zebra_crossing",
        objects=objects,
        action_objects=["red_ball"],
        success_condition=success_condition,
        metadata={
            "description": "Navigate the green ball through diagonal bars and hole to reach the purple ground.",
        },
    )
