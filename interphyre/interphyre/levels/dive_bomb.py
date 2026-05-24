from __future__ import annotations

import numpy as np
from interphyre.objects import Ball, Bar
from interphyre.level import Level
from interphyre.levels import register_level
from interphyre.config import MAX_X, MAX_Y, MIN_X, MIN_Y


def success_condition(engine):
    success_time = engine.config.default_success_time
    return engine.is_in_contact_for_duration("green_ball", "purple_pad", success_time)


@register_level
def build_level(seed=None, variant=0, scene=None) -> Level:
    rng = np.random.default_rng(seed if variant == 0 else (seed, variant))

    # Level parameters
    green_ball_radius = rng.uniform(0.2, 0.47)
    bar_thickness = 0.2
    purple_pad_length = 1
    target_position = rng.uniform(0.35, 0.65)

    purple_pad_left = MIN_X + target_position * (MAX_X - MIN_X)
    purple_pad_right = purple_pad_left + purple_pad_length
    purple_pad = Bar(
        left=purple_pad_left,
        right=purple_pad_right,
        y=MIN_Y + bar_thickness / 2,
        thickness=bar_thickness,
        color="purple",
        dynamic=False,
    )

    left_floor = Bar(
        left=MIN_X,
        right=purple_pad.left,
        y=MIN_Y + bar_thickness / 2,
        thickness=bar_thickness,
        color="black",
        dynamic=False,
    )

    right_floor = Bar(
        left=purple_pad.right,
        right=MAX_X,
        y=MIN_Y + bar_thickness / 2,
        thickness=bar_thickness,
        color="black",
        dynamic=False,
    )

    # Short barrier at left edge of pad
    short_barrier_length = 1
    short_barrier_x = purple_pad.left - bar_thickness / 2
    short_barrier_y = MIN_Y + bar_thickness / 2 + short_barrier_length / 2
    short_barrier = Bar.from_point_and_angle(
        x=short_barrier_x,
        y=short_barrier_y,
        angle=90.0,
        length=short_barrier_length,
        thickness=bar_thickness,
        color="black",
        dynamic=False,
    )

    # Position cannon bottom at fixed height
    cannon_bottom_target = MIN_Y + 3.0

    # Calculate ramp properties
    ramp_length = 1.0
    ramp_angle = 10

    # Position cannon relative to barriers
    blocker_left_x = short_barrier_x - bar_thickness / 2
    cannon_right_x = blocker_left_x - 1.0

    # Sample cannon angle and length with conditional sampling to keep ball in bounds
    # At ball_position=0.0: ball_x = cannon_left_x = cannon_right_x - cannon_length * cos(angle)
    # Need: cannon_left_x - green_ball_radius >= MIN_X
    # Therefore: cannon_length <= (cannon_right_x - MIN_X - green_ball_radius) / cos(angle)

    # First attempt to sample angle
    cannon_angle = rng.uniform(-50, -30)
    max_cannon_length = (cannon_right_x - MIN_X - green_ball_radius) / np.cos(
        np.radians(cannon_angle)
    )

    # If max_cannon_length < 3, we need a steeper angle
    if max_cannon_length < 3:
        # Solve: (cannon_right_x - MIN_X - green_ball_radius) / cos(angle) >= 3
        # cos(angle) <= (cannon_right_x - MIN_X - green_ball_radius) / 3
        required_cos = (cannon_right_x - MIN_X - green_ball_radius) / 3
        required_angle_rad = np.arccos(np.clip(required_cos, -1, 1))
        required_angle = np.degrees(required_angle_rad)
        # Sample steeper angle (more negative). If the required angle exceeds
        # the current sampling bound, fall back to the minimum steepness that
        # satisfies the constraint.
        if required_angle <= 50:
            cannon_angle = rng.uniform(-50, -required_angle)
        else:
            cannon_angle = -required_angle
        max_cannon_length = (cannon_right_x - MIN_X - green_ball_radius) / np.cos(
            np.radians(cannon_angle)
        )

    min_cannon_length = 3.0
    max_cannon_length = min(6, max_cannon_length)
    if max_cannon_length < min_cannon_length:
        cannon_length = max_cannon_length
    else:
        cannon_length = rng.uniform(min_cannon_length, max_cannon_length)

    cannon_left_x = cannon_right_x - cannon_length * np.cos(np.radians(cannon_angle))

    angle_rad = np.radians(cannon_angle)
    cannon_left_y = (
        cannon_bottom_target
        - cannon_length * np.sin(angle_rad)
        + (bar_thickness / 2) * np.cos(angle_rad)
    )
    cannon_right_y = cannon_left_y + cannon_length * np.sin(angle_rad)

    cannon_bottom = Bar.from_endpoints(
        x1=cannon_left_x,
        y1=cannon_left_y,
        x2=cannon_right_x,
        y2=cannon_right_y,
        thickness=bar_thickness,
        color="black",
        dynamic=False,
    )

    # Exit ramp
    ramp_left_x = cannon_right_x - 0.2
    ramp_left_y = cannon_right_y

    ramp_right_x = ramp_left_x + ramp_length * np.cos(np.radians(ramp_angle))
    ramp_right_y = ramp_left_y + ramp_length * np.sin(np.radians(ramp_angle))

    ramp = Bar.from_endpoints(
        x1=ramp_left_x,
        y1=ramp_left_y,
        x2=ramp_right_x,
        y2=ramp_right_y,
        thickness=bar_thickness,
        color="black",
        dynamic=False,
    )

    # Build barrier stack on right side
    barrier_stack = []
    flat_barrier_width = 1.0
    base_top = right_floor.y + bar_thickness / 2

    for i in range(5):
        # Each plank's bottom rests on previous bar's top
        stack_bar_bottom = base_top + i * bar_thickness
        stack_bar_y = stack_bar_bottom + bar_thickness / 2
        stack_bar_left = right_floor.left
        stack_bar_right = stack_bar_left + flat_barrier_width

        stack_bar = Bar(
            left=stack_bar_left,
            right=stack_bar_right,
            y=stack_bar_y,
            thickness=bar_thickness,
            color="black",
            dynamic=False,
        )
        barrier_stack.append(stack_bar)

    # Tall vertical barrier on top of stack
    top_plank = barrier_stack[-1]
    tall_barrier_length = short_barrier_length
    tall_barrier_bottom = top_plank.y + bar_thickness / 2
    tall_barrier_y = tall_barrier_bottom + tall_barrier_length / 2
    tall_barrier_x = top_plank.left + bar_thickness / 2

    tall_barrier = Bar.from_point_and_angle(
        x=tall_barrier_x,
        y=tall_barrier_y,
        angle=90.0,
        length=tall_barrier_length,
        thickness=bar_thickness,
        color="black",
        dynamic=False,
    )

    # Green ball on cannon surface
    ball_position_along_cannon = rng.uniform(0.0, 0.8)

    green_ball_x = cannon_left_x + ball_position_along_cannon * (
        cannon_right_x - cannon_left_x
    )

    cannon_surface_y_at_ball = (
        cannon_left_y
        + ball_position_along_cannon * (cannon_right_y - cannon_left_y)
        + bar_thickness / 2
    )
    green_ball_y = cannon_surface_y_at_ball + green_ball_radius * 1.7

    green_ball = Ball(
        x=green_ball_x,
        y=green_ball_y,
        radius=green_ball_radius,
        color="green",
        dynamic=True,
    )

    gray_ball_radius = green_ball_radius * 0.5

    # Place gray ball ahead of green ball, constrained to not overlap ramp
    # Need: gray_ball_x + gray_ball_radius < ramp.x1
    # So: gray_ball_position < (ramp.x1 - cannon_left_x - gray_ball_radius) / (cannon_right_x - cannon_left_x)
    max_gray_position = (ramp.x1 - cannon_left_x - gray_ball_radius) / (
        cannon_right_x - cannon_left_x
    )

    # Determine sampling range, relaxing spacing if needed to respect ramp constraint
    min_gray_position = ball_position_along_cannon + 0.15
    upper_gray_position = min(0.95, max_gray_position)

    if upper_gray_position < min_gray_position:
        # Relax spacing requirement to fit within ramp constraint
        min_gray_position = ball_position_along_cannon
        upper_gray_position = max_gray_position

    if upper_gray_position <= min_gray_position:
        # Ramp starts at or before the green ball: no valid gap for gray ball.
        # Clamp to min_gray_position — oracle will reject this seed as impossible.
        gray_ball_position = min_gray_position
    else:
        gray_ball_position = rng.uniform(min_gray_position, upper_gray_position)
    gray_ball_x = cannon_left_x + gray_ball_position * (cannon_right_x - cannon_left_x)

    cannon_surface_y_at_gray = (
        cannon_left_y
        + gray_ball_position * (cannon_right_y - cannon_left_y)
        + bar_thickness / 2
    )
    gray_ball_y = cannon_surface_y_at_gray + gray_ball_radius * 2.6

    gray_ball = Ball(
        x=gray_ball_x,
        y=gray_ball_y,
        radius=gray_ball_radius,
        color="gray",
        dynamic=True,
    )

    # Cannon middle and top bars
    cannon_middle_spacing = green_ball_radius * 6

    cannon_middle = Bar.offset_along_angle(
        base_x=cannon_bottom.x,
        base_y=cannon_bottom.y,
        angle=90,
        distance=cannon_middle_spacing,
        thickness=bar_thickness,
        color="black",
        dynamic=False,
    )
    cannon_middle.angle = cannon_angle
    cannon_middle.length = cannon_length

    cannon_top_spacing = green_ball_radius * 10

    cannon_top = Bar.offset_along_angle(
        base_x=cannon_bottom.x,
        base_y=cannon_bottom.y,
        angle=90,
        distance=cannon_top_spacing,
        thickness=bar_thickness,
        color="black",
        dynamic=False,
    )
    cannon_top.angle = cannon_angle
    cannon_top.length = cannon_length

    # Diagonal deflector bar to prevent overshooting
    deflector_angle = -30.0
    deflector_length = 5.0

    # Position deflector at ramp exit, aligned with cannon top
    ramp_top_surface_y = ramp_right_y + bar_thickness / 2
    vertical_offset = cannon_top.y - cannon_bottom.y
    deflector_start_y = ramp_top_surface_y + vertical_offset
    deflector_start_x = ramp_right_x

    cannon_top_extension = Bar.from_corner(
        corner_x=deflector_start_x,
        corner_y=deflector_start_y,
        angle=deflector_angle,
        length=deflector_length,
        thickness=bar_thickness,
        color="black",
        dynamic=False,
    )

    # Randomly place a red ball in the scene
    red_ball = Ball(
        x=rng.uniform(MIN_X + 1, MAX_X - 1),
        y=rng.uniform(MIN_Y + 2, MAX_Y - 2),
        radius=0.4,
        color="red",
        dynamic=True,
    )

    objects = {
        "cannon_bottom": cannon_bottom,
        "cannon_middle": cannon_middle,
        "cannon_top": cannon_top,
        "cannon_top_extension": cannon_top_extension,
        "ramp": ramp,
        "short_barrier": short_barrier,
        "tall_barrier": tall_barrier,
        "purple_pad": purple_pad,
        "left_floor": left_floor,
        "right_floor": right_floor,
        "green_ball": green_ball,
        "gray_ball": gray_ball,
        "red_ball": red_ball,
    }

    # Add stack bars to objects
    for i, stack_bar in enumerate(barrier_stack):
        objects[f"stack_bar_{i}"] = stack_bar

    return Level(
        name="dive_bomb",
        objects=objects,
        action_objects=["red_ball"],
        success_condition=success_condition,
        metadata={
            "description": "Get the green ball to touch the purple pad by navigating through the cannon obstacles."
        },
    )
