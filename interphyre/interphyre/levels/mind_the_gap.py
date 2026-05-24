import numpy as np
from interphyre.objects import Ball, Bar
from interphyre.level import Level
from interphyre.levels import register_level
from interphyre.config import MIN_X, MAX_X


def success_condition(engine):
    success_time = engine.config.default_success_time
    return engine.is_in_contact_for_duration(
        "green_ball", "purple_ground", success_time
    )


@register_level
def build_level(seed=None, variant=0, scene=None) -> Level:
    rng = np.random.default_rng(seed if variant == 0 else (seed, variant))

    # Level parameters
    hole_left_x = rng.uniform(-2.1, 1.1)
    hole_width = 1.15  # was 1.05: tripling the post-displacement slack (0.05→0.15) resolves seeds
    # where hole_width≈green_ball_diameter left insufficient geometric tolerance for most variants.
    # Seed 6719 had only 2/30 variants solvable at hole_width=1.05 —
    # RNG unchanged — hole_width is a constant, not a sampled value.
    hole_right_x = hole_left_x + hole_width

    platform_y = rng.uniform(-3.5, 1.0)
    block_left_side = rng.choice([True, False])

    # Ball sizes
    green_ball_radius = 0.5
    blocking_ball_radius = 0.55
    green_ball_y = 3.5

    # Ensure blocker stays in reasonable range relative to green ball
    # Green ball is at y=3.5, blocker should be between y=0.5 and y=3.0
    min_blocker_y = 0.5
    max_blocker_y = 3.0

    left_platform = Bar(
        left=MIN_X,
        right=hole_left_x,
        y=platform_y,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    right_platform = Bar(
        left=hole_right_x,
        right=MAX_X,
        y=platform_y,
        thickness=0.2,
        color="black",
        dynamic=False,
    )

    # Green ball starts at center-top
    green_ball = Ball(
        x=0.0,
        y=green_ball_y,
        radius=green_ball_radius,
        color="green",
        dynamic=True,
    )

    # Calculate blocker distance to keep it in reasonable range
    # blocker_y = platform_y + distance + radius, so:
    # distance = blocker_y - platform_y - radius
    min_distance = max(1.0, min_blocker_y - platform_y - blocking_ball_radius)
    max_distance = min(3.0, max_blocker_y - platform_y - blocking_ball_radius)

    # Ensure valid range exists
    if min_distance > max_distance:
        blocker_distance_from_platform = min_distance
    else:
        blocker_distance_from_platform = rng.uniform(min_distance, max_distance)

    blocker_bottom = platform_y + blocker_distance_from_platform
    blocker_y = blocker_bottom + blocking_ball_radius

    # Position blocker at hole edge with 0.25 offset, accounting for radius
    if block_left_side:
        blocker_x = hole_left_x + 0.25 + blocking_ball_radius
    else:
        blocker_x = hole_right_x - 0.25 - blocking_ball_radius

    blocking_ball = Ball(
        x=blocker_x,
        y=blocker_y,
        radius=blocking_ball_radius,
        color="gray",
        dynamic=True,
    )

    # Ensure green ball isn't directly over the hole
    if (
        green_ball.x - green_ball_radius >= hole_left_x
        and green_ball.x + green_ball_radius <= hole_right_x
    ):
        green_ball.x = hole_left_x - green_ball_radius - 0.1

    # Ground
    purple_ground = Bar(
        left=MIN_X,
        right=MAX_X,
        y=-4.9,
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
        "red_ball": red_ball,
        "blocking_ball": blocking_ball,
        "purple_ground": purple_ground,
        "left_platform": left_platform,
        "right_platform": right_platform,
    }

    return Level(
        name="mind_the_gap",
        objects=objects,
        action_objects=["red_ball"],
        success_condition=success_condition,
        metadata={
            "description": "Push the green ball through the gap to reach the ground."
        },
    )
