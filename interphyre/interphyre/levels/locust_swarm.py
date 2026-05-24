import numpy as np
from interphyre.objects import Ball, Bar
from interphyre.level import Level
from interphyre.levels import register_level
from interphyre.config import MAX_X, MAX_Y, MIN_X, MIN_Y, WORLD_WIDTH, WORLD_HEIGHT


def success_condition(engine):
    success_time = engine.config.default_success_time
    return engine.is_in_contact_for_duration("green_ball", "purple_floor", success_time)


@register_level
def build_level(seed=None, variant=0, scene=None) -> Level:
    """Build locust swarm level.

    NOTE: This level has inherent difficulty variability across seeds due to random
    chain-based obstacle generation. Some seeds may be impossible while others may
    be trivial. Seed filtering during data collection is recommended.
    """
    rng = np.random.default_rng(seed if variant == 0 else (seed, variant))

    ball_x = rng.uniform(MIN_X + 1, MAX_X - 1)
    ball_y = MAX_Y - 0.1 * WORLD_HEIGHT
    ball_radius = 0.5
    star_radius = 0.25

    green_ball = Ball(
        x=ball_x,
        y=ball_y,
        radius=ball_radius,
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

    top = ball_y - 5 * ball_radius

    if scene is not None and any(k.startswith("star_") for k in scene):
        # Scene-driven reconstruction: create placeholder stars with names matching
        # the stored scene. _apply_scene_overrides sets geometry after construction.
        # This allows build_level_from_scene to work correctly for variable-count
        # levels where the star count is seed-dependent.
        star_objects = {
            name: Ball(x=0.0, y=0.0, radius=star_radius, color="black", dynamic=False)
            for name in scene
            if name.startswith("star_")
        }
    else:

        def gen_chain(start_x, start_y):
            """
            Generate a chain of stars with normal random steps.
            """
            angle = rng.uniform() * 2 * np.pi
            angle_diff = rng.uniform() * 2 * np.pi / 10
            chain_stars = [(start_x, start_y)]
            line_length = 1
            n_valid = 0
            max_points = rng.integers(
                10, 20
            )  # was (15,30): reduced to lower barrier-forming density.
            # Stars are static (dynamic=False) — red ball deflects green_ball through EXISTING GAPS.
            # Grid search confirmed remaining 85/10001 impossible seeds all have
            # genuine geometric barriers — step size [0.5,1.55] means ~93% of adjacent star pairs
            # have gaps < ball diameter (1.0). v5 fix: min_step raised to 1.5 (passability threshold)
            # ensuring each consecutive star pair in the chain has a navigable gap ≥ 1.0 units.
            # max_step raised to 3.0 for natural-looking sparse-to-dense variation.

            while n_valid < max_points:
                if line_length >= 3 and rng.uniform() < 0.2:
                    # Branch to random existing point
                    x, y = chain_stars[rng.integers(len(chain_stars))]
                    line_length = 1
                    angle = rng.uniform() * 2 * np.pi
                    angle_diff = rng.uniform() * 2 * np.pi / 10
                else:
                    line_length += 1
                    # Normal random step — min 1.5 ensures gap ≥ ball diameter (passability threshold)
                    step = rng.uniform(1.5, 3.0)

                    angle += angle_diff
                    dx, dy = step * np.cos(angle), step * np.sin(angle)
                    x, y = chain_stars[-1]
                    x += dx
                    y += dy

                if y >= top:
                    continue

                chain_stars.append((x, y))

                # Convert to normalized coordinates for bounds check
                norm_x = (x - MIN_X) / WORLD_WIDTH
                norm_y = (y - MIN_Y) / WORLD_HEIGHT
                if 0.0 < norm_x < 1 and 0.0 < norm_y < 1:
                    n_valid += 1

            return chain_stars

        stars = []
        # Chain 1 is anchored to green_ball.x so it covers the ball's direct fall
        # corridor. Prior design used a fixed start at MIN_X + 0.2*W = -3.0, which
        # left ~48% of variants trivially solvable (no stars blocking green_ball's
        # natural fall path). Anchoring to green_ball.x reduces the trivial rate.
        # Chain 2 is anchored at the right-side fixed offset for visual variety.
        chain_starts = [green_ball.x, MIN_X + 0.7 * WORLD_WIDTH]
        for start_x in chain_starts:
            start_y = MIN_Y + 0.5 * WORLD_HEIGHT
            stars.extend(gen_chain(start_x, start_y))

        # Create star objects
        star_objects = {}
        for i, (x, y) in enumerate(stars):
            # Convert to normalized coordinates for bounds check
            norm_x = (x - MIN_X) / WORLD_WIDTH
            norm_y = (y - MIN_Y) / WORLD_HEIGHT
            if 0 <= norm_x <= 1 and 0 <= norm_y <= 1:
                star_ball = Ball(
                    x=x,
                    y=y,
                    radius=star_radius,
                    color="black",
                    dynamic=False,
                )
                star_objects[f"star_{i}"] = star_ball

    purple_floor = Bar.from_point_and_angle(
        x=0.0,
        y=-4.9,
        length=WORLD_WIDTH,
        thickness=0.2,
        angle=0.0,
        color="purple",
        dynamic=False,
    )

    objects = {
        "green_ball": green_ball,
        "red_ball": red_ball,
        "purple_floor": purple_floor,
        **star_objects,
    }

    return Level(
        name="locust_swarm",
        objects=objects,
        action_objects=["red_ball"],
        success_condition=success_condition,
        metadata={
            "description": "Get the green ball to the purple floor by using the red ball to navigate through the two separate clouds of star obstacles."
        },
    )
