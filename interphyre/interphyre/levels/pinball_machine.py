import numpy as np
from interphyre.objects import Ball, Bar
from interphyre.level import Level
from interphyre.levels import register_level
from interphyre.config import MIN_X, MIN_Y, WORLD_WIDTH, WORLD_HEIGHT


def success_condition(engine):
    success_time = engine.config.default_success_time
    return engine.is_in_contact_for_duration("green_ball", "purple_floor", success_time)


@register_level
def build_level(seed=None, variant=0, scene=None) -> Level:
    """Build pinball machine level.

    NOTE: This level has inherent difficulty variability across seeds due to random
    obstacle placement. Some seeds may be trivial (>50% random success) while others
    may be very difficult (<5% random success). Seed filtering during data collection
    is recommended.
    """
    rng = np.random.default_rng(seed if variant == 0 else (seed, variant))

    ball_radius = 0.5
    bar_thickness = 0.2
    ball_center_norm = rng.uniform(0.2, 0.5)
    ball_x = MIN_X + ball_center_norm * WORLD_WIDTH
    ball_y = MIN_Y + 0.9 * WORLD_HEIGHT

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

    star_radius = 0.2

    if scene is not None and any(k.startswith("star_") for k in scene):
        # Scene-driven reconstruction: create placeholder stars with names matching
        # the stored scene. _apply_scene_overrides sets geometry after construction.
        star_objects = {
            name: Ball(x=0.0, y=0.0, radius=star_radius, color="black", dynamic=False)
            for name in scene
            if name.startswith("star_")
        }
    else:
        stars = []

        def _generate_line(start_x, start_y, base_angle, num_nodes, max_x):
            """Generate a random walk of star positions."""
            line_stars = [(start_x, start_y)]
            x, y = start_x, start_y
            for _ in range(num_nodes):
                step = rng.uniform(0.05, 0.15)
                angle = base_angle + rng.normal() * 2 * np.pi / 40
                x += step * np.cos(angle)
                y += step * np.sin(angle)
                if x >= max_x:
                    break
                line_stars.append((x, y))
            return line_stars

        # Generate zigzag lines of obstacle stars
        ball_radius_norm = ball_radius / WORLD_HEIGHT
        ball_bottom_norm = 0.9 - ball_radius_norm
        ball_height_norm = 2 * ball_radius_norm
        top_norm = ball_bottom_norm - 2 * ball_height_norm

        for i, y in enumerate(reversed(np.linspace(0.1, top_norm, 4))):
            num_stars = rng.integers(
                3, 6
            )  # was (3, 8); cap reduces dense-obstacle seeds
            base_angle = np.deg2rad(5)
            new_stars = _generate_line(0, y, base_angle, num_stars, 0.8)
            # Alternate direction for zigzag pattern
            if i % 2:
                new_stars = [(1 - x, y) for x, y in new_stars]
            stars.extend(new_stars)

        # Create star objects
        star_objects = {}
        for i, (x, y) in enumerate(stars):
            x_world = MIN_X + x * WORLD_WIDTH
            y_world = MIN_Y + y * WORLD_HEIGHT
            if (
                MIN_X <= x_world <= MIN_X + WORLD_WIDTH
                and MIN_Y <= y_world <= MIN_Y + WORLD_HEIGHT
            ):
                star_ball = Ball(
                    x=x_world,
                    y=y_world,
                    radius=star_radius,
                    color="black",
                    dynamic=False,
                )
                star_objects[f"star_{i}"] = star_ball

    purple_floor = Bar.from_point_and_angle(
        x=0.0,
        y=MIN_Y + bar_thickness / 2,
        length=WORLD_WIDTH,
        thickness=bar_thickness,
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
        name="pinball_machine",
        objects=objects,
        action_objects=["red_ball"],
        success_condition=success_condition,
        metadata={"description": "Get the green ball to reach the floor."},
    )
