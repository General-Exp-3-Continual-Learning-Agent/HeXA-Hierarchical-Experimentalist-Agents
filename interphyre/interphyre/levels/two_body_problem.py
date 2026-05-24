import numpy as np
from interphyre.objects import Ball
from interphyre.level import Level
from interphyre.levels import register_level


def success_condition(engine):
    success_time = engine.config.default_success_time
    return engine.is_in_contact_for_duration("green_ball", "blue_ball", success_time)


@register_level
def build_level(seed=None, variant=0, scene=None) -> Level:
    scene = scene or {}
    green_spec = scene.get("green_ball", {})
    blue_spec = scene.get("blue_ball", {})
    red_spec = scene.get("red_ball", {})

    full_scene = (
        all(k in green_spec for k in ("x", "y", "radius"))
        and all(k in blue_spec for k in ("x", "y", "radius"))
        and "radius" in red_spec
    )

    if full_scene:
        # Bypass RNG entirely; geometry is deterministic regardless of seed
        green_ball_x = green_spec["x"]
        green_ball_y = green_spec["y"]
        green_ball_radius = green_spec["radius"]
        blue_ball_x = blue_spec["x"]
        blue_ball_y = blue_spec["y"]
        blue_ball_radius = blue_spec["radius"]
        red_ball_radius = red_spec["radius"]
    else:
        rng = np.random.default_rng(seed if variant == 0 else (seed, variant))

        # Always draw in fixed order to preserve RNG sequence across partial overrides
        green_ball_radius = rng.uniform(0.3, 0.6)
        blue_ball_radius = rng.uniform(0.3, 0.6)
        red_ball_radius = rng.uniform(0.3, 0.6)

        # Both balls at the same height
        ball_bottom = rng.uniform(-3, 3)

        # Green ball on left, blue ball on right with a gap
        min_gap = max(green_ball_radius, blue_ball_radius)
        # Ensure the blue ball's minimum x never exceeds its maximum x.
        max_green_x = 4.5 - (2 * blue_ball_radius) - min_gap - green_ball_radius
        green_ball_x = rng.uniform(-4.5 + green_ball_radius, max_green_x)
        min_blue_x = green_ball_x + green_ball_radius + min_gap + blue_ball_radius
        blue_ball_x = rng.uniform(min_blue_x, 4.5 - blue_ball_radius)

        green_ball_y = ball_bottom + green_ball_radius
        blue_ball_y = ball_bottom + blue_ball_radius

        # Apply partial scene overrides after all draws are complete
        green_ball_x = green_spec.get("x", green_ball_x)
        green_ball_y = green_spec.get("y", green_ball_y)
        green_ball_radius = green_spec.get("radius", green_ball_radius)
        blue_ball_x = blue_spec.get("x", blue_ball_x)
        blue_ball_y = blue_spec.get("y", blue_ball_y)
        blue_ball_radius = blue_spec.get("radius", blue_ball_radius)
        red_ball_radius = red_spec.get("radius", red_ball_radius)

    green_ball = Ball(
        x=green_ball_x,
        y=green_ball_y,
        radius=green_ball_radius,
        color="green",
        dynamic=True,
    )

    blue_ball = Ball(
        x=blue_ball_x,
        y=blue_ball_y,
        radius=blue_ball_radius,
        color="blue",
        dynamic=True,
    )

    red_ball = Ball(
        x=0,
        y=0,
        radius=red_ball_radius,
        color="red",
        dynamic=True,
    )

    objects = {
        "green_ball": green_ball,
        "blue_ball": blue_ball,
        "red_ball": red_ball,
    }

    return Level(
        name="two_body_problem",
        objects=objects,
        action_objects=["red_ball"],
        success_condition=success_condition,
        metadata={"description": "Make the green ball touch the blue ball"},
    )
