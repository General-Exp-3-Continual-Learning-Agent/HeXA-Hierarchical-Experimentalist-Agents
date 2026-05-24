"""
Tool wrappers around the Interphyre API for the ReAct agent.
Supports: down_to_earth, two_body_problem, catapult, falling_into_place
Each tool returns a text observation string that gets fed back to the LLM.
"""

import sys
import os
import math
import cv2

# Add the interphyre package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from interphyre import InterphyreEnv
from interphyre.levels import load_level
from interphyre.config import SimulationConfig

# At 60Hz, 2000 steps ≈ 33s gives enough buffer for late-contact scenarios.
# Success requires 3s (180 steps) of contact.
_SIM_CONFIG = SimulationConfig(max_steps=2000)


class InterphyreToolkit:
    """Toolkit wrapping Interphyre API for physics puzzle levels."""

    def __init__(self, level_name: str = "down_to_earth", seed: int = 42, is_oss: bool = False):
        self.level_name = level_name
        self.seed = seed
        self.is_oss = is_oss

        # Create an environment with extended max_steps
        self.env = InterphyreEnv(load_level(level_name, seed), config=_SIM_CONFIG)

        # Cache the level info at init
        self._level_info = self._extract_level_info()
        self._action_objects = list(self.env.level.action_objects)

    def _extract_level_info(self) -> dict:
        """Extract level object info from the level definition."""
        info = {}
        level = self.env.level
        for obj_name, obj in level.objects.items():
            obj_info = {
                "x": obj.x, "y": obj.y,
                "color": obj.color, "dynamic": obj.dynamic,
            }
            if hasattr(obj, "radius"):
                obj_info["type"] = "Ball"
                obj_info["radius"] = obj.radius
            elif hasattr(obj, "length"):
                obj_info["type"] = "Bar"
                obj_info["length"] = getattr(obj, "length", None)
                obj_info["thickness"] = getattr(obj, "thickness", None)
                obj_info["left"] = getattr(obj, "left", None)
                obj_info["right"] = getattr(obj, "right", None)
                obj_info["top"] = getattr(obj, "top", None)
                obj_info["bottom"] = getattr(obj, "bottom", None)
                obj_info["angle"] = getattr(obj, "angle", 0)
            else:
                obj_info["type"] = type(obj).__name__
                obj_info["angle"] = getattr(obj, "angle", 0)
                obj_info["scale"] = getattr(obj, "scale", None)
            info[obj_name] = obj_info
        return info

    def _find_platform(self) -> tuple:
        """Find the horizontal platform (static Bar that isn't the ground or a ramp)."""
        for name, obj in self._level_info.items():
            if (obj["type"] == "Bar" and not obj["dynamic"]
                    and obj.get("color") != "purple"
                    and abs(obj.get("angle", 0)) < 5):  # near-horizontal only
                return name, obj
        return None, None

    def _find_green_ball(self) -> tuple:
        """Find the green ball dynamically."""
        for name, obj in self._level_info.items():
            if obj["type"] == "Ball" and obj.get("color") == "green":
                return name, obj
        return None, None

    def _find_green_bar(self) -> tuple:
        """Find the green bar (dynamic green Bar — used in cliffhanger)."""
        for name, obj in self._level_info.items():
            if obj["type"] == "Bar" and obj.get("color") == "green" and obj.get("dynamic"):
                return name, obj
        return None, None

    def _find_ceiling(self) -> tuple:
        """Find the ceiling (static black Bar at the top of the box, used in cliffhanger)."""
        # Pick the highest-y static black horizontal Bar that isn't the platform.
        best_name, best_obj = None, None
        for name, obj in self._level_info.items():
            if (obj["type"] == "Bar" and not obj["dynamic"]
                    and obj.get("color") == "black"
                    and abs(obj.get("angle", 0)) < 5):
                y = obj.get("y", -math.inf)
                if y > 1.0 and (best_obj is None or y > best_obj.get("y", -math.inf)):
                    best_name, best_obj = name, obj
        return best_name, best_obj

    def _find_blue_ball(self) -> tuple:
        """Find the blue ball dynamically."""
        for name, obj in self._level_info.items():
            if obj["type"] == "Ball" and obj.get("color") == "blue":
                return name, obj
        return None, None

    def _find_catapult_bar(self) -> tuple:
        """Find the catapult arm bar (dynamic gray bar) by name or fallback."""
        for candidate in ("gray_platform", "catapult_bar"):
            if candidate in self._level_info:
                return candidate, self._level_info[candidate]
        # Fallback: first dynamic gray bar
        for name, obj in self._level_info.items():
            if obj["type"] == "Bar" and obj["dynamic"] and obj.get("color") == "gray":
                return name, obj
        return None, None

    def _find_pivot_ball(self) -> tuple:
        """Find the catapult pivot ball (gray ball acting as fulcrum) by name or fallback."""
        for candidate in ("gray_ball", "pivot_ball"):
            if candidate in self._level_info:
                return candidate, self._level_info[candidate]
        return None, None

    def _find_blue_jar(self) -> tuple:
        """Find the blue jar/basket (falling_into_place level)."""
        for name, obj in self._level_info.items():
            if obj.get("color") == "blue" and obj["type"] == "Basket":
                return name, obj
        # Fallback: any blue non-ball object
        for name, obj in self._level_info.items():
            if obj.get("color") == "blue" and obj["type"] not in ("Ball",):
                return name, obj
        return None, None

    def get_image(self):
        """Get an RGB rendering of the current environment state.
        Returns a numpy array (height, width, 3) in RGB format."""
        img_rgb = self.env._get_image_observation()
        # Convert RGB to BGR for OpenCV saving
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return img_bgr

    def _reset_env(self):
        """Reset the environment for a fresh simulation."""
        self.env = InterphyreEnv(load_level(self.level_name, self.seed), config=_SIM_CONFIG)

    def get_level_state(self) -> str:
        """Return a text description of all objects in the level."""
        lines = [f"=== Level State: {self.level_name} ===", ""]

        for name, obj in self._level_info.items():
            if name in self._action_objects:
                # Skip action objects (red ball) — agent places these
                continue

            if obj["type"] == "Ball":
                lines.append(
                    f"* {name} ({obj['color']} {obj['type']}): "
                    f"position=({obj['x']:.4f}, {obj['y']:.4f}), "
                    f"radius={obj['radius']:.4f}, "
                    f"dynamic={obj['dynamic']}"
                )
            elif obj["type"] == "Bar":
                top_val = f"{obj['top']:.4f}" if obj['top'] is not None else "N/A"
                bottom_val = f"{obj['bottom']:.4f}" if obj['bottom'] is not None else "N/A"
                left_val = f"{obj['left']:.4f}" if obj['left'] is not None else "N/A"
                right_val = f"{obj['right']:.4f}" if obj['right'] is not None else "N/A"
                lines.append(
                    f"* {name} ({obj['color']} {obj['type']}): "
                    f"position=({obj['x']:.4f}, {obj['y']:.4f}), "
                    f"left={left_val}, right={right_val}, "
                    f"top={top_val}, bottom={bottom_val}, "
                    f"thickness={obj['thickness']:.4f}, "
                    f"dynamic={obj['dynamic']}"
                )
            else:
                lines.append(f"* {name}: {obj}")

        lines.append("")
        lines.append("World bounds: x in [-5, 5], y in [-5, 5]")
        lines.append(f"Action objects: {self._action_objects} -- you control the red ball placement")

        # Level-specific success condition
        if self.level_name in ("two_body_problem", "catapult", "pass_the_parcel"):
            lines.append("Success condition: green_ball must touch blue_ball for 3 seconds")
        elif self.level_name == "falling_into_place":
            lines.append("Success condition: green_ball must touch blue_jar for 3 seconds")
        elif self.level_name == "basket_case":
            lines.append("Success condition: green_ball must touch purple_ground for 3 seconds")
        else:
            lines.append("Success condition: green_ball must touch purple_ground for 3 seconds")

        lines.append("=" * 60)
        return "\n".join(lines)

    def _validate_placement(self, x: float, y: float, radius: float):
        """Return an error string if the placement is invalid, else None."""
        errors = []

        if radius < 0.1 or radius > 2.0:
            errors.append(f"Radius {radius:.4f} out of range [0.1, 2.0].")

        if x - radius < -5.0:
            errors.append(f"Left edge (x-radius = {x - radius:.4f}) exceeds left wall (-5).")
        if x + radius > 5.0:
            errors.append(f"Right edge (x+radius = {x + radius:.4f}) exceeds right wall (5).")
        if y - radius < -5.0:
            errors.append(f"Bottom edge (y-radius = {y - radius:.4f}) exceeds floor (-5).")
        if y + radius > 5.0:
            errors.append(f"Top edge (y+radius = {y + radius:.4f}) exceeds ceiling (5).")

        for obj_name, obj in self._level_info.items():
            if obj_name in self._action_objects:
                continue
            if obj["type"] != "Ball":
                continue
            dist = math.sqrt((x - obj["x"]) ** 2 + (y - obj["y"]) ** 2)
            min_dist = radius + obj["radius"]
            if dist <= min_dist:
                shortfall = min_dist - dist + 0.05
                errors.append(
                    f"Overlaps with {obj_name}: dist={dist:.4f}, "
                    f"sum_of_radii={min_dist:.4f}. "
                    f"Move at least {shortfall:.4f} units further away."
                )

        plat_name, platform = self._find_platform()
        if platform:
            plat_left = platform["left"] if platform["left"] is not None else platform["x"] - platform["length"] / 2
            plat_right = platform["right"] if platform["right"] is not None else platform["x"] + platform["length"] / 2
            plat_top = platform["top"] if platform["top"] is not None else platform["y"] + platform["thickness"] / 2
            plat_bottom = platform["bottom"] if platform["bottom"] is not None else platform["y"] - platform["thickness"] / 2

            closest_x = max(plat_left, min(x, plat_right))
            closest_y = max(plat_bottom, min(y, plat_top))
            dist_to_rect = math.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2)

            if dist_to_rect < radius:
                shortfall = radius - dist_to_rect + 0.05
                errors.append(
                    f"Overlaps with {plat_name}: dist_to_platform={dist_to_rect:.4f} < "
                    f"radius={radius:.4f}. "
                    f"Move at least {shortfall:.4f} units further away from the platform."
                )

        if errors:
            return "INVALID placement:\n" + "\n".join(f"  - {e}" for e in errors)
        return None

    def simulate_action(self, x: float, y: float, radius: float) -> str:
        """Place red ball and run full simulation. Returns outcome."""
        err = self._validate_placement(x, y, radius)
        if err:
            return f"{err}\nThe simulation was not run. Fix the placement and try again."

        self._reset_env()

        action = [(x, y, radius)]
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
        except Exception as e:
            return f"ERROR: Simulation failed -- {str(e)}"

        success = info.get("success", False)
        step_count = info.get("step_count", "unknown")

        lines = []
        if success:
            if self.level_name in ("two_body_problem", "pass_the_parcel"):
                lines.append("SUCCESS! The green ball contacted the blue ball.")
            elif self.level_name == "catapult":
                lines.append("SUCCESS! The green ball contacted the blue ball in the basket.")
            elif self.level_name == "falling_into_place":
                lines.append("SUCCESS! The green ball contacted the blue jar.")
            elif self.level_name == "cliffhanger":
                lines.append("SUCCESS! The green bar reached the purple ground")
            elif self.level_name == "tipping_point":
                lines.append("SUCCESS! The green bar contacted the purple wall.")
            else:
                lines.append("SUCCESS! The green ball reached the purple ground.")
        else:
            if self.level_name in ("two_body_problem", "pass_the_parcel"):
                lines.append("FAILURE. The green ball did NOT contact the blue ball.")
            elif self.level_name == "catapult":
                lines.append("FAILURE. The green ball did NOT contact the blue ball in the basket.")
            elif self.level_name == "falling_into_place":
                lines.append("FAILURE. The green ball did NOT contact the blue jar.")
            elif self.level_name == "cliffhanger":
                lines.append("FAILURE. The green bar did NOT reach the purple ground.")
            elif self.level_name == "tipping_point":
                lines.append("FAILURE. The green bar did NOT contact the purple wall.")
            else:
                lines.append("FAILURE. The green ball did NOT reach the purple ground.")
        

        lines.append(f"Total simulation steps: {step_count}")
        lines.append(f"Reward: {reward}")
        lines.append("")

        # Report final object positions
        if isinstance(obs, dict) and "objects" in obs:
            lines.append("Final object positions:")
            for obj_name, obj_data in obs["objects"].items():
                pos = obj_data.get("position", [0, 0])
                vel = obj_data.get("velocity", [0, 0])
                lines.append(
                    f"  {obj_name}: pos=({pos[0]:.3f}, {pos[1]:.3f}), "
                    f"vel=({vel[0]:.3f}, {vel[1]:.3f})"
                )

        return "\n".join(lines)

    def simulate_partial(self, x: float, y: float, radius: float, stop_step: int) -> str:
        """Run simulation up to stop_step steps. Returns positions/velocities."""
        err = self._validate_placement(x, y, radius)
        if err:
            return f"{err}\nThe simulation was not run. Fix the placement and try again."

        self._reset_env()

        try:
            self.env.reset(seed=self.seed)
            action = [(x, y, radius)]
            validation_result = self.env._validate_action_with_failure(action)
            if validation_result["invalid"]:
                return f"INVALID ACTION: {validation_result['error']}"
            self.env._place_action_objects(validation_result["action"])

            actual_steps = 0
            for _ in range(stop_step):
                self.env._step_physics()
                actual_steps += 1
                if self.env.level.success_condition(self.env.engine):
                    break
        except Exception as e:
            return f"ERROR: Partial simulation failed -- {str(e)}"

        lines = [f"Simulation state at step {actual_steps}:", ""]

        try:
            for obj_name in self.env.level.objects:
                if obj_name in self.env.engine.bodies:
                    body = self.env.engine.bodies[obj_name]
                    lines.append(
                        f"  {obj_name}: pos=({body.position.x:.3f}, {body.position.y:.3f}), "
                        f"vel=({body.linearVelocity.x:.3f}, {body.linearVelocity.y:.3f})"
                    )
        except Exception as e:
            lines.append(f"  (error reading state: {e})")

        return "\n".join(lines)

    def get_contact_log(self) -> str:
        """Return contact events from the most recent simulation."""
        try:
            log = self.env.get_contact_log()
        except Exception as e:
            return f"ERROR: Could not retrieve contact log -- {str(e)}"

        if not log:
            return "No contact events recorded. (Run a simulation first with simulate_action.)"

        lines = ["Contact events:"]
        for entry in log[:20]:  # Limit output
            lines.append(f"  {entry}")

        if len(log) > 20:
            lines.append(f"  ... and {len(log) - 20} more events")

        return "\n".join(lines)

    def compute_gap_analysis(self) -> str:
        """Analyze gaps on each side of the platform (for down_to_earth)."""
        _, platform = self._find_platform()
        _, green = self._find_green_ball()

        if not platform or not green:
            return "ERROR: Could not find platform or green ball in level."

        plat_left = platform["left"] if platform["left"] is not None else platform["x"] - platform["length"] / 2
        plat_right = platform["right"] if platform["right"] is not None else platform["x"] + platform["length"] / 2
        green_diameter = green["radius"] * 2

        left_gap = plat_left - (-5.0)
        right_gap = 5.0 - plat_right

        lines = [
            "=== Gap Analysis ===",
            f"Platform spans: x in [{plat_left:.4f}, {plat_right:.4f}]",
            f"Green ball diameter: {green_diameter:.4f}",
            "",
            f"Left gap (left wall to platform left edge): {left_gap:.4f}",
            f"  Green ball can fall left: {'YES' if left_gap > green_diameter else 'NO'} "
            f"(gap {'>' if left_gap > green_diameter else '<='} diameter)",
            "",
            f"Right gap (platform right edge to right wall): {right_gap:.4f}",
            f"  Green ball can fall right: {'YES' if right_gap > green_diameter else 'NO'} "
            f"(gap {'>' if right_gap > green_diameter else '<='} diameter)",
        ]

        if left_gap > green_diameter and right_gap > green_diameter:
            if left_gap > right_gap:
                lines.append("\nRecommendation: Push green ball LEFT (larger gap).")
            else:
                lines.append("\nRecommendation: Push green ball RIGHT (larger gap).")
        elif left_gap > green_diameter:
            lines.append("\nRecommendation: Push green ball LEFT (only viable direction).")
        elif right_gap > green_diameter:
            lines.append("\nRecommendation: Push green ball RIGHT (only viable direction).")
        else:
            lines.append(
                "\nWARNING: Neither gap is wide enough for the green ball! "
                "A very forceful collision may still push it off the edge."
            )

        return "\n".join(lines)

    def compute_relative_positions(self) -> str:
        """Analyze relative positions of green and blue balls (for two_body_problem)."""
        _, green = self._find_green_ball()
        _, blue = self._find_blue_ball()

        if not green or not blue:
            return "ERROR: Could not find green ball or blue ball in level."

        dx = blue["x"] - green["x"]
        dy = blue["y"] - green["y"]
        dist = math.sqrt(dx ** 2 + dy ** 2)

        direction = "RIGHT" if dx > 0 else "LEFT"
        red_side = "LEFT" if dx > 0 else "RIGHT"  # opposite side

        lines = [
            "=== Relative Positions ===",
            f"Green ball: ({green['x']:.4f}, {green['y']:.4f}), radius={green['radius']:.4f}",
            f"Blue ball:  ({blue['x']:.4f}, {blue['y']:.4f}), radius={blue['radius']:.4f}",
            "",
            f"Horizontal separation (dx): {dx:.4f}",
            f"Vertical separation (dy): {dy:.4f}",
            f"Center-to-center distance: {dist:.4f}",
            f"Min contact distance (sum of radii): {green['radius'] + blue['radius']:.4f}",
            "",
            f"Blue is to the {direction} of Green.",
            f"Push the Green Ball to the {direction}."
        ]

        return "\n".join(lines)

    # ─── Catapult-level strategy-neutral tools ──────────────────

    def _bar_endpoints(self, obj: dict) -> tuple:
        """Return true (x1, y1), (x2, y2) endpoints of a bar from center+angle+length."""
        x, y = obj["x"], obj["y"]
        length = obj.get("length") or 0.0
        angle_deg = obj.get("angle") or 0.0
        angle_rad = math.radians(angle_deg)
        dx = 0.5 * length * math.cos(angle_rad)
        dy = 0.5 * length * math.sin(angle_rad)
        return (x - dx, y - dy), (x + dx, y + dy)

    def _basket_geometry(self, obj: dict) -> dict:
        """Compute key basket geometry: floor center, wall tips, opening direction."""
        # Interphyre Basket anchor stored in _extract_level_info as the raw object.
        # We fall back to reading the underlying level object for accurate geometry.
        name = None
        for n, o in self._level_info.items():
            if o is obj:
                name = n
                break
        level_obj = self.env.level.objects.get(name) if name else None
        if level_obj is None:
            return {}

        angle_rad = math.radians(getattr(level_obj, "angle", 0.0) or 0.0)
        bw = getattr(level_obj, "bottom_width", 0.0)
        tw = getattr(level_obj, "top_width", 0.0)
        h = getattr(level_obj, "height", 0.0)
        wt = getattr(level_obj, "wall_thickness", 0.0)
        anchor = getattr(level_obj, "anchor", "bottom_center")
        cx, cy = level_obj.x, level_obj.y

        # Anchor offset: Basket stores position relative to named anchor; we normalize
        # to "bottom_center" so (cx0, cy0) is the center of the basket floor top.
        if anchor == "bottom_center":
            bx, by = cx, cy
        elif anchor == "center":
            bx, by = cx, cy - h / 2
        elif anchor == "top_center":
            bx, by = cx, cy - h
        else:
            bx, by = cx, cy

        # Local (pre-rotation) key points: bottom-floor-center, bottom-floor-tips, top-wall-tips.
        def rot(px, py):
            rx = bx + (px * math.cos(angle_rad) - py * math.sin(angle_rad))
            ry = by + (px * math.sin(angle_rad) + py * math.cos(angle_rad))
            return (rx, ry)

        floor_center = rot(0.0, 0.0)
        bottom_left = rot(-bw / 2 - wt / 2, 0.0)
        bottom_right = rot(bw / 2 + wt / 2, 0.0)
        top_left = rot(-tw / 2 - wt / 2, h)
        top_right = rot(tw / 2 + wt / 2, h)
        # Opening direction = local +y rotated: (-sin, cos).
        opening_dir = (-math.sin(angle_rad), math.cos(angle_rad))

        return {
            "floor_center": floor_center,
            "bottom_left_corner": bottom_left,
            "bottom_right_corner": bottom_right,
            "top_left_rim": top_left,
            "top_right_rim": top_right,
            "opening_direction": opening_dir,
            "interior_height": h,
            "bottom_width": bw,
            "top_width": tw,
            "angle_deg": math.degrees(angle_rad),
        }

    def _live_body_mass(self, name: str):
        """Return the Box2D body mass if the body is instantiated, else None."""
        try:
            body = self.env.engine.bodies.get(name)
            if body is None:
                return None
            return float(body.mass)
        except Exception:
            return None

    def describe_scene_geometry(self) -> str:
        """Strategy-neutral derived geometry for the catapult level (optimized)."""
        lines = ["=== Scene Geometry ==="]
        lines.append("World bounds: x ∈ [-5, 5], y ∈ [-5, 5]")
        lines.append("")

        # Balls (minimal: name, pos, radius, dynamic)
        lines.append("Balls:")
        for name, obj in self._level_info.items():
            if obj["type"] != "Ball":
                continue
            if name in self._action_objects:
                continue
            lines.append(
                f"  {name}: pos=({obj['x']:.2f}, {obj['y']:.2f}), r={obj['radius']:.2f}, "
                f"dynamic={obj['dynamic']}"
            )
        lines.append("")

        # Bars (minimal: name, pos, angle, length)
        lines.append("Bars:")
        for name, obj in self._level_info.items():
            if obj["type"] != "Bar":
                continue
            angle_deg = obj.get("angle") or 0.0
            length = obj.get('length') or 0.0
            lines.append(
                f"  {name}: pos=({obj['x']:.2f}, {obj['y']:.2f}), angle={angle_deg:.1f}°, "
                f"len={length:.2f}, dynamic={obj['dynamic']}"
            )
        lines.append("")

        # Baskets (minimal: name, pos)
        basket_entries = [
            (name, obj) for name, obj in self._level_info.items() if obj["type"] == "Basket"
        ]
        if basket_entries:
            lines.append("Baskets:")
            for name, obj in basket_entries:
                lines.append(f"  {name}: pos=({obj['x']:.2f}, {obj['y']:.2f}), dynamic={obj['dynamic']}")
            lines.append("")

        # Pairwise distances (fast computation, no helper calls)
        def dist(a_pos, b_pos):
            return math.hypot(a_pos[0] - b_pos[0], a_pos[1] - b_pos[1])

        _, green = self._find_green_ball()
        _, blue = self._find_blue_ball()

        dist_lines = []
        if green and blue:
            d = dist((green['x'], green['y']), (blue['x'], blue['y']))
            dist_lines.append(f"  green↔blue: {d:.2f}")

        if dist_lines:
            lines.append("Key distances:")
            lines.extend(dist_lines)
            lines.append("")

        lines.append("Success condition: green_ball ↔ blue_ball for 3.0s")
        lines.append("=" * 50)
        return "\n".join(lines)

    def simulate_with_trace(
        self,
        x: float,
        y: float,
        radius: float,
        object_names: list | None = None,
        n_samples: int = 12,
        stop_step: int | None = None,
    ) -> str:
        """Run a simulation and return kinematic summary + contacts for the chosen objects (no waypoint traces)."""
        err = self._validate_placement(x, y, radius)
        if err:
            return f"{err}\nThe simulation was not run. Fix the placement and try again."

        if not object_names:
            return (
                "ERROR: 'object_names' is required. Pick at least one object to trace, e.g. "
                '["green_ball", "catapult_arm"] or ["basket", "blue_ball"].'
            )

        n_samples = max(2, min(int(n_samples), 20))
        # Cap stop_step to avoid runaway sims; None => run until success or max_steps.
        max_steps = int(self.env.config.max_steps)
        if stop_step is None:
            stop_step = max_steps
        stop_step = max(1, min(int(stop_step), max_steps))

        self._reset_env()
        try:
            self.env.reset(seed=self.seed)
            action = [(x, y, radius)]
            validation_result = self.env._validate_action_with_failure(action)
            if validation_result["invalid"]:
                return f"INVALID ACTION: {validation_result['error']}"
            self.env._place_action_objects(validation_result["action"])
        except Exception as e:
            return f"ERROR: could not place red ball -- {e}"

        # Resolve requested object names; keep only ones present in the engine.
        available = list(self.env.level.objects.keys())
        traced = []
        missing = []
        for name in object_names:
            if name in available:
                traced.append(name)
            else:
                missing.append(name)
        if not traced:
            return (
                f"ERROR: None of the requested object_names exist. "
                f"Available: {', '.join(available)}"
            )

        summaries = {name: {
            "peak_y": -1e9, "min_y": 1e9, "max_speed": 0.0,
            "peak_angular_speed": 0.0,
            "min_angle_deg": 1e9, "max_angle_deg": -1e9,
            "start_pos": None, "end_pos": None,
        } for name in traced}

        success = False
        actual_steps = 0
        try:
            for step_idx in range(1, stop_step + 1):
                self.env._step_physics()
                actual_steps = step_idx
                # Update summaries every step.
                for name in traced:
                    body = self.env.engine.bodies.get(name)
                    if body is None:
                        continue
                    px, py = body.position.x, body.position.y
                    vx, vy = body.linearVelocity.x, body.linearVelocity.y
                    speed = math.hypot(vx, vy)
                    ang_deg = math.degrees(body.angle)
                    ang_speed = abs(body.angularVelocity)
                    s = summaries[name]
                    if s["start_pos"] is None:
                        s["start_pos"] = (px, py)
                    s["end_pos"] = (px, py)
                    if py > s["peak_y"]: s["peak_y"] = py
                    if py < s["min_y"]: s["min_y"] = py
                    if speed > s["max_speed"]: s["max_speed"] = speed
                    if ang_speed > s["peak_angular_speed"]: s["peak_angular_speed"] = ang_speed
                    if ang_deg < s["min_angle_deg"]: s["min_angle_deg"] = ang_deg
                    if ang_deg > s["max_angle_deg"]: s["max_angle_deg"] = ang_deg

                if self.env.level.success_condition(self.env.engine):
                    success = True
                    break
        except Exception as e:
            return f"ERROR: simulation failed at step {actual_steps} -- {e}"

        # Build contact list (only events involving the red ball OR the traced objects).
        contact_events = []
        try:
            raw = self.env.engine.get_contact_log()
            relevant = set(traced) | {"red_ball"}
            for ev in raw:
                if ev.get("event") != "begin":
                    continue
                a, b = ev.get("objects", (None, None))
                if a in relevant or b in relevant:
                    contact_events.append({
                        "step": int(round(ev.get("time", 0.0) * 60)),
                        "a": a, "b": b,
                    })
        except Exception:
            pass

        # Format output (sparse: summary + contacts only, no waypoint traces).
        lines = [f"=== Simulation Result (steps: {actual_steps}) ==="]
        lines.append(f"Success: {success}")
        if missing:
            lines.append(f"Note: unknown object_names: {', '.join(missing)}")
        lines.append("")

        lines.append("Contacts (red_ball + traced objects):")
        if not contact_events:
            lines.append("  (none)")
        else:
            for ev in contact_events[:15]:
                lines.append(f"  step~{ev['step']:3d}: {ev['a']} ↔ {ev['b']}")
            if len(contact_events) > 15:
                lines.append(f"  ... +{len(contact_events) - 15} more")
        lines.append("")

        lines.append("Object summary (key kinematic extrema):")
        for name in traced:
            s = summaries[name]
            if s["start_pos"] is None:
                continue
            dx = s["end_pos"][0] - s["start_pos"][0]
            dy = s["end_pos"][1] - s["start_pos"][1]
            lines.append(
                f"  {name}: y_peak={s['peak_y']:.2f}, y_min={s['min_y']:.2f}, "
                f"v_max={s['max_speed']:.2f}, Δpos=({dx:+.2f}, {dy:+.2f})"
            )
            ang_range = s["max_angle_deg"] - s["min_angle_deg"]
            if ang_range > 0.5 or s["peak_angular_speed"] > 0.05:
                lines.append(
                    f"    ω_peak={s['peak_angular_speed']:.2f} rad/s, "
                    f"θ∈[{s['min_angle_deg']:+.1f}°, {s['max_angle_deg']:+.1f}°]"
                )
        return "\n".join(lines)

    def predict_first_contact(self, x: float, y: float, radius: float) -> str:
        """Run a short (≤90 step) simulation and report the red ball's first contact."""
        err = self._validate_placement(x, y, radius)
        if err:
            return f"{err}\nplacement_valid=False; no contact check performed."

        self._reset_env()
        try:
            self.env.reset(seed=self.seed)
            action = [(x, y, radius)]
            validation_result = self.env._validate_action_with_failure(action)
            if validation_result["invalid"]:
                return f"INVALID ACTION: {validation_result['error']}"
            self.env._place_action_objects(validation_result["action"])
        except Exception as e:
            return f"ERROR: could not place red ball -- {e}"

        max_check_steps = 90
        hit = None
        try:
            for step_idx in range(1, max_check_steps + 1):
                prev_len = len(self.env.engine.contact_listener.contact_events)
                self.env._step_physics()
                events = self.env.engine.contact_listener.contact_events
                for ev in events[prev_len:]:
                    if ev.get("event") != "begin":
                        continue
                    a, b = ev.get("objects", (None, None))
                    if a == "red_ball" or b == "red_ball":
                        other = b if a == "red_ball" else a
                        # Ignore self-walls if uninteresting? keep walls — they matter for strategy reasoning.
                        red_body = self.env.engine.bodies.get("red_ball")
                        other_body = self.env.engine.bodies.get(other)
                        if red_body is None:
                            hit = {"other": other, "step": step_idx}
                            break
                        rx, ry = red_body.position.x, red_body.position.y
                        rvx, rvy = red_body.linearVelocity.x, red_body.linearVelocity.y
                        approach_speed = math.hypot(rvx, rvy)
                        if other_body is not None:
                            ox, oy = other_body.position.x, other_body.position.y
                            nx = rx - ox
                            ny = ry - oy
                            n_len = math.hypot(nx, ny) or 1.0
                            normal = (nx / n_len, ny / n_len)
                            contact_point = ((rx + ox) / 2, (ry + oy) / 2)
                        else:
                            normal = (0.0, 0.0)
                            contact_point = (rx, ry)
                        hit = {
                            "other": other,
                            "step": step_idx,
                            "approach_speed": approach_speed,
                            "contact_point": contact_point,
                            "surface_normal": normal,
                        }
                        break
                if hit is not None:
                    break
        except Exception as e:
            return f"ERROR: pre-simulation failed -- {e}"

        lines = ["=== Predicted First Contact ==="]
        lines.append(f"placement_valid: True")
        lines.append(f"red_ball drop: x={x:.3f}, y={y:.3f}, radius={radius:.3f}")
        if hit is None:
            lines.append(
                f"No contact within {max_check_steps} steps. "
                "Red ball is still in free-fall — consider whether it will eventually miss."
            )
            return "\n".join(lines)

        lines.append(f"first_contact_object: {hit['other']}")
        lines.append(f"contact_step: {hit['step']}")
        if "approach_speed" in hit:
            lines.append(f"approach_speed: {hit['approach_speed']:.3f}")
            cp = hit["contact_point"]
            lines.append(f"contact_point (approx midpoint): ({cp[0]:.3f}, {cp[1]:.3f})")
            sn = hit["surface_normal"]
            lines.append(f"contact_normal (red->other inverted): ({sn[0]:.3f}, {sn[1]:.3f})")
        return "\n".join(lines)

    def trace_green_ball(self, x: float, y: float, radius: float) -> str:
        """Cheap trajectory probe: place a red ball and sample only the green
        ball's (x, y) at fixed intervals. Stops early once the green ball is at
        rest (speed below threshold for several consecutive samples) or after a
        capped number of steps. Returns a compact waypoint list.

        Designed as a lighter replacement for simulate_with_trace when the only
        thing the agent needs is the green ball's path.
        """
        err = self._validate_placement(x, y, radius)
        if err:
            return f"{err}\nThe simulation was not run. Fix the placement and try again."

        green_name, _ = self._find_green_ball()
        if green_name is None:
            return "ERROR: no green ball found in this level."

        # Tunables
        sample_every = 30          # physics steps between samples
        max_total_steps = 600      # hard ceiling regardless of motion
        settle_disp_eps = 0.10     # |Δpos| between samples below this counts as "settled"
        settle_consec = 3          # need this many consecutive settled samples to stop

        self._reset_env()
        try:
            self.env.reset(seed=self.seed)
            action = [(x, y, radius)]
            validation_result = self.env._validate_action_with_failure(action)
            if validation_result["invalid"]:
                return f"INVALID ACTION: {validation_result['error']}"
            self.env._place_action_objects(validation_result["action"])
        except Exception as e:
            return f"ERROR: could not place red ball -- {e}"

        waypoints = []  # list of (step, x, y, speed)
        settle_count = 0
        success = False
        actual_steps = 0
        try:
            for step_idx in range(1, max_total_steps + 1):
                self.env._step_physics()
                actual_steps = step_idx

                if step_idx % sample_every == 0 or step_idx == 1:
                    body = self.env.engine.bodies.get(green_name)
                    if body is None:
                        break
                    px, py = body.position.x, body.position.y
                    speed = math.hypot(body.linearVelocity.x, body.linearVelocity.y)
                    waypoints.append((step_idx, px, py, speed))

                    if len(waypoints) >= 2:
                        prev = waypoints[-2]
                        disp = math.hypot(px - prev[1], py - prev[2])
                        if disp < settle_disp_eps:
                            settle_count += 1
                            if settle_count >= settle_consec:
                                break
                        else:
                            settle_count = 0

                if self.env.level.success_condition(self.env.engine):
                    body = self.env.engine.bodies.get(green_name)
                    if body is not None:
                        px, py = body.position.x, body.position.y
                        speed = math.hypot(body.linearVelocity.x, body.linearVelocity.y)
                        waypoints.append((step_idx, px, py, speed))
                    success = True
                    break
        except Exception as e:
            return f"ERROR: simulation failed at step {actual_steps} -- {e}"

        # Compact summary
        lines = [f"=== Green Ball Trajectory (steps run: {actual_steps}) ==="]
        lines.append(f"red_ball drop: x={x:.3f}, y={y:.3f}, radius={radius:.3f}")
        lines.append(f"Success: {success}")
        if not waypoints:
            lines.append("No waypoints recorded (green ball missing or simulation aborted).")
            return "\n".join(lines)

        sx, sy = waypoints[0][1], waypoints[0][2]
        ex, ey = waypoints[-1][1], waypoints[-1][2]
        peak_y = max(p[2] for p in waypoints)
        max_speed = max(p[3] for p in waypoints)
        lines.append(
            f"green_ball: start=({sx:+.2f}, {sy:+.2f}) end=({ex:+.2f}, {ey:+.2f}) "
            f"Δ=({ex - sx:+.2f}, {ey - sy:+.2f}) peak_y={peak_y:+.2f} v_max={max_speed:.2f}"
        )
        lines.append("")
        lines.append("Waypoints (step: x, y):")
        for step_idx, px, py, _spd in waypoints:
            lines.append(f"  step {step_idx:4d}: ({px:+.2f}, {py:+.2f})")
        return "\n".join(lines)

    def compute_intercept_setup(self) -> str:
        """Compute intercept geometry for the falling_into_place level."""
        _, green = self._find_green_ball()
        jar_name, blue_jar = self._find_blue_jar()

        if not green or not blue_jar:
            return "ERROR: Could not find green ball or blue jar. Is this the falling_into_place level?"

        # Find the two horizontal platforms (exclude angled ramp: ramp has |top - bottom| >> thickness)
        left_bar = right_bar = None
        left_bar_name = right_bar_name = None
        for name, obj in self._level_info.items():
            if (obj["type"] == "Bar" and not obj["dynamic"]
                    and obj.get("color") == "black"
                    and abs(obj.get("top", 0) - obj.get("bottom", 0)) <= obj.get("thickness", 0.2) * 3):
                if left_bar is None or obj["x"] < left_bar["x"]:
                    right_bar = left_bar
                    right_bar_name = left_bar_name
                    left_bar = obj
                    left_bar_name = name
                else:
                    right_bar = obj
                    right_bar_name = name

        # Determine which platform green is on
        if left_bar and right_bar:
            lb_right = left_bar["right"] if left_bar["right"] is not None else left_bar["x"] + left_bar.get("length", 0) / 2
            rb_left = right_bar["left"] if right_bar["left"] is not None else right_bar["x"] - right_bar.get("length", 0) / 2

            if green["x"] <= lb_right:
                on_platform = "LEFT"
                nearest_edge = lb_right
            else:
                on_platform = "RIGHT"
                nearest_edge = rb_left

            travel_dir = "RIGHT" if blue_jar["x"] > green["x"] else "LEFT"

            # Estimate jar fall time to platform height
            jar_y = blue_jar["y"]
            plat_y = left_bar["y"]
            fall_dist = jar_y - plat_y
            g = 10.0
            fall_time = math.sqrt(2 * max(fall_dist, 0) / g) if fall_dist > 0 else 0.0

            lines = [
                "=== Intercept Setup ===",
                f"Green ball: pos=({green['x']:.4f}, {green['y']:.4f}), on {on_platform} platform",
                f"Blue jar: pos=({blue_jar['x']:.4f}, {blue_jar['y']:.4f})",
                "",
                f"Green must travel: {travel_dir} (toward jar x={blue_jar['x']:.4f})",
                f"Nearest platform edge to cross: x={nearest_edge:.4f}",
                f"Gap center: x={((lb_right + rb_left) / 2):.4f}",
                "",
                f"Jar fall time to platform height: ~{fall_time:.2f}s ({int(fall_time * 60)} steps at 60Hz)",
            ]
        else:
            lines = [
                "=== Intercept Setup ===",
                f"Green ball: pos=({green['x']:.4f}, {green['y']:.4f})",
                f"Blue jar: pos=({blue_jar['x']:.4f}, {blue_jar['y']:.4f})",
            ]

        return "\n".join(lines)

    def compute_basket_analysis(self) -> str:
        """Analyze the basket_case level setup and recommend a push direction."""
        _, green = self._find_green_ball()

        basket_info = None
        for name, obj in self._level_info.items():
            if obj["type"] == "Basket":
                basket_info = obj
                break

        purple_ground = None
        for name, obj in self._level_info.items():
            if obj.get("color") == "purple" and obj["type"] == "Bar":
                purple_ground = obj
                break

        if not green or not basket_info:
            return "ERROR: Could not find green ball or basket. Is this the basket_case level?"

        lines = [
            "=== Basket Analysis ===",
            f"Green ball: pos=({green['x']:.4f}, {green['y']:.4f}), radius={green['radius']:.4f}",
            f"Basket: pos=({basket_info['x']:.4f}, {basket_info['y']:.4f}), scale={basket_info.get('scale', '?')}",
        ]
        if purple_ground:
            lines.append(f"Purple ground: y={purple_ground['y']:.4f}")

        dx = green["x"] - basket_info["x"]
        basket_scale = basket_info.get("scale", 1.0) or 1.0
        lines.append("")
        lines.append(f"Green ball is {abs(dx):.4f} units {'RIGHT' if dx > 0 else 'LEFT'} of basket center.")
        lines.append(f"Basket opening half-width ~ {basket_scale:.2f} units.")
        lines.append("")

        # Recommend push direction: push green further away from basket center, or pick the wider side
        push_dir = "LEFT" if green["x"] >= basket_info["x"] else "RIGHT"
        offset_sign = -1 if push_dir == "LEFT" else 1
        rec_x = round(green["x"] + offset_sign * (basket_scale + 0.3), 2)
        rec_y = round(green["y"] + 0.1, 2)

        # lines.append(f"Recommended push direction: {push_dir}")
        # lines.append(f"Place red ball to the {'RIGHT' if push_dir == 'LEFT' else 'LEFT'} of green ball to push it {push_dir}.")
        # lines.append(f"Example starting placement: x~{rec_x}, y~{rec_y}, radius~0.6")
        # lines.append("Increase radius or adjust x offset if the green ball still falls into the basket.")

        return "\n".join(lines)

    def compute_cliffhanger_analysis(self) -> str:
        """Analyse cliffhanger geometry: green bar endpoints, platform extents,
        which edge of the platform the bar is closer to, and the falling gap on
        that side. Strategy-light — names directions and gaps, does not pick a
        placement.
        """
        _, bar = self._find_green_bar()
        plat_name, platform = self._find_platform()
        _, ceiling = self._find_ceiling()

        purple_ground = None
        for _name, obj in self._level_info.items():
            if obj.get("color") == "purple" and obj["type"] == "Bar":
                purple_ground = obj
                break

        if not bar or not platform:
            return ("ERROR: Could not find green bar or black platform. "
                    "Is this the cliffhanger level?")

        bx, by = bar["x"], bar["y"]
        L = bar.get("length") or 0.0
        thk = bar.get("thickness") or 0.2
        angle_deg = bar.get("angle", 90)
        # Cliffhanger bars are constructed vertical (angle=90); compute endpoints
        # generically so non-vertical orientations still report sensibly.
        rad = math.radians(angle_deg)
        half_dx = (L / 2.0) * math.cos(rad)
        half_dy = (L / 2.0) * math.sin(rad)
        end1 = (bx + half_dx, by + half_dy)
        end2 = (bx - half_dx, by - half_dy)
        # Whichever endpoint has the smaller y is the one resting on the platform.
        bottom_pt, top_pt = (end2, end1) if end2[1] < end1[1] else (end1, end2)

        plat_left = platform["left"] if platform["left"] is not None else platform["x"] - (platform.get("length") or 0) / 2
        plat_right = platform["right"] if platform["right"] is not None else platform["x"] + (platform.get("length") or 0) / 2
        plat_y = platform["y"]
        plat_thk = platform.get("thickness") or 0.2
        plat_top_y = plat_y + plat_thk / 2

        dist_left = bottom_pt[0] - plat_left
        dist_right = plat_right - bottom_pt[0]
        if dist_left <= dist_right:
            close_edge = "LEFT"
            close_edge_x = plat_left
            falling_gap_width = plat_left - (-5.0)  # space between platform left and world wall
            push_direction = "LEFT"
        else:
            close_edge = "RIGHT"
            close_edge_x = plat_right
            falling_gap_width = 5.0 - plat_right
            push_direction = "RIGHT"

        lines = [
            "=== Cliffhanger Geometry Analysis ===",
            f"Green bar: center=({bx:.4f}, {by:.4f}), length={L:.4f}, "
            f"thickness={thk:.4f}, angle={angle_deg:.1f}°",
            f"  Bottom point (resting on platform): ({bottom_pt[0]:.4f}, {bottom_pt[1]:.4f})",
            f"  Top point (opposite end): ({top_pt[0]:.4f}, {top_pt[1]:.4f})",
            "",
            f"Black platform `{plat_name}`: x ∈ [{plat_left:.4f}, {plat_right:.4f}], "
            f"y={plat_y:.4f}, top surface y={plat_top_y:.4f}, thickness={plat_thk:.4f}",
        ]
        if ceiling:
            lines.append(f"Ceiling: y={ceiling['y']:.4f}")
        if purple_ground:
            lines.append(f"Purple ground: y={purple_ground['y']:.4f}")

        lines.extend([
            "",
            f"Bar's bottom-point distance to platform LEFT edge:  {dist_left:.4f}",
            f"Bar's bottom-point distance to platform RIGHT edge: {dist_right:.4f}",
            f"Closer platform edge: {close_edge} (x = {close_edge_x:.4f})",
            f"Falling gap on the {close_edge} side (between platform edge and world wall): "
            f"{falling_gap_width:.4f} units wide",
            "",
            f"To tip the bar off the platform, it must fall past the {close_edge} edge — "
            f"i.e. its top end must rotate {push_direction} until the bar's centre of mass "
            f"crosses the {close_edge} edge.",
        ])
        return "\n".join(lines)

    def compute_tipping_point_analysis(self) -> str:
        """Analyse tipping_point geometry: green bar endpoints, basket centre,
        purple wall x and side relative to the bar, horizontal distance, and
        suggested tip direction. Strategy-light — names directions and
        distances, does not pick a placement.
        """
        _, bar = self._find_green_bar()

        purple_wall = None
        for _name, obj in self._level_info.items():
            if (obj["type"] == "Bar" and obj.get("color") == "purple"
                    and not obj.get("dynamic", False)):
                purple_wall = obj
                break

        basket = None
        for _name, obj in self._level_info.items():
            if obj["type"] == "Basket":
                basket = obj
                break

        if not bar or not purple_wall:
            return ("ERROR: Could not find green bar or purple wall. "
                    "Is this the tipping_point level?")

        bx, by = bar["x"], bar["y"]
        L = bar.get("length") or 0.0
        thk = bar.get("thickness") or 0.2
        angle_deg = bar.get("angle", 90)
        rad = math.radians(angle_deg)
        half_dx = (L / 2.0) * math.cos(rad)
        half_dy = (L / 2.0) * math.sin(rad)
        end1 = (bx + half_dx, by + half_dy)
        end2 = (bx - half_dx, by - half_dy)
        bottom_pt, top_pt = (end2, end1) if end2[1] < end1[1] else (end1, end2)

        wall_x = purple_wall["x"]
        wall_top = purple_wall.get("top")
        wall_bottom = purple_wall.get("bottom")
        if wall_top is None:
            wall_top = wall_x  # fallback shouldn't happen for tipping_point
        if wall_bottom is None:
            wall_bottom = wall_x

        if wall_x < bx:
            wall_side = "LEFT"
            tip_direction = "LEFT"
            horizontal_distance = bx - wall_x
        else:
            wall_side = "RIGHT"
            tip_direction = "RIGHT"
            horizontal_distance = wall_x - bx

        # Angle the bar (treated as a rigid stick pivoting at its bottom) must
        # rotate from vertical so its top end reaches the wall horizontally.
        # If horizontal_distance >= L the wall is unreachable by pure rotation;
        # report None for that case.
        if L > 0 and horizontal_distance < L:
            tip_angle_deg = math.degrees(math.asin(
                max(0.0, min(1.0, horizontal_distance / L))
            ))
            tip_angle_str = f"{tip_angle_deg:.2f}°"
        else:
            tip_angle_str = "N/A (wall farther than bar length; bar must slide or basket must move)"

        opposite_side = "RIGHT" if wall_side == "LEFT" else "LEFT"
        # Suggested placement region: above the bar's free top end, offset to the
        # side opposite the wall. This is a starting region, not a recipe.
        if opposite_side == "RIGHT":
            target_x_lo = top_pt[0] + 0.3
            target_x_hi = top_pt[0] + 1.5
        else:
            target_x_lo = top_pt[0] - 1.5
            target_x_hi = top_pt[0] - 0.3
        target_y_lo = top_pt[1] + 1.0
        target_y_hi = min(4.5, top_pt[1] + 3.0)

        lines = [
            "=== Tipping Point Geometry Analysis ===",
            f"Green bar: center=({bx:.4f}, {by:.4f}), length={L:.4f}, "
            f"thickness={thk:.4f}, angle={angle_deg:.1f}°",
            f"  Top point (FREE end): "
            f"({top_pt[0]:.4f}, {top_pt[1]:.4f})",
            f"  Bottom point (PINNED in basket): "
            f"({bottom_pt[0]:.4f}, {bottom_pt[1]:.4f})",
        ]
        if basket:
            lines.append(
                f"Gray basket: center=({basket['x']:.4f}, {basket['y']:.4f}) "
                f"(holds the bar's base)"
            )
        lines.extend([
            f"Purple wall: x={wall_x:.4f}, top={wall_top:.4f}, "
            f"bottom={wall_bottom:.4f} (static)",
            "",
            f"Wall side relative to green bar: {wall_side}",
            f"Horizontal distance (bar centre → wall): {horizontal_distance:.4f}",
            f"Approx angle bar must tip (pivoting at base) to reach wall: {tip_angle_str}",
            ""
        ])
        return "\n".join(lines)

    def get_ramp_center(self) -> str:
        """Compute geometry-derived guidance for the pass_the_parcel level."""
        _, green = self._find_green_ball()

        top_basket = bottom_basket = None
        for name, obj in self._level_info.items():
            if obj["type"] == "Basket":
                angle = obj.get("angle", 0)
                if angle and abs(angle - 180) < 45:
                    top_basket = obj
                else:
                    bottom_basket = obj

        ramp = platform = None
        for name, obj in self._level_info.items():
            if obj["type"] == "Bar" and obj.get("color") == "black" and not obj["dynamic"]:
                bar_angle = abs(obj.get("angle", 0))
                if bar_angle < 5:  # near-horizontal → platform
                    platform = obj
                else:              # angled → ramp
                    ramp = obj

        if not top_basket or not ramp:
            return "ERROR: Could not identify ramp or top basket. Is this the pass_the_parcel level?"

        ramp_x0 = ramp["left"]
        ramp_y0 = ramp["bottom"]
        ramp_x1 = ramp["right"]
        ramp_y1 = ramp["top"]
        ramp_dx = ramp_x1 - ramp_x0
        ramp_dy = ramp_y1 - ramp_y0

        mid_x = round(ramp_x0 + 0.5 * ramp_dx, 3)
        mid_y = round(ramp_y0 + 0.5 * ramp_dy, 3)

        # Perpendicular offset from ramp surface: normal = (-sinθ, cosθ)
        ramp_angle_rad = math.atan2(ramp_dy, ramp_dx)
        nx = -math.sin(ramp_angle_rad)
        ny = math.cos(ramp_angle_rad)
        r = 0.7
        gap = 0.2
        sug_x = round(mid_x + (r + gap) * nx, 3)
        sug_y = min(round(mid_y + (r + gap) * ny, 3), round(5.0 - r - 0.1, 3))

        lines = [
            "=== Pass The Parcel Analysis ===",
            f"Ramp center: ({mid_x}, {mid_y})",
            # f"Suggested placement (r=0.6, perpendicular to ramp surface): x={sug_x}, y={sug_y}  [increase gap if overlap] try around the ramp center avoiding overlap with the ramp.",
        ]

        return "\n".join(lines)

    def dispatch_tool(self, tool_name: str, args: dict) -> str:
        """Dispatch a tool call by name and return the observation string."""
        tool_name = tool_name.strip().lower()

        if tool_name == "get_level_state":
            return self.get_level_state()
        elif tool_name == "simulate_action":
            return self.simulate_action(
                float(args.get("x", 0)),
                float(args.get("y", 0)),
                float(args.get("radius", 0.5)),
            )
        elif tool_name == "simulate_partial":
            # OSS models do not have access to simulate_partial; must use simulate_with_trace instead
            if self.is_oss:
                return f"ERROR: Tool 'simulate_partial' not available for OSS models. Use 'simulate_with_trace' instead: it accepts object_names parameter to trace specific objects."
            return self.simulate_partial(
                float(args.get("x", 0)),
                float(args.get("y", 0)),
                float(args.get("radius", 0.5)),
                int(args.get("stop_step", 50)),
            )
        elif tool_name == "get_contact_log":
            return self.get_contact_log()
        elif tool_name == "compute_gap_analysis":
            return self.compute_gap_analysis()
        elif tool_name == "compute_relative_positions":
            return self.compute_relative_positions()
        elif tool_name == "describe_scene_geometry":
            return self.describe_scene_geometry()
        elif tool_name == "simulate_with_trace":
            raw_names = args.get("object_names")
            if isinstance(raw_names, str):
                # Allow comma-separated strings from looser callers.
                object_names = [s.strip() for s in raw_names.split(",") if s.strip()]
            elif isinstance(raw_names, list):
                object_names = [str(s) for s in raw_names]
            else:
                object_names = None
            stop_step = args.get("stop_step")
            stop_step = int(stop_step) if stop_step is not None else None
            return self.simulate_with_trace(
                float(args.get("x", 0)),
                float(args.get("y", 0)),
                float(args.get("radius", 0.5)),
                object_names=object_names,
                n_samples=int(args.get("n_samples", 12)),
                stop_step=stop_step,
            )
        elif tool_name == "predict_first_contact":
            return self.predict_first_contact(
                float(args.get("x", 0)),
                float(args.get("y", 0)),
                float(args.get("radius", 0.5)),
            )
        elif tool_name == "trace_green_ball":
            return self.trace_green_ball(
                float(args.get("x", 0)),
                float(args.get("y", 0)),
                float(args.get("radius", 0.5)),
            )
        elif tool_name == "compute_intercept_setup":
            return self.compute_intercept_setup()
        elif tool_name == "compute_basket_analysis":
            return self.compute_basket_analysis()
        elif tool_name == "compute_cliffhanger_analysis":
            return self.compute_cliffhanger_analysis()
        elif tool_name == "compute_tipping_point_analysis":
            return self.compute_tipping_point_analysis()
        elif tool_name == "get_ramp_center":
            return self.get_ramp_center()
        elif tool_name == "finish":
            return "FINISH"
        else:
            available = "get_level_state, simulate_action, simulate_partial, get_contact_log, "
            if self.level_name == "two_body_problem":
                available += "compute_relative_positions, "
            elif self.level_name == "catapult":
                available += "describe_scene_geometry, simulate_with_trace, predict_first_contact, trace_green_ball, "
            elif self.level_name == "falling_into_place":
                available += "compute_intercept_setup, "
            elif self.level_name == "basket_case":
                available += "compute_basket_analysis, "
            elif self.level_name == "cliffhanger":
                available += "compute_cliffhanger_analysis, "
            elif self.level_name == "tipping_point":
                available += "compute_tipping_point_analysis, "
            elif self.level_name == "pass_the_parcel":
                available += "get_ramp_center, "
            else:
                available += "compute_gap_analysis, "
            available += "finish"
            return f"ERROR: Unknown tool '{tool_name}'. Available: {available}"

    def close(self):
        """Clean up the environment."""
        self.env.close()
