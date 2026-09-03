"""
Interphyre tool server for verl-tool RL training.
Wraps the Interphyre physics puzzle environment as a ReAct-style tool.

Supported levels: down_to_earth, two_body_problem, cliffhanger, tipping_point, catapult

Action format expected from the LLM:
    Thought: <reasoning>
    Action: <tool_name>
    Action Input: <json_args>

Stop token: "\nObservation:" — generation halts here and the tool injects the result.
"""

import sys
import os
import json
import re
import math
import logging
from typing import Tuple, Dict, Any, Union

from .base import BaseTool, register_tool

logger = logging.getLogger(__name__)

# Add repo root to path so `interphyre` package is importable
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Regex to parse ReAct format from LLM output
_ACTION_RE = re.compile(r"Action:\s*([A-Za-z_][A-Za-z0-9_]*)", re.DOTALL)
_INPUT_RE = re.compile(r"Action Input:\s*(\{.*?\})", re.DOTALL | re.IGNORECASE)

LEVEL_NAMES = ["down_to_earth", "two_body_problem", "cliffhanger", "tipping_point", "catapult"]

AVAILABLE_TOOLS = (
    "get_level_state, simulate_action, simulate_partial, get_contact_log, "
    "compute_gap_analysis, compute_relative_positions, compute_tipping_analysis, "
    "compute_wall_distance_analysis, compute_catapult_analysis, finish"
)


# ---------------------------------------------------------------------------
# Local toolkit (mirrors react_agent/tools.py; avoids cross-repo imports)
# ---------------------------------------------------------------------------

class _PhysicsToolkit:
    """Wraps InterphyreEnv with all the ReAct tool methods."""

    def __init__(self, level_name: str, seed: int = 42):
        from interphyre import InterphyreEnv
        self.level_name = level_name
        self.seed = seed
        self.env = InterphyreEnv(level_name, seed=seed)
        self._level_info = self._extract_level_info()
        self._action_objects = list(self.env.level.action_objects)

    def _extract_level_info(self) -> dict:
        info = {}
        level = self.env.level
        for obj_name, obj in level.objects.items():
            obj_info = {"x": obj.x, "y": obj.y, "color": obj.color, "dynamic": obj.dynamic}
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
            elif hasattr(obj, "bottom_width"):
                obj_info["type"] = "Basket"
                obj_info["bottom_width"] = getattr(obj, "bottom_width", None)
                obj_info["top_width"] = getattr(obj, "top_width", None)
                obj_info["height"] = getattr(obj, "height", None)
                obj_info["wall_thickness"] = getattr(obj, "wall_thickness", None)
                obj_info["floor_thickness"] = getattr(obj, "floor_thickness", None)
            else:
                obj_info["type"] = type(obj).__name__
            info[obj_name] = obj_info
        return info

    def _reset_env(self):
        from interphyre import InterphyreEnv
        self.env = InterphyreEnv(self.level_name, seed=self.seed)

    # ---- tool implementations ----

    def get_level_state(self) -> str:
        lines = [f"=== Level State: {self.level_name} ===", ""]
        for name, obj in self._level_info.items():
            if name in self._action_objects:
                continue
            if obj["type"] == "Ball":
                lines.append(
                    f"* {name} ({obj['color']} {obj['type']}): "
                    f"position=({obj['x']:.4f}, {obj['y']:.4f}), "
                    f"radius={obj['radius']:.4f}, dynamic={obj['dynamic']}"
                )
            elif obj["type"] == "Bar":
                top_v = f"{obj['top']:.4f}" if obj["top"] is not None else "N/A"
                bot_v = f"{obj['bottom']:.4f}" if obj["bottom"] is not None else "N/A"
                lft_v = f"{obj['left']:.4f}" if obj["left"] is not None else "N/A"
                rgt_v = f"{obj['right']:.4f}" if obj["right"] is not None else "N/A"
                lines.append(
                    f"* {name} ({obj['color']} {obj['type']}): "
                    f"position=({obj['x']:.4f}, {obj['y']:.4f}), "
                    f"left={lft_v}, right={rgt_v}, top={top_v}, bottom={bot_v}, "
                    f"thickness={obj['thickness']:.4f}, dynamic={obj['dynamic']}"
                )
            elif obj["type"] == "Basket":
                lines.append(
                    f"* {name} ({obj['color']} {obj['type']}): "
                    f"position=({obj['x']:.4f}, {obj['y']:.4f}), "
                    f"bottom_width={obj['bottom_width']:.4f}, "
                    f"top_width={obj['top_width']:.4f}, height={obj['height']:.4f}, "
                    f"dynamic={obj['dynamic']}"
                )
            else:
                lines.append(f"* {name}: {obj}")
        lines.append("")
        lines.append("World bounds: x in [-5, 5], y in [-5, 5]")
        lines.append(f"Action objects: {self._action_objects} -- you control the red ball placement")
        if self.level_name in ("two_body_problem", "catapult"):
            lines.append("Success condition: green_ball must touch blue_ball for 3 seconds")
        elif self.level_name == "cliffhanger":
            lines.append("Success condition: green_bar must touch purple_ground for 3 seconds")
        elif self.level_name == "tipping_point":
            lines.append("Success condition: green_bar must touch purple_wall for 3 seconds")
        else:
            lines.append("Success condition: green_ball must touch purple_ground for 3 seconds")
        return "\n".join(lines)

    def simulate_action(self, x: float, y: float, radius: float) -> str:
        self._reset_env()
        action = [(x, y, radius)]
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
        except Exception as e:
            return f"ERROR: Simulation failed -- {e}"

        if info.get("invalid_action", False):
            error = info.get("validation_error", "Unknown validation error")
            return f"INVALID ACTION: {error}\nFix the placement and try again."

        success = info.get("success", False)
        step_count = info.get("step_count", "unknown")
        lines = []
        if success:
            if self.level_name in ("two_body_problem", "catapult"):
                lines.append("SUCCESS! The green ball contacted the blue ball.")
            elif self.level_name == "cliffhanger":
                lines.append("SUCCESS! The green bar fell and contacted the purple ground.")
            elif self.level_name == "tipping_point":
                lines.append("SUCCESS! The green bar tipped over and contacted the purple wall.")
            else:
                lines.append("SUCCESS! The green ball reached the purple ground.")
        else:
            if self.level_name in ("two_body_problem", "catapult"):
                lines.append("FAILURE. The green ball did NOT contact the blue ball.")
            elif self.level_name == "cliffhanger":
                lines.append("FAILURE. The green bar did NOT contact the purple ground.")
            elif self.level_name == "tipping_point":
                lines.append("FAILURE. The green bar did NOT contact the purple wall.")
            else:
                lines.append("FAILURE. The green ball did NOT reach the purple ground.")
        lines.append(f"Total simulation steps: {step_count}")
        lines.append(f"Reward: {reward}")
        if isinstance(obs, dict) and "objects" in obs:
            lines.append("")
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
            return f"ERROR: Partial simulation failed -- {e}"

        lines = [f"Simulation state at step {actual_steps}:", ""]
        try:
            for obj_name in self.env._level.objects:
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
        try:
            log = self.env.get_contact_log()
        except Exception as e:
            return f"ERROR: Could not retrieve contact log -- {e}"
        if not log:
            return "No contact events recorded. (Run a simulation first with simulate_action.)"
        lines = ["Contact events:"]
        for entry in log[:20]:
            lines.append(f"  {entry}")
        if len(log) > 20:
            lines.append(f"  ... and {len(log) - 20} more events")
        return "\n".join(lines)

    def _find_platform(self):
        for name, obj in self._level_info.items():
            if obj["type"] == "Bar" and not obj["dynamic"] and obj.get("color") != "purple":
                return name, obj
        return None, None

    def _find_green_ball(self):
        for name, obj in self._level_info.items():
            if obj["type"] == "Ball" and obj.get("color") == "green":
                return name, obj
        return None, None

    def _find_blue_ball(self):
        for name, obj in self._level_info.items():
            if obj["type"] == "Ball" and obj.get("color") == "blue":
                return name, obj
        return None, None

    def _find_green_bar(self):
        for name, obj in self._level_info.items():
            if obj["type"] == "Bar" and obj.get("color") == "green":
                return name, obj
        return None, None

    def _find_purple_wall(self):
        for name, obj in self._level_info.items():
            if obj["type"] == "Bar" and obj.get("color") == "purple" and not obj["dynamic"]:
                top = obj.get("top")
                bottom = obj.get("bottom")
                if top is not None and bottom is not None and (top - bottom) > 5.0:
                    return name, obj
        return None, None

    def _find_basket(self):
        for name, obj in self._level_info.items():
            if obj["type"] == "Basket":
                return name, obj
        return None, None

    def compute_gap_analysis(self) -> str:
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
            f"  Green ball can fall left: {'YES' if left_gap > green_diameter else 'NO'}",
            "",
            f"Right gap (platform right edge to right wall): {right_gap:.4f}",
            f"  Green ball can fall right: {'YES' if right_gap > green_diameter else 'NO'}",
        ]
        return "\n".join(lines)

    def compute_relative_positions(self) -> str:
        _, green = self._find_green_ball()
        _, blue = self._find_blue_ball()
        if not green or not blue:
            return "ERROR: Could not find green ball or blue ball in level."
        dx = blue["x"] - green["x"]
        dy = blue["y"] - green["y"]
        dist = math.sqrt(dx ** 2 + dy ** 2)
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
            f"Blue is to the {'RIGHT' if dx > 0 else 'LEFT'} of Green.",
        ]
        return "\n".join(lines)

    def compute_tipping_analysis(self) -> str:
        _, green_bar = self._find_green_bar()
        _, platform = self._find_platform()
        if not green_bar:
            return "ERROR: No green bar found in this level."
        if not platform:
            return "ERROR: No platform found in this level."
        plat_left = platform["left"] if platform["left"] is not None else platform["x"] - platform["length"] / 2
        plat_right = platform["right"] if platform["right"] is not None else platform["x"] + platform["length"] / 2
        bar_x = green_bar["x"]
        dist_to_left = bar_x - plat_left
        dist_to_right = plat_right - bar_x
        closer = "left" if dist_to_left < dist_to_right else "right"
        lines = [
            "=== Tipping Analysis ===",
            f"Green bar: position=({green_bar['x']:.4f}, {green_bar['y']:.4f}), "
            f"length={green_bar.get('length', 'N/A')}, angle={green_bar.get('angle', 90):.1f}°",
            f"Platform spans: x in [{plat_left:.4f}, {plat_right:.4f}]",
            "",
            f"Distance to left edge: {dist_to_left:.4f}",
            f"Distance to right edge: {dist_to_right:.4f}",
            f"Closer edge: {closer}",
        ]
        return "\n".join(lines)

    def compute_wall_distance_analysis(self) -> str:
        _, green_bar = self._find_green_bar()
        _, wall = self._find_purple_wall()
        _, basket = self._find_basket()
        if not green_bar:
            return "ERROR: Could not find green bar in level."
        if not wall:
            return "ERROR: Could not find purple wall in level."
        wall_x = wall["x"]
        bar_x = green_bar["x"]
        bar_length = green_bar.get("length", 2.0)
        distance_to_wall = abs(wall_x - bar_x)
        wall_side = "LEFT" if wall_x < bar_x else "RIGHT"
        lines = [
            "=== Wall Distance Analysis ===",
            f"Green bar: position=({green_bar['x']:.4f}, {green_bar['y']:.4f}), "
            f"length={bar_length:.4f}, angle={green_bar.get('angle', 90):.1f}°",
            f"Purple wall: x={wall_x:.4f} ({wall_side} side)",
        ]
        if basket:
            lines.append(f"Basket: position=({basket['x']:.4f}, {basket['y']:.4f})")
        lines += [
            "",
            f"Horizontal distance from bar to wall: {distance_to_wall:.4f}",
        ]
        return "\n".join(lines)

    def compute_catapult_analysis(self) -> str:
        _, green = self._find_green_ball()
        _, blue = self._find_blue_ball()
        _, basket = self._find_basket()
        if not green:
            return "ERROR: No green ball found."
        if not blue:
            return "ERROR: No blue ball found."
        catapult_bar = next(
            (obj for _, obj in self._level_info.items()
             if obj["type"] == "Bar" and obj["dynamic"] and obj.get("color") == "gray"),
            None
        )
        pivot = next(
            (obj for _, obj in self._level_info.items()
             if obj["type"] == "Ball" and obj["dynamic"] and obj.get("color") == "gray"),
            None
        )
        dx = blue["x"] - green["x"]
        dy = blue["y"] - green["y"]
        dist = math.sqrt(dx ** 2 + dy ** 2)
        lines = [
            "=== Catapult Analysis ===",
            f"Green ball (launch): ({green['x']:.4f}, {green['y']:.4f}), radius={green['radius']:.4f}",
            f"Blue ball (target):  ({blue['x']:.4f}, {blue['y']:.4f}), radius={blue['radius']:.4f}",
        ]
        if basket:
            lines.append(f"Basket: position=({basket['x']:.4f}, {basket['y']:.4f})")
        if catapult_bar:
            lines.append(
                f"Catapult bar: position=({catapult_bar['x']:.4f}, {catapult_bar['y']:.4f}), "
                f"left={catapult_bar.get('left')}, right={catapult_bar.get('right')}"
            )
        if pivot:
            lines.append(f"Pivot ball: ({pivot['x']:.4f}, {pivot['y']:.4f}), radius={pivot['radius']:.4f}")
        lines += [
            "",
            f"Horizontal distance (green to blue): {dx:.4f}",
            f"Vertical distance (green to blue): {dy:.4f}",
            f"Straight-line distance: {dist:.4f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# verl-tool BaseTool subclass
# ---------------------------------------------------------------------------

@register_tool
class InterphyreTool(BaseTool):
    """Tool server for Interphyre physics puzzle environments."""

    tool_type = "interphyre"
    # Stop token: after the model writes "Action Input: {...}" it will write
    # "\nObservation:" — we intercept there and inject the real observation.
    stop_tokens = ["\nObservation:"]

    def __init__(self, num_workers: int = 1, **kwargs):
        super().__init__(num_workers)
        # env_cache: trajectory_id -> {"toolkit": _PhysicsToolkit, "turns": int}

    def get_usage_inst(self) -> str:
        return (
            "Use the following tools to solve the physics puzzle.\n"
            "Format each step as:\n"
            "  Thought: <reasoning>\n"
            "  Action: <tool_name>\n"
            "  Action Input: <json>\n"
            "Available tools: " + AVAILABLE_TOOLS
        )

    # ---- action parsing ----

    def parse_action(self, action: str) -> Tuple[str, bool]:
        """Check that action contains a parseable Action: line."""
        m = _ACTION_RE.search(action)
        if not m:
            return action, False
        return action, True

    # ---- core execution ----

    def conduct_action(
        self,
        trajectory_id: str,
        action: str,
        extra_field: Dict[str, Any],
    ) -> Tuple[str, bool, bool]:
        """
        Execute one ReAct step.
        Returns: (observation, done, valid)
        """
        extra_field = extra_field or {}
        level_name = extra_field.get("level_name", "down_to_earth")
        if level_name not in LEVEL_NAMES:
            logger.warning(f"Unknown level '{level_name}', defaulting to down_to_earth")
            level_name = "down_to_earth"
        seed = int(extra_field.get("seed", 42))

        # Parse tool name
        action_match = _ACTION_RE.search(action)
        if not action_match:
            obs = (
                "ERROR: Could not parse your action. Use the format:\n"
                "  Action: <tool_name>\n"
                "  Action Input: {\"key\": value}\n"
                f"Available tools: {AVAILABLE_TOOLS}"
            )
            return obs, False, False

        tool_name = action_match.group(1).strip().lower()

        # Parse JSON args (empty dict if no args needed)
        input_match = _INPUT_RE.search(action)
        args = {}
        if input_match:
            try:
                raw_json = input_match.group(1)
                # Fix bare decimals like 2. or .5 that are invalid JSON
                raw_json = re.sub(r'(\d+)\.\s*([,}\]])', r'\1.0\2', raw_json)
                raw_json = re.sub(r':\s*(\d+)\.\s*$', r': \1.0', raw_json)
                args = json.loads(raw_json)
            except json.JSONDecodeError as e:
                obs = f"ERROR: Invalid JSON in Action Input: {e}\nRaw input: {input_match.group(1)}\nHint: use 2.0 not 2."
                return obs, False, False

        # Lazy-init toolkit for this trajectory
        if trajectory_id not in self.env_cache:
            try:
                toolkit = _PhysicsToolkit(level_name, seed=seed)
            except Exception as e:
                return f"ERROR: Failed to initialize environment for level '{level_name}': {e}", True, False
            self.env_cache[trajectory_id] = {"toolkit": toolkit, "turns": 0}

        state = self.env_cache[trajectory_id]
        toolkit: _PhysicsToolkit = state["toolkit"]
        state["turns"] += 1

        done = False
        valid = True
        obs = ""

        try:
            if tool_name == "finish":
                x = float(args.get("x", 0))
                y = float(args.get("y", 0))
                radius = float(args.get("radius", 0.5))
                sim_result = toolkit.simulate_action(x, y, radius)
                # Prefix with [FINAL_RESULT] so the reward manager can find it
                obs = f"[FINAL_RESULT]\n{sim_result}"
                done = True
                self.delete_env(trajectory_id)

            elif tool_name == "get_level_state":
                obs = toolkit.get_level_state()

            elif tool_name == "simulate_action":
                x = float(args.get("x", 0))
                y = float(args.get("y", 0))
                radius = float(args.get("radius", 0.5))
                obs = toolkit.simulate_action(x, y, radius)

            elif tool_name == "simulate_partial":
                x = float(args.get("x", 0))
                y = float(args.get("y", 0))
                radius = float(args.get("radius", 0.5))
                stop_step = int(args.get("stop_step", 50))
                obs = toolkit.simulate_partial(x, y, radius, stop_step)

            elif tool_name == "get_contact_log":
                obs = toolkit.get_contact_log()

            elif tool_name == "compute_gap_analysis":
                obs = toolkit.compute_gap_analysis()

            elif tool_name == "compute_relative_positions":
                obs = toolkit.compute_relative_positions()

            elif tool_name == "compute_tipping_analysis":
                obs = toolkit.compute_tipping_analysis()

            elif tool_name == "compute_wall_distance_analysis":
                obs = toolkit.compute_wall_distance_analysis()

            elif tool_name == "compute_catapult_analysis":
                obs = toolkit.compute_catapult_analysis()

            else:
                obs = f"ERROR: Unknown tool '{tool_name}'.\nAvailable: {AVAILABLE_TOOLS}"
                valid = False

        except Exception as e:
            obs = f"ERROR: Tool '{tool_name}' raised an exception: {e}"
            valid = False

        return obs, done, valid
