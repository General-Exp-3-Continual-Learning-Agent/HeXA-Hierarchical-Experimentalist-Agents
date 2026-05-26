"""
Prompt templates for the Catapult level.

Strategy-neutral: the puzzle admits multiple valid solutions (lever launch, wall/ceiling
deflection, ceiling-ball ricochet, knocking the basket, using the arm as a bridge, ...).
The prompts and level-specific tools deliberately avoid prescribing any one of them.
"""

from .shared import SHARED_TOOLS, FINISH_TOOL, FINISH_AS_5

# ─── Catapult ───────────────────────────────────────────────

CATAPULT_SCENE_TOOL = """\
1. describe_scene_geometry
   Description: Return strategy-neutral geometry: every ball (position, radius, dynamic flag), every bar (position, angle, length, dynamic flag), every basket (position, dynamic flag), and the key distance (green ↔ blue). No prescriptive advice; you interpret the layout to form a strategy.
   Arguments: None
   Usage: Action: describe_scene_geometry
"""

CATAPULT_TRACE_TOOL = """\
6. simulate_with_trace
   Description: Place a red ball and run the simulation. Returns: success flag, contact events involving the red ball or YOUR chosen objects (via object_names), and per-object kinematic extrema (peak_y, min_y, max_speed, displacement, and angular stats for moving bars/baskets). You choose which objects to trace—e.g., ["green_ball"] to see if it launches, ["basket","blue_ball"] to see if the basket is disturbed, ["catapult_arm","green_ball"] to see how the lever moves the green ball.
   Arguments: x (float), y (float), radius (float), object_names (list of strings), n_samples (int, optional, unused), stop_step (int, optional, default = run to completion)
   Usage: Action: simulate_with_trace
          Action Input: {"x": 1.2, "y": 3.5, "radius": 0.6, "object_names": ["green_ball", "catapult_arm"]}
"""

CATAPULT_PREDICT_TOOL = """\
7. predict_first_contact
   Description: Cheap pre-simulation check (≤90 physics steps, ~1.5s of sim time). Runs just long enough to find the FIRST object the red ball touches after it is released, and reports: placement validity, the other object's name, the step of impact, approach speed, approximate contact point, and surface normal. Use this to verify that your red ball actually reaches the object you intended to hit BEFORE burning a full simulation budget.
   Arguments: x (float), y (float), radius (float)
   Usage: Action: predict_first_contact
          Action Input: {"x": 1.2, "y": 3.5, "radius": 0.6}
"""

CATAPULT_GREEN_TRACE_TOOL = """\
6. trace_green_ball
   Description: Lightweight trajectory probe — only the green ball is sampled. Places a red ball, runs the simulation, and returns the green ball's (x, y) waypoints at fixed step intervals plus start/end/peak summary. Stops early once the green ball comes to rest (capped at ~600 steps). Use this when you only care about WHERE the green ball travels, not contact events or other objects — much cheaper than simulate_with_trace.
   Arguments: x (float), y (float), radius (float)
   Usage: Action: trace_green_ball
          Action Input: {"x": 1.2, "y": 3.5, "radius": 0.6}
"""

CATAPULT_TOOL_DESCRIPTIONS = f"""\
You have access to the following tools to interact with the physics simulation:

{CATAPULT_SCENE_TOOL}
{SHARED_TOOLS}
{CATAPULT_GREEN_TRACE_TOOL}
{CATAPULT_PREDICT_TOOL}

9. finish
   Description: Submit your final answer. Use this when you are confident in your solution.
   Arguments: x (float), y (float), radius (float)
   Usage: Action: finish
          Action Input: {{"x": 0.5, "y": 4.0, "radius": 0.6}}"""

# Ablation: no catapult-specific tools (agent must rely on shared tools only).
CATAPULT_TOOL_DESCRIPTIONS_NO_LEVEL_TOOL = f"""\
You have access to the following tools to interact with the physics simulation:

{SHARED_TOOLS}
{FINISH_AS_5}"""

CATAPULT_SYSTEM = """\
You are an expert physics reasoning agent solving a 2D physics puzzle. You have access to a physics simulator and can test your ideas before submitting a final answer.

**Puzzle: Catapult**

The environment is a 2D box with coordinates ranging from -5 to 5 on both axes. Gravity pulls objects downward.

**Key Elements (factual — no implied approach):**
- **Green Ball:** A small dynamic ball sitting on the LEFT end of a gray bar.
- **Gray Bar (Catapult Arm):** A dynamic lever resting on a gray ball (pivot). The green ball sits on its left end.
- **Gray Ball (Pivot):** A dynamic ball acting as the fulcrum. It sits on the left black platform.
- **Black Ball (Ceiling Blocker):** A static ball near the top of the scene.
- **Black Platform (Left):** A static horizontal platform on the left side.
- **Black Ledge (Right):** A static (possibly angled) platform on the right side.
- **Basket (Gray):** A dynamic basket sitting on the right ledge.
- **Blue Ball (Target):** A dynamic ball inside the basket.

Use more radius for better energy (r>1)
**The Goal:**
Place ONE Red Ball somewhere in the box so that, once the simulation runs, the green ball contacts the blue ball for at least 3 seconds. The success condition is ONLY the green-blue contact — how you achieve it is your choice.

**Placement Constraints:**
- The red ball must be completely inside the box: -5 + radius <= x <= 5 - radius, -5 + radius <= y <= 5 - radius.
- The red ball must NOT overlap with any existing object at t=0.
- 0.1 <= radius <= 2.0


{tool_descriptions}

{react_format}"""

CATAPULT_INITIAL = """\
Solve the Catapult puzzle. You have 12 iterations to solve it. The success condition is: green_ball must contact blue_ball. Use tools effectively and think about alternate approaches if one does not work.
- **ALWAYS call predict_first_contact first** — it is cheap and tells you if your red ball hits the intended object.
- Only if predict confirms the right contact: call trace_green_ball to track green ball.
"""

# ─── OSS-specific Catapult prompts ─────────────────────────

OSS_CATAPULT_SYSTEM = """\
You are an expert physics reasoning agent solving a 2D physics puzzle.

Puzzle: Catapult

The environment is a 2D box with coordinates ranging from -5 to 5 on both axes. Gravity pulls objects downward.

Key Elements (factual):
- Green Ball: small dynamic ball on the LEFT end of the gray bar.
- Gray Bar: dynamic lever resting on a gray ball pivot; green ball sits on its left end.
- Gray Ball: dynamic fulcrum on the left black platform.
- Black Ball: static ball near the ceiling.
- Black Platform (Left): static left-side platform.
- Black Ledge (Right): static (possibly angled) right-side platform.
- Basket (Gray): dynamic basket on the right ledge.
- Blue Ball: dynamic ball inside the basket.

Goal: Place ONE Red Ball so that green_ball contacts blue_ball for at least 3 seconds. The path to that outcome is yours to design.

Placement Constraints:
- Red ball must be inside the box: -5 + radius <= x <= 5 - radius, -5 + radius <= y <= 5 - radius.
- No overlap with existing objects at t=0.
- 0.1 <= radius <= 2.0

When your trace shows green_ball in continuous contact with blue_ball for 3 seconds, call finish."""

OSS_CATAPULT_INITIAL = """\
Solve the Catapult puzzle. You have 25 iterations to solve it. The success condition is: green_ball must contact blue_ball. Use tools effectively and think about alternate approaches if one does not work.
- **ALWAYS call predict_first_contact first** — it is cheap and tells you if your red ball hits the intended object.
- Only if predict confirms the right contact: call trace_green_ball to track green ball."""

# Ablation blocks for Catapult — retained for explicit strategy-1 experiments only.
# These are NOT included in the default prompts above.
_OSS_CATAPULT_MECHANICS_BLOCK = """
Catapult Mechanics (strategy-1 ablation):
- Drop red ball on the RIGHT side of the catapult arm (right of pivot) to launch green ball.
- Right arm goes down → left arm (with green ball) goes UP → green ball launches toward basket.
"""

_OSS_CATAPULT_FAILURE_MODES_BLOCK = """
Lever mechanics — two independent controls (strategy-1 ablation):
- x position on arm (distance from pivot) = LAUNCH ANGLE. Closer to pivot → steeper. Farther (toward tip) → flatter.
- Drop height × radius = LAUNCH SPEED. Height = impact velocity. Radius = mass. Raise height first, radius second.

Mandatory failure diagnosis — after EVERY failure, classify WHY:
- wrong_direction: green went left → move x further RIGHT of pivot.
- arc_too_short: green fell short → increase energy (more height or larger radius).
- arc_too_flat: green reached right x but hit floor → move x CLOSER to pivot for steeper arc.
- overshot: green flew past basket → reduce energy or move x closer to pivot.

Exploration rule: after 2 failures in same zone, switch zone. Never repeat exact (x, y, radius).
"""

_OSS_CATAPULT_STRATEGY_BLOCK = """
Strategy (strategy-1 ablation):
- Call describe_scene_geometry to map the scene.
- Drop red ball RIGHT of pivot, above arm.
- Call simulate_with_trace with ["green_ball","catapult_arm"] — if green_y is below arm_top mid-flight, launch is too flat.
- Classify each failure, apply matching fix, switch zones if stuck.
- When it succeeds, call finish."""
