"""
Prompt templates for the Tipping Point level.

A vertical green bar rests in a small gray basket on the ground. Somewhere on
the LEFT or RIGHT side of the box there is a static purple wall. The agent
places one red ball so the green bar tips over and contacts the purple wall
for the success-time duration.
"""

from .shared import SHARED_TOOLS, FINISH_TOOL, FINISH_AS_5

# ─── Tipping Point ─────────────────────────────────────────

TIPPING_POINT_ANALYSIS_TOOL = """\
5. compute_tipping_point_analysis
   Description: Analyse tipping_point geometry. Returns the green bar's centre, length, angle, and the (x, y) coordinates of its top and bottom endpoints; the basket's centre and floor; the purple wall's x position and its top/bottom y; the purple wall's side relative to the green bar (LEFT or RIGHT); the horizontal distance from the bar's centre to the wall; the angle the bar must tip through to touch the wall (assuming it pivots near the basket); and the suggested tip direction (LEFT or RIGHT).
   Arguments: None
   Usage: Action: compute_tipping_point_analysis
"""

TIPPING_POINT_TOOL_DESCRIPTIONS = f"""\
You have access to the following tools to interact with the physics simulation:

{SHARED_TOOLS}
{TIPPING_POINT_ANALYSIS_TOOL}
{FINISH_TOOL}"""

# Ablation: tipping_point-specific tool removed (level-tool ablation).
TIPPING_POINT_TOOL_DESCRIPTIONS_NO_LEVEL_TOOL = f"""\
You have access to the following tools to interact with the physics simulation:

{SHARED_TOOLS}
{FINISH_AS_5}"""

TIPPING_POINT_SYSTEM = """\
You are an expert physics reasoning agent solving a 2D physics puzzle. You have access to a physics simulator and can test your ideas before submitting a final answer.

**Puzzle: Tipping Point**

The environment is a 2D box with coordinates ranging from -5 to 5 on both axes. Gravity pulls objects downward.

**Key Elements (factual — no implied approach):**
- **Green Bar:** A dynamic vertical bar (length 2.0–5.0) resting upright inside a small gray basket on the ground. Its centre of mass sits above the basket floor; small horizontal pushes at the top can topple it.
- **Gray Basket:** A small dynamic basket sitting near the bottom of the box. The green bar rests inside it; the basket can slide if struck hard.
- **Purple Wall:** A static vertical bar (top ≈ 5, bottom ≈ -5) standing flush against either the LEFT side (x ≈ -4.9) or the RIGHT side (x ≈ 4.9) of the box. The bar must contact this wall.
- **Box Walls / Floor:** Standard box boundary at x = ±5, y = ±5.

**The Goal:**
Place ONE Red Ball somewhere in the box so that, once the simulation runs, the green bar contacts the purple wall and stays in contact for the success-time duration. The success condition is ONLY the green-bar / purple-wall contact — how you achieve it is your choice.

**Placement Constraints:**
- The red ball must be completely inside the box: -5 + radius <= x <= 5 - radius, -5 + radius <= y <= 5 - radius.
- The red ball must NOT overlap with any existing object at t=0.
- 0.1 <= radius <= 2.0

{tool_descriptions}

{react_format}"""

TIPPING_POINT_INITIAL = """\
Solve the Tipping Point puzzle. You have 25 iterations to solve it. The success condition is: green_bar must contact purple_wall for the success-time duration. Use tools effectively and think about alternate approaches if one does not work.

Start by calling compute_tipping_point_analysis — it returns the green bar's endpoints, the purple wall's position, which side of the bar the wall is on, the suggested tip direction, and a starting placement region. """


# ─── OSS-specific Tipping Point prompts ────────────────────

OSS_TIPPING_POINT_SYSTEM = """\
You are an expert physics reasoning agent solving a 2D physics puzzle.

Puzzle: Tipping Point

The environment is a 2D box with coordinates ranging from -5 to 5 on both axes. Gravity pulls objects downward.

Key Elements (factual):
- Green Bar: dynamic vertical bar (length 2.0–5.0) resting upright in a small gray basket on the ground.
- Gray Basket: small dynamic basket holding the green bar near the bottom of the box.
- Purple Wall: static vertical bar flush against the LEFT (x ≈ -4.9) or RIGHT (x ≈ 4.9) side of the box; spans top to bottom.
- Box bounds: x ∈ [-5, 5], y ∈ [-5, 5].

Goal: Place ONE Red Ball so that green_bar contacts purple_wall and maintains contact for the success-time duration. The path to that outcome is yours to design.

Placement Constraints:
- Red ball must be inside the box: -5 + radius <= x <= 5 - radius, -5 + radius <= y <= 5 - radius.
- No overlap with existing objects at t=0.
- 0.1 <= radius <= 2.0

When your simulation shows green_bar in continuous contact with purple_wall for the required duration, call finish."""

OSS_TIPPING_POINT_INITIAL = """\
Solve the Tipping Point puzzle. You have 25 iterations to solve it. The success condition is: green_bar must contact purple_wall for the success-time duration.

Start with compute_tipping_point_analysis to get the green bar's endpoints, the purple wall's x position, which side of the bar the wall is on, the suggested tip direction, and a starting placement region. The basket pins the bar's base, so applying force near the bar's TOP end (typically by dropping the red ball from above) tips it much more efficiently than placements at basket level. Use simulate_action to test placements; when one succeeds, call finish."""
