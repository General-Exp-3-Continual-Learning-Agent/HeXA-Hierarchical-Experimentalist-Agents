"""
Prompt templates for the Cliffhanger level.

Strategy-neutral: the puzzle admits multiple valid solutions (knock the bar off
the platform with a direct hit, drop a ball on top to topple it, ricochet off
the ceiling, ...). The prompts deliberately avoid prescribing any one of them.
"""

from .shared import SHARED_TOOLS, FINISH_TOOL, FINISH_AS_5

# ─── Cliffhanger ───────────────────────────────────────────

CLIFFHANGER_ANALYSIS_TOOL = """\
5. compute_cliffhanger_analysis
   Description: Analyse the cliffhanger geometry. Returns the green bar's centre, length, and the (x, y) coordinates of its bottom point (resting on the platform) and top point (opposite end); the platform's left/right extents and top-surface y; the ceiling y and purple-ground y; the bar's distance to each platform edge; which edge is closer (LEFT or RIGHT) — i.e. the edge the bar must fall past — and the width of the falling gap on that side.
   Arguments: None
   Usage: Action: compute_cliffhanger_analysis
"""

CLIFFHANGER_TOOL_DESCRIPTIONS = f"""\
You have access to the following tools to interact with the physics simulation:

{SHARED_TOOLS}
{CLIFFHANGER_ANALYSIS_TOOL}
{FINISH_TOOL}"""

# Ablation: cliffhanger-specific tool removed (level-tool ablation).
CLIFFHANGER_TOOL_DESCRIPTIONS_NO_LEVEL_TOOL = f"""\
You have access to the following tools to interact with the physics simulation:

{SHARED_TOOLS}
{FINISH_AS_5}"""

CLIFFHANGER_SYSTEM = """\
You are an expert physics reasoning agent solving a 2D physics puzzle. You have access to a physics simulator and can test your ideas before submitting a final answer.

**Puzzle: Cliffhanger**

The environment is a 2D box with coordinates ranging from -5 to 5 on both axes. Gravity pulls objects downward.

**Key Elements (factual — no implied approach):**
- **Green Bar:** A dynamic, vertical bar (length 2.0–3.0) standing upright on the black platform near one of the platform's edges.
- **Black Platform:** A static horizontal bar (length 4.0–6.0) at variable height y ∈ [-3, 0]; the green bar stands on top of it.
- **Ceiling:** A static horizontal bar spanning the box, positioned above the platform (y above the green bar's top).
- **Purple Ground:** The static floor at the bottom of the box (y ≈ -5).

**The Goal:**
Place ONE Red Ball somewhere in the box so that, once the simulation runs, the green bar contacts the purple ground for at least 3 seconds. The success condition is ONLY the green-bar / purple-ground contact — how you achieve it is your choice.

**Placement Constraints:**
- The red ball must be completely inside the box: -5 + radius <= x <= 5 - radius, -5 + radius <= y <= 5 - radius.
- The red ball must NOT overlap with any existing object at t=0.
- 0.1 <= radius <= 2.0

{tool_descriptions}

{react_format}"""

CLIFFHANGER_INITIAL = """\
Solve the Cliffhanger puzzle. You have 25 iterations to solve it. The success condition is: green_bar must contact purple_ground for at least 3 seconds. Use tools effectively and think about alternate approaches if one does not work.

Start by calling compute_cliffhanger_analysis — it returns the green bar's bottom and top point coordinates, the platform extents, which edge is closer, and the falling gap width — then inspect the level state for any extra context before placing the red ball."""


# ─── OSS-specific Cliffhanger prompts ──────────────────────

OSS_CLIFFHANGER_SYSTEM = """\
You are an expert physics reasoning agent solving a 2D physics puzzle.

Puzzle: Cliffhanger

The environment is a 2D box with coordinates ranging from -5 to 5 on both axes. Gravity pulls objects downward.

Key Elements (factual):
- Green Bar: dynamic vertical bar (length 2.0–3.0) standing upright on the black platform near one edge.
- Black Platform: static horizontal bar (length 4.0–6.0) at variable height y ∈ [-3, 0].
- Ceiling: static horizontal bar above the platform.
- Purple Ground: static floor at the bottom of the box (y ≈ -5).

Goal: Place ONE Red Ball so that green_bar contacts purple_ground for at least 3 seconds. The path to that outcome is yours to design.

Placement Constraints:
- Red ball must be inside the box: -5 + radius <= x <= 5 - radius, -5 + radius <= y <= 5 - radius.
- No overlap with existing objects at t=0.
- 0.1 <= radius <= 2.0

When your simulation shows green_bar in continuous contact with purple_ground for 3 seconds, call finish."""

OSS_CLIFFHANGER_INITIAL = """\
Solve the Cliffhanger puzzle. You have 25 iterations to solve it. The success condition is: green_bar must contact purple_ground for at least 3 seconds.

Start with compute_cliffhanger_analysis to get the green bar's bottom/top point coordinates, the platform extents, and the closer edge (the side the bar must topple past). Use simulate_action to test placements; when one succeeds, call finish."""
