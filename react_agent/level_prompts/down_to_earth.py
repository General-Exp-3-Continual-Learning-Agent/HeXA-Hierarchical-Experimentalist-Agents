"""
Prompt templates for the Down to Earth level.
"""

from .shared import SHARED_TOOLS, FINISH_TOOL, FINISH_AS_5

# ─── Down to Earth ─────────────────────────────────────────

DOWN_TO_EARTH_ANALYSIS_TOOL = """\
5. compute_gap_analysis
   Description: Analyze the gaps on each side of the platform.
    Returns the left gap and right gap, and whether the green ball can fit through each gap.
   Arguments: None
   Usage: Action: compute_gap_analysis
"""

DOWN_TO_EARTH_TOOL_DESCRIPTIONS = f"""\
You have access to the following tools to interact with the physics simulation:

{SHARED_TOOLS}
{DOWN_TO_EARTH_ANALYSIS_TOOL}
{FINISH_TOOL}"""

# Ablation: no compute_gap_analysis
DOWN_TO_EARTH_TOOL_DESCRIPTIONS_NO_LEVEL_TOOL = f"""\
You have access to the following tools to interact with the physics simulation:

{SHARED_TOOLS}
{FINISH_AS_5}"""

DOWN_TO_EARTH_SYSTEM = """\
You are an expert physics reasoning agent solving a 2D physics puzzle. You have access to a physics simulator and can test your ideas before submitting a final answer.

**Puzzle: Down to Earth**

The environment is a 2D box with coordinates ranging from -5 to 5 on both axes. Gravity pulls objects downward.

**Key Elements:**
- **Green Ball (Target):** A dynamic ball that will fall due to gravity.
- **Black High Platform:** A static horizontal platform below the green ball. Without intervention, the green ball lands on this platform and stays there.
- **Purple Ground:** The floor at the very bottom of the box (y ~ -5).

**The Goal:**
Introduce a Red Ball into the scene so that the Green Ball is knocked off the platform and touches the Purple Ground for at least 3 seconds.

**Placement Constraints:**
- The red ball must be completely inside the box: -5 + radius <= x <= 5 - radius, -5 + radius <= y <= 5 - radius.
- The red ball must NOT overlap with the green ball at t=0: distance between centers > sum of radii.
- The red ball must NOT overlap with the black platform at t=0.

{tool_descriptions}

{react_format}"""

DOWN_TO_EARTH_INITIAL = """\
Solve the Down to Earth puzzle. You have exactly 25 iterations to solve the puzzle. Start by inspecting the level state, then figure out where to place the red ball to knock the green ball off the platform and onto the purple ground.

Remember: use the tools to test your ideas before submitting your final answer with the finish tool."""


# ─── OSS-specific Down to Earth prompts ────────────────────

OSS_DOWN_TO_EARTH_SYSTEM = """\
You are an expert physics reasoning agent solving a 2D physics puzzle.

Puzzle: Down to Earth

The environment is a 2D box with coordinates ranging from -5 to 5 on both axes. Gravity pulls objects downward.

Key Elements:
- Green Ball (Target): A dynamic ball that will fall due to gravity.
- Black High Platform: A static horizontal platform below the green ball. Without intervention, the green ball lands on this platform and stays there.
- Purple Ground: The floor at the very bottom of the box (y ~ -5).

The Goal:
Introduce a Red Ball into the scene so that the Green Ball is knocked off the platform and touches the Purple Ground for at least 3 seconds.



Placement Constraints:
- The red ball must be completely inside the box: -5 + radius <= x <= 5 - radius, -5 + radius <= y <= 5 - radius.
- The red ball must NOT overlap with the green ball at t=0: distance between centers > sum of radii.
- The red ball must NOT overlap with the black platform at t=0.
"""

OSS_DOWN_TO_EARTH_INITIAL = """\
Solve the Down to Earth puzzle. Start by calling get_level_state to inspect the level, then call compute_gap_analysis to find which side has a wider gap.

Use simulate_action to test placements. When one succeeds, call finish with those coordinates."""

# Ablation blocks for Down to Earth
_OSS_DTE_STRATEGY_BLOCK = """
Strategy:
- First, call get_level_state to understand the scene layout.
- Call compute_gap_analysis to determine which side the green ball can fall off.
- The red ball should collide with the green ball on the platform to push it toward the wider gap.
- Call simulate_action to test placements and observe what happens.
- If a simulation fails, analyze WHY and adjust your approach.
- When a simulation succeeds, call finish with those coordinates."""

_OSS_DTE_PHYSICS_LAST_TWO = (
    "\n- To push the green ball RIGHT, place the red ball to its LEFT. To push LEFT, place to the RIGHT."
    "\n- The green ball can only fall off the platform if the gap between the platform edge and the wall is wider than the green ball's diameter."
)

# Non-OSS ablation blocks
_DTE_STRATEGY_BLOCK = """**Strategy Tips:**
- First, use get_level_state to understand the scene layout.
- Use compute_gap_analysis to determine which side the green ball can fall off.
- The red ball should collide with the green ball on the platform to push it toward the wider gap.
- Use simulate_action to test your ideas and observe what happens.
- If a simulation fails, analyze WHY and adjust your approach."""
