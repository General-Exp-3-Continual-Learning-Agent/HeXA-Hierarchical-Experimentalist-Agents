"""
Prompt templates for the Two Body Problem level.
"""

from .shared import SHARED_TOOLS, FINISH_TOOL, FINISH_AS_5

# ─── Two Body Problem ──────────────────────────────────────

TWO_BODY_ANALYSIS_TOOL = """\
5. compute_relative_positions
   Description: Analyze the positions of the green and blue balls.
   Returns their coordinates, distance, on which side the blue ball is
    relative to green, and recommended red ball placement direction.
   Arguments: None
   Usage: Action: compute_relative_positions
"""

TWO_BODY_TOOL_DESCRIPTIONS = f"""\
You have access to the following tools to interact with the physics simulation:

{SHARED_TOOLS}
{TWO_BODY_ANALYSIS_TOOL}
{FINISH_TOOL}"""

# Ablation: no compute_relative_positions
TWO_BODY_TOOL_DESCRIPTIONS_NO_LEVEL_TOOL = f"""\
You have access to the following tools to interact with the physics simulation:

{SHARED_TOOLS}
{FINISH_AS_5}"""

TWO_BODY_SYSTEM = """\
You are an expert physics reasoning agent solving a 2D physics puzzle. You have access to a physics simulator and can test your ideas before submitting a final answer.

**Puzzle: Two Body Problem**

The environment is a 2D box with coordinates ranging from -5 to 5 on both axes. Gravity pulls objects downward.

**Key Elements:**
- **Green Ball:** A dynamic ball.
- **Blue Ball:** A dynamic ball, separated horizontally from the green ball.
- Both balls fall under gravity from rest.

**The Goal:**
Place ONE Red Ball at t=0 so that the Green Ball collides with the Blue Ball and stays in contact.

**Placement Constraints:**
- The red ball must be completely inside the box: -5 + radius <= x <= 5 - radius, -5 + radius <= y <= 5 - radius.
- The red ball must NOT overlap with the green ball or the blue ball at t=0.
- 0.1 <= radius <= 1.5

{tool_descriptions}

{react_format}"""

TWO_BODY_INITIAL = """\
Solve the Two Body Problem puzzle. You have 25 iterations to solve the puzzle. Start by inspecting the level state with get_level_state, then use compute_relative_positions to understand where the blue ball is relative to green.

Remember: use the tools to test your ideas before submitting your final answer with the finish tool."""


# ─── OSS-specific Two Body Problem prompts ─────────────────

OSS_TWO_BODY_SYSTEM = """\
You are an expert physics reasoning agent solving a 2D physics puzzle.

Puzzle: Two Body Problem

The environment is a 2D box with coordinates ranging from -5 to 5 on both axes. Gravity pulls objects downward.

Key Elements:
- Green Ball: A dynamic ball.
- Blue Ball: A dynamic ball, separated horizontally from the green ball.
- Both balls fall under gravity from rest.

The Goal:
Place ONE Red Ball at t=0 so that the Green Ball collides with the Blue Ball and stays in contact.

Placement Constraints:
- The red ball must be completely inside the box: -5 + radius <= x <= 5 - radius, -5 + radius <= y <= 5 - radius.
- The red ball must NOT overlap with the green ball or the blue ball at t=0.
- 0.1 <= radius <= 1.5
"""

OSS_TWO_BODY_INITIAL = """\
Solve the Two Body Problem.You have 25 iterations to solve. Start by calling get_level_state, then call compute_relative_positions to see where blue is relative to green.

"""

# Ablation blocks for Two Body Problem
_OSS_TBP_HOW_TO_PUSH_BLOCK = """
How to Push Toward Blue:
1. Determine if Blue is left or right of Green.
2. If Blue is to the RIGHT: place Red slightly LEFT of Green and ABOVE it.
3. If Blue is to the LEFT: place Red slightly RIGHT of Green and ABOVE it.
This creates a diagonal impact that pushes Green toward Blue.
"""

_OSS_TBP_PRACTICAL_FULL = """
Practical Guidelines:
- Keep |x_red - x_green| small but nonzero (approx 0.2-0.6).
- Keep y_red moderately above Green (enough to avoid overlap but not too high).
- Use radius between 0.5 and 1.0 for reliable impulse.
"""

_OSS_TBP_PRACTICAL_GEOMETRY_ONLY = """
Practical Guidelines:
- Keep |x_red - x_green| small but nonzero (approx 0.2-0.6).
- Keep y_red moderately above Green (enough to avoid overlap but not too high).
"""

_OSS_TBP_PRACTICAL_RADIUS_ONLY = """
Practical Guidelines:
- Use radius between 0.5 and 1.0 for reliable impulse.
"""

_OSS_TBP_CORE_FACT_BLOCK = """
CORE FACT:
Green and Blue start separated horizontally. Without intervention, they fall straight down and NEVER meet. Therefore, Green must gain HORIZONTAL velocity from a collision with Red.
"""

_OSS_TBP_CRITICAL_COLLISION_BLOCK = """
Critical Collision Rules:
- If Red and Green share the same y, they fall in parallel with NO collision.
- Red must start ABOVE Green (y_red > y_green) so it falls onto Green first.
- Red must NOT be directly centered at the same x — a slight horizontal offset is required to create a sideways push.
- Collision direction is along the line from Red center to Green center at impact.
"""

_OSS_TBP_STRATEGY_BLOCK = """
Strategy:
- Call get_level_state to see object positions.
- Call compute_relative_positions to determine blue's side relative to green.
- Place red ABOVE green with a slight offset AWAY from blue.
- Call simulate_action to test. If it fails, adjust and retry.
- When it succeeds, call finish."""
