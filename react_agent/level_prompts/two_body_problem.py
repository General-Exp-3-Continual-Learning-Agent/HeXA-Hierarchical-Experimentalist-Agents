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
Solve the Two Body Problem puzzle. You have {max_iterations} iterations to solve the puzzle. Start by inspecting the level state with get_level_state, then use compute_relative_positions to understand where the blue ball is relative to green.

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
Solve the Two Body Problem.You have {max_iterations} iterations to solve. Start by calling get_level_state, then call compute_relative_positions to see where blue is relative to green.

"""
