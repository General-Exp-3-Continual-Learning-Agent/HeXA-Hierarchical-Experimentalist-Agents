"""
Prompt templates for the Pass The Parcel level.
"""

from .shared import SHARED_TOOLS, FINISH_TOOL, FINISH_AS_5

# ─── Pass The Parcel ────────────────────────────────────────

PASS_THE_PARCEL_ANALYSIS_TOOL = """\
5. get_ramp_center
   Description: Analyze the pass_the_parcel setup. Returns the center of the ramp.
   Arguments: None
   Usage: Action: get_ramp_center
"""

PASS_THE_PARCEL_TOOL_DESCRIPTIONS = f"""\
You have access to the following tools to interact with the physics simulation:

{SHARED_TOOLS}
{PASS_THE_PARCEL_ANALYSIS_TOOL}
{FINISH_TOOL}"""

PASS_THE_PARCEL_TOOL_DESCRIPTIONS_NO_LEVEL_TOOL = f"""\
You have access to the following tools to interact with the physics simulation:

{SHARED_TOOLS}
{FINISH_AS_5}"""

PASS_THE_PARCEL_SYSTEM = """\
You are an expert physics reasoning agent solving a 2D physics puzzle. You have access to a physics simulator and can test your ideas before submitting a final answer.

**Puzzle: Pass The Parcel**

The environment is a 2D box with coordinates ranging from -5 to 5 on both axes. Gravity pulls objects downward.

**Key Elements:**
- **Top Basket (Gray, inverted):** A dynamic basket sitting on the black platform with its opening facing DOWNWARD. It traps the green ball underneath it.
- **Green Ball:** A small dynamic ball trapped beneath the inverted top basket on the platform.
- **Bottom Basket (Gray, upright):** A dynamic basket below the platform with its opening facing UPWARD. It holds the blue ball.
- **Blue Ball:** A dynamic ball sitting inside the bottom basket. This is the target — the green ball must touch it.
- **Black Platform:** A static horizontal bar. The top basket and (initially) red ball sit on it.
- **Ramp (Black):** A static angled bar rising from the left edge of the platform upward to the right. Useful for rolling the red ball down onto the top basket.

**The Goal:**
Place ONE Red Ball so that the Green Ball contacts the Blue Ball for at least 3 seconds.

**Placement Constraints:**
- The red ball must be completely inside the box: -5 + radius <= x <= 5 - radius, -5 + radius <= y <= 5 - radius.
- The red ball must NOT overlap with existing objects at t=0.
- 0.1 <= radius <= 2.0


{tool_descriptions}

{react_format}"""

PASS_THE_PARCEL_INITIAL = """\
Solve the Pass The Parcel puzzle. You have 12 iterations to solve this. Start by calling get_level_state.

Your goal: knock the inverted top basket off the platform to release the green ball, which must then contact the blue ball in the bottom basket for 3 seconds.

Remember: use the tools to test your ideas before submitting your final answer with the finish tool."""


# ─── OSS-specific Pass The Parcel prompts ──────────────────

OSS_PASS_THE_PARCEL_SYSTEM = """\
You are an expert physics reasoning agent solving a 2D physics puzzle.

Puzzle: Pass The Parcel

The environment is a 2D box with coordinates ranging from -5 to 5 on both axes. Gravity pulls objects downward.

Key Elements:
- Top Basket (Gray, inverted): Dynamic basket on the platform, opening faces DOWN, trapping the green ball underneath.
- Green Ball: Small dynamic ball trapped under the inverted top basket on the platform.
- Bottom Basket (Gray, upright): Dynamic basket below the platform, opening faces UP, holds the blue ball.
- Blue Ball: Dynamic ball inside the bottom basket. Green ball must contact it.
- Black Platform: Static horizontal bar. Top basket sits on it.
- Ramp (Black): Static angled bar rising from the platform's left edge upward-right.

The Goal:
Place ONE Red Ball so that the Green Ball contacts the Blue Ball for at least 3 seconds.

Placement Constraints:
- Red ball must be inside box: -5 + radius <= x <= 5 - radius, -5 + radius <= y <= 5 - radius.
- No overlap with existing objects at t=0.
- 0.1 <= radius <= 2.0
."""

OSS_PASS_THE_PARCEL_INITIAL = """\
Solve the Pass The Parcel puzzle. Call get_level_state first, then get_ramp_center to find the ramp angle and basket positions.

Knock the inverted top basket off the platform to release the green ball, which must then contact the blue ball. Use simulate_action to test placements. When it succeeds, call finish."""
