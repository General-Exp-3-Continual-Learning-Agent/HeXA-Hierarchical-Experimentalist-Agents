"""
Prompt templates for the Falling Into Place level.
"""

from .shared import SHARED_TOOLS, FINISH_TOOL, FINISH_AS_5

# ─── Falling Into Place ─────────────────────────────────────

FALLING_INTERCEPT_TOOL = """\
5. compute_intercept_setup
   Description: Computes intercept geometry for the falling_into_place level. Returns which platform the green ball is on, which direction it must travel to reach the jar, the platform edge it must cross, the gap center, and the estimated time before the jar reaches platform height.
   Arguments: None
   Usage: Action: compute_intercept_setup
"""

FALLING_TOOL_DESCRIPTIONS = f"""\
You have access to the following tools to interact with the physics simulation:

{SHARED_TOOLS}
{FALLING_INTERCEPT_TOOL}
{FINISH_TOOL}"""

FALLING_TOOL_DESCRIPTIONS_NO_LEVEL_TOOL = f"""\
You have access to the following tools to interact with the physics simulation:

{SHARED_TOOLS}
{FINISH_AS_5}"""

FALLING_INTO_PLACE_SYSTEM = """\
You are an expert physics reasoning agent solving a 2D physics puzzle. You have access to a physics simulator and can test your ideas before submitting a final answer.

**Puzzle: Falling Into Place**

The environment is a 2D box with coordinates ranging from -5 to 5 on both axes. Gravity pulls objects downward.

**Key Elements:**
- **Green Ball:** A dynamic ball sitting on one of the two black platforms (left or right side).
- **Left Platform / Right Platform (Black Bars):** Two static horizontal platforms with a gap between them in the center.
- **Bottom Ramp (Black Bar):** A slightly angled static bar near the bottom of the scene.
- **Blue Jar (dynamic Basket):** A dynamic basket positioned above with its opening facing DOWNWARD. It falls due to gravity.

**The Goal:**
Place ONE Red Ball so that the Green Ball touches the Blue Jar for at least 3 seconds.

**Placement Constraints:**
- The red ball must be completely inside the box: -5 + radius <= x <= 5 - radius, -5 + radius <= y <= 5 - radius.
- The red ball must NOT overlap with any existing object at t=0.
- 0.1 <= radius <= 2.0

{tool_descriptions}

{react_format}"""

FALLING_INTO_PLACE_INITIAL = """\
Solve the Falling Into Place puzzle. You have 25 iterations to solve. Start by inspecting the level state, then use compute_intercept_setup to find which direction the green ball must travel.

."""


# ─── OSS-specific Falling Into Place prompts ───────────────

OSS_FALLING_INTO_PLACE_SYSTEM = """\
You are an expert physics reasoning agent solving a 2D physics puzzle.

Puzzle: Falling Into Place

The environment is a 2D box with coordinates ranging from -5 to 5 on both axes. Gravity pulls objects downward.

Key Elements:
- Green Ball: A dynamic ball sitting on one of the two black platforms.
- Left Platform / Right Platform (Black Bars): Two static horizontal platforms with a gap in the center.
- Bottom Ramp (Black Bar): A slightly angled static bar near the bottom.
- Blue Jar (dynamic Basket): A dynamic basket that falls due to gravity, opening facing downward.

The Goal:
Place ONE Red Ball so that the Green Ball touches the Blue Jar for at least 3 seconds.

Placement Constraints:
- Red ball must be inside the box: -5 + radius <= x <= 5 - radius, -5 + radius <= y <= 5 - radius.
- No overlap with existing objects at t=0.
- 0.1 <= radius <= 2.0

."""

OSS_FALLING_INTO_PLACE_INITIAL = """\
Solve the Falling Into Place puzzle. Call get_level_state first, then call compute_intercept_setup to find the push direction and jar fall timing.

Remember: place the red ball HIGH ABOVE the green ball with a slight horizontal offset. Use simulate_action to test placements. When it succeeds, call finish."""

# Ablation blocks for Falling Into Place
_OSS_FIP_HOW_TO_PLACE_BLOCK = """
How to Place the Red Ball:
- Place the red ball HIGH ABOVE the green ball (at least 2-3 units higher in y) so it gains speed while falling.
- Offset the red ball slightly in the OPPOSITE direction of where green needs to travel. For example, if green must go RIGHT, place red slightly to the LEFT of green.
- Use a radius between 0.5 and 1.5 for strong impact.
- Do NOT place the red ball at the same height as the green ball — it must fall onto the green ball from above.
"""
