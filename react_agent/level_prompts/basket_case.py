"""
Prompt templates for the Basket Case level.
"""

from .shared import SHARED_TOOLS, FINISH_TOOL, FINISH_AS_5

# ─── Basket Case ────────────────────────────────────────────

BASKET_CASE_ANALYSIS_TOOL = """\
5. compute_basket_analysis
   Description: Analyze the basket case setup. Returns the green ball position, basket position and scale, purple ground position, and recommended push direction to deflect the green ball away from the basket.
   Arguments: None
   Usage: Action: compute_basket_analysis
"""

BASKET_CASE_TOOL_DESCRIPTIONS = f"""\
You have access to the following tools to interact with the physics simulation:

{SHARED_TOOLS}
{BASKET_CASE_ANALYSIS_TOOL}
{FINISH_TOOL}"""

BASKET_CASE_TOOL_DESCRIPTIONS_NO_LEVEL_TOOL = f"""\
You have access to the following tools to interact with the physics simulation:

{SHARED_TOOLS}
{FINISH_AS_5}"""

BASKET_CASE_SYSTEM = """\
You are an expert physics reasoning agent solving a 2D physics puzzle. You have access to a physics simulator and can test your ideas before submitting a final answer.

**Puzzle: Basket Case**

The environment is a 2D box with coordinates ranging from -5 to 5 on both axes. Gravity pulls objects downward.

**Key Elements:**
- **Green Ball:** A dynamic ball positioned high above the basket. It falls straight down due to gravity.
- **Basket (Gray):** A dynamic container sitting near the purple ground. Its opening faces UPWARD. Without intervention, the green ball falls directly into it and gets trapped.
- **Purple Ground:** A static bar at the very bottom of the scene (y ~ -5). This is the target surface.

**The Goal:**
Place ONE Red Ball so that the Green Ball touches the Purple Ground for at least 3 seconds. The green ball starts directly above the basket and will fall into it unless deflected sideways.

**Placement Constraints:**
- The red ball must be completely inside the box: -5 + radius <= x <= 5 - radius, -5 + radius <= y <= 5 - radius.
- The red ball must NOT overlap with the green ball or basket at t=0.
- 0.1 <= radius <= 2.0


{tool_descriptions}

{react_format}"""

BASKET_CASE_INITIAL = """\
Solve the Basket Case puzzle. You have 25 iterations to solve it. Start by inspecting the level state, then use compute_basket_analysis to find the basket position, scale, and recommended push direction.

Your goal: deflect the green ball sideways so it misses the basket and lands on the purple ground for 3 seconds.

Remember: use the tools to test your ideas before submitting your final answer with the finish tool."""


# ─── OSS-specific Basket Case prompts ──────────────────────

OSS_BASKET_CASE_SYSTEM = """\
You are an expert physics reasoning agent solving a 2D physics puzzle.

Puzzle: Basket Case

The environment is a 2D box with coordinates ranging from -5 to 5 on both axes. Gravity pulls objects downward.

Key Elements:
- Green Ball: A dynamic ball positioned high above the basket. Falls straight down due to gravity.
- Basket (Gray): A dynamic container sitting near the purple ground, opening facing UPWARD. Without intervention the green ball falls into it and gets trapped.
- Purple Ground: A static bar at the very bottom (y ~ -5). The target surface.

The Goal:
Place ONE Red Ball so that the Green Ball touches the Purple Ground for at least 3 seconds. The green ball starts directly above the basket — you must deflect it sideways so it misses the basket and lands on the ground.

Placement Constraints:
- Red ball must be inside box: -5 + radius <= x <= 5 - radius, -5 + radius <= y <= 5 - radius.
- No overlap with green ball or basket at t=0.
- 0.1 <= radius <= 2.0
"""

OSS_BASKET_CASE_INITIAL = """\
Solve the Basket Case puzzle. Call get_level_state first, then compute_basket_analysis to find basket position and scale.

Deflect the green ball sideways so it misses the basket and lands on the purple ground. Use simulate_action to test placements. When it succeeds, call finish."""
