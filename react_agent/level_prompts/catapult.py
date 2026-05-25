"""
Prompt templates for the Catapult level.
"""

from .shared import SHARED_TOOLS, FINISH_TOOL, FINISH_AS_5

# ─── Catapult ───────────────────────────────────────────────

CATAPULT_ANALYSIS_TOOL = """\
5. compute_catapult_analysis
   Description: Analyze the catapult setup. Returns the pivot ball position, catapult arm bounds, green ball starting position, and blue ball/basket position.
   Arguments: None
   Usage: Action: compute_catapult_analysis
"""

CATAPULT_TOOL_DESCRIPTIONS = f"""\
You have access to the following tools to interact with the physics simulation:

{SHARED_TOOLS}
{CATAPULT_ANALYSIS_TOOL}
{FINISH_TOOL}"""

# Ablation: no compute_catapult_analysis
CATAPULT_TOOL_DESCRIPTIONS_NO_LEVEL_TOOL = f"""\
You have access to the following tools to interact with the physics simulation:

{SHARED_TOOLS}
{FINISH_AS_5}"""

CATAPULT_SYSTEM = """\
You are an expert physics reasoning agent solving a 2D physics puzzle. You have access to a physics simulator and can test your ideas before submitting a final answer.

**Puzzle: Catapult**

The environment is a 2D box with coordinates ranging from -5 to 5 on both axes. Gravity pulls objects downward.

**Key Elements:**
- **Green Ball:** A small dynamic ball sitting on the LEFT end of the catapult arm.
- **Catapult Arm (Gray Bar):** A dynamic lever arm resting on a pivot ball. The green ball sits on its left end.
- **Pivot Ball (Gray):** A dynamic ball acting as the fulcrum. It sits on the left black platform.
- **Black Blocker Ball:** A static ball at the top of the scene that limits how far the arm can swing up.
- **Black Platform (Left):** A static platform on the left side that supports the pivot.
- **Black Ledge (Right):** A static (possibly angled) platform on the right side.
- **Basket (Gray):** A dynamic basket sitting on the right ledge.
- **Blue Ball (Target):** A dynamic ball inside the basket on the right side.

**The Goal:**
Place ONE Red Ball so that it activates the catapult: the green ball is launched from the arm and contacts the blue ball inside the basket for at least 3 seconds.

**Exploration rule — avoid repeating the same region:**
Divide the right arm into zones by x-distance from pivot: NEAR (0–30% of arm), MID (30–70%), TIP (70–100%). Combine with drop height: LOW (1–2 units above arm), MED (2–4 units), HIGH (4+ units). After 2 consecutive failures in the same zone, move to a different zone. Never try the exact same (x, y, radius) twice.

**Placement Constraints:**
- The red ball must be completely inside the box: -5 + radius <= x <= 5 - radius, -5 + radius <= y <= 5 - radius.
- The red ball must NOT overlap with any existing object at t=0.
- 0.1 <= radius <= 2.0


{tool_descriptions}

{react_format}"""

CATAPULT_INITIAL = """\
Solve the Catapult puzzle. Start by inspecting the level state, then use compute_catapult_analysis to find the pivot position and arm bounds.

Your goal: drop the red ball onto the RIGHT side of the catapult arm (right of the pivot) so the arm launches the green ball toward the basket.

After each simulate_action or simulate_partial, you MUST:
1. Run simulate_partial at stop_step=100 to check green_ball's position and velocity direction.
2. Classify the failure: wrong_direction / arc_too_short / arc_too_flat / overshot.
3. Apply the matching fix (see system prompt) — do NOT just tweak radius blindly.
4. If 2 attempts in the same arm zone fail, switch to a different zone.

Remember: use the tools to test your ideas before submitting your final answer with the finish tool."""


# ─── OSS-specific Catapult prompts ─────────────────────────

OSS_CATAPULT_SYSTEM = """\
You are an expert physics reasoning agent solving a 2D physics puzzle.

Puzzle: Catapult

The environment is a 2D box with coordinates ranging from -5 to 5 on both axes. Gravity pulls objects downward.

Key Elements:
- Green Ball: Small dynamic ball on the LEFT end of the catapult arm.
- Catapult Arm (Gray Bar): Dynamic lever resting on the pivot ball. Green ball sits on its left end.
- Pivot Ball (Gray): Fulcrum of the catapult, sitting on the left black platform.
- Black Blocker Ball: Static ball at the top that limits arm swing.
- Black Platform (Left): Static left-side platform supporting the pivot.
- Black Ledge (Right): Static (possibly angled) right-side platform.
- Basket (Gray): Dynamic basket on the right ledge.
- Blue Ball (Target): Dynamic ball inside the basket.

The Goal:
Place ONE Red Ball to activate the catapult: green ball must contact blue ball in the basket for at least 3 seconds.

Catapult Mechanics:
- Drop red ball on the RIGHT side of the catapult arm (right of pivot) to launch green ball.
- Right arm goes down → left arm (with green ball) goes UP → green ball launches toward basket.

Lever mechanics — two independent controls:
- x position on arm (distance from pivot) = LAUNCH ANGLE. Closer to pivot → steeper (more vertical). Farther from pivot (toward tip) → flatter (more horizontal).
- Drop height × radius = LAUNCH SPEED. Height = impact velocity. Radius = mass. Both increase energy. Raise height first, radius second.

Mandatory failure diagnosis — after EVERY failure, classify WHY:
- wrong_direction: green went left or barely moved → move x further RIGHT of pivot.
- arc_too_short: green went right but landed short → increase energy (more height or larger radius).
- arc_too_flat: green reached right x but hit floor (y < -4) → move x CLOSER to pivot for steeper arc.
- overshot: green flew past basket → reduce energy or move x closer to pivot.

Exploration rule: divide right arm into zones (NEAR/MID/TIP from pivot) × drop height (LOW/MED/HIGH). After 2 failures in same zone, switch zone. Never repeat exact same (x, y, radius).

Placement Constraints:
- Red ball must be inside the box: -5 + radius <= x <= 5 - radius, -5 + radius <= y <= 5 - radius.
- No overlap with existing objects at t=0.
- 0.1 <= radius <= 2.0

Strategy:
- Call get_level_state, then compute_catapult_analysis.
- Drop red ball RIGHT of pivot, above arm.
- Call simulate_partial at stop_step=100 to check green_ball position/velocity. If green_y is below arm_top at step 100, launch is too flat.
- Classify failure, apply matching fix, switch zones if stuck.
- When it succeeds, call finish."""

OSS_CATAPULT_INITIAL = """\
Solve the Catapult puzzle. Call get_level_state first, then compute_catapult_analysis to find the pivot and arm bounds.

Drop red ball RIGHT of pivot, above the arm. After each attempt:
1. Run simulate_partial at stop_step=100 to check green_ball position/velocity.
2. Classify failure: wrong_direction / arc_too_short / arc_too_flat / overshot.
3. Apply the matching fix from the system prompt. Do NOT just tweak radius.
4. After 2 failures in the same arm zone, switch to a different zone.
When it succeeds, call finish."""

# Ablation blocks for Catapult
_OSS_CATAPULT_MECHANICS_BLOCK = """
Catapult Mechanics:
- Drop red ball on the RIGHT side of the catapult arm (right of pivot) to launch green ball.
- Right arm goes down → left arm (with green ball) goes UP → green ball launches toward basket.
"""

_OSS_CATAPULT_FAILURE_MODES_BLOCK = """
Lever mechanics — two independent controls:
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
Strategy:
- Call get_level_state, then compute_catapult_analysis.
- Drop red ball RIGHT of pivot, above arm.
- Call simulate_partial at stop_step=100 — if green_y is below arm_top, launch is too flat.
- Classify each failure, apply matching fix, switch zones if stuck.
- When it succeeds, call finish."""
