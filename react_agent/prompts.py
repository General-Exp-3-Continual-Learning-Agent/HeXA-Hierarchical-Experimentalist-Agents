"""
Prompt templates for the ReAct agent solving Interphyre physics puzzles.
Supports: down_to_earth, two_body_problem, catapult, falling_into_place
"""


# ─── Shared ────────────────────────────────────────────────

REACT_FORMAT_INSTRUCTIONS = """\
To solve this puzzle, you will reason step-by-step and use tools to test your ideas.

At each step, you MUST follow this exact format:

Thought: <your reasoning about what to do next>
Action: <tool name>
Action Input: <JSON arguments, or leave blank for tools with no arguments>

After you take an action, you will receive:

Observation: <result from the tool>

Then you continue with another Thought/Action cycle.

When you are confident in your answer, use the "finish" tool to submit it.

Important rules:
- Always start with a Thought before taking an Action.
- Only call ONE tool per step.
- Parse observation results carefully before your next thought.
- You can simulate multiple different actions to compare results.
- Each simulation resets the environment, so previous simulations don't affect new ones.
"""

SHARED_TOOLS = """\
1. get_level_state
   Description: Get the current level layout including all object positions, sizes, and properties.
   Arguments: None
   Usage: Action: get_level_state

2. simulate_action
   Description: Place a red ball at (x, y) with the given radius and run the full physics simulation to completion. Returns whether the goal was achieved, final positions of all objects, and total simulation steps. If the placement is invalid (out of bounds or overlaps), returns a detailed error with how far to move the ball.
   Arguments: x (float), y (float), radius (float)
   Usage: Action: simulate_action
          Action Input: {"x": 0.5, "y": 4.0, "radius": 0.6}

3. simulate_partial
   Description: Place a red ball and run the simulation only up to the specified step. Returns object positions and velocities at that point. Useful for observing mid-simulation dynamics.
   Arguments: x (float), y (float), radius (float), stop_step (int)
   Usage: Action: simulate_partial
          Action Input: {"x": 0.5, "y": 4.0, "radius": 0.6, "stop_step": 50}

4. get_contact_log
   Description: After running a simulation, returns the contact events: which objects touched and when.
   Arguments: None
   Usage: Action: get_contact_log
"""

FINISH_TOOL = """\
6. finish
   Description: Submit your final answer. Use this when you are confident in your solution.
   Arguments: x (float), y (float), radius (float)
   Usage: Action: finish
          Action Input: {"x": 0.5, "y": 4.0, "radius": 0.6}
"""

# When level-specific tool (5) is omitted, finish is numbered 5
FINISH_AS_5 = """\
5. finish
   Description: Submit your final answer. Use this when you are confident in your solution.
   Arguments: x (float), y (float), radius (float)
   Usage: Action: finish
          Action Input: {"x": 0.5, "y": 4.0, "radius": 0.6}
"""


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

**Physics Rules:**
- Both balls are released at t=0 and fall under gravity simultaneously.
- A larger ball has more mass (constant density). Bigger balls push smaller ones in collisions.
- To push the green ball RIGHT, place the red ball to its LEFT. To push LEFT, place to the RIGHT.
- The green ball can only fall off the platform if the gap between the platform edge and the wall is wider than the green ball's diameter.

**Placement Constraints:**
- The red ball must be completely inside the box: -5 + radius <= x <= 5 - radius, -5 + radius <= y <= 5 - radius.
- The red ball must NOT overlap with the green ball at t=0: distance between centers > sum of radii.
- The red ball must NOT overlap with the black platform at t=0.

**Strategy Tips:**
- First, use get_level_state to understand the scene layout.
- Use compute_gap_analysis to determine which side the green ball can fall off.
- The red ball should collide with the green ball on the platform to push it toward the wider gap.
- Use simulate_action to test your ideas and observe what happens.
- If a simulation fails, analyze WHY and adjust your approach.

{tool_descriptions}

{react_format}"""

DOWN_TO_EARTH_INITIAL = """\
Solve the Down to Earth puzzle. Start by inspecting the level state, then figure out where to place the red ball to knock the green ball off the platform and onto the purple ground.

Remember: use the tools to test your ideas before submitting your final answer with the finish tool."""


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
Place ONE Red Ball at t=0 so that the Green Ball collides with the Blue Ball and stays in contact for at least 3 seconds.

**CORE FACT:**
Green and Blue start separated horizontally. Without intervention, they fall straight down and NEVER meet. Therefore, Green must gain HORIZONTAL velocity from a collision with Red.

**Critical Collision Rules:**
- If Red and Green share the same y, they fall in parallel with NO collision.
- Red must start ABOVE Green (y_red > y_green) so it falls onto Green first.
- Red must NOT be directly centered at the same x -- a slight horizontal offset is required to create a sideways push.
- Collision direction is along the line from Red center to Green center at impact.

**How to Push Toward Blue:**
1. Determine if Blue is left or right of Green.
2. If Blue is to the RIGHT: place Red slightly LEFT of Green and ABOVE it.
3. If Blue is to the LEFT: place Red slightly RIGHT of Green and ABOVE it.
This creates a diagonal impact that pushes Green toward Blue.

**Practical Guidelines:**
- Keep |x_red - x_green| small but nonzero (approx 0.2-0.6).
- Keep y_red moderately above Green (enough to avoid overlap but not too high).
- Use radius between 0.5 and 1.0 for reliable impulse.

**Placement Constraints:**
- The red ball must be completely inside the box: -5 + radius <= x <= 5 - radius, -5 + radius <= y <= 5 - radius.
- The red ball must NOT overlap with the green ball or the blue ball at t=0.
- 0.1 <= radius <= 1.5

{tool_descriptions}

{react_format}"""

TWO_BODY_INITIAL = """\
Solve the Two Body Problem puzzle. Start by inspecting the level state with get_level_state, then use compute_relative_positions to understand where the blue ball is relative to green.

Remember: use the tools to test your ideas before submitting your final answer with the finish tool."""


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

**Catapult Mechanics:**
- The catapult arm is a lever with the pivot ball as the fulcrum.
- The green ball sits on the LEFT end of the arm.
- To LAUNCH the green ball: drop the red ball on the RIGHT side of the arm (to the right of the pivot).
- When red lands on the RIGHT arm → right side goes DOWN → left side (with green) goes UP → green ball launches toward the basket.
- If you drop red on the LEFT side of the pivot, the green ball is pushed DOWN, not launched.
- Choose a large enough radius so the red ball lands on the catapult arm and launches the green ball.

**Lever mechanics — two independent controls:**
- **x position on arm (distance from pivot)** controls the LAUNCH ANGLE via the lever arm ratio.
  - Closer to pivot → steeper launch (more vertical, higher arc).
  - Farther from pivot (toward arm tip) → flatter launch (more horizontal distance, lower arc).
  This is NOT an energy knob — it shapes the trajectory.
- **Drop height × radius** together control LAUNCH SPEED.
  - Height above arm = impact velocity. Radius = mass. Both increase the energy transferred to the arm.
  - To increase launch speed: raise height first, increase radius second.

**Mandatory failure diagnosis — after EVERY failed simulation, classify WHY before trying again:**
- `wrong_direction`: green ball went left or barely moved → drop point is on the wrong side of pivot, or too close to it. Move x further RIGHT of pivot.
- `arc_too_short`: green ball traveled rightward but landed short of the basket → increase energy (more height or larger radius). Do NOT decrease radius.
- `arc_too_flat`: green ball reached the right x-range but hit the floor (y < -4) → the launch angle is too shallow. Move x CLOSER to pivot for a steeper arc. Keep energy the same or increase it slightly.
- `overshot`: green ball flew past the basket (x > basket_x + 1) → reduce energy (less height or smaller radius), or move x closer to pivot for steeper arc.
You MUST state the diagnosis label in your Thought before choosing the next action.

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
Solve the Basket Case puzzle. Start by inspecting the level state, then use compute_basket_analysis to find the basket position, scale, and recommended push direction.

Your goal: deflect the green ball sideways so it misses the basket and lands on the purple ground for 3 seconds.

Remember: use the tools to test your ideas before submitting your final answer with the finish tool."""


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

**How to Place the Red Ball:**
- Place the red ball HIGH ABOVE the green ball (at least 2-3 units higher in y) so it gains speed while falling.
- Offset the red ball slightly in the OPPOSITE direction of where green needs to travel. For example, if green must go RIGHT, place red slightly to the LEFT of green.
- Use a radius between 0.5 and 1.5 for strong impact.
- Do NOT place the red ball at the same height as the green ball — it must fall onto the green ball from above.

**Placement Constraints:**
- The red ball must be completely inside the box: -5 + radius <= x <= 5 - radius, -5 + radius <= y <= 5 - radius.
- The red ball must NOT overlap with any existing object at t=0.
- 0.1 <= radius <= 2.0

{tool_descriptions}

{react_format}"""

FALLING_INTO_PLACE_INITIAL = """\
Solve the Falling Into Place puzzle. Start by inspecting the level state, then use compute_intercept_setup to find which direction the green ball must travel.

Strategy:
- Call get_level_state to see all object positions.
- Call compute_intercept_setup to get push direction, platform edge, and jar fall time.
- Place red ball HIGH ABOVE green ball with a slight horizontal offset to push green in the needed direction.
- Call simulate_action to test. Adjust and retry as needed.
- When it succeeds, call finish."""


# ─── OSS-model format override ──────────────────────────────

OSS_FORMAT_ADDENDUM = """\

IMPORTANT: You have access to tools as functions. Call exactly ONE function per turn, then STOP and wait for the result. Do NOT guess or imagine results.
Keep your analysis brief (2-3 sentences max). Focus on WHAT to try, not lengthy calculations.
"""

# ─── OSS-specific prompts ──────────────────────────────────
# These mirror the standard prompts but are formatted for the OSS model's
# native function-calling flow (no ReAct format instructions, no tool
# descriptions — those come via tool schemas in the chat template).

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

Physics Rules:
- Both balls are released at t=0 and fall under gravity simultaneously.
- A larger ball has more mass (constant density). Bigger balls push smaller ones in collisions.
- To push the green ball RIGHT, place the red ball to its LEFT. To push LEFT, place to the RIGHT.
- The green ball can only fall off the platform if the gap between the platform edge and the wall is wider than the green ball's diameter.

Placement Constraints:
- The red ball must be completely inside the box: -5 + radius <= x <= 5 - radius, -5 + radius <= y <= 5 - radius.
- The red ball must NOT overlap with the green ball at t=0: distance between centers > sum of radii.
- The red ball must NOT overlap with the black platform at t=0.

Strategy:
- First, call get_level_state to understand the scene layout.
- Call compute_gap_analysis to determine which side the green ball can fall off.
- The red ball should collide with the green ball on the platform to push it toward the wider gap.
- Call simulate_action to test placements and observe what happens.
- If a simulation fails, analyze WHY and adjust your approach.
- When a simulation succeeds, call finish with those coordinates."""

# Strategy block for removal during ablation
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

OSS_TWO_BODY_SYSTEM = """\
You are an expert physics reasoning agent solving a 2D physics puzzle.

Puzzle: Two Body Problem

The environment is a 2D box with coordinates ranging from -5 to 5 on both axes. Gravity pulls objects downward.

Key Elements:
- Green Ball: A dynamic ball.
- Blue Ball: A dynamic ball, separated horizontally from the green ball.
- Both balls fall under gravity from rest.

The Goal:
Place ONE Red Ball at t=0 so that the Green Ball collides with the Blue Ball and stays in contact for at least 3 seconds.

CORE FACT:
Green and Blue start separated horizontally. Without intervention, they fall straight down and NEVER meet. Therefore, Green must gain HORIZONTAL velocity from a collision with Red.

Critical Collision Rules:
- If Red and Green share the same y, they fall in parallel with NO collision.
- Red must start ABOVE Green (y_red > y_green) so it falls onto Green first.
- Red must NOT be directly centered at the same x — a slight horizontal offset is required to create a sideways push.
- Collision direction is along the line from Red center to Green center at impact.

How to Push Toward Blue:
1. Determine if Blue is left or right of Green.
2. If Blue is to the RIGHT: place Red slightly LEFT of Green and ABOVE it.
3. If Blue is to the LEFT: place Red slightly RIGHT of Green and ABOVE it.
This creates a diagonal impact that pushes Green toward Blue.

Practical Guidelines:
- Keep |x_red - x_green| small but nonzero (approx 0.2-0.6).
- Keep y_red moderately above Green (enough to avoid overlap but not too high).
- Use radius between 0.5 and 1.0 for reliable impulse.

Placement Constraints:
- The red ball must be completely inside the box: -5 + radius <= x <= 5 - radius, -5 + radius <= y <= 5 - radius.
- The red ball must NOT overlap with the green ball or the blue ball at t=0.
- 0.1 <= radius <= 1.5

Strategy:
- Call get_level_state to see object positions.
- Call compute_relative_positions to determine blue's side relative to green.
- Place red ABOVE green with a slight offset AWAY from blue.
- Call simulate_action to test. If it fails, adjust and retry.
- When it succeeds, call finish."""

# Blocks for ablation removal
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

# Blocks for ablation removal (catapult)
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

# Blocks for ablation removal (falling into place)
_OSS_FIP_HOW_TO_PLACE_BLOCK = """
How to Place the Red Ball:
- Place the red ball HIGH ABOVE the green ball (at least 2-3 units higher in y) so it gains speed while falling.
- Offset the red ball slightly in the OPPOSITE direction of where green needs to travel. For example, if green must go RIGHT, place red slightly to the LEFT of green.
- Use a radius between 0.5 and 1.5 for strong impact.
- Do NOT place the red ball at the same height as the green ball — it must fall onto the green ball from above.
"""

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

How to Place the Red Ball:
- Place the red ball HIGH ABOVE the green ball (at least 2-3 units higher in y) so it gains speed while falling.
- Offset the red ball slightly in the OPPOSITE direction of where green needs to travel. For example, if green must go RIGHT, place red slightly to the LEFT of green.
- Use a radius between 0.5 and 1.5 for strong impact.
- Do NOT place the red ball at the same height as the green ball — it must fall onto the green ball from above.

Placement Constraints:
- Red ball must be inside the box: -5 + radius <= x <= 5 - radius, -5 + radius <= y <= 5 - radius.
- No overlap with existing objects at t=0.
- 0.1 <= radius <= 2.0

Strategy:
- Call get_level_state to see all object positions.
- Call compute_intercept_setup to get push direction, platform edge, and jar fall time.
- Place red ball HIGH ABOVE green ball with a slight horizontal offset to push green in the needed direction.
- Call simulate_action to test. Adjust and retry as needed.
- When it succeeds, call finish."""

OSS_DOWN_TO_EARTH_INITIAL = """\
Solve the Down to Earth puzzle. Start by calling get_level_state to inspect the level, then call compute_gap_analysis to find which side has a wider gap.

Use simulate_action to test placements. When one succeeds, call finish with those coordinates."""

OSS_TWO_BODY_INITIAL = """\
Solve the Two Body Problem. Start by calling get_level_state, then call compute_relative_positions to see where blue is relative to green.

Place red ABOVE green with a slight horizontal offset AWAY from blue. Use simulate_action to test. When it succeeds, call finish."""

OSS_CATAPULT_INITIAL = """\
Solve the Catapult puzzle. Call get_level_state first, then compute_catapult_analysis to find the pivot and arm bounds.

Drop red ball RIGHT of pivot, above the arm. After each attempt:
1. Run simulate_partial at stop_step=100 to check green_ball position/velocity.
2. Classify failure: wrong_direction / arc_too_short / arc_too_flat / overshot.
3. Apply the matching fix from the system prompt. Do NOT just tweak radius.
4. After 2 failures in the same arm zone, switch to a different zone.
When it succeeds, call finish."""

OSS_FALLING_INTO_PLACE_INITIAL = """\
Solve the Falling Into Place puzzle. Call get_level_state first, then call compute_intercept_setup to find the push direction and jar fall timing.

Remember: place the red ball HIGH ABOVE the green ball with a slight horizontal offset. Use simulate_action to test placements. When it succeeds, call finish."""

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

Physics Rules:
- All dynamic objects fall under gravity from t=0.
- The green ball starts directly above the basket (same x). It falls in and gets trapped unless deflected.
- Place red ball to the LEFT or RIGHT of the green ball at roughly the same height to push it sideways.
- A larger red ball delivers a stronger push. Use radius 0.5–1.5.

Placement Constraints:
- Red ball must be inside box: -5 + radius <= x <= 5 - radius, -5 + radius <= y <= 5 - radius.
- No overlap with green ball or basket at t=0.
- 0.1 <= radius <= 2.0

Strategy:
- Call get_level_state to see all positions.
- Call compute_basket_analysis to get basket scale and recommended push direction.
- Place red ball beside the green ball (same height, offset left or right) to knock it past the basket edge.
- Call simulate_action to test. Adjust x offset and radius until green ball lands on purple ground.
- When it succeeds, call finish."""

OSS_BASKET_CASE_INITIAL = """\
Solve the Basket Case puzzle. Call get_level_state first, then compute_basket_analysis to find basket position and scale.

Deflect the green ball sideways so it misses the basket and lands on the purple ground. Use simulate_action to test placements. When it succeeds, call finish."""

# The retry nudge for OSS when no action is parsed — short and direct.
OSS_RETRY_NUDGE = "Call a function now. For example, to inspect the level: call get_level_state with arguments {}."

# ─── OSS tool schemas (OpenAI function-calling format) ──────

_XYR_PARAMS = {
    "type": "object",
    "properties": {
        "x": {"type": "number", "description": "x coordinate"},
        "y": {"type": "number", "description": "y coordinate"},
        "radius": {"type": "number", "description": "ball radius"},
    },
    "required": ["x", "y", "radius"],
}

_EMPTY_PARAMS = {"type": "object", "properties": {}}

OSS_TOOLS_SHARED = [
    {"type": "function", "function": {
        "name": "get_level_state",
        "description": "Get current level layout: object positions, sizes, properties.",
        "parameters": _EMPTY_PARAMS}},
    {"type": "function", "function": {
        "name": "simulate_action",
        "description": "Place red ball and run full physics simulation. Returns success/failure and final positions. If placement is invalid, returns a detailed error with how far to move the ball instead of simulating.",
        "parameters": _XYR_PARAMS}},
    {"type": "function", "function": {
        "name": "simulate_partial",
        "description": "Place red ball and simulate up to stop_step. Returns positions/velocities at that point.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number"}, "y": {"type": "number"},
                "radius": {"type": "number"}, "stop_step": {"type": "integer"},
            },
            "required": ["x", "y", "radius", "stop_step"],
        }}},
    {"type": "function", "function": {
        "name": "get_contact_log",
        "description": "Get collision events from the last simulation.",
        "parameters": _EMPTY_PARAMS}},
    {"type": "function", "function": {
        "name": "finish",
        "description": "Submit final answer. Use when confident in your solution.",
        "parameters": _XYR_PARAMS}},
]

OSS_TOOL_GAP_ANALYSIS = {"type": "function", "function": {
    "name": "compute_gap_analysis",
    "description": "Analyze gaps on each side of the platform. Returns whether green ball can fit through each gap.",
    "parameters": _EMPTY_PARAMS}}

OSS_TOOL_RELATIVE_POS = {"type": "function", "function": {
    "name": "compute_relative_positions",
    "description": "Analyze positions of green and blue balls. Returns distance, relative side, and recommended placement direction.",
    "parameters": _EMPTY_PARAMS}}

OSS_TOOL_CATAPULT_ANALYSIS = {"type": "function", "function": {
    "name": "compute_catapult_analysis",
    "description": "Analyze the catapult setup. Returns pivot position, arm bounds, green/blue ball positions.",
    "parameters": _EMPTY_PARAMS}}

OSS_TOOL_FALLING_ANALYSIS = {"type": "function", "function": {
    "name": "compute_intercept_setup",
    "description": "Computes intercept geometry: which platform green is on, direction it must travel, platform edge to cross, gap center, and estimated jar fall time to platform height.",
    "parameters": _EMPTY_PARAMS}}

OSS_TOOL_BASKET_ANALYSIS = {"type": "function", "function": {
    "name": "compute_basket_analysis",
    "description": "Analyze the basket setup: green ball position, basket position and scale, and recommended direction to push the green ball to avoid the basket.",
    "parameters": _EMPTY_PARAMS}}


def get_oss_tools(level_name: str = "down_to_earth", include_level_tool: bool = True) -> list:
    """Return the list of tool schemas for the OSS model's chat template.
    When include_level_tool is False (ablation), the level-specific
    analysis tool is omitted.
    """
    tools = list(OSS_TOOLS_SHARED)
    if include_level_tool:
        if level_name == "down_to_earth":
            tools.append(OSS_TOOL_GAP_ANALYSIS)
        elif level_name == "two_body_problem":
            tools.append(OSS_TOOL_RELATIVE_POS)
        elif level_name == "catapult":
            tools.append(OSS_TOOL_CATAPULT_ANALYSIS)
        elif level_name == "falling_into_place":
            tools.append(OSS_TOOL_FALLING_ANALYSIS)
        elif level_name == "basket_case":
            tools.append(OSS_TOOL_BASKET_ANALYSIS)
    return tools


# ─── Dispatch ───────────────────────────────────────────────

def build_system_prompt(level_name: str = "down_to_earth", is_oss: bool = False) -> str:
    """Build the full system prompt for the given level."""
    if is_oss:
        if level_name == "two_body_problem":
            return OSS_TWO_BODY_SYSTEM
        elif level_name == "catapult":
            return OSS_CATAPULT_SYSTEM
        elif level_name == "falling_into_place":
            return OSS_FALLING_INTO_PLACE_SYSTEM
        elif level_name == "basket_case":
            return OSS_BASKET_CASE_SYSTEM
        else:
            return OSS_DOWN_TO_EARTH_SYSTEM
    if level_name == "two_body_problem":
        return TWO_BODY_SYSTEM.format(
            tool_descriptions=TWO_BODY_TOOL_DESCRIPTIONS,
            react_format=REACT_FORMAT_INSTRUCTIONS,
        )
    elif level_name == "catapult":
        return CATAPULT_SYSTEM.format(
            tool_descriptions=CATAPULT_TOOL_DESCRIPTIONS,
            react_format=REACT_FORMAT_INSTRUCTIONS,
        )
    elif level_name == "falling_into_place":
        return FALLING_INTO_PLACE_SYSTEM.format(
            tool_descriptions=FALLING_TOOL_DESCRIPTIONS,
            react_format=REACT_FORMAT_INSTRUCTIONS,
        )
    elif level_name == "basket_case":
        return BASKET_CASE_SYSTEM.format(
            tool_descriptions=BASKET_CASE_TOOL_DESCRIPTIONS,
            react_format=REACT_FORMAT_INSTRUCTIONS,
        )
    else:
        return DOWN_TO_EARTH_SYSTEM.format(
            tool_descriptions=DOWN_TO_EARTH_TOOL_DESCRIPTIONS,
            react_format=REACT_FORMAT_INSTRUCTIONS,
        )


def build_initial_user_message(level_name: str = "down_to_earth", is_oss: bool = False) -> str:
    """Build the initial user message for the given level."""
    if is_oss:
        if level_name == "two_body_problem":
            return OSS_TWO_BODY_INITIAL
        elif level_name == "catapult":
            return OSS_CATAPULT_INITIAL
        elif level_name == "falling_into_place":
            return OSS_FALLING_INTO_PLACE_INITIAL
        elif level_name == "basket_case":
            return OSS_BASKET_CASE_INITIAL
        else:
            return OSS_DOWN_TO_EARTH_INITIAL
    if level_name == "two_body_problem":
        return TWO_BODY_INITIAL
    elif level_name == "catapult":
        return CATAPULT_INITIAL
    elif level_name == "falling_into_place":
        return FALLING_INTO_PLACE_INITIAL
    elif level_name == "basket_case":
        return BASKET_CASE_INITIAL
    else:
        return DOWN_TO_EARTH_INITIAL


def format_observation(observation_text: str) -> str:
    """Format an observation to append to the conversation."""
    return f"Observation: {observation_text}"
