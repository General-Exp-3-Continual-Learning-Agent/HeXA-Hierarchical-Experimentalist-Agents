"""
Shared constants and utilities used across all level prompt modules.
"""

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

3. get_contact_log
   Description: After running a simulation, returns the contact events: which objects touched and when.
   Arguments: None
   Usage: Action: get_contact_log

4. simulate_partial
   Description: Place a red ball and run the simulation only up to the specified step. Returns object positions and velocities at that point. Useful for observing mid-simulation dynamics.
   Arguments: x (float), y (float), radius (float), stop_step (int)
   Usage: Action: simulate_partial
          Action Input: {"x": 0.5, "y": 4.0, "radius": 0.6, "stop_step": 50}
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

# ─── OSS-model format override ──────────────────────────────

OSS_FORMAT_ADDENDUM = """
IMPORTANT: You have access to tools as functions. Call exactly ONE function per turn, then STOP and wait for the result. Do NOT guess or imagine results.
Keep your analysis brief (2-3 sentences max). Focus on WHAT to try, not lengthy calculations.
"""

# The retry nudge for OSS when no action is parsed — short and direct.
OSS_RETRY_NUDGE = "Call a function now. For example, to inspect the level: call get_level_state with arguments {}."

# ─── OSS tool schema helpers ────────────────────────────────

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
        "name": "get_contact_log",
        "description": "Get collision events from the last simulation.",
        "parameters": _EMPTY_PARAMS}},
    {"type": "function", "function": {
        "name": "finish",
        "description": "Submit final answer. Use when confident in your solution.",
        "parameters": _XYR_PARAMS}},
]


def format_observation(observation_text: str) -> str:
    """Format an observation to append to the conversation."""
    return f"Observation: {observation_text}"
