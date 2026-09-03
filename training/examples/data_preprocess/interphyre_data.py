"""
Data preprocessing for Interphyre physics puzzle RL training.

Generates train/val parquet files with one entry per (level, index) pair.
The prompt uses the same system prompt and ReAct format as the vanilla react_agent,
with only the tools relevant to the target level included.

Usage:
    python examples/data_preprocess/interphyre_data.py \
        --output_dir data/interphyre_two_body \
        --levels two_body_problem \
        --num_train_per_level 40 \
        --num_val_per_level 2
"""

import argparse
import json
import os
import pandas as pd
from datasets import Dataset

# ---------------------------------------------------------------------------
# Prompts (inlined from react_agent/prompts.py to avoid cross-repo dependency)
# ---------------------------------------------------------------------------

LEVEL_TITLES = {
    "down_to_earth": "Down to Earth",
    "two_body_problem": "Two Body Problem",
    "cliffhanger": "Cliffhanger",
    "tipping_point": "Tipping Point",
    "catapult": "Catapult",
}

LEVEL_DESCRIPTIONS = {
    "down_to_earth": """\
The environment is a 2D box with coordinates ranging from -5 to 5 on both axes. Gravity pulls objects downward.

**Objects:**
- **Green Ball:** A dynamic ball resting on a black platform. Falls under gravity.
- **Black Platform:** A static horizontal platform. Without intervention, the green ball lands here and stays.
- **Purple Ground:** The static floor at the very bottom of the box (y ~ -5).

**Goal:**
Place a Red Ball so that the Green Ball ends up touching the Purple Ground for at least 3 seconds.""",

    "two_body_problem": """\
The environment is a 2D box with coordinates ranging from -5 to 5 on both axes. Gravity pulls objects downward.

**Objects:**
- **Green Ball:** A dynamic ball.
- **Blue Ball:** A dynamic ball, separated horizontally from the green ball.
- Both balls fall under gravity from rest.

**Goal:**
Place a Red Ball so that the Green Ball contacts the Blue Ball and stays in contact for at least 3 seconds.""",

    "cliffhanger": """\
The environment is a 2D box with coordinates ranging from -5 to 5 on both axes. Gravity pulls objects downward.

**Objects:**
- **Green Bar:** A dynamic vertical bar standing upright on a black platform.
- **Black Platform:** A static horizontal platform that the green bar rests on.
- **Ceiling:** A static horizontal bar at the top that limits upward motion.
- **Purple Ground:** The static floor at the very bottom of the box (y ~ -5).

**Goal:**
Place a Red Ball so that the Green Bar falls off the platform and touches the Purple Ground for at least 3 seconds.""",

    "tipping_point": """\
The environment is a 2D box with coordinates ranging from -5 to 5 on both axes. Gravity pulls objects downward.

**Objects:**
- **Green Bar:** A dynamic vertical bar standing upright inside a basket at the bottom of the scene.
- **Basket:** A dynamic U-shaped container at the bottom that holds the green bar.
- **Purple Wall:** A static vertical wall on one side of the scene.

**Goal:**
Place a Red Ball so that the Green Bar tips over and touches the Purple Wall for at least 3 seconds.""",

    "catapult": """\
The environment is a 2D box with coordinates ranging from -5 to 5 on both axes. Gravity pulls objects downward.

**Objects:**
- **Catapult Bar:** A dynamic horizontal bar balanced on a pivot ball, forming a lever/seesaw.
- **Pivot Ball:** A dynamic gray ball on a black platform, acting as the fulcrum for the catapult bar.
- **Green Ball:** A dynamic ball sitting on the left end of the catapult bar (the launch end).
- **Black Platform:** A static platform on the left side supporting the pivot.
- **Ledge:** A static platform on the right side (may be slightly angled).
- **Basket:** A dynamic U-shaped container sitting on the ledge.
- **Blue Ball:** A dynamic ball inside the basket on the ledge.
- **Blocker Ball:** A static black ball at the top of the scene.

**Goal:**
Place a Red Ball so that the Green Ball is launched off the catapult and contacts the Blue Ball for at least 3 seconds.""",
}

# ---------------------------------------------------------------------------
# Tool descriptions — each tool defined individually for level-specific selection
# ---------------------------------------------------------------------------

_TOOL_DESCRIPTIONS = {
    "get_level_state": """\
1. get_level_state
   Description: Get the current level layout including all object positions, sizes, and properties.
   Arguments: None
   Usage: Action: get_level_state
          Action Input: {}""",

    "simulate_action": """\
2. simulate_action
   Description: Place a red ball at (x, y) with the given radius and run the full physics simulation to completion. Returns whether the goal was achieved, final positions of all objects, and total simulation steps.
   Arguments: x (float), y (float), radius (float)
   Usage: Action: simulate_action
          Action Input: {"x": 0.5, "y": 4.0, "radius": 0.6}""",

    "simulate_partial": """\
3. simulate_partial
   Description: Place a red ball and run the simulation only up to the specified step. Returns object positions and velocities at that point. Useful for observing mid-simulation dynamics.
   Arguments: x (float), y (float), radius (float), stop_step (int)
   Usage: Action: simulate_partial
          Action Input: {"x": 0.5, "y": 4.0, "radius": 0.6, "stop_step": 50}""",

    "get_contact_log": """\
4. get_contact_log
   Description: After running a simulation, returns the contact events: which objects touched and when.
   Arguments: None
   Usage: Action: get_contact_log
          Action Input: {}""",

    "compute_gap_analysis": """\
5. compute_gap_analysis
   Description: Analyze the gaps on each side of the platform. Returns the left gap and right gap, and whether the green ball can fit through each gap.
   Arguments: None
   Usage: Action: compute_gap_analysis
          Action Input: {}""",

    "compute_relative_positions": """\
5. compute_relative_positions
   Description: Analyze the positions of the green and blue balls. Returns their coordinates, distance, and relative positioning. Use this to understand where to place the red ball to bridge the gap between them.
   Arguments: None
   Usage: Action: compute_relative_positions
          Action Input: {}""",

    "compute_tipping_analysis": """\
5. compute_tipping_analysis
   Description: Analyze the green bar's position on the platform. Returns which side the bar is near the edge, how far it is from the edge, and which direction to push it.
   Arguments: None
   Usage: Action: compute_tipping_analysis
          Action Input: {}""",

    "compute_wall_distance_analysis": """\
5. compute_wall_distance_analysis
   Description: Analyze the green bar's position relative to the target wall. Returns the bar position, basket position, wall position, and distance to wall.
   Arguments: None
   Usage: Action: compute_wall_distance_analysis
          Action Input: {}""",

    "compute_catapult_analysis": """\
5. compute_catapult_analysis
   Description: Analyze the catapult setup. Returns positions of the catapult bar, pivot, green ball (launch end), basket, blue ball (target), and the horizontal/vertical distances between them.
   Arguments: None
   Usage: Action: compute_catapult_analysis
          Action Input: {}""",

    "finish": """\
6. finish
   Description: Submit your final answer. Use this when you are confident in your solution.
   Arguments: x (float), y (float), radius (float)
   Usage: Action: finish
          Action Input: {"x": 0.5, "y": 4.0, "radius": 0.6}""",
}

# Tools shown per level — only the relevant ones
LEVEL_TOOLS = {
    "down_to_earth":    ["get_level_state", "simulate_action", "simulate_partial", "get_contact_log", "compute_gap_analysis", "finish"],
    "two_body_problem": ["get_level_state", "simulate_action", "simulate_partial", "get_contact_log", "compute_relative_positions", "finish"],
    "cliffhanger":      ["get_level_state", "simulate_action", "simulate_partial", "get_contact_log", "compute_wall_distance_analysis", "finish"],
    "tipping_point":    ["get_level_state", "simulate_action", "simulate_partial", "get_contact_log", "compute_tipping_analysis", "finish"],
    "catapult":         ["get_level_state", "simulate_action", "simulate_partial", "get_contact_log", "compute_catapult_analysis", "finish"],
}

REACT_FORMAT_INSTRUCTIONS = """\
To solve this puzzle, you will reason step-by-step and use tools to test your ideas.

At each step, you MUST follow this exact format:

Thought: <your reasoning about what to do next>
Action: <tool name>
Action Input: <JSON arguments, or {} for tools with no arguments>
Observation: <result from the tool>

Then you continue with another Thought/Action cycle.

When you are confident in your answer, use the "finish" tool to submit it.

CRITICAL: After every Action Input line, you MUST write "Observation:" on the next line and wait for the result. Never skip the Observation line.

Important rules:
- Always start with a Thought before taking an Action.
- Only call ONE tool per step.
- Parse observation results carefully before your next thought.
- You can simulate multiple different actions to compare results.
- Each simulation resets the environment, so previous simulations do not affect new ones.
- When a simulation fails repeatedly, investigate WHY before trying again.
"""

# ---------------------------------------------------------------------------
# Few-shot examples — one per level, showing the full ReAct format including
# "Observation:" after every Action Input. This teaches the model to write
# "\nObservation:" which is the stop token that triggers real tool calls.
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = {
    "two_body_problem": """\
Here is an example interaction showing the correct format:

Thought: I'll start by checking the positions of the green and blue balls.
Action: get_level_state
Action Input: {}
Observation: === Level State: two_body_problem ===
Green Ball: center=(-2.50, 2.00), radius=0.40, dynamic=True
Blue Ball:  center=( 2.50, 2.00), radius=0.40, dynamic=True
Both balls are at the same height and will fall under gravity.

Thought: The balls are 5 units apart horizontally at y=2.0. I need to place a red ball between them so they collide and stay in contact. I'll try placing it at the midpoint with a large enough radius.
Action: simulate_action
Action Input: {"x": 0.0, "y": 1.5, "radius": 0.8}
Observation: === Simulation Result ===
Red ball placed at (0.00, 1.50) with radius=0.80
Simulation ran for 312 steps.
GOAL NOT ACHIEVED — green and blue balls contact duration: 0.6s (need 3.0s)
Final positions: Green=(-1.20, -3.80), Blue=(1.20, -3.80)

Thought: They touched briefly but didn't stay in contact. I need the red ball to be bigger or positioned so both balls land on top of it and stay pressed together. Let me try a larger radius and slightly higher position to catch them earlier.
Action: simulate_action
Action Input: {"x": 0.0, "y": 2.5, "radius": 1.2}
Observation: === Simulation Result ===
Red ball placed at (0.00, 2.50) with radius=1.20
Simulation ran for 289 steps.
GOAL ACHIEVED — green and blue balls contact duration: 5.1s
SUCCESS!

Thought: Placing the red ball at (0.0, 2.5) with radius=1.2 achieves the goal. I'll submit this.
Action: finish
Action Input: {"x": 0.0, "y": 2.5, "radius": 1.2}
Observation: [FINAL_RESULT] SUCCESS — puzzle solved!
""",

    "down_to_earth": """\
Here is an example interaction showing the correct format:

Thought: I need to understand the layout — where is the black platform and what gaps exist for the green ball to fall through?
Action: get_level_state
Action Input: {}
Observation: === Level State: down_to_earth ===
Green Ball: center=(0.00, 3.50), radius=0.40, dynamic=True
Black Platform: center=(0.00, 0.50), half-width=2.00, static=True
Purple Ground: y=-5.00, static=True

Thought: The platform spans from x=-2.0 to x=2.0. The green ball is directly above it. I need to place a red ball to knock the green ball off the platform edge so it falls to the ground.
Action: compute_gap_analysis
Action Input: {}
Observation: === Gap Analysis ===
Platform left edge: x=-2.00, Gap left of platform: 3.00 units (ball radius 0.40 fits: YES)
Platform right edge: x=2.00, Gap right of platform: 3.00 units (ball radius 0.40 fits: YES)
Recommend pushing green ball left or right off platform edge.

Thought: I'll place a red ball just to the right of the green ball to push it leftward off the platform.
Action: simulate_action
Action Input: {"x": 1.0, "y": 3.5, "radius": 0.5}
Observation: === Simulation Result ===
Red ball placed at (1.00, 3.50) with radius=0.50
GOAL ACHIEVED — green ball on purple ground for 4.8s
SUCCESS!

Thought: This works. I'll submit the answer.
Action: finish
Action Input: {"x": 1.0, "y": 3.5, "radius": 0.5}
Observation: [FINAL_RESULT] SUCCESS — puzzle solved!
""",

    "cliffhanger": """\
Here is an example interaction showing the correct format:

Thought: Let me check where the green bar is standing on the platform.
Action: get_level_state
Action Input: {}
Observation: === Level State: cliffhanger ===
Green Bar: center=(1.00, 1.80), half-height=1.20, dynamic=True
Black Platform: center=(0.50, 0.50), half-width=1.50, static=True
Purple Ground: y=-5.00, static=True

Thought: The green bar is near the right edge of the platform. I should push it rightward to tip it off.
Action: simulate_action
Action Input: {"x": -0.5, "y": 2.5, "radius": 0.5}
Observation: === Simulation Result ===
Red ball placed at (-0.50, 2.50) with radius=0.50
GOAL ACHIEVED — green bar on purple ground for 5.2s
SUCCESS!

Thought: This works. I'll submit.
Action: finish
Action Input: {"x": -0.5, "y": 2.5, "radius": 0.5}
Observation: [FINAL_RESULT] SUCCESS — puzzle solved!
""",

    "tipping_point": """\
Here is an example interaction showing the correct format:

Thought: Let me check the positions of the green bar and basket relative to the purple wall.
Action: compute_wall_distance_analysis
Action Input: {}
Observation: === Wall Distance Analysis ===
Green Bar: center=(0.50, -2.00), upright in basket
Basket: center=(0.50, -3.00)
Purple Wall: x=4.50 (right side)
Distance from bar to wall: 4.00 units
Recommendation: push bar rightward to tip it toward the wall.

Thought: I'll place the red ball to the left of the bar to push it rightward toward the purple wall.
Action: simulate_action
Action Input: {"x": -0.5, "y": -0.5, "radius": 0.6}
Observation: === Simulation Result ===
Red ball placed at (-0.50, -0.50) with radius=0.60
GOAL ACHIEVED — green bar touching purple wall for 4.0s
SUCCESS!

Thought: This works. Submitting.
Action: finish
Action Input: {"x": -0.5, "y": -0.5, "radius": 0.6}
Observation: [FINAL_RESULT] SUCCESS — puzzle solved!
""",

    "catapult": """\
Here is an example interaction showing the correct format:

Thought: Let me analyze the catapult setup to understand the geometry.
Action: compute_catapult_analysis
Action Input: {}
Observation: === Catapult Analysis ===
Catapult Bar: center=(0.00, 0.20), half-length=2.50, balanced on pivot
Pivot Ball: center=(0.50, -0.20)
Green Ball (launch end): center=(-2.00, 0.60)
Basket: center=(3.50, 1.50) on ledge
Blue Ball (target): center=(3.50, 2.20)
Horizontal distance launch→target: 5.50 units, Vertical: 1.60 units

Thought: I need to drop a heavy red ball on the right end of the catapult bar to launch the green ball upward toward the blue ball. I'll place it above the right end of the bar.
Action: simulate_action
Action Input: {"x": 2.5, "y": 2.0, "radius": 0.8}
Observation: === Simulation Result ===
Red ball placed at (2.50, 2.00) with radius=0.80
GOAL ACHIEVED — green ball contacts blue ball for 3.5s
SUCCESS!

Thought: This works. Submitting.
Action: finish
Action Input: {"x": 2.5, "y": 2.0, "radius": 0.8}
Observation: [FINAL_RESULT] SUCCESS — puzzle solved!
""",
}

SYSTEM_PROMPT_TEMPLATE = """\
You are a physics reasoning agent solving a 2D physics puzzle. You have access to a physics simulator and can test your ideas before submitting a final answer.

**Puzzle: {level_title}**

{level_description}

**Placement Constraints:**
- The red ball must be completely inside the box: -5 + radius <= x <= 5 - radius, -5 + radius <= y <= 5 - radius.
- The red ball must NOT overlap with any existing objects at t=0.
- 0.1 <= radius <= 2.0

You have access to the following tools to interact with the physics simulation:

{tools}

{react_format}

{few_shot_example}"""

INITIAL_USER_MESSAGE = "Solve the {level_title} puzzle. Use the tools to understand the level and test your ideas, then submit your final answer with the finish tool."


def build_tools_section(level_name: str) -> str:
    tool_names = LEVEL_TOOLS[level_name]
    return "\n\n".join(_TOOL_DESCRIPTIONS[t] for t in tool_names)


from verl_tool.utils.skill_injection import load_skill_bank, format_skills_block


def build_system_prompt(level_name: str, skill_bank: dict = None) -> str:
    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        level_title=LEVEL_TITLES[level_name],
        level_description=LEVEL_DESCRIPTIONS[level_name],
        tools=build_tools_section(level_name),
        react_format=REACT_FORMAT_INSTRUCTIONS,
        few_shot_example=FEW_SHOT_EXAMPLES[level_name],
    )
    if skill_bank:
        skills_block = format_skills_block(level_name, skill_bank)
        if skills_block:
            prompt += "\n\n" + skills_block
    return prompt


def build_initial_user_message(level_name: str) -> str:
    return INITIAL_USER_MESSAGE.format(level_title=LEVEL_TITLES[level_name])


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def make_sample(level_name: str, idx: int, split: str, skill_bank: dict = None,
                runtime_skill_bank_path: str = None) -> dict:
    # Static-skilled variant bakes the bank into the prompt at gen-time.
    # Evolving variant leaves the prompt base-only and stores a path that
    # rl_dataset.py reads at __getitem__ to inject the current bank.
    system_prompt = build_system_prompt(level_name, skill_bank=skill_bank)
    user_message = build_initial_user_message(level_name)
    extra_info = {
        "level_name": level_name,
        "split": split,
        "index": idx,
        "seed": idx,
    }
    if runtime_skill_bank_path:
        extra_info["skill_bank_path"] = runtime_skill_bank_path
    return {
        "data_source": "interphyre",
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "ability": "physics_reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": level_name,
        },
        "extra_info": extra_info,
    }


def generate_dataset(output_dir: str, levels: list, num_train_per_level: int, num_val_per_level: int,
                     train_seed_start: int = 0, val_seed_start: int = None, skill_bank: dict = None,
                     runtime_skill_bank_path: str = None):
    os.makedirs(output_dir, exist_ok=True)

    if val_seed_start is None:
        val_seed_start = train_seed_start + num_train_per_level

    train_records, val_records = [], []

    for level_name in levels:
        for i in range(num_train_per_level):
            train_records.append(make_sample(level_name, train_seed_start + i, "train",
                                             skill_bank=skill_bank,
                                             runtime_skill_bank_path=runtime_skill_bank_path))
        for i in range(num_val_per_level):
            val_records.append(make_sample(level_name, val_seed_start + i, "val",
                                           skill_bank=skill_bank,
                                           runtime_skill_bank_path=runtime_skill_bank_path))

    Dataset.from_list(train_records).to_parquet(os.path.join(output_dir, "train.parquet"))
    Dataset.from_list(val_records).to_parquet(os.path.join(output_dir, "val.parquet"))

    print(f"Saved {len(train_records)} train samples -> {output_dir}/train.parquet")
    print(f"Saved {len(val_records)} val samples   -> {output_dir}/val.parquet")
    print(f"Levels: {levels}")
    print(f"Train per level: {num_train_per_level}, Val per level: {num_val_per_level}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/interphyre")
    parser.add_argument("--levels", type=str, default="two_body_problem",
                        help="Comma-separated list of levels to include")
    parser.add_argument("--num_train_per_level", type=int, default=40)
    parser.add_argument("--num_val_per_level", type=int, default=2)
    parser.add_argument("--train_seed_start", type=int, default=0)
    parser.add_argument("--val_seed_start", type=int, default=None)
    parser.add_argument("--skill_bank", type=str, default=None,
                        help="Path to skill bank JSON to bake into the system prompt at data-gen "
                             "time (static-skilled variant). Mutually exclusive with "
                             "--runtime_skill_bank_path.")
    parser.add_argument("--runtime_skill_bank_path", type=str, default=None,
                        help="Path to skill bank JSON. Stored in extra_info so rl_dataset.py "
                             "reads it fresh on every __getitem__ and injects skills at runtime "
                             "(evolving variant). The file does not need to exist at data-gen time.")
    args = parser.parse_args()
    if args.skill_bank and args.runtime_skill_bank_path:
        parser.error("--skill_bank and --runtime_skill_bank_path are mutually exclusive")
    levels = [l.strip() for l in args.levels.split(",")]
    skill_bank = load_skill_bank(args.skill_bank) if args.skill_bank else None
    generate_dataset(args.output_dir, levels, args.num_train_per_level, args.num_val_per_level,
                     args.train_seed_start, args.val_seed_start, skill_bank=skill_bank,
                     runtime_skill_bank_path=args.runtime_skill_bank_path)
