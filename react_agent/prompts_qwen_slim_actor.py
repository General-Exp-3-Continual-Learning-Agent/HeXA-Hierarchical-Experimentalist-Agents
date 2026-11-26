"""Actor-side prompt slimming for small (~3B) Qwen actors.

The mirror of ``skillrl.distillation.teacher_prompts_qwen_slim``, which does the
same job for the teacher. Both exist because a 3B model loses the task in a long
prompt; neither changes the method.

Applied as a TRANSFORM over the stock prompts rather than as a second copy of
each level's text, so ``react_agent/level_prompts/`` stays the single source of
truth and no other model or level is affected. Three changes:

  1. The placement-constraints block is tightened and points at the legal ranges
     the rejection prints. The radius bound itself now lives in the STOCK
     down_to_earth prompt for every model (it is a rule the simulator already
     enforces, not a hint): in dte_hexa_evolution_rep1, where it was stated
     nowhere, 270 of the 500 rejections were out-of-range radii.
  2. The redundant per-tool ``Usage:`` lines and the restated ReAct rules are
     dropped in favour of one worked example of each call form. Shortens the
     prompt without removing a single fact.
  3. The first user message drops the goal restatement already in the system
     prompt.

Every substitution asserts that it matched. A silent no-op would leave a run
looking slimmed when it was not, which is worse than a crash at episode start.
"""

from __future__ import annotations

import re

# Reuse the teacher-side detector so one regex decides "is this a small Qwen".
from skillrl.distillation.teacher_prompts_qwen_slim import (  # noqa: F401
    is_small_qwen_teacher as _is_small_qwen,
)


def is_small_qwen_actor(model: str | None) -> bool:
    """True when `model` names a ~3B Qwen, e.g. ``openai:Qwen/Qwen2.5-3B-Instruct``."""
    return _is_small_qwen(model)


# ── 1. placement constraints, with the radius bound stated ───────────────
_DTE_CONSTRAINTS_OLD = """\
**Placement Constraints:**
- The red ball's radius must be between 0.1 and 2.0.
- The red ball must be completely inside the box: -5 + radius <= x <= 5 - radius, -5 + radius <= y <= 5 - radius.
- The red ball must NOT overlap with the green ball at t=0: distance between centers > sum of radii.
- The red ball must NOT overlap with the black platform at t=0."""

_DTE_CONSTRAINTS_NEW = """\
**Placement Constraints (the simulator enforces these and rejects anything that breaks them):**
- Radius must be in [0.1, 2.0].
- The whole ball must be inside the box: -5 + radius <= x <= 5 - radius, and the same for y.
- At t=0 the red ball must not overlap the green ball (distance between centers > sum of radii) or the black platform.
- A rejected placement is NOT simulated, and the rejection lists the legal x ranges for your radius -- take your next value from those ranges instead of guessing again."""

# ── 2. one worked example instead of per-tool Usage lines ────────────────
_FORMAT_OLD_START = "To solve this puzzle, you will reason step-by-step"
_FORMAT_NEW = """\
Reason step by step and use the tools to test your ideas. At each step you MUST
use this exact format, one tool per step:

Thought: <your reasoning about what to do next>
Action: <tool name>
Action Input: <JSON arguments; omit this line for tools that take none>

You will then receive:

Observation: <result from the tool>

Example of a tool with arguments:

Thought: I will test a placement with the simulator.
Action: simulate_action
Action Input: {"x": 0.5, "y": 4.0, "radius": 0.6}

Example of a tool without arguments:

Thought: I need the scene layout first.
Action: get_level_state

Each simulation resets the environment, so earlier simulations do not affect
later ones. Submit your answer with the finish tool."""

_USAGE_LINE_RE = re.compile(
    r"\n   Usage: Action: [^\n]*(?:\n *Action Input: [^\n]*)?", re.MULTILINE
)


def slim_system_prompt(system_prompt: str, level_name: str = "") -> str:
    """Return `system_prompt` with the slim-actor transforms applied."""
    out = system_prompt

    # (1) constraints — down_to_earth only; other levels word theirs differently
    # and get no constraint rewrite until one is written and checked for them.
    if _DTE_CONSTRAINTS_OLD in out:
        out = out.replace(_DTE_CONSTRAINTS_OLD, _DTE_CONSTRAINTS_NEW)
    elif level_name == "down_to_earth":
        raise AssertionError(
            "slim_system_prompt: down_to_earth constraints block not found — "
            "the level prompt changed, so this transform needs updating."
        )

    # (2a) drop the per-tool Usage lines
    out, n_usage = _USAGE_LINE_RE.subn("", out)
    if n_usage == 0:
        raise AssertionError("slim_system_prompt: no 'Usage:' lines matched.")

    # (2b) swap the ReAct format block for the compact one
    idx = out.find(_FORMAT_OLD_START)
    if idx == -1:
        raise AssertionError("slim_system_prompt: ReAct format block not found.")
    out = out[:idx] + _FORMAT_NEW + "\n"

    return out


def slim_initial_message(initial_message: str, max_iterations: int) -> str:
    """Drop the goal restatement; the system prompt already states the goal."""
    return (
        f"Solve the puzzle. You have {max_iterations} iterations. Start by "
        f"inspecting the level state, then test placements with the simulator "
        f"and submit the one that works with finish."
    )


# ── 3. the skill-block caveat (passed to format_skills_as_prompt) ────────
EXAMPLE_CAVEAT = (
    "An Example line shows the FORM of a call from an earlier seed. That seed's "
    "scene had different object positions, so its x, y and radius do not apply "
    "here: recompute them from this seed's level state. Never copy an example's "
    "numbers unchanged."
)
