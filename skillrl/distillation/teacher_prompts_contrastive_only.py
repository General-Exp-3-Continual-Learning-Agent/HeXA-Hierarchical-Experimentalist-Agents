"""Teacher prompts for the *contrastive-only* SkillRL variant.

This variant deliberately removes the reward / confidence machinery used by the
default pipeline. Skills are formed purely by contrasting successful and failed
trajectories — no reward scores, no per-trajectory weighting, no per-skill
confidence calibration.

Provided here:
  - format_trajectory_no_reward / format_trajectories_block_no_reward
        Trajectory rendering helpers that omit the reward header and don't
        re-sort by reward (presentation order is preserved).

  - CONTRASTIVE_DISTILLATION_PROMPT_NO_REWARD
        Round-1 contrastive skill extraction (analogue of the standard
        CONTRASTIVE_DISTILLATION_PROMPT, minus reward language).

  - COMMON_MISTAKES_PROMPT_NO_REWARD
        Round-1 common-mistakes pass on failures (no reward language).

  - format_skill_bank_for_teacher_no_conf
        Renders the previous skill bank for the evolution prompt without
        showing confidence percentages.

  - SKILL_BANK_EVOLUTION_PROMPT_NO_REWARD
        Round-2+ evolution prompt: teacher sees prior bank + new trajectories
        and decides keep/remove/add, without confidence values.
"""

from __future__ import annotations

from skillrl.core.skill_bank import SkillBank


# ── Trajectory rendering (no reward header, no reward sort) ──────────────────

def format_trajectory_no_reward(trajectory: dict, max_steps: int = 10) -> str:
    """Render a single trajectory without a Reward field in the header.

    Truncates to the last ``max_steps`` reasoning steps to stay within context.
    """
    lines = [
        f"--- Seed {trajectory.get('seed', '?')} | "
        f"Success: {trajectory.get('success', '?')} | "
        f"Iterations: {trajectory.get('iterations', '?')} ---"
    ]

    steps = trajectory.get("trajectory", [])
    if len(steps) > max_steps:
        lines.append(f"  (showing last {max_steps} of {len(steps)} steps)\n")
        steps = steps[-max_steps:]

    for i, step in enumerate(steps, 1):
        if isinstance(step, dict):
            thought = step.get("thought", "")
            action = step.get("action", "")
            obs = step.get("observation", "")
        elif isinstance(step, (list, tuple)) and len(step) >= 3:
            thought, action, obs = step[0], step[1], step[2]
        else:
            continue
        if len(obs) > 500:
            obs = obs[:500] + "... [truncated]"
        lines.append(f"  Step {i}:")
        lines.append(f"    Thought: {thought}")
        lines.append(f"    Action: {action}")
        lines.append(f"    Observation: {obs}")
        lines.append("")

    final_action = trajectory.get("action")
    if final_action:
        lines.append(
            f"  Final action: x={final_action[0]}, y={final_action[1]}, r={final_action[2]}"
        )

    return "\n".join(lines)


def format_trajectories_block_no_reward(
    trajectories: list[dict], max_trajs: int = 5
) -> str:
    """Render multiple trajectories without reward-based sorting.

    Order is preserved (caller is responsible for any selection); the first
    ``max_trajs`` trajectories are shown.
    """
    sel = trajectories[:max_trajs] if len(trajectories) > max_trajs else trajectories
    return "\n\n".join(format_trajectory_no_reward(t) for t in sel)


# ── Round 1: contrastive distillation (no reward weighting) ──────────────────

CONTRASTIVE_DISTILLATION_PROMPT_NO_REWARD = """\
You are an expert physics analyst distilling agent behavior into concise, actionable skills.

**Environment**: 2D physics simulation (Box2D). Gravity = -9.8 m/s². World bounds [-5, 5] on both axes. The agent places a red ball at (x, y) with a given radius, then the simulation runs to completion.

**Level: {level_name}**
{level_description}

Below are SUCCESSFUL and FAILED trajectories from the same level. Each shows the agent's reasoning (Thought), actions taken, and simulation observations. **Treat all trajectories as equally informative** — there are no reward scores or quality weights. Your job is to contrast what the successful runs do that the failed runs do not.

=== SUCCESSFUL TRAJECTORIES ({n_successes}) ===
{success_block}

=== FAILED TRAJECTORIES ({n_failures}) ===
{failure_block}

---

Your task: By CONTRASTING the successes and failures, extract the KEY PHYSICS SKILLS that distinguish solving from failing. What insight let the successful agents solve the puzzle? What did the failed agents miss or get wrong?

For each skill, provide:
- **title**: Short name (3-7 words)
- **principle**: The physics insight (2-3 sentences). What mechanism was exploited? Why does it work?
- **when_to_apply**: Specific trigger condition (1 sentence)
- **source_seeds**: List of seed numbers from the trajectories above whose behavior directly motivates this skill (used purely for provenance — do NOT estimate quality from this).

Output a JSON array:
```json
[
  {{
    "title": "...",
    "principle": "...",
    "when_to_apply": "...",
    "source_seeds": [1, 5, 16]
  }}
]
```

Extract 4-6 skills. Each should capture a DISTINCT insight from the success/failure contrast. Avoid redundancy. Do NOT output any "confidence" or "reward" fields.
"""


# ── Round 1: mistakes pass (no reward weighting) ─────────────────────────────

COMMON_MISTAKES_PROMPT_NO_REWARD = """\
You are an expert at analyzing agent failures and distilling them into avoidable mistake patterns.

**Environment**: 2D physics simulation (Box2D). Gravity = -9.8 m/s². World bounds [-5, 5].

**Level: {level_name}**
{level_description}

Below are FAILED trajectories. Each shows the agent's reasoning, actions, and simulation results. There are no reward scores — treat all failures as equally worth analyzing.

{failure_block}

---

Your task has TWO parts:

**Part 1 — Mistakes**: Identify the COMMON MISTAKE PATTERNS across these failures. For each mistake, analyze:
1. What exactly the agent did wrong
2. WHY the agent made this error (what broken causal belief led to it)
3. A concrete actionable fix

**Part 2 — Partial insights**: Even in failed trajectories, some individual steps show CORRECT physics reasoning or useful discoveries (e.g., the agent found a valid placement region but then abandoned it, or correctly identified a mechanism but applied it with wrong parameters). Extract 1-2 skills from these "good steps within bad trajectories". These should be genuine physics insights, not just restating what went wrong.

Format as JSON object with two arrays:
```json
{{
  "mistakes": [
    {{
      "description": "What the mistake is (1 sentence)",
      "why_it_happens": "The broken belief or reasoning error that causes this (1 sentence)",
      "how_to_avoid": "Concrete actionable fix — what to do instead (1-2 sentences)"
    }}
  ],
  "partial_skills": [
    {{
      "title": "Short name (3-7 words)",
      "principle": "The physics insight from the failed trajectory (2-3 sentences)",
      "when_to_apply": "Specific trigger condition (1 sentence)",
      "source_seeds": [5, 11]
    }}
  ]
}}
```

Extract 3-5 mistakes and 1-2 partial skills. Do NOT output any "confidence" or "reward" fields.
"""


# ── Round 2+: evolution (no confidence preservation) ─────────────────────────

def format_skill_bank_for_teacher_no_conf(bank: SkillBank, level_name: str) -> str:
    """Render the existing skill bank for the evolution prompt without confidence.

    Mirrors evolution_prompts.format_skill_bank_for_teacher but omits the
    confidence percentage prefix.
    """
    lines: list[str] = []

    level_skills = bank.level_skills.get(level_name, [])
    if level_skills:
        lines.append("EXISTING LEVEL-SPECIFIC SKILLS:")
        for skill in level_skills:
            principle = skill.principle or ""
            lines.append(f"  - {skill.title}: {principle[:100]}")
            if skill.when_to_apply:
                lines.append(f"      Apply when: {skill.when_to_apply}")
        lines.append("")

    level_mistakes = bank.level_mistakes.get(level_name, [])
    if level_mistakes:
        lines.append("EXISTING MISTAKES:")
        for mistake in level_mistakes:
            lines.append(f"  - {mistake.description}")
        lines.append("")

    if not level_skills and not level_mistakes:
        lines.append("(No existing skills or mistakes yet)")

    return "\n".join(lines)


SKILL_BANK_EVOLUTION_PROMPT_NO_REWARD = """\
You are a physics teacher evolving a skill bank for puzzle-solving agents.

LEVEL: {level_name}
LEVEL DESCRIPTION: {level_description}

CURRENT SKILL BANK (from previous rounds):
{existing_skills_block}

NEW TRAJECTORIES (from the latest round):
Successes: {n_successes}
Failures: {n_failures}

{new_trajectories_block}

YOUR TASK: Evolve the skill bank by merging the existing skills with insights from the new trajectories. There are NO reward scores or confidence values — treat every trajectory and every existing skill as equally weighted, and decide what to keep / add / remove based purely on the physics evidence.

RULES:
1. Output the COMPLETE FINAL skill bank (not a diff) — include both retained existing skills and any new ones you want to add.
2. Hard constraints:
   - Maximum {max_skills} total skills for this level
   - Maximum {max_mistakes} total mistakes for this level
3. For each skill you include:
   - If it's a RETAINED skill from the existing bank: set "is_new": false and copy its title/principle/when_to_apply faithfully (you may lightly tighten the wording).
   - If it's a NEW skill extracted from the new trajectories: set "is_new": true.
   - Include "source_seeds" listing seed numbers where this skill is observed (provenance only).
   - Do NOT output any "confidence" or "reward" fields.
4. Drop a skill if it is:
   - Redundant or fully subsumed by another skill in your output
   - Directly contradicted by the new trajectories
   - Too narrow to be reused on future seeds
5. Add new skills when the new trajectories clearly demonstrate an insight not already in the bank.
6. Mistakes follow the same logic — keep the ones the new failures still exemplify, add new ones for new failure modes.

OUTPUT JSON OBJECT:
{{
  "skills": [
    {{
      "title": "<short name of skill>",
      "principle": "<2-3 sentence physics insight>",
      "when_to_apply": "<condition for applicability>",
      "example": "<optional concrete coordinate example>",
      "source_seeds": [<seed numbers>],
      "is_new": <true|false>
    }}
    ...
  ],
  "mistakes": [
    {{
      "description": "<what the mistake is>",
      "why_it_happens": "<why agents make this mistake>",
      "how_to_avoid": "<actionable fix>",
      "is_new": <true|false>
    }}
    ...
  ],
  "removed_skill_titles": ["<title of removed skill 1>", ...],
  "reasoning": "<brief explanation of key changes: what was removed, what was added, why>"
}}

Be concise but precise. Focus on physics insights that directly help puzzle-solving.
"""


# ─────────────────────────────────────────────────────────────────────────
# Catapult-specific variants
#
# Mirror the discipline of `teacher_prompts_catapult.py`: bake the FACTUAL
# catapult scene block (no strategy hints) into the prompts so the teacher
# discovers strategies from the trajectories alone. Combined with the
# contrastive-only ablation's removal of rewards and confidence.
# ─────────────────────────────────────────────────────────────────────────

from skillrl.distillation.teacher_prompts_catapult import CATAPULT_LEVEL_BLOCK


CONTRASTIVE_DISTILLATION_PROMPT_NO_REWARD_CATAPULT = """\
You are an expert physics analyst distilling agent behavior into concise, actionable skills.

**Environment**: 2D physics simulation (Box2D). Gravity = -9.8 m/s². World bounds [-5, 5] on both axes. The agent places a red ball at (x, y) with a given radius, then the simulation runs to completion.

**Level: {level_name}**
""" + CATAPULT_LEVEL_BLOCK + """

Below are SUCCESSFUL and FAILED trajectories from this catapult level. Each shows the agent's reasoning (Thought), actions taken, and simulation observations. **Treat all trajectories as equally informative** — there are no reward scores or quality weights. Your job is to contrast what the successful runs do that the failed runs do not.

=== SUCCESSFUL TRAJECTORIES ({n_successes}) ===
{success_block}

=== FAILED TRAJECTORIES ({n_failures}) ===
{failure_block}

---

Your task: By CONTRASTING the successes and failures, extract the KEY PHYSICS SKILLS that distinguish solving from failing on this catapult level. Let the trajectories — not your prior expectations about catapults — determine the mechanisms that actually work.

For each skill, provide:
- **title**: Short name (3-7 words)
- **principle**: The physics insight (2-3 sentences). What mechanism was exploited? Why does it work?
- **when_to_apply**: Specific trigger condition (1 sentence)
- **source_seeds**: List of seed numbers from the trajectories above whose behavior directly motivates this skill (used purely for provenance — do NOT estimate quality from this).

Output a JSON array:
```json
[
  {{
    "title": "...",
    "principle": "...",
    "when_to_apply": "...",
    "source_seeds": [1, 5, 16]
  }}
]
```

Extract 4-6 skills. Each should capture a DISTINCT insight from the success/failure contrast. Avoid redundancy. Do NOT output any "confidence" or "reward" fields.
"""


COMMON_MISTAKES_PROMPT_NO_REWARD_CATAPULT = """\
You are an expert at analyzing agent failures and distilling them into avoidable mistake patterns.

**Environment**: 2D physics simulation (Box2D). Gravity = -9.8 m/s². World bounds [-5, 5].

**Level: {level_name}**
""" + CATAPULT_LEVEL_BLOCK + """

Below are FAILED trajectories from this catapult level. Each shows the agent's reasoning, actions, and simulation results. There are no reward scores — treat all failures as equally worth analyzing.

{failure_block}

---

Your task has TWO parts:

**Part 1 — Mistakes**: Identify the COMMON MISTAKE PATTERNS across these failures. For each mistake, analyze:
1. What exactly the agent did wrong
2. WHY the agent made this error (what broken causal belief led to it)
3. A concrete actionable fix

**Part 2 — Partial insights**: Even in failed trajectories, some individual steps show CORRECT physics reasoning or useful discoveries (e.g., the agent found a promising placement region but then abandoned it, or correctly identified a mechanism but applied it with wrong parameters). Extract 1-2 skills from these "good steps within bad trajectories". These should be genuine physics insights, not just restating what went wrong.

Format as a JSON object with two arrays:
```json
{{
  "mistakes": [
    {{
      "description": "What the mistake is (1 sentence)",
      "why_it_happens": "The broken belief or reasoning error that causes this (1 sentence)",
      "how_to_avoid": "Concrete actionable fix — what to do instead (1-2 sentences)"
    }}
  ],
  "partial_skills": [
    {{
      "title": "Short name (3-7 words)",
      "principle": "The physics insight from the failed trajectory (2-3 sentences)",
      "when_to_apply": "Specific trigger condition (1 sentence)",
      "source_seeds": [5, 11]
    }}
  ]
}}
```

Extract 3-5 mistakes and 1-2 partial skills. Do NOT output any "confidence" or "reward" fields.
"""


SKILL_BANK_EVOLUTION_PROMPT_NO_REWARD_CATAPULT = """\
You are a physics teacher evolving a skill bank for a catapult-puzzle-solving agent.

**Environment**: 2D physics simulation (Box2D). Gravity = -9.8 m/s². World bounds [-5, 5].

**Level: {level_name}**
""" + CATAPULT_LEVEL_BLOCK + """

CURRENT SKILL BANK (from previous rounds):
{existing_skills_block}

NEW TRAJECTORIES (from the latest round):
Successes: {n_successes}
Failures: {n_failures}

{new_trajectories_block}

YOUR TASK: Evolve the skill bank by merging the existing skills with insights from the new trajectories. There are NO reward scores or confidence values — treat every trajectory and every existing skill as equally weighted, and decide what to keep / add / remove based purely on the physics evidence.

RULES:
1. Output the COMPLETE FINAL skill bank (not a diff) — include both retained existing skills and any new ones you want to add.
2. Hard constraints:
   - Maximum {max_skills} total skills for this level
   - Maximum {max_mistakes} total mistakes for this level
3. For each skill you include:
   - If it's a RETAINED skill from the existing bank: set "is_new": false and copy its title/principle/when_to_apply faithfully (you may lightly tighten the wording).
   - If it's a NEW skill extracted from the new trajectories: set "is_new": true.
   - Include "source_seeds" listing seed numbers where this skill is observed (provenance only).
   - Do NOT output any "confidence" or "reward" fields.
4. Drop a skill if it is:
   - Redundant or fully subsumed by another skill in your output
   - Directly contradicted by the new trajectories
   - Too narrow to be reused on future seeds
5. Add new skills when the new trajectories clearly demonstrate an insight not already in the bank.
6. Mistakes follow the same logic — keep the ones the new failures still exemplify, add new ones for new failure modes.

OUTPUT JSON OBJECT:
{{
  "skills": [
    {{
      "title": "<short name of skill>",
      "principle": "<2-3 sentence physics insight>",
      "when_to_apply": "<condition for applicability>",
      "example": "<optional concrete coordinate example>",
      "source_seeds": [<seed numbers>],
      "is_new": <true|false>
    }}
    ...
  ],
  "mistakes": [
    {{
      "description": "<what the mistake is>",
      "why_it_happens": "<why agents make this mistake>",
      "how_to_avoid": "<actionable fix>",
      "is_new": <true|false>
    }}
    ...
  ],
  "removed_skill_titles": ["<title of removed skill 1>", ...],
  "reasoning": "<brief explanation of key changes: what was removed, what was added, why>"
}}

Be concise but precise. Focus on physics insights that directly help puzzle-solving.
"""


def patch() -> None:
    """Swap the catapult-specific contrastive-only prompts into the consumers.

    `contrastive_only_distill.py` imports the round-1 prompts by value, and
    `contrastive_only_evolve.py` imports the round-2+ prompt by value. To swap
    them at runtime we patch the consumer modules' globals. Call this BEFORE
    invoking `run_distillation_contrastive_only()` or
    `evolve_skill_bank_contrastive_only()`.

    Safe and idempotent.
    """
    import skillrl.distillation.contrastive_only_distill as _co_distill
    import skillrl.distillation.contrastive_only_evolve as _co_evolve

    _co_distill.CONTRASTIVE_DISTILLATION_PROMPT_NO_REWARD = (
        CONTRASTIVE_DISTILLATION_PROMPT_NO_REWARD_CATAPULT
    )
    _co_distill.COMMON_MISTAKES_PROMPT_NO_REWARD = (
        COMMON_MISTAKES_PROMPT_NO_REWARD_CATAPULT
    )
    _co_evolve.SKILL_BANK_EVOLUTION_PROMPT_NO_REWARD = (
        SKILL_BANK_EVOLUTION_PROMPT_NO_REWARD_CATAPULT
    )
