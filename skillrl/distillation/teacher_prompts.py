"""Prompt templates for the teacher model used in skill distillation and evolution.

Three distillation passes per level (matching the SkillRL paper structure):
1. Contrastive skill extraction — successes + failures shown together
2. Common mistakes — dedicated pass on failures with richer schema
3. Cross-level generalization — general physics principles from all levels
"""

# ── Phase 1a: Contrastive skill distillation (successes + failures together) ─

CONTRASTIVE_DISTILLATION_PROMPT = """\
You are an expert physics analyst distilling agent behavior into concise, actionable skills.

**Environment**: 2D physics simulation (Box2D). Gravity = -9.8 m/s². World bounds [-5, 5] on both axes. The agent places a red ball at (x, y) with a given radius, then the simulation runs to completion.

**Level: {level_name}**
{level_description}

Below are SUCCESSFUL and FAILED trajectories from the same level. Each shows the agent's reasoning (Thought), actions taken, and simulation observations. Each trajectory has a **Reward** score:
- Successes: +1.0 (solved fast, 1-3 iters) to +0.25 (solved slowly, 16-25 iters)
- Failures: -0.5 (tried all iterations) to -0.75 (gave up early)

**Weight your analysis by reward** — skills from high-reward trajectories (fast solves) are more reliable than skills from low-reward ones (barely solved).

=== SUCCESSFUL TRAJECTORIES ({n_successes}) ===
{success_block}

=== FAILED TRAJECTORIES ({n_failures}) ===
{failure_block}

---

Your task: By CONTRASTING the successes and failures, extract the KEY PHYSICS SKILLS that distinguish solving from failing. Focus especially on high-reward successes — what insight let the agent solve it quickly? What did the failed agents miss?

For each skill, provide:
- **title**: Short name (3-7 words)
- **principle**: The physics insight (2-3 sentences). What mechanism was exploited? Why does it work?
- **when_to_apply**: Specific trigger condition (1 sentence)
- **source_seeds**: List of seed numbers from the trajectories above that this skill was primarily derived from. Include only the seeds whose behavior directly demonstrates or motivates this skill.

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

Extract 4-6 skills. Each should capture a DISTINCT insight from the success/failure contrast. Avoid redundancy.
"""

# ── Phase 1b: Common mistakes extraction ───────────────────────────────

COMMON_MISTAKES_PROMPT = """\
You are an expert at analyzing agent failures and distilling them into avoidable mistake patterns.

**Environment**: 2D physics simulation (Box2D). Gravity = -9.8 m/s². World bounds [-5, 5].

**Level: {level_name}**
{level_description}

Below are FAILED trajectories. Each shows the agent's reasoning, actions, and simulation results.

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

Extract 3-5 mistakes and 1-2 partial skills. For mistakes, group similar failures into one and focus on ROOT CAUSES. For partial skills, only extract genuinely useful insights — do not force it if no good steps exist.
"""

# ── Phase 1c: Cross-level generalization ───────────────────────────────

CROSS_LEVEL_GENERALIZATION_PROMPT = """\
Below are level-specific physics skills extracted from multiple puzzle types
in a 2D physics simulation environment (gravity = -9.8 m/s², world bounds [-5,5]).

{all_level_skills}

---

Your task: Identify 5-10 GENERAL physics principles that apply across multiple
levels. These should be fundamental insights about:
- Collision mechanics (momentum transfer, impact angles, center-to-center impulse)
- Gravity and free-fall trajectories
- Lever/catapult mechanics
- Object placement strategy (offsets, heights, radii as mass proxy)
- Diagnostic reasoning (how to interpret simulation failures)

For each general skill:
- **title**: Short name (3-7 words)
- **principle**: The fundamental physics insight (2-3 sentences)
- **when_to_apply**: Broad condition for applicability (1 sentence)
- **example**: A concrete example from any level illustrating the principle

Output a JSON array:
```json
[
  {{
    "title": "...",
    "principle": "...",
    "when_to_apply": "...",

  }}
]
```
"""

# ── Phase 3: Evolution — generate skills for weak levels ───────────────

EVOLUTION_PROMPT = """\
The agent is performing poorly on the following level.

**Level**: {level_name}
{level_description}
**Current accuracy**: {accuracy:.0%}

Here are the failure trajectories from the latest evaluation:

{failure_trajectories}

Here are the EXISTING skills for this level (already in the skill bank):

{existing_skills}

Here are the EXISTING mistakes for this level:

{existing_mistakes}

---

Your task: Generate 2-3 NEW skills and 1-2 NEW mistakes that specifically address
the failure modes observed above. The new entries must NOT duplicate existing ones.

Analyze:
1. What specific misconceptions or reasoning errors caused these failures?
2. What physics principles is the agent failing to apply correctly?
3. What concrete, actionable guidance would prevent these failures?

Output as JSON with two arrays:
```json
{{
  "skills": [
    {{
      "title": "...",
      "principle": "...",
      "when_to_apply": "...",
    }}
  ],
  "mistakes": [
    {{
      "description": "...",
      "why_it_happens": "...",
      "how_to_avoid": "..."
    }}
  ]
}}
```
"""


# ── Reward computation ─────────────────────────────────────────────────

def compute_trajectory_reward(trajectory: dict) -> float:
    """Compute a reward score for a trajectory based on outcome and iterations.

    Successful trajectories: reward inversely proportional to iterations.
      1-3 iters → 1.0  |  4-7 → 0.75  |  8-15 → 0.5  |  16-25 → 0.25

    Failed trajectories: penalized, slightly less harsh if agent explored fully.
      Used all 25 iterations → -0.5  (at least tried)
      Gave up early (<10) → -0.75  (didn't explore)
    """
    success = trajectory.get("success", False)
    iters = trajectory.get("iterations", 25)

    if success:
        if iters <= 3:
            return 1.0
        elif iters <= 7:
            return 0.75
        elif iters <= 15:
            return 0.5
        else:
            return 0.25
    else:
        if iters < 10:
            return -0.75
        else:
            return -0.5


# ── Trajectory formatting helpers ──────────────────────────────────────

def format_trajectory_for_teacher(trajectory: dict, max_steps: int = 10) -> str:
    """Format a single trajectory dict into a readable text block for the teacher.

    Includes reward score so the teacher can weight trajectories by quality.
    Truncates to the last *max_steps* reasoning steps to stay within context limits.
    """
    reward = compute_trajectory_reward(trajectory)
    lines = [f"--- Seed {trajectory.get('seed', '?')} | "
             f"Success: {trajectory.get('success', '?')} | "
             f"Iterations: {trajectory.get('iterations', '?')} | "
             f"Reward: {reward:+.2f} ---"]

    steps = trajectory.get("trajectory", [])
    if len(steps) > max_steps:
        lines.append(f"  (showing last {max_steps} of {len(steps)} steps)\n")
        steps = steps[-max_steps:]

    for i, step in enumerate(steps, 1):
        # Handle both dict format {"thought":..., "action":..., "observation":...}
        # and list/tuple format [thought, action, observation]
        if isinstance(step, dict):
            thought = step.get("thought", "")
            action = step.get("action", "")
            obs = step.get("observation", "")
        elif isinstance(step, (list, tuple)) and len(step) >= 3:
            thought, action, obs = step[0], step[1], step[2]
        else:
            continue
        # Truncate long observations
        if len(obs) > 500:
            obs = obs[:500] + "... [truncated]"
        lines.append(f"  Step {i}:")
        lines.append(f"    Thought: {thought}")
        lines.append(f"    Action: {action}")
        lines.append(f"    Observation: {obs}")
        lines.append("")

    final_action = trajectory.get("action")
    if final_action:
        lines.append(f"  Final action: x={final_action[0]}, y={final_action[1]}, r={final_action[2]}")

    return "\n".join(lines)


def format_trajectories_block(trajectories: list[dict], max_trajs: int = 5) -> str:
    """Format multiple trajectories into a single block for the teacher prompt.

    Sorts by reward (highest first) so the teacher sees the best evidence first.
    """
    sorted_trajs = sorted(
        trajectories, key=lambda t: compute_trajectory_reward(t), reverse=True
    )
    if len(sorted_trajs) > max_trajs:
        sorted_trajs = sorted_trajs[:max_trajs]
    return "\n\n".join(
        format_trajectory_for_teacher(t) for t in sorted_trajs
    )
