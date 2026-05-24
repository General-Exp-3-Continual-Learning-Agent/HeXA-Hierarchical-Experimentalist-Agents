"""Teacher prompts for the *skills-only* ablation of HeXA.

Two ways this ablation differs from the standard evolving distill (whose prompts
live in `teacher_prompts.py` and `evolution_prompts.py`):

1. **Trajectory filter** — only SUCCESSFUL trajectories are sent to the teacher
   (failures are dropped entirely before formatting).
2. **Output schema** — the teacher emits ONLY level-specific skills. There is
   no `mistakes` block, no `common_mistakes`, and no contrastive failure
   analysis. The resulting SkillBank therefore has zero Mistake entries for
   the level.

The round-1 prompt is adapted from `CONTRASTIVE_DISTILLATION_PROMPT` (with the
failure contrast removed), and the round-2+ prompt is adapted from
`SKILL_BANK_EVOLUTION_PROMPT` (with the mistakes section removed).
"""

from __future__ import annotations

# ── Round 1: success-only distillation ──────────────────────────────────

SUCCESS_ONLY_DISTILLATION_PROMPT = """\
You are an expert physics analyst distilling agent behavior into concise, actionable skills.

**Environment**: 2D physics simulation (Box2D). Gravity = -9.8 m/s². World bounds [-5, 5] on both axes. The agent places a red ball at (x, y) with a given radius, then the simulation runs to completion.

**Level: {level_name}**
{level_description}

Below are SUCCESSFUL trajectories from the same level. Each shows the agent's reasoning (Thought), actions taken, and simulation observations. Each trajectory has a **Reward** score:
- +1.0 (solved fast, 1-3 iters) to +0.25 (solved slowly, 16-25 iters)

**Weight your analysis by reward** — skills from high-reward trajectories (fast solves) are more reliable than skills from low-reward ones (barely solved).

=== SUCCESSFUL TRAJECTORIES ({n_successes}) ===
{success_block}

---

Your task: From these SUCCESSFUL trajectories, extract the KEY PHYSICS SKILLS that explain why these solves worked. Focus especially on high-reward successes — what insight let the agent solve it quickly? What mechanism was exploited?

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

Extract 4-6 skills. Each should capture a DISTINCT physics insight from the successful trajectories. Avoid redundancy. Do NOT include any "mistakes" or "common errors" — emit only positive skills.
"""

# ── Round 2+: success-only, skills-only evolution ───────────────────────

SKILL_BANK_EVOLUTION_SKILLS_ONLY_PROMPT = """\
You are a physics teacher evolving a skill bank for puzzle-solving agents.

LEVEL: {level_name}
LEVEL DESCRIPTION: {level_description}

CURRENT SKILL BANK (from previous rounds):
{existing_skills_block}

NEW SUCCESSFUL TRAJECTORIES (from the latest round):
Successes: {n_successes}

{new_trajectories_block}

YOUR TASK: Evolve the LEVEL-SPECIFIC SKILLS by merging the existing skills with insights from the new successful trajectories. Do NOT emit any mistakes or common-error entries — this ablation evaluates a skills-only bank.

RULES:
1. Output the COMPLETE FINAL list of skills (not a diff) — include both retained existing skills and any new ones extracted from the new successes.
2. Hard constraint: maximum {max_skills} total skills for this level.
3. For each skill you include:
   - If it's a RETAINED skill from the existing bank: set "is_new": false
   - If it's a NEW skill extracted from the new successful trajectories: set "is_new": true
   - Include "source_seeds" listing seed numbers from the NEW trajectories where this skill was observed (required for confidence calibration of new skills)
   - Include "confidence": a float in [0.1, 1.0] representing your confidence in this skill
4. For retained skills, preserve their existing confidence values (they've been validated by earlier rounds).
5. For new skills, estimate confidence based on:
   - How many of the new successes corroborate the underlying mechanism
   - Universality (applies across multiple seed conditions = higher confidence)
   - Clarity and actionability of the principle
6. Do not include duplicate skills. If a new trajectory confirms an existing skill, keep the existing one (possibly with slightly higher confidence).
7. Remove skills that are:
   - Redundant or subsumed by other skills
   - Contradicted by the new successes (e.g. a skill said "always X" but the new solves used "not X")
   - Too specific or rarely applicable
   - Low confidence (< 0.3) and not directly observed in the new trajectories

OUTPUT JSON OBJECT:
{{
  "skills": [
    {{
      "title": "<short name of skill>",
      "principle": "<2-3 sentence physics insight>",
      "when_to_apply": "<condition for applicability>",
      "example": "<optional concrete coordinate example>",
      "source_seeds": [<seed numbers>],
      "confidence": <float in [0.1, 1.0]>,
      "is_new": <true|false>
    }}
    ...
  ],
  "removed_skill_titles": ["<title of removed skill 1>", ...],
  "reasoning": "<brief explanation of key changes: what was removed, what was added, why>"
}}

Be concise but precise. Focus on physics insights that directly help puzzle-solving. DO NOT include a "mistakes" array — this ablation forbids it.
"""


def format_skill_bank_for_teacher_skills_only(bank, level_name: str) -> str:
    """Format the existing skill bank for the teacher prompt, **skills only**.

    Mirrors `evolution_prompts.format_skill_bank_for_teacher` but never emits
    a MISTAKES section — matching the ablation's bank shape.
    """
    lines = []
    level_skills = bank.level_skills.get(level_name, [])
    if level_skills:
        lines.append("EXISTING LEVEL-SPECIFIC SKILLS:")
        for skill in level_skills:
            conf_pct = f"{skill.confidence * 100:.0f}%"
            lines.append(f"  - [{conf_pct}] {skill.title}: {skill.principle[:100]}")
        lines.append("")
    else:
        lines.append("(No existing skills yet)")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# Catapult-specific variants
#
# Mirror the discipline of `teacher_prompts_catapult.py`: bake the FACTUAL
# catapult scene block (no strategy hints) into the prompts so the teacher
# discovers strategies from the trajectories alone. Combined with the
# skills-only ablation's input filter (no failures) and output filter
# (no mistakes).
# ─────────────────────────────────────────────────────────────────────────

from skillrl.distillation.teacher_prompts_catapult import CATAPULT_LEVEL_BLOCK


SUCCESS_ONLY_DISTILLATION_PROMPT_CATAPULT = """\
You are an expert physics analyst distilling agent behavior into concise, actionable skills.

**Environment**: 2D physics simulation (Box2D). Gravity = -9.8 m/s². World bounds [-5, 5] on both axes. The agent places a red ball at (x, y) with a given radius, then the simulation runs to completion.

**Level: {level_name}**
""" + CATAPULT_LEVEL_BLOCK + """

Below are SUCCESSFUL trajectories from this catapult level. Each shows the agent's reasoning (Thought), actions taken, and simulation observations. Each trajectory has a **Reward** score:
- +1.0 (solved fast, 1-3 iters) to +0.25 (solved slowly, 16-25 iters)

**Weight your analysis by reward** — skills from high-reward trajectories (fast solves) are more reliable than skills from low-reward ones.

=== SUCCESSFUL TRAJECTORIES ({n_successes}) ===
{success_block}

---

Your task: From these SUCCESSFUL trajectories, extract the KEY PHYSICS SKILLS that explain why these solves worked on the catapult. Let the trajectories — not your prior expectations about catapults — determine the mechanisms that actually work.

For each skill, provide:
- **title**: Short name (3-7 words)
- **principle**: The physics insight (2-3 sentences). What mechanism was exploited? Why does it work?
- **when_to_apply**: Specific trigger condition (1 sentence)
- **source_seeds**: Seed numbers from the trajectories above that this skill was primarily derived from.

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

Extract 4-6 skills. Each should capture a DISTINCT physics insight from the successful trajectories. Avoid redundancy. Do NOT include any "mistakes" or "common errors" — emit only positive skills.
"""


SKILL_BANK_EVOLUTION_SKILLS_ONLY_PROMPT_CATAPULT = """\
You are a physics teacher evolving a skill bank for a catapult-puzzle-solving agent.

**Environment**: 2D physics simulation (Box2D). Gravity = -9.8 m/s². World bounds [-5, 5].

**Level: {level_name}**
""" + CATAPULT_LEVEL_BLOCK + """

CURRENT SKILL BANK (from previous rounds):
{existing_skills_block}

NEW SUCCESSFUL TRAJECTORIES (from the latest round):
Successes: {n_successes}

{new_trajectories_block}

YOUR TASK: Evolve the LEVEL-SPECIFIC SKILLS by merging the existing skills with insights from the new successful trajectories. Do NOT emit any mistakes or common-error entries — this ablation evaluates a skills-only bank for the catapult level.

RULES:
1. Output the COMPLETE FINAL list of skills (not a diff) — include both retained existing skills and any new ones extracted from the new successes.
2. Hard constraint: maximum {max_skills} total skills for this level.
3. For each skill you include:
   - If it's a RETAINED skill from the existing bank: set "is_new": false
   - If it's a NEW skill extracted from the new successful trajectories: set "is_new": true
   - Include "source_seeds" listing seed numbers from the NEW trajectories where this skill was observed.
   - Include "confidence": a float in [0.1, 1.0].
4. For retained skills, preserve their existing confidence values (they've been validated by earlier rounds).
5. For new skills, estimate confidence based on:
   - How many of the new successes corroborate the underlying mechanism
   - Universality (applies across multiple seed conditions = higher confidence)
   - Clarity and actionability of the principle
6. Do not include duplicate skills. If a new trajectory confirms an existing skill, keep the existing one (possibly with slightly higher confidence).
7. Remove skills that are:
   - Redundant or subsumed by another skill
   - Contradicted by the new successes
   - Too specific or rarely applicable
   - Low confidence (< 0.3) and not directly observed in the new trajectories

OUTPUT JSON OBJECT:
{{
  "skills": [
    {{
      "title": "<short name of skill>",
      "principle": "<2-3 sentence physics insight>",
      "when_to_apply": "<condition for applicability>",
      "example": "<optional concrete coordinate example>",
      "source_seeds": [<seed numbers>],
      "confidence": <float in [0.1, 1.0]>,
      "is_new": <true|false>
    }}
    ...
  ],
  "removed_skill_titles": ["<title of removed skill 1>", ...],
  "reasoning": "<brief explanation of key changes: what was removed, what was added, why>"
}}

Be concise but precise. Focus on physics insights grounded in the trajectories. DO NOT include a "mistakes" array — this ablation forbids it.
"""


def patch() -> None:
    """Swap the catapult-specific skills-only prompts into the distill module.

    `skills_only_distill.py` imports the prompt names by value (standard
    Python `from X import Y`), so to swap them at runtime we patch the
    consumer module's globals. Call this BEFORE invoking
    `run_skills_only_distillation()` or `evolve_skill_bank_skills_only()`.

    Safe and idempotent.
    """
    import skillrl.distillation.skills_only_distill as _so
    _so.SUCCESS_ONLY_DISTILLATION_PROMPT = SUCCESS_ONLY_DISTILLATION_PROMPT_CATAPULT
    _so.SKILL_BANK_EVOLUTION_SKILLS_ONLY_PROMPT = SKILL_BANK_EVOLUTION_SKILLS_ONLY_PROMPT_CATAPULT
