"""Catapult-specific teacher prompts for offline→online evolving skill distillation.

Three prompt templates (same 3 phases as the generic teacher) but with the
factual catapult scene baked in and NO strategy hints leaked to the teacher:
    1. CONTRASTIVE_DISTILLATION_PROMPT_CATAPULT  — round 1 contrastive skill extraction
    2. COMMON_MISTAKES_PROMPT_CATAPULT           — round 1 mistakes + partial skills
    3. SKILL_BANK_EVOLUTION_PROMPT_CATAPULT      — round 2+ bank evolution

Usage (preferred — monkey-patches the existing pipeline, batch size 3):

    python -m skillrl.teacher_prompts_catapult \\
        --initial-traj-dir results/catapult_claude \\
        --num-rounds 3

Or from Python:

    import skillrl.distillation.teacher_prompts_catapult as tpc
    tpc.patch()                  # swaps in catapult prompts
    from skillrl.loops.evolving_loop import run_evolving_loop
    run_evolving_loop(level_name="catapult", seeds_per_round=3, ...)
"""

from __future__ import annotations


# ── Factual level description (no strategy hints) ──────────────────────
# Source of scene facts: prompts/catapult_refine.txt.
# Strategy language (tool-use loops, exploration rules, suggested approaches)
# intentionally REMOVED so the teacher discovers strategies from trajectories.

CATAPULT_LEVEL_BLOCK = """\
**Scene (factual — no approach implied):**
- Green Ball: small dynamic ball on the LEFT end of a gray bar.
- Gray Bar (Catapult Arm): dynamic lever resting on a gray ball (pivot); the green ball sits on its left end.
- Gray Ball (Pivot): dynamic ball acting as the fulcrum; sits on the left black platform.
- Black Ball (Ceiling Blocker): static ball near the top of the scene.
- Black Platform (Left): static horizontal platform on the left side.
- Black Ledge (Right): static (possibly angled) platform on the right side.
- Basket (Gray): dynamic basket sitting on the right ledge.
- Blue Ball (Target): dynamic ball inside the basket.

**Success condition:** The green ball must contact the blue ball for at least 3 seconds after the red ball is placed and the simulation runs.

**Placement constraints:**
- The red ball must be completely inside the box: -5 + r ≤ x ≤ 5 - r, -5 + r ≤ y ≤ 5 - r.
- The red ball must NOT overlap with any existing object at t=0.
- 0.1 ≤ radius ≤ 2.0.
"""


# ── Phase 1a: Contrastive distillation (round 1 seeding)iterative  ───────────────

CONTRASTIVE_DISTILLATION_PROMPT_CATAPULT = """\
You are an expert physics analyst distilling agent behavior into concise, actionable skills.

**Environment**: 2D physics simulation (Box2D). Gravity = -9.8 m/s². World bounds [-5, 5] on both axes. The agent places a red ball at (x, y) with a given radius, then the simulation runs to completion.

**Level: {level_name}**
""" + CATAPULT_LEVEL_BLOCK + """

Below are SUCCESSFUL and FAILED trajectories from this level. Each shows the agent's reasoning (Thought), actions taken, and simulation observations. Each trajectory has a **Reward** score:
- Successes: +1.0 (solved fast, 1-3 iters), +0.75 (4-7 iters), +0.5 (8-15 iters), +0.25 (solved slowly, 16-25 iters)
- Failures: -0.5 (tried all 25 iters), -0.75 (gave up early, <10 iters)

**Weight your analysis by reward** — skills extracted from high-reward trajectories (fast solves) are more reliable than those from low-reward ones.

=== SUCCESSFUL TRAJECTORIES ({n_successes}) ===
{success_block}

=== FAILED TRAJECTORIES ({n_failures}) ===
{failure_block}

---

Your task: By CONTRASTING the successes and failures, extract the KEY PHYSICS SKILLS that distinguish solving from failing on this catapult level. Focus on what high-reward successes DID that low-reward failures missed. Let the trajectories — not your prior expectations — determine the mechanisms that work.

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

Extract 4-6 skills. Each should capture a DISTINCT insight from the success/failure contrast. Avoid redundancy.
"""


# ── Phase 1b: Common mistakes + partial skills (round 1 seeding) ───────

COMMON_MISTAKES_PROMPT_CATAPULT = """\
You are an expert at analyzing agent failures and distilling them into avoidable mistake patterns.

**Environment**: 2D physics simulation (Box2D). Gravity = -9.8 m/s². World bounds [-5, 5].

**Level: {level_name}**
""" + CATAPULT_LEVEL_BLOCK + """

Below are FAILED trajectories. Each shows the agent's reasoning, actions, and simulation results.

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

Extract 3-5 mistakes and 1-2 partial skills. For mistakes, group similar failures into one and focus on ROOT CAUSES. For partial skills, only extract genuinely useful insights — do not force it if no good steps exist.
"""


# ── Phase 2 (v2): Skill bank evolution (round 2+) ──────────────────────

SKILL_BANK_EVOLUTION_PROMPT_CATAPULT = """\
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

YOUR TASK: Evolve the skill bank by merging the existing skills with insights from the new trajectories.

RULES:
1. Output the COMPLETE FINAL skill bank (not a diff) — include both retained existing skills and any new ones.
2. Hard constraints:
   - Maximum {max_skills} total skills for this level.
   - Maximum {max_mistakes} total mistakes for this level.
3. For each skill you include:
   - If it's a RETAINED skill from the existing bank: set "is_new": false.
   - If it's a NEW skill extracted from the new trajectories: set "is_new": true.
   - Include "source_seeds" listing seed numbers where this skill was observed (required for confidence calibration).
   - Include "confidence": a float in [0.1, 1.0] representing your confidence in this skill.
4. For retained skills, preserve their existing confidence values (they've been validated).
5. For new skills, estimate confidence based on:
   - Success rate among source trajectories (high success = high confidence).
   - Universality (applies across multiple seed conditions = higher confidence).
   - Clarity and actionability of the principle.
6. Do not include duplicate skills. If a new trajectory confirms an existing skill, keep the existing one (optionally raising its confidence slightly).
7. Remove skills that are:
   - Redundant or subsumed by another skill.
   - Contradicted by the new trajectories.
   - Too specific or rarely applicable.
   - Low confidence (< 0.3) AND not observed in the new trajectories.
8. Do not remove mistakes unless the new trajectories show they're no longer common.

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
  ],
  "mistakes": [
    {{
      "description": "<what the mistake is>",
      "why_it_happens": "<why agents make this mistake>",
      "how_to_avoid": "<actionable fix>",
      "is_new": <true|false>
    }}
  ],
  "removed_skill_titles": ["<title of removed skill 1>", ...],
  "reasoning": "<brief explanation of key changes: what was removed, what was added, why>"
}}

Be concise but precise. Focus on physics insights grounded in the trajectories.
"""


# ── Hook into the generic pipeline via monkey-patch ────────────────────

def patch() -> None:
    """Swap the catapult prompts into distill.py and evolving_distill.py.

    Call this BEFORE invoking run_distillation() or evolve_skill_bank().
    Patches module-level bindings in-place — safe because each function
    resolves the prompt name at call time from its own module globals.
    """
    import skillrl.distillation.distill as _distill
    import skillrl.distillation.evolving_distill as _evolving_distill

    _distill.CONTRASTIVE_DISTILLATION_PROMPT = CONTRASTIVE_DISTILLATION_PROMPT_CATAPULT
    _distill.COMMON_MISTAKES_PROMPT = COMMON_MISTAKES_PROMPT_CATAPULT
    _evolving_distill.SKILL_BANK_EVOLUTION_PROMPT = SKILL_BANK_EVOLUTION_PROMPT_CATAPULT


# ── CLI entrypoint: patch + run evolving loop with batch size 3 ────────

def _main() -> None:
    import argparse
    from pathlib import Path

    from skillrl.core.config import (
        DEFAULT_MAX_ITERATIONS,
        DEFAULT_MAX_NEW_TOKENS,
        DEFAULT_TEMPERATURE,
        MAX_SKILLS_PER_LEVEL,
        TEACHER_MODEL,
    )

    parser = argparse.ArgumentParser(
        description="Catapult offline→online evolving distillation (uses catapult-specific teacher prompts)."
    )
    parser.add_argument(
        "--initial-traj-dir", type=str, required=True,
        help="Directory with initial seed trajectories (offline phase).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: skillrl/data/evolving/catapult).",
    )
    parser.add_argument("--num-rounds", type=int, default=3,
                        help="Number of rounds — round 1 seeds from offline trajs, rounds 2+ evolve online (default: 3).")
    parser.add_argument("--seeds-per-round", type=int, default=3,
                        help="Batch size — new seeds per online round (default: 3).")
    parser.add_argument("--start-seed", type=int, default=6,
                        help="First seed for round 1 online rollouts (default: 6).")
    parser.add_argument("--max-skills", type=int, default=MAX_SKILLS_PER_LEVEL)
    parser.add_argument("--max-mistakes", type=int, default=5)
    parser.add_argument("--model", type=str, default="claude", help="Agent model.")
    parser.add_argument("--teacher-model", type=str, default=TEACHER_MODEL)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-general-skills", type=int, default=0)
    parser.add_argument("--max-specific-skills", type=int, default=6)
    parser.add_argument("--max-mistakes-agent", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Apply the catapult prompt overrides BEFORE importing the loop.
    patch()

    from skillrl.loops.evolving_loop import run_evolving_loop

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(__file__).resolve().parent / "data" / "evolving_oss" / "catapult"
    )

    print("\n[catapult teacher prompts] Patched — round 1 uses catapult CONTRASTIVE+MISTAKES; rounds 2+ use catapult EVOLUTION.")
    print(f"[catapult teacher prompts] Batch size = {args.seeds_per_round}, rounds = {args.num_rounds}\n")

    run_evolving_loop(
        level_name="catapult",
        initial_traj_dir=Path(args.initial_traj_dir),
        output_dir=output_dir,
        num_rounds=args.num_rounds,
        seeds_per_round=args.seeds_per_round,
        start_seed=args.start_seed,
        max_skills=args.max_skills,
        max_mistakes=args.max_mistakes,
        model_name=args.model,
        teacher_model=args.teacher_model,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        skip_general=True,
        max_general_skills=args.max_general_skills,
        max_specific_skills=args.max_specific_skills,
        max_mistakes_agent=args.max_mistakes_agent,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    _main()
