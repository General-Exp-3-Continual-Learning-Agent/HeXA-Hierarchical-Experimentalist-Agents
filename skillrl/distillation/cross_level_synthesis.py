"""Cross-level offline skill transfer.

Stage 1 of the cross-level meta-RL experiment: a teacher model reads expert
(fully-evolved) skill banks from N source levels, plus factual descriptions
of the source levels and a target level, and synthesises a target-level
skill bank WITHOUT ever seeing target-level trajectories.

Design constraints (mirroring teacher_prompts_catapult.py's discipline):
- The target-level description is FACTUAL only — no strategy hints leaked.
- Each synthesised skill must cite the source skill_ids it transferred from
  (audit trail written to a sidecar JSON; the core SkillBank format is left
  untouched so the bank loads with existing tooling).
- The teacher is explicitly forbidden from inventing target-specific
  coordinates, since no target trajectories exist.

Outputs (under ``output_dir``):
- ``skill_bank_xl_<target>.json``  — SkillBank-compatible JSON
- ``skill_bank_xl_<target>_audit.json``  — transfer attributions + reasoning

Usage:

    python -m skillrl.distillation.cross_level_synthesis \\
        --target catapult \\
        --source-bank down_to_earth=skillrl/cross/down_to_earth.json \\
        --source-bank two_body_problem=skillrl/cross/two_body_problem.json \\
        --source-bank pass_the_parcel=skillrl/cross/pass_the_parcel.json \\
        --output-dir skillrl/data/cross_level/catapult
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from collections.abc import Mapping
from pathlib import Path

from skillrl.core.config import TEACHER_MODEL
from skillrl.core.skill_bank import Mistake, Skill, SkillBank
from skillrl.distillation.distill import _parse_json_object, call_teacher
from skillrl.distillation.teacher_prompts_catapult import CATAPULT_LEVEL_BLOCK


# ── Factual source-level descriptions (no strategy hints) ──────────────
# Kept short and factual: scene + success condition + action space.
# Strategy is carried by the *skill banks themselves*, so the descriptions
# only need to ground the teacher in what each level looks like.

SOURCE_LEVEL_BLOCKS: dict[str, str] = {
    "down_to_earth": """\
**Scene (factual):**
- Green Ball: dynamic ball resting on a raised black platform.
- Black Platform: static horizontal bar holding the green ball, with gaps on either side.
- Purple Ground: static floor at the bottom of the box (y ≈ -5).

**Success condition:** The green ball must contact the purple ground for at least 3 seconds.

**Action:** Place ONE red ball at (x, y, radius), 0.1 ≤ r ≤ 2.0, fully inside [-5, 5] × [-5, 5], no overlap at t=0.
""",

    "two_body_problem": """\
**Scene (factual):**
- Green Ball: dynamic ball.
- Blue Ball: dynamic ball, separated horizontally from the green ball.
- Both balls fall under gravity from rest from the start of the simulation.

**Success condition:** The green ball must contact the blue ball for at least 3 seconds.

**Action:** Place ONE red ball at (x, y, radius), 0.1 ≤ r ≤ 2.0, fully inside [-5, 5] × [-5, 5], no overlap at t=0.
""",

    "pass_the_parcel": """\
**Scene (factual):**
- Top Basket (Gray, inverted): dynamic basket sitting on a black platform with its opening facing DOWNWARD; it traps the green ball beneath it.
- Green Ball: small dynamic ball trapped under the inverted top basket on the platform.
- Bottom Basket (Gray, upright): dynamic basket below the platform with its opening facing UPWARD; it holds the blue ball.
- Blue Ball: dynamic ball inside the bottom basket.
- Black Platform: static horizontal bar; the top basket and the agent's red ball can sit on it.
- Black Ramp: static angled bar rising from the left edge of the platform upward to the right.

**Success condition:** The green ball must contact the blue ball for at least 3 seconds.

**Action:** Place ONE red ball at (x, y, radius), 0.1 ≤ r ≤ 2.0, fully inside [-5, 5] × [-5, 5], no overlap at t=0.
""",
}


# ── Synthesis prompt ───────────────────────────────────────────────────

CROSS_LEVEL_SYNTHESIS_PROMPT = """\
You are an expert physics analyst performing CROSS-LEVEL OFFLINE SKILL TRANSFER.

You will be given expert skill banks from {n_sources} SOURCE physics levels.
Each source bank was distilled and evolved over many rounds of an agent solving
that level. You will also be given a factual description of a TARGET level
that the agent has NEVER attempted yet — no target-level trajectories,
no target-level skills, no target-level success/failure data exist.

Your task: synthesise a TARGET skill bank by extracting transferable physics
PRINCIPLES from the source banks, re-grounding them in the target scene, and
predicting which principles the target level will reward.

═══════════════════════════════════════════════════════════════════════════
**Environment (shared across all levels)**: 2D physics simulation (Box2D).
Gravity = -9.8 m/s². World bounds [-5, 5] on both axes. The agent places ONE
red ball at (x, y) with a chosen radius (0.1 ≤ r ≤ 2.0) at t=0; the simulation
then runs to completion. Mass scales as r³; momentum and impact force scale
with both mass and the ball's velocity at first contact.
═══════════════════════════════════════════════════════════════════════════

═══ SOURCE LEVELS ═══

{source_blocks}

═══ TARGET LEVEL: {target_level} ═══

{target_block}
{structural_hint_block}
═══ TASK ═══

Produce a SKILL BANK for the target level. The bank must contain:
- 6 to 10 SKILLS (physics principles you predict will help on the target).
- 2 to 4 MISTAKES (anti-patterns transferable from source failures).

═══ HARD CONSTRAINTS ═══

1. **Every skill MUST cite source skills.** For each output skill, include
   `source_skills`: a non-empty list of objects of the form
   `{{"source_level": "<level>", "skill_id": "<id>"}}`, listing the source-bank
   entries that motivated this skill. If you cannot cite at least one source
   skill, do NOT emit the skill — synthesise something else instead.

2. **No invented coordinates.** You have NEVER seen the target level solved.
   Do NOT predict specific (x, y, r) values that "work" for the target. The
   `example` field, if used, must describe a *qualitative* placement
   (e.g. "place near the end of the lever opposite the green ball, with
   r large enough to dominate the lever's mass") — never specific numbers.

3. **Use target-level entities.** Phrase each skill in terms of the entities
   that exist in the TARGET scene description above. Do NOT mention source-
   level-only entities (e.g. "ramp", "inverted basket") in the principle or
   when_to_apply, even if those entities motivated the transfer. The
   `transfer_rationale` field is where you may name source entities to
   explain the analogy.

4. **No platitudes.** Skills like "use physics intuition", "consider gravity",
   "be careful" are forbidden. Each skill must name a SPECIFIC mechanism
   (e.g. "torque about a pivot scales with moment arm × force"; "the line of
   centres at first contact determines the post-collision velocity direction").

5. **Each skill must include a `transfer_rationale`** (1-2 sentences) explaining
   what physics primitive bridges the source observation to the target
   prediction. This is the audit trail for your synthesis.

6. **Confidence calibration.** Weight a skill's confidence by:
   - How many source banks corroborate the underlying primitive (more = higher).
   - How directly the primitive applies to the target scene (direct = higher).
   - How load-bearing the source skill was (high source confidence = higher).
   Confidence is a float in [0.1, 1.0]. Assign 0.7+ only when the primitive
   appears in ≥2 source banks AND the target scene clearly invokes it.

7. **Avoid redundancy.** Each skill must capture a DISTINCT mechanism. Do not
   emit two skills that are paraphrases of the same idea.

8. **Mistakes**: derive each from source-bank mistakes or from the failure
   modes those mistakes imply. Apply the same hard constraints (cite sources,
   no invented coordinates, target-side entities, specific mechanism).

═══ OUTPUT FORMAT (single JSON object) ═══

```json
{{
  "skills": [
    {{
      "title": "<3-7 word name>",
      "principle": "<2-3 sentence physics insight, in target-level terms>",
      "when_to_apply": "<specific condition, in target-level terms>",
      "example": "<qualitative placement description, NO numbers — or omit>",
      "confidence": <float in [0.1, 1.0]>,
      "source_skills": [
        {{"source_level": "<level>", "skill_id": "<id>"}}
      ],
      "transfer_rationale": "<1-2 sentences naming the bridging physics primitive>"
    }}
  ],
  "mistakes": [
    {{
      "description": "<1 sentence in target-level terms>",
      "why_it_happens": "<1 sentence root cause>",
      "how_to_avoid": "<1-2 sentence concrete fix>",
      "source_skills": [
        {{"source_level": "<level>", "skill_id": "<id>"}}
      ],
      "transfer_rationale": "<1-2 sentences>"
    }}
  ],
  "synthesis_reasoning": "<2-4 sentences: which source levels contributed most heavily, which primitives you treated as load-bearing, which source skills you deliberately did NOT transfer and why>"
}}
```

Be precise and specific. Do not pad. Do not produce extra fields. Output ONLY
the JSON object inside a single fenced ```json block.
"""


# ── Slim Qwen-friendly synthesis prompt ────────────────────────────────
# For smaller models (Qwen 7B etc.) the longer 8-rule prompt produced
# platitude-heavy, hallucinated mechanisms (e.g. inventing levers on
# levels that have no pivot). This slim variant: 3 rules, one concrete
# few-shot example, drops audit fields (source_skills, transfer_rationale)
# to keep output budget for skill content.

CROSS_LEVEL_SYNTHESIS_PROMPT_QWEN = """\
You are extracting transferable physics PRINCIPLES from one or more source level skill banks for a NEW target level.

ENVIRONMENT: 2D physics (Box2D), gravity = -9.8 m/s², bounds [-5, 5]. Agent places ONE red ball (x, y, radius 0.1-2.0) at t=0; simulation runs to completion.

═══ SOURCE LEVELS ═══

{source_blocks}

═══ TARGET LEVEL: {target_level} ═══

{target_block}

═══ TASK ═══

Output a skill bank for the target level. Produce 4-6 skills and 1-3 mistakes.

RULES (FOLLOW STRICTLY):

1. SKILLS ARE PRINCIPLES, NOT COMMANDS. Each skill states a physics observation, mechanism, or relationship the agent can REASON FROM. Skills MUST NOT tell the agent what to do. Each field has a strict FORM:
   - `principle`: starts with a condition or relationship ("When X holds, Y occurs", "X scales with Y", "A and B differ in Z"). NEVER an imperative verb. FORBIDDEN openings: place / position / put / drop / move / set / use / adjust.
   - `when_to_apply`: describes a REASONING context the agent enters ("When evaluating whether…", "When comparing alternatives that…"). NEVER a target outcome to achieve ("when you need to tip…", "when knocking the bar off…").
   - `example`: contrasts two physical scenarios or illustrates the principle abstractly. NEVER a placement, never a specific scene action.

2. Each skill names a SPECIFIC mechanism. FORBIDDEN platitudes: "use physics", "place strategically", "consider gravity", "adjust as needed", "be careful".

3. Use ONLY entities present in the TARGET scene above. Do not mention source-only entities anywhere. If the target scene has no pivot, do NOT write skills about levers or torque-around-a-pivot.

4. NO coordinates and NO placements anywhere in any field.

5. Mistakes describe MISCONCEPTIONS the agent might form (wrong mental models), NOT actions to avoid. `how_to_avoid` is a CORRECTED UNDERSTANDING, not a placement instruction.

OUTPUT FORMAT (single JSON object, fenced ```json block, no extra fields):
```json
{{
  "skills": [
    {{"title": "...", "principle": "...", "when_to_apply": "...", "example": "...", "confidence": 0.X}}
  ],
  "mistakes": [
    {{"description": "<misconception, NOT an action>", "why_it_happens": "...", "how_to_avoid": "<corrected understanding, NOT a placement command>"}}
  ],
  "synthesis_reasoning": "<1-2 sentences: which source primitive transferred>"
}}
```

Be concise. Output ONLY the JSON object.
"""


# ── Domain-neutral synthesis prompt (turn-based text games) ────────────
# The two prompts above hardcode InterPhyre's Box2D framing ("place ONE red
# ball at (x, y, r)") in their bodies, so neither can be pointed at ORBIT's
# turn-based games. This variant takes the shared-environment paragraph as a
# `{domain_block}` parameter instead of baking it in, and restates the hard
# constraints in domain-neutral terms: "no invented coordinates" becomes "no
# invented target-task state", since a Maze cell or a Mastermind code is the
# same kind of unseen specific that a catapult (x, y) was.
#
# It keeps the 8-rule discipline and the audit fields (source_skills,
# transfer_rationale) rather than following the Qwen-slim variant: Design C's
# Phase 3 manual inspection reads those fields, and the ORBIT teacher is
# gpt-oss-120b, not a 7B model.

CROSS_TASK_SYNTHESIS_PROMPT = """\
You are an expert analyst performing CROSS-TASK OFFLINE SKILL TRANSFER.

You will be given expert skill banks from {n_sources} SOURCE tasks. Each source
bank was distilled and evolved over many rounds of an agent playing that task.
You will also be given a factual description of a TARGET task that the agent has
NEVER attempted — no target-task trajectories, no target-task skills, no
target-task success/failure data exist.

Your task: synthesise a TARGET skill bank by extracting transferable REASONING
STRATEGIES from the source banks, re-grounding them in the target task's own
state and action vocabulary, and predicting which strategies the target task
will reward.

═══════════════════════════════════════════════════════════════════════════
{domain_block}
═══════════════════════════════════════════════════════════════════════════

═══ SOURCE TASKS ═══

{source_blocks}

═══ TARGET TASK: {target_level} ═══

{target_block}
{structural_hint_block}
═══ TASK ═══

Produce a SKILL BANK for the target task. The bank must contain:
- 6 to 10 SKILLS (reasoning strategies you predict will help on the target).
- 2 to 4 MISTAKES (anti-patterns transferable from source failures).

═══ HARD CONSTRAINTS ═══

1. **Every skill MUST cite source skills.** For each output skill, include
   `source_skills`: a non-empty list of objects of the form
   `{{"source_level": "<task>", "skill_id": "<id>"}}`, listing the source-bank
   entries that motivated this skill. Cite `skill_id`s exactly as they appear in
   the source banks above — invented ids are detected and flagged. If you cannot
   cite at least one real source skill, do NOT emit the skill.

2. **No invented target state.** You have NEVER seen the target task played. Do
   NOT assert specific target-task facts that could only come from playing it —
   no particular board positions, hidden codes, word lists, layouts, or "the
   answer is usually X" claims. The `example` field, if used, must describe a
   *qualitative* situation ("when two prior guesses share exactly one confirmed
   position"), never a concrete solved instance.

3. **Use target-task entities and actions.** Phrase each skill in terms of the
   state and the action vocabulary that exist in the TARGET description above.
   Do NOT mention source-task-only entities (mines, flags, letters, cards, bets)
   in `principle` or `when_to_apply`, even where those entities motivated the
   transfer. The `transfer_rationale` field is the one place you may name source
   entities, to explain the analogy.

4. **No platitudes.** Skills like "think carefully", "use logic", "learn from
   feedback", "be systematic" are forbidden. Each skill must name a SPECIFIC
   mechanism — what information a particular observation carries, what it rules
   out, and how that changes the next decision.

5. **Each skill must include a `transfer_rationale`** (1-2 sentences) naming the
   task-independent primitive that bridges the source observation to the target
   prediction (e.g. "both tasks return a per-position match count, so a guess
   that varies one position isolates that position's contribution").

6. **Confidence calibration.** Weight a skill's confidence by:
   - How many source banks corroborate the underlying primitive (more = higher).
   - How directly the primitive applies to the target task (direct = higher).
   - How load-bearing the source skill was (high source confidence = higher).
   Confidence is a float in [0.1, 1.0]. Assign 0.7+ only when the primitive
   appears in ≥2 source banks AND the target task clearly invokes it.

7. **Avoid redundancy.** Each skill must capture a DISTINCT mechanism. Do not
   emit two skills that are paraphrases of the same idea.

8. **Mistakes**: derive each from source-bank mistakes or from the failure modes
   those mistakes imply. Apply the same hard constraints (cite sources, no
   invented target state, target-side entities, specific mechanism).

9. **Multi-episode structure.** The target task is played for several episodes
   against the same hidden instance, and the agent carries its full transcript
   across them. Skills that say what a *later* episode should do with what an
   earlier episode revealed are especially valuable — but only if the source
   primitive genuinely supports the claim.

═══ OUTPUT FORMAT (single JSON object) ═══

```json
{{
  "skills": [
    {{
      "title": "<3-7 word name>",
      "principle": "<2-3 sentence strategy insight, in target-task terms>",
      "when_to_apply": "<specific condition, in target-task terms>",
      "example": "<qualitative situation, no concrete solved instance — or omit>",
      "confidence": <float in [0.1, 1.0]>,
      "source_skills": [
        {{"source_level": "<task>", "skill_id": "<id>"}}
      ],
      "transfer_rationale": "<1-2 sentences naming the bridging primitive>"
    }}
  ],
  "mistakes": [
    {{
      "description": "<1 sentence in target-task terms>",
      "why_it_happens": "<1 sentence root cause>",
      "how_to_avoid": "<1-2 sentence concrete fix>",
      "source_skills": [
        {{"source_level": "<task>", "skill_id": "<id>"}}
      ],
      "transfer_rationale": "<1-2 sentences>"
    }}
  ],
  "synthesis_reasoning": "<2-4 sentences: which source tasks contributed most heavily, which primitives you treated as load-bearing, which source skills you deliberately did NOT transfer and why>"
}}
```

Be precise and specific. Do not pad. Do not produce extra fields. Output ONLY
the JSON object inside a single fenced ```json block.
"""


#: Prompt variants selectable by name. ``None`` (the default) keeps the historic
#: teacher-model-prefix routing, so existing InterPhyre invocations are unchanged.
PROMPT_VARIANTS: dict[str, tuple[str, str]] = {
    "physics": (CROSS_LEVEL_SYNTHESIS_PROMPT, "physics (8-rule, with audit fields)"),
    "physics-slim": (CROSS_LEVEL_SYNTHESIS_PROMPT_QWEN, "physics-slim (3-rule, no audit fields)"),
    "text-game": (CROSS_TASK_SYNTHESIS_PROMPT, "text-game (9-rule, with audit fields)"),
}


def _select_prompt_template(
    teacher_model: str,
    prompt_variant: str | None = None,
) -> tuple[str, str]:
    """Return (prompt_template, label) for the given teacher.

    An explicit *prompt_variant* wins. Without one, routing falls back to the
    historic behaviour: Claude models get the long 8-rule physics prompt with
    audit fields, everything else gets the slim Qwen-friendly physics variant.

    That fallback is a trap for any non-Claude strong teacher — gpt-oss-120b
    would silently receive the small-model prompt and drop the audit fields the
    transfer inspection reads — so non-physics callers must name their variant.
    The label is used in logs.
    """
    if prompt_variant is not None:
        try:
            return PROMPT_VARIANTS[prompt_variant]
        except KeyError:
            raise ValueError(
                f"Unknown prompt_variant {prompt_variant!r}. "
                f"Known: {sorted(PROMPT_VARIANTS)}."
            ) from None
    if teacher_model.startswith("claude-"):
        return PROMPT_VARIANTS["physics"]
    return PROMPT_VARIANTS["physics-slim"]


# ── Source bank rendering for the prompt ───────────────────────────────

def format_source_bank_for_teacher(bank: SkillBank, level_name: str) -> str:
    """Render a source skill bank as a markdown block.

    Includes skill_ids verbatim so the teacher can cite them in source_skills.
    Skills are sorted by confidence (highest first) so the teacher reads
    high-confidence evidence before lower-confidence skills.
    """
    skills = sorted(
        bank.level_skills.get(level_name, []),
        key=lambda s: s.confidence,
        reverse=True,
    )
    mistakes = bank.level_mistakes.get(level_name, [])

    lines: list[str] = [f"#### Skill bank for `{level_name}`"]
    if not skills and not mistakes:
        lines.append("(empty)")
        return "\n".join(lines)

    if skills:
        lines.append("\n**Skills:**")
        for s in skills:
            lines.append(
                f"- `{s.skill_id}` (confidence={s.confidence:.2f}, gen={s.generation}) "
                f"**{s.title}** — {s.principle}"
            )
            if s.when_to_apply:
                lines.append(f"  - Apply when: {s.when_to_apply}")
            if s.example:
                lines.append(f"  - Example: {s.example}")

    if mistakes:
        lines.append("\n**Mistakes:**")
        for m in mistakes:
            lines.append(
                f"- `{m.mistake_id}` **{m.description}** "
                f"(why: {m.why_it_happens}; avoid: {m.how_to_avoid})"
            )

    return "\n".join(lines)


# ── Target-level block selection ───────────────────────────────────────

# Target-level FACTUAL blocks. Add more here as new target levels are
# attempted. We deliberately re-use the strategy-free CATAPULT_LEVEL_BLOCK
# rather than the strategy-flavoured config.LEVEL_DESCRIPTIONS["catapult"].

FALLING_INTO_PLACE_LEVEL_BLOCK = """\
**Scene (factual — no approach implied):**
- Green Ball: small dynamic ball resting on a black platform.
- Black Platform: two static horizontal segments separated by a gap (hole) of width 2.0; the green ball sits on one of the two segments, randomised left or right.
- Blue Basket: dynamic basket positioned above the platform with its opening facing DOWNWARD; falls under gravity from the start of the simulation.
- Bottom Ramp: static angled bar near the floor of the box.
- Red Ball: dynamic ball with radius drawn from [0.3, 0.6] — placed by the agent at t=0.

**Success condition:** The green ball must contact the blue basket for at least 3 seconds.

**Placement constraints:**
- The red ball must be completely inside the box: -5 + r ≤ x ≤ 5 - r, -5 + r ≤ y ≤ 5 - r.
- The red ball must NOT overlap any existing object at t=0.
- 0.1 ≤ radius ≤ 2.0.
"""

CLIFFHANGER_LEVEL_BLOCK = """\
**Scene (factual — no approach implied):**
- Green Bar: dynamic vertical bar (length drawn from [2.0, 3.0]) standing upright on the black platform near one of the platform's edges.
- Black Platform: static horizontal bar of length drawn from [4.0, 6.0] at variable height y ∈ [-3, 0]; the green bar stands on top of it.
- Ceiling: static horizontal bar spanning the box, positioned above the platform (y above the green bar's top).
- Purple Ground: static floor at the bottom of the box (y ≈ -5).
- Red Ball: dynamic ball with radius drawn from [0.3, 0.6] — placed by the agent at t=0.

**Success condition:** The green bar must contact the purple ground for at least 3 seconds.

**Placement constraints:**
- The red ball must be completely inside the box: -5 + r ≤ x ≤ 5 - r, -5 + r ≤ y ≤ 5 - r.
- The red ball must NOT overlap any existing object at t=0.
- 0.1 ≤ radius ≤ 2.0.
"""

TWO_BODY_PROBLEM_BLOCK = """\
**Scene (factual — no approach implied):**
- **Green Ball:** A dynamic ball.
- **Blue Ball:** A dynamic ball, separated horizontally from the green ball.
- Both balls fall under gravity from rest.

**The Goal:**
Place ONE Red Ball at t=0 so that the Green Ball collides with the Blue Ball and stays in contact for at least 3 seconds.
"""

TARGET_LEVEL_BLOCKS: dict[str, str] = {
    "catapult": CATAPULT_LEVEL_BLOCK,
    "falling_into_place": FALLING_INTO_PLACE_LEVEL_BLOCK,
    "cliffhanger": CLIFFHANGER_LEVEL_BLOCK,
    "two_body_problem": TWO_BODY_PROBLEM_BLOCK
}


# ── Citation verification ──────────────────────────────────────────────

def _known_ids(bank: SkillBank, level: str) -> set[str]:
    """Every skill/mistake id the teacher could legitimately cite from *bank*."""
    ids = {s.skill_id for s in bank.level_skills.get(level, [])}
    ids |= {s.skill_id for s in bank.general_skills}
    ids |= {m.mistake_id for m in bank.level_mistakes.get(level, [])}
    ids |= {m.mistake_id for m in bank.common_mistakes}
    return ids


def verify_citations(
    audit_entries: list[dict],
    loaded_sources: dict[str, SkillBank],
) -> dict:
    """Check every cited (source_level, skill_id) against the source banks.

    Cross-task synthesis has no target trajectories by construction, so the
    per-turn citation check that guards within-task evolution
    (``evolving_distill.py:140``) has nothing to check against here. What *is*
    checkable is that each cited id exists in the source bank it names — the
    only automatic guard against a fabricated audit trail. Design B measured a
    58% turn-citation fabrication rate on a weak teacher; assume nothing.

    Reports rather than filters: a bogus citation invalidates the audit trail
    for that skill, not necessarily the skill, and the call on what to do with
    it belongs to the manual inspection step.
    """
    known = {lvl: _known_ids(bank, lvl) for lvl, bank in loaded_sources.items()}
    invalid: list[dict] = []
    uncited: list[str] = []
    n_citations = 0

    for entry in audit_entries:
        entry_id = entry.get("skill_id") or entry.get("mistake_id", "?")
        citations = entry.get("source_skills") or []
        if not citations:
            uncited.append(entry_id)
            continue
        for cite in citations:
            n_citations += 1
            lvl = cite.get("source_level", "")
            sid = cite.get("skill_id", "")
            if lvl not in known:
                invalid.append({"entry": entry_id, "cite": cite, "why": "unknown source_level"})
            elif sid not in known[lvl]:
                invalid.append({"entry": entry_id, "cite": cite, "why": "skill_id not in source bank"})

    return {
        "n_citations": n_citations,
        "n_invalid": len(invalid),
        "invalid_citations": invalid,
        "uncited_entries": uncited,
    }


# ── Synthesis runner ───────────────────────────────────────────────────

def synthesize_target_bank(
    target_level: str,
    source_bank_paths: dict[str, Path],
    output_dir: Path,
    teacher_model: str = TEACHER_MODEL,
    target_block_override: str | None = None,
    structural_hint: str | None = None,
    source_block_overrides: Mapping[str, str] | None = None,
    domain_block: str | None = None,
    prompt_variant: str | None = None,
    dry_run: bool = False,
) -> tuple[SkillBank | None, dict]:
    """Run the cross-level synthesis pipeline.

    Parameters
    ----------
    target_level : Name of the target level (e.g. "catapult").
    source_bank_paths : {source_level_name: path_to_evolved_bank_json}.
    output_dir : Directory where the bank + audit sidecar are written.
    teacher_model : Teacher LLM identifier (default: claude-sonnet-4-6).
    target_block_override : Optional factual block for the target level. If
        omitted, falls back to ``TARGET_LEVEL_BLOCKS[target_level]``.
    source_block_overrides : Optional {source_level: factual scene block},
        consulted before ``SOURCE_LEVEL_BLOCKS``. Required for any source
        outside InterPhyre, whose blocks are not in the module-level registry.
    domain_block : Shared-environment paragraph for the ``text-game`` variant,
        which parameterises it rather than hardcoding Box2D. Ignored by the
        physics variants, which bake their own in.
    prompt_variant : One of ``PROMPT_VARIANTS``. ``None`` keeps the historic
        teacher-prefix routing (see :func:`_select_prompt_template`).
    dry_run : Build and dump the prompt, then stop without calling the teacher.
        Returns ``(None, audit)`` with ``audit["dry_run"] is True``.

    Returns
    -------
    (SkillBank | None, audit_dict)
    """
    if target_block_override is not None:
        target_block = target_block_override
    elif target_level in TARGET_LEVEL_BLOCKS:
        target_block = TARGET_LEVEL_BLOCKS[target_level]
    else:
        raise ValueError(
            f"No factual target-level block registered for '{target_level}'. "
            f"Add one to TARGET_LEVEL_BLOCKS or pass target_block_override."
        )

    # Load source banks and render them.
    source_bank_blocks: list[str] = []
    loaded_sources: dict[str, SkillBank] = {}
    overrides = dict(source_block_overrides or {})
    for source_level, bank_path in source_bank_paths.items():
        if source_level in overrides:
            scene = overrides[source_level]
        elif source_level in SOURCE_LEVEL_BLOCKS:
            scene = SOURCE_LEVEL_BLOCKS[source_level]
        else:
            raise ValueError(
                f"No factual scene block registered for source level "
                f"'{source_level}'. Add it to SOURCE_LEVEL_BLOCKS or pass it "
                f"in source_block_overrides."
            )
        bank = SkillBank.load(bank_path)
        loaded_sources[source_level] = bank

        skills_md = format_source_bank_for_teacher(bank, source_level)
        source_bank_blocks.append(
            f"### Source level: `{source_level}`\n\n{scene}\n{skills_md}"
        )
        n_sk = len(bank.level_skills.get(source_level, []))
        n_err = len(bank.level_mistakes.get(source_level, []))
        print(f"  Loaded {source_level}: {n_sk} skills, {n_err} mistakes from {bank_path}")

    prompt_template, prompt_label = _select_prompt_template(teacher_model, prompt_variant)
    print(f"  Prompt template: {prompt_label}")

    if "{domain_block}" in prompt_template and not domain_block:
        raise ValueError(
            f"Prompt variant '{prompt_label}' parameterises the shared-environment "
            "paragraph; pass domain_block."
        )

    structural_hint_block = (
        f"\n═══ STRUCTURAL ANALOGY HINT (use this to ground the transfer) ═══\n"
        f"{structural_hint.strip()}\n"
        if structural_hint else ""
    )

    prompt = prompt_template.format(
        n_sources=len(source_bank_paths),
        source_blocks="\n\n---\n\n".join(source_bank_blocks),
        target_level=target_level,
        target_block=target_block,
        structural_hint_block=structural_hint_block,
        domain_block=(domain_block or ""),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_dump_path = output_dir / f"_synthesis_prompt_{target_level}.txt"
    prompt_dump_path.write_text(prompt)
    print(f"  Synthesis prompt saved to {prompt_dump_path}")

    if dry_run:
        print("  Dry run — teacher not called.")
        return None, {
            "dry_run": True,
            "target_level": target_level,
            "source_levels": list(source_bank_paths),
            "prompt_path": str(prompt_dump_path),
            "prompt_chars": len(prompt),
            "prompt_variant": prompt_label,
        }

    print(f"  Calling teacher ({teacher_model}) for cross-level synthesis...")
    response = call_teacher(prompt, model=teacher_model)
    if not response:
        raise RuntimeError("Teacher returned empty response — aborting.")

    raw_response_path = output_dir / f"_synthesis_raw_{target_level}.txt"
    raw_response_path.write_text(response)
    print(f"  Raw teacher response saved to {raw_response_path}")

    parsed = _parse_json_object(response)
    if not parsed:
        raise RuntimeError(
            "Could not parse teacher response as JSON. "
            f"See {raw_response_path} for the raw output."
        )

    bank = SkillBank()
    audit_skills: list[dict] = []
    audit_mistakes: list[dict] = []

    raw_skills = parsed.get("skills", [])
    prefix = target_level[:3]
    for i, raw in enumerate(raw_skills):
        skill_id = f"{prefix}_xl_{i:03d}"
        confidence = float(raw.get("confidence", 0.5))
        confidence = max(0.1, min(1.0, confidence))

        bank.add_skill(Skill(
            skill_id=skill_id,
            title=raw.get("title", ""),
            principle=raw.get("principle", ""),
            when_to_apply=raw.get("when_to_apply", ""),
            example=raw.get("example", ""),
            source_level=target_level,
            source_type="cross_level_synthesized",
            source_seeds=[],
            generation=0,
            confidence=confidence,
        ))
        audit_skills.append({
            "skill_id": skill_id,
            "title": raw.get("title", ""),
            "confidence": confidence,
            "source_skills": raw.get("source_skills", []),
            "transfer_rationale": raw.get("transfer_rationale", ""),
        })
        n_sources_cited = len(raw.get("source_skills", []))
        print(
            f"    [SKILL {skill_id}] '{raw.get('title', '')[:60]}' "
            f"conf={confidence:.2f} cites={n_sources_cited}"
        )

    raw_mistakes = parsed.get("mistakes", [])
    for i, raw in enumerate(raw_mistakes):
        mistake_id = f"{prefix}_xl_err_{i:03d}"
        bank.add_mistake(Mistake(
            mistake_id=mistake_id,
            description=raw.get("description", ""),
            why_it_happens=raw.get("why_it_happens", ""),
            how_to_avoid=raw.get("how_to_avoid", ""),
            source_level=target_level,
            source_seeds=[],
            generation=0,
        ))
        audit_mistakes.append({
            "mistake_id": mistake_id,
            "description": raw.get("description", ""),
            "source_skills": raw.get("source_skills", []),
            "transfer_rationale": raw.get("transfer_rationale", ""),
        })
        print(f"    [MISTAKE {mistake_id}] '{raw.get('description', '')[:60]}'")

    citation_check = verify_citations(audit_skills + audit_mistakes, loaded_sources)
    if citation_check["n_invalid"] or citation_check["uncited_entries"]:
        print(
            f"  [CITATIONS] {citation_check['n_invalid']}/{citation_check['n_citations']} "
            f"cited ids do not exist in the source banks; "
            f"{len(citation_check['uncited_entries'])} entries cite nothing. "
            "See the audit sidecar."
        )
    else:
        print(f"  [CITATIONS] all {citation_check['n_citations']} cited ids resolve.")

    bank_path = output_dir / f"skill_bank_xl_{target_level}.json"
    bank.save(bank_path)
    print(f"\n  SkillBank saved: {bank_path}")
    print(f"  {bank}")

    audit = {
        "target_level": target_level,
        "source_levels": list(source_bank_paths.keys()),
        "source_bank_paths": {k: str(v) for k, v in source_bank_paths.items()},
        "source_bank_skill_counts": {
            lvl: len(b.level_skills.get(lvl, [])) for lvl, b in loaded_sources.items()
        },
        "teacher_model": teacher_model,
        "synthesis_reasoning": parsed.get("synthesis_reasoning", ""),
        "prompt_variant": prompt_label,
        "citation_check": citation_check,
        "skills": audit_skills,
        "mistakes": audit_mistakes,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    audit_path = output_dir / f"skill_bank_xl_{target_level}_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2))
    print(f"  Audit sidecar saved: {audit_path}")

    return bank, audit


# ── CLI ────────────────────────────────────────────────────────────────

def _parse_source_bank_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"--source-bank expects 'level=path', got: {raw!r}"
        )
    level, path = raw.split("=", 1)
    return level.strip(), Path(path.strip())


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-level offline skill synthesis (no target trajectories used).",
    )
    parser.add_argument(
        "--target", required=True,
        help="Target level/task name. Must be registered in TARGET_LEVEL_BLOCKS, "
             "or in the TARGET_BLOCKS of a --blocks-module.",
    )
    parser.add_argument(
        "--source-bank", action="append", required=True,
        type=_parse_source_bank_arg,
        help="Repeatable. Format: '<source_level>=<path_to_evolved_bank.json>'.",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory for the synthesised bank + audit sidecar.",
    )
    parser.add_argument(
        "--teacher-model", default=TEACHER_MODEL,
        help=f"Teacher LLM (default: {TEACHER_MODEL}). The prompt template is "
             f"auto-selected: claude-* gets the 8-rule version with audit "
             f"fields; everything else gets the slim Qwen-friendly variant.",
    )
    parser.add_argument(
        "--structural-hint", default=None,
        help="Optional one-paragraph hint telling the teacher how the source "
             "and target levels are structurally related (e.g. shared "
             "mechanism) and what NOT to transfer. High-leverage for small "
             "teachers — prevents hallucinated mechanisms.",
    )
    parser.add_argument(
        "--blocks-module", default=None,
        help="Import path of a module supplying factual blocks for a non-"
             "InterPhyre domain, e.g. 'hexa_orbit.cross_task_blocks'. Must "
             "expose SOURCE_BLOCKS and TARGET_BLOCKS dicts, and may expose "
             "DOMAIN_BLOCK and PROMPT_VARIANT. Entries here take precedence "
             "over this module's InterPhyre registries.",
    )
    parser.add_argument(
        "--prompt-variant", default=None, choices=sorted(PROMPT_VARIANTS),
        help="Prompt template to use. Default: auto by teacher model prefix "
             "(claude-* -> physics, else physics-slim), which is correct only "
             "for InterPhyre. Overrides --blocks-module's PROMPT_VARIANT.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build and dump the synthesis prompt, then stop without calling "
             "the teacher.",
    )
    args = parser.parse_args()

    source_block_overrides: dict[str, str] = {}
    target_block_override: str | None = None
    domain_block: str | None = None
    prompt_variant: str | None = args.prompt_variant
    if args.blocks_module:
        import importlib

        blocks = importlib.import_module(args.blocks_module)
        source_block_overrides = dict(getattr(blocks, "SOURCE_BLOCKS", {}))
        target_block_override = dict(getattr(blocks, "TARGET_BLOCKS", {})).get(args.target)
        domain_block = getattr(blocks, "DOMAIN_BLOCK", None)
        if prompt_variant is None:
            prompt_variant = getattr(blocks, "PROMPT_VARIANT", None)
        if target_block_override is None:
            parser.error(
                f"{args.blocks_module} has no TARGET_BLOCKS entry for "
                f"'{args.target}' (has: {sorted(getattr(blocks, 'TARGET_BLOCKS', {}))})."
            )

    source_bank_paths: dict[str, Path] = dict(args.source_bank)
    if not source_bank_paths:
        parser.error("Provide at least one --source-bank entry.")
    if len(source_bank_paths) == 1:
        print(
            "[cross-level synthesis] WARNING: only 1 source bank supplied — "
            "single-source transfer is allowed but yields a weaker headline "
            "than multi-source synthesis."
        )

    print(f"\n[cross-level synthesis] target={args.target} sources={list(source_bank_paths)}")
    if args.structural_hint:
        print(f"[cross-level synthesis] structural hint: {args.structural_hint[:100]}...")
    synthesize_target_bank(
        target_level=args.target,
        source_bank_paths=source_bank_paths,
        output_dir=args.output_dir,
        teacher_model=args.teacher_model,
        structural_hint=args.structural_hint,
        source_block_overrides=source_block_overrides,
        target_block_override=target_block_override,
        domain_block=domain_block,
        prompt_variant=prompt_variant,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    _main()
