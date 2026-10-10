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
import re
from concurrent.futures import ThreadPoolExecutor
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
   never specific numbers.

3. **Use target-level entities.** Phrase each skill in terms of the entities
   that exist in the TARGET scene description above. Do NOT mention source-
   level-only entities (e.g. "ramp", "inverted basket") in the principle or
   when_to_apply, even if those entities motivated the transfer. The
   `transfer_rationale` field is where you may name source entities to
   explain the analogy.

4. **No platitudes.** Skills like "use physics intuition", "consider gravity",
   "be careful" are forbidden. Each skill must name a SPECIFIC mechanism

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
{transfer_guard_block}
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


# ── Neutral full prompt (--prompt-variant full-neutral) ────────────────
# Same structure and audit fields as the full prompt, but nothing in it can
# hint at how any particular level is solved: no example mechanisms, no
# example placements, no example entities, and the shared environment block
# states only the action interface (same as the slim prompt). Every piece of
# physics content in the output must therefore come from the source banks or
# from the factual target description.

CROSS_LEVEL_SYNTHESIS_PROMPT_NEUTRAL = """\
You are an expert physics analyst performing CROSS-LEVEL OFFLINE SKILL TRANSFER.

You will be given expert skill banks from {n_sources} SOURCE physics levels.
Each source bank was distilled and evolved over many rounds of an agent solving
that level. You will also be given a factual description of a TARGET level
that the agent has NEVER attempted yet — no target-level trajectories,
no target-level skills, no target-level success/failure data exist.

Your task: synthesise a TARGET skill bank by extracting transferable knowledge
from the source banks, re-grounding it in the target scene, and predicting
which of it the target level will reward.

═══════════════════════════════════════════════════════════════════════════
**Environment (shared across all levels)**: 2D physics simulation (Box2D).
Gravity = -9.8 m/s². World bounds [-5, 5] on both axes. The agent places ONE
red ball at (x, y) with a chosen radius (0.1 ≤ r ≤ 2.0) at t=0; the simulation
then runs to completion.
═══════════════════════════════════════════════════════════════════════════

═══ SOURCE LEVELS ═══

{source_blocks}

═══ TARGET LEVEL: {target_level} ═══

{target_block}
{structural_hint_block}
═══ TASK ═══

Produce a SKILL BANK for the target level. The bank must contain:
- 3 to 8 SKILLS. Fewer is correct when the sources support fewer; do not pad.
- 1 to 4 MISTAKES.

═══ HARD CONSTRAINTS ═══

1. **Every skill MUST cite source skills.** For each output skill, include
   `source_skills`: a non-empty list of objects of the form
   `{{"source_level": "<level>", "skill_id": "<id>"}}`, listing the source-bank
   entries that motivated this skill. If you cannot cite at least one source
   entry, do NOT emit the skill.

2. **Only source content.** Every claim in a skill must be carried by the
   cited source entries, re-expressed for the target. Do not add physics,
   strategies, or facts about the target that no cited source entry states,
   even if you believe them to be true.

3. **No invented numbers.** You have NEVER seen the target level solved. Do
   NOT write coordinates, radii, distances, or other magnitudes, except those
   stated in the environment or the target description. Describe actions
   qualitatively, in terms of target entities.

4. **Use target-level entities.** Phrase `principle`, `when_to_apply` and
   `example` only in terms of entities that exist in the TARGET scene. Entities
   that appear only in a source scene may be named in `transfer_rationale`.

5. **Actionable.** Each skill must say what the agent should DO or CHECK, and
   why, in target terms. Statements that only say one quantity "affects" or
   "influences" another, without saying in which direction or what to do
   about it, are forbidden, as are platitudes ("be careful", "consider
   gravity", "place strategically").

6. **Transfer rationale.** Each skill includes `transfer_rationale`
   (1-2 sentences): what the source entry observed, and why the same cause
   operates in the target scene. If you cannot state why it operates in the
   target, do not emit the skill.

7. **Confidence calibration.** Confidence is a float in [0.1, 1.0], weighted by
   how many source banks support the entry, how directly its cause operates in
   the target scene, and the cited entries' own confidence. Assign 0.7+ only
   when the entry is supported by ≥2 source banks AND its cause clearly
   operates in the target.

8. **Avoid redundancy.** Each skill captures a DISTINCT idea.

9. **Mistakes**: derive each from source-bank mistakes, under the same rules
   (cite sources, only source content, no invented numbers, target entities).
   `how_to_avoid` must say what to do instead.
{transfer_guard_block}
═══ OUTPUT FORMAT (single JSON object) ═══

```json
{{
  "skills": [
    {{
      "title": "<3-7 word name>",
      "principle": "<2-3 sentences, in target-level terms>",
      "when_to_apply": "<specific condition, in target-level terms>",
      "example": "<qualitative description, NO numbers — or omit>",
      "confidence": <float in [0.1, 1.0]>,
      "source_skills": [
        {{"source_level": "<level>", "skill_id": "<id>"}}
      ],
      "transfer_rationale": "<1-2 sentences>"
    }}
  ],
  "mistakes": [
    {{
      "description": "<1 sentence in target-level terms>",
      "why_it_happens": "<1 sentence root cause>",
      "how_to_avoid": "<1-2 sentences: what to do instead>",
      "source_skills": [
        {{"source_level": "<level>", "skill_id": "<id>"}}
      ],
      "transfer_rationale": "<1-2 sentences>"
    }}
  ],
  "synthesis_reasoning": "<2-4 sentences: which source entries you transferred and which you deliberately did NOT transfer and why>"
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



# ── Transfer guard (opt-in, --transfer-guard) ──────────────────────────
# Source banks carry magnitudes calibrated to source physics (e.g. "default
# r=0.6" in down_to_earth, where a bigger ball tightens overlap tolerance).
# Copied verbatim into a target whose mechanism differs, they become wrong
# defaults. The guard (a) forbids carrying source magnitudes across, (b)
# requires a per-item check that the source mechanism plays the same
# physical role in the target, and (c) validates the output in code: any
# number not stated in the shared environment or the target description is
# a violation -> one repair round with the teacher, then the item is dropped.
# Target-agnostic: uses only the source banks and the factual target block.

TRANSFER_GUARD_BLOCK = """
═══ TRANSFER GUARD (additional hard constraints) ═══

9. **No source calibrations.** Numeric magnitudes in source skills (radii,
   coordinates, offsets, step sizes, angles, thresholds) were tuned to the
   SOURCE level's physics and are meaningless in the target. Do NOT write any
   number in title / principle / when_to_apply / example / description /
   why_it_happens / how_to_avoid, except numbers stated in the shared
   environment or the target description (e.g. the allowed radius range).
   Transfer the ADJUSTMENT RULE instead: which parameter to change, in which
   direction, and what observation triggers the change. If a parameter's good
   value is unknown for the target, say so and tell the agent to explore its
   allowed range, not to start from a source default.

10. **Mechanism check.** Every skill and mistake must carry a
   `mechanism_check` field (1-2 sentences): what physical role the source
   mechanism played in the source level, and whether the SAME role exists in
   the target given only the target description. If the role differs (e.g.
   in the source a larger ball hurt because it tightened an overlap
   tolerance, but in the target ball mass is what drives the effect), do NOT
   emit the item. This applies with extra force to mistakes: a source
   anti-pattern is only a target anti-pattern if its cause carries over.

The `mechanism_check` field is REQUIRED on every skill and mistake (it is an
allowed extra field despite the instruction below about extra fields).
"""

_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
_GUARDED_FIELDS = (
    "title", "principle", "when_to_apply", "example",
    "description", "why_it_happens", "how_to_avoid",
)


def _numbers_in(text: str) -> set[float]:
    return {float(m) for m in _NUMBER_RE.findall(text or "")}


def _allowed_numbers(prompt_template: str, target_block: str) -> set[float]:
    """Numbers the teacher may legitimately state: shared env + target facts."""
    env = re.search(r"\*\*Environment.*?═══", prompt_template, re.S)
    return _numbers_in(env.group(0) if env else "") | _numbers_in(target_block)


def _source_numbers(source_bank_paths: dict[str, Path]) -> set[float]:
    return set().union(*(_numbers_in(Path(p).read_text())
                         for p in source_bank_paths.values()))


def _guard_violations(parsed: dict, allowed: set[float],
                      source_nums: set[float]) -> list[dict]:
    """List items using numbers outside ``allowed`` or lacking mechanism_check."""
    out = []
    for kind in ("skills", "mistakes"):
        for i, item in enumerate(parsed.get(kind, []) or []):
            bad = set()
            for f in _GUARDED_FIELDS:
                bad |= _numbers_in(str(item.get(f, ""))) - allowed
            problems = []
            if bad:
                tags = [f"{n:g}" + (" (in a source bank)" if n in source_nums else "")
                        for n in sorted(bad)]
                problems.append("uses numbers not in the env/target description: "
                                + ", ".join(tags))
            if not str(item.get("mechanism_check", "")).strip():
                problems.append("missing mechanism_check")
            if problems:
                out.append({"kind": kind, "index": i,
                            "title": item.get("title") or item.get("description", "")[:60],
                            "problems": problems})
    return out


GUARD_REPAIR_PROMPT = """{original_prompt}

═══ YOUR PREVIOUS OUTPUT ═══

```json
{previous_json}
```

═══ VALIDATION FAILED ═══

These items violate the transfer guard (rules 9-10):
{violations}

Rewrite ONLY the offending items so they comply (remove the numbers and
express the adjustment rule qualitatively; add the missing mechanism_check;
or drop an item whose mechanism does not carry over). Keep every other item
unchanged. Output the FULL corrected JSON object in a single fenced ```json
block.
"""



# ── Source critic (opt-in, --source-critic) ────────────────────────────
# Failure seen in catapult v3/v4: three source banks each said "default to a
# small/moderate radius", for three different reasons (hit tolerance,
# overlap avoidance, limiting the green ball's travel), none of which holds
# in the target. Rule 6 counted that as 3-bank corroboration (conf 0.70), and
# the self-graded mechanism_check justified it with a reason none of the
# sources gave. The critic fixes both: rule 11 defines corroboration as the
# same REASON, and a separate teacher call, which sees only one transferred
# item, its cited source entries and the factual target description, judges
# each cited source on its OWN stated reason. Items keep only accepted
# sources (rewritten if needed), and confidence is capped by how many
# distinct source levels and seeds actually back them. Target-agnostic.

SOURCE_CRITIC_BLOCK = """
11. **Corroboration means the same REASON.** When rule 6 asks how many
   source banks support a skill, count a source only if its OWN stated
   reason for its advice is the same physical reason you rely on in the
   target. Several sources giving the same advice for different reasons is
   NOT corroboration. Weigh evidence size too: `seeds=N` shows how many
   successful seeds a source entry was distilled from; entries from few seeds
   are weak. A default or preferred value for a free parameter needs a source
   reason that operates in the target; otherwise present the parameter as an
   open question the agent should test, not as a default.
"""

SOURCE_CRITIC_PROMPT = """\
You are auditing ONE entry of a skill bank that was transferred from other
physics levels to a new TARGET level the agent has never attempted. Judge each
cited source entry independently and strictly. You are not the author of the
entry; do not defend it.

**Environment (shared):** 2D Box2D physics, gravity -9.8, world [-5, 5]^2. The
agent places ONE red ball (x, y, radius 0.1-2.0) at t=0, then the simulation runs.

═══ TARGET LEVEL (factual description only) ═══
{target_block}

═══ TRANSFERRED ENTRY ({kind}) ═══
{item_text}

═══ CITED SOURCE ENTRIES ═══
{sources_text}

For each cited source entry:
a. `source_reason`: the physical reason, as stated or directly implied BY THE
   ENTRY ITSELF, why its advice helped in its OWN level. Paraphrase the entry;
   do not substitute a better reason of your own.
b. `holds_in_target`: does that SAME reason operate in the target, judging only
   from the target description? False if it depends on entities, constraints
   or goals the target lacks, or that play a different role there (e.g. a
   reason about placement feasibility does not support a claim about how
   strong an effect is).
c. `supports_entry`: under that reason, does the source support what the
   transferred entry actually recommends, including any default value,
   direction or condition it states?

Then list `unsupported_parts`: recommendations in the transferred entry
(defaults, directions, conditions) that no source with holds_in_target AND
supports_entry backs. Empty string if none.

Output ONLY a JSON object in a single fenced ```json block:
```json
{{"sources": [{{"index": 1, "source_reason": "...", "holds_in_target": true, "supports_entry": true, "why": "<1 sentence>"}}],
  "unsupported_parts": ""}}
```
"""

SOURCE_CRITIC_REWRITE_PROMPT = """\
An entry in a transferred skill bank cites some source entries whose reasons
do NOT carry over to the target level. Rewrite the entry so it states only
what the ACCEPTED sources support, under their accepted reasons. Remove every
default, direction or condition that came only from rejected sources or is
listed as unsupported. Where the entry used to prescribe a value or direction
for a free parameter without accepted support, say instead that the best
value is unknown for this level and should be tested across its range.

Constraints: use only entities from the target description; no numbers
except those in the shared environment or target description; no platitudes.
If nothing specific and useful remains, output {{"drop": true}}.

═══ TARGET LEVEL ═══
{target_block}

═══ ENTRY ({kind}) ═══
```json
{item_json}
```

═══ ACCEPTED SOURCES ═══
{accepted}

═══ REJECTED SOURCES ═══
{rejected}

═══ UNSUPPORTED PARTS ═══
{unsupported}

Output ONLY the rewritten entry as a JSON object with the same fields (or
{{"drop": true}}) in a single fenced ```json block.
"""

_ITEM_TEXT_FIELDS = ("title", "principle", "when_to_apply", "example",
                     "description", "why_it_happens", "how_to_avoid")


def _source_index(loaded_sources: dict[str, SkillBank]) -> dict[str, dict]:
    """skill/mistake id -> {level, text, n_seeds}, for the critic."""
    idx: dict[str, dict] = {}
    for lvl, b in loaded_sources.items():
        for sk in b.level_skills.get(lvl, []):
            text = f"[skill] {sk.title} -- {sk.principle}"
            if sk.when_to_apply:
                text += f" Apply when: {sk.when_to_apply}"
            if sk.example:
                text += f" Example: {sk.example}"
            idx[sk.skill_id] = {"level": lvl, "text": text, "n_seeds": len(sk.source_seeds)}
        for m in b.level_mistakes.get(lvl, []):
            text = (f"[mistake] {m.description} Why: {m.why_it_happens} "
                    f"Avoid: {m.how_to_avoid}")
            idx[m.mistake_id] = {"level": lvl, "text": text, "n_seeds": len(m.source_seeds)}
    return idx


def _critic_confidence_cap(n_levels: int, n_seeds: int) -> float:
    cap = {0: 0.0, 1: 0.5, 2: 0.65}.get(n_levels, 0.8)
    return min(cap, 0.5) if n_seeds < 5 else cap


def _critique_item(kind: str, item: dict, src_idx: dict[str, dict],
                   target_block: str, teacher_model: str) -> dict:
    """Run the critic (and, if needed, the rewrite) for one item."""
    cites = item.get("source_skills", []) or []
    resolved = [(c, src_idx.get(c.get("skill_id", ""))) for c in cites]
    known = [(c, s) for c, s in resolved if s]
    log = {"title": item.get("title") or item.get("description", "")[:60],
           "unresolved_cites": [c for c, s in resolved if not s]}
    if not known:
        return {**log, "action": "drop", "reason": "no resolvable cited sources"}

    item_text = "\n".join(f"{f}: {item[f]}" for f in _ITEM_TEXT_FIELDS if item.get(f))
    sources_text = "\n\n".join(
        f"[{i}] level `{s['level']}`, entry `{c['skill_id']}` "
        f"(distilled from {s['n_seeds']} seeds)\n"
        f"Scene of `{s['level']}`:\n{SOURCE_LEVEL_BLOCKS.get(s['level'], '')}\n"
        f"Entry: {s['text']}"
        for i, (c, s) in enumerate(known, 1))
    raw = call_teacher(SOURCE_CRITIC_PROMPT.format(
        target_block=target_block, kind=kind[:-1], item_text=item_text,
        sources_text=sources_text), model=teacher_model)
    verdict = _parse_json_object(raw or "") or {}
    by_idx = {v.get("index"): v for v in verdict.get("sources", []) if isinstance(v, dict)}
    judged = []
    for i, (c, s) in enumerate(known, 1):
        v = by_idx.get(i, {})
        judged.append({"source_level": s["level"], "skill_id": c["skill_id"],
                       "n_seeds": s["n_seeds"],
                       "source_reason": v.get("source_reason", ""),
                       "holds_in_target": bool(v.get("holds_in_target")),
                       "supports_entry": bool(v.get("supports_entry")),
                       "why": v.get("why", "")})
    accepted = [j for j in judged if j["holds_in_target"] and j["supports_entry"]]
    rejected = [j for j in judged if j not in accepted]
    unsupported = str(verdict.get("unsupported_parts", "") or "").strip()
    log.update(verdicts=judged, unsupported_parts=unsupported,
               critic_parsed=bool(verdict))

    if not verdict:
        return {**log, "action": "drop", "reason": "critic output unparseable"}
    if not accepted:
        return {**log, "action": "drop", "reason": "no cited source accepted"}

    new_item = dict(item)
    if rejected or unsupported:
        fmt = lambda js: "\n".join(
            f"- `{j['skill_id']}` ({j['source_level']}): reason = {j['source_reason']} "
            f"-- {j['why']}" for j in js) or "(none)"
        raw_rw = call_teacher(SOURCE_CRITIC_REWRITE_PROMPT.format(
            target_block=target_block, kind=kind[:-1],
            item_json=json.dumps({k: v for k, v in item.items()
                                  if k not in ("source_skills",)}, indent=2),
            accepted=fmt(accepted), rejected=fmt(rejected),
            unsupported=unsupported or "(none)"), model=teacher_model)
        rw = _parse_json_object(raw_rw or "")
        if not rw or rw.get("drop"):
            return {**log, "action": "drop",
                    "reason": "rewrite dropped it" if rw else "rewrite unparseable"}
        new_item.update({k: v for k, v in rw.items() if k in _ITEM_TEXT_FIELDS
                         or k in ("transfer_rationale", "mechanism_check")})
        log["action"] = "rewritten"
        log["before"] = {f: item.get(f, "") for f in _ITEM_TEXT_FIELDS if item.get(f)}
    else:
        log["action"] = "kept"

    new_item["source_skills"] = [{"source_level": j["source_level"],
                                  "skill_id": j["skill_id"]} for j in accepted]
    if kind == "skills":
        n_levels = len({j["source_level"] for j in accepted})
        n_seeds = sum(j["n_seeds"] for j in accepted)
        cap = _critic_confidence_cap(n_levels, n_seeds)
        before = float(item.get("confidence", 0.5))
        new_item["confidence"] = min(before, cap)
        log.update(confidence_before=before, confidence_cap=cap,
                   accepted_levels=n_levels, accepted_seeds=n_seeds)
    log["item"] = new_item
    return log


def _run_source_critic(parsed: dict, loaded_sources: dict[str, SkillBank],
                       target_block: str, teacher_model: str,
                       allowed: set[float] | None, source_nums: set[float] | None,
                       workers: int = 4) -> tuple[dict, list[dict]]:
    src_idx = _source_index(loaded_sources)
    jobs = [(kind, item) for kind in ("skills", "mistakes")
            for item in (parsed.get(kind) or [])]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        logs = list(ex.map(lambda ki: _critique_item(ki[0], ki[1], src_idx,
                                                     target_block, teacher_model), jobs))
    out = {k: v for k, v in parsed.items() if k not in ("skills", "mistakes")}
    out["skills"], out["mistakes"] = [], []
    for (kind, _), log in zip(jobs, logs):
        log["kind"] = kind
        if log["action"] == "drop":
            continue
        item = log["item"]
        if allowed is not None:  # rewrites must still pass the numeric guard
            v = _guard_violations({kind: [item]}, allowed, source_nums or set())
            if v and any("numbers" in p for p in v[0]["problems"]):
                log["action"], log["reason"] = "drop", "rewrite failed numeric guard"
                continue
        out[kind].append(item)
    for log in logs:
        log.pop("item", None)
    return out, logs


# ── Default check on transferred entries (opt-in, --default-check) ─────
# Rule 9 forbids defaults for parameters whose good value is unknown in the
# target, but nothing enforced it on transferred entries. v6 kept two
# pass_the_parcel-derived entries saying "start at or near the arm end's
# height, don't drop from high" -- a default the critic accepted. This applies
# the target-prior checker's default test to every kept transferred skill and
# mistake (one batched call), gives each offender one rewrite (state the
# trade-off and say to test it), re-checks, and drops what still prescribes.

DEFAULT_CHECK_PROMPT = """\
You are checking entries of a skill bank for a 2D physics puzzle the agent has
never attempted. The agent's free parameters are the red ball's x, y and radius.

═══ LEVEL (factual description) ═══
{target_block}

═══ ENTRIES ═══
{entries}

For each entry answer `prescribes_default`: does it recommend a default,
starting or preferred value, region or direction for any free parameter
(e.g. "start near X", "prefer small radius", "do not place high on the first
attempt"), rather than describing a mechanism, a trade-off to test, or a
procedure (such as how to resolve an overlap)? Corrective rules triggered by a
specific observed outcome ("if A happened, change B") are NOT defaults.
Output ONLY a JSON object in a fenced ```json block:
```json
{{"entries": [{{"index": 1, "prescribes_default": false, "why": "<1 sentence>"}}]}}
```
"""

DEFAULT_REWRITE_PROMPT = """\
This skill-bank entry recommends a default value, region or direction for a
free parameter (x, y or radius) of a puzzle the agent has never attempted:
{why}

Rewrite it so it keeps its mechanism and any outcome-triggered corrective rule,
but replaces the default with the trade-off: what each direction gains and
loses, and that the best choice is unknown for this level and should be tested.
Use only entities from the level description; no numbers except those in it.
If nothing specific remains without the default, output {{"drop": true}}.

═══ LEVEL ═══
{target_block}

═══ ENTRY ({kind}) ═══
```json
{item_json}
```

Output ONLY the rewritten entry as a JSON object with the same fields (or
{{"drop": true}}) in a single fenced ```json block.
"""


def _default_verdicts(items: list[tuple[str, dict]], target_block: str,
                      teacher_model: str) -> dict[int, dict]:
    entries = "\n\n".join(
        f"[{i}] ({kind[:-1]})\n" + "\n".join(
            f"{f}: {it[f]}" for f in _ITEM_TEXT_FIELDS if it.get(f))
        for i, (kind, it) in enumerate(items, 1))
    raw = call_teacher(DEFAULT_CHECK_PROMPT.format(
        target_block=target_block, entries=entries), model=teacher_model)
    return {v.get("index"): v for v in
            ((_parse_json_object(raw or "") or {}).get("entries", []) or [])
            if isinstance(v, dict)}


def _run_default_check(parsed: dict, target_block: str, teacher_model: str,
                       allowed: set[float] | None) -> tuple[dict, list[dict]]:
    items = [(k, it) for k in ("skills", "mistakes") for it in (parsed.get(k) or [])]
    if not items:
        return parsed, []
    first = _default_verdicts(items, target_block, teacher_model)
    log, new_items = [], []
    for i, (kind, it) in enumerate(items, 1):
        v = first.get(i) or {}
        title = it.get("title") or it.get("description", "")[:60]
        if not v.get("prescribes_default"):
            new_items.append((kind, it))
            continue
        raw = call_teacher(DEFAULT_REWRITE_PROMPT.format(
            why=v.get("why", ""), target_block=target_block, kind=kind[:-1],
            item_json=json.dumps({k: x for k, x in it.items()
                                  if k in _ITEM_TEXT_FIELDS}, indent=2)),
            model=teacher_model)
        rw = _parse_json_object(raw or "")
        entry = {"kind": kind, "title": title, "why": v.get("why", ""),
                 "before": {f: it.get(f, "") for f in _ITEM_TEXT_FIELDS if it.get(f)}}
        if not rw or rw.get("drop"):
            log.append({**entry, "action": "drop",
                        "reason": "rewrite dropped it" if rw else "rewrite unparseable"})
            continue
        new_it = dict(it)
        new_it.update({k: x for k, x in rw.items() if k in _ITEM_TEXT_FIELDS})
        new_items.append((kind, new_it))
        log.append({**entry, "action": "rewritten", "pending_recheck": True,
                    "after": {f: new_it.get(f, "") for f in _ITEM_TEXT_FIELDS if new_it.get(f)}})

    # Re-check the rewrites (one batched call); drop any that still prescribe
    # or that broke the numeric guard.
    rewritten = [(n, ki) for n, ki in enumerate(new_items)
                 if any(l.get("pending_recheck") and l["after"] == {f: ki[1].get(f, "")
                        for f in _ITEM_TEXT_FIELDS if ki[1].get(f)} for l in log)]
    drop_idx = set()
    if rewritten:
        second = _default_verdicts([ki for _, ki in rewritten], target_block, teacher_model)
        for j, (n, (kind, it)) in enumerate(rewritten, 1):
            v = second.get(j) or {}
            entry = next(l for l in log if l.get("pending_recheck")
                         and l["after"] == {f: it.get(f, "") for f in _ITEM_TEXT_FIELDS if it.get(f)})
            entry.pop("pending_recheck", None)
            bad = (allowed is not None and
                   _guard_violations({kind: [it]}, allowed, set()) and
                   any("numbers" in p for p in _guard_violations({kind: [it]}, allowed, set())[0]["problems"]))
            if v.get("prescribes_default") or bad:
                drop_idx.add(n)
                entry["action"] = "drop"
                entry["reason"] = ("still prescribes a default after rewrite: " + v.get("why", "")
                                   if v.get("prescribes_default") else "rewrite failed numeric guard")
    out = {k: x for k, x in parsed.items() if k not in ("skills", "mistakes")}
    out["skills"] = [it for n, (k, it) in enumerate(new_items) if k == "skills" and n not in drop_idx]
    out["mistakes"] = [it for n, (k, it) in enumerate(new_items) if k == "mistakes" and n not in drop_idx]
    return out, log


# ── Target priors (opt-in, --target-priors) ────────────────────────────
# The source critic showed that for catapult almost no physics transfers
# from the source levels (none has a lever), and the actor's own
# small-radius habit persists even with no radius advice (v5: 70% of
# simulations at r < 0.73). Target priors are a separate, honestly labelled
# step: the teacher sees ONLY the shared environment and the factual target
# description (no source banks, no trajectories) and states how each free
# parameter scales, the trade-offs, and every way the target's controlling
# quantity can be changed. No defaults or preferred values. A separate check
# call drops entries that do not follow from the given facts or that
# prescribe a default; the numeric guard applies too. Target-agnostic: the
# same prompt for every target, and nothing from target runs.

TARGET_PRIOR_PROMPT = """\
You are a physicist preparing an agent for a 2D physics puzzle it has NEVER
attempted. You have no solutions, no trajectories and no strategy hints: only
the environment facts and the factual level description below. Write what
follows from those facts alone.

{env_block}

═══ LEVEL: {target_level} (factual description) ═══
{target_block}

═══ TASK ═══
Write 2 to 4 PRIOR entries. Each states physics that follows directly from the
facts above and that bears on how the agent should explore its free parameters
(x, y, radius). Cover:
- how each free parameter scales the physical effect it controls (e.g. what
  radius does to mass and momentum, what drop height does to impact speed);
- the trade-offs each parameter carries (what gets worse as it grows or shrinks);
- for the quantity that most likely decides success in this level, EVERY
  distinct way the agent can raise or lower it, so a failed attempt can be
  corrected along more than one parameter.

HARD RULES:
1. No default, starting or preferred value or direction for any parameter
   ("start small", "use a moderate radius", "prefer the far end" are all
   forbidden). Where a choice matters, say what each direction trades off and
   that it should be tested.
2. No numbers except those stated in the facts above.
3. Every entry must name, in `derivation`, the specific facts above it follows
   from. If it needs a fact that is not stated, do not write it.
4. Use only entities named in the level description. No platitudes.

OUTPUT a single JSON object in a fenced ```json block:
```json
{{"priors": [{{"title": "<3-7 words>", "principle": "<2-3 sentences>",
  "when_to_apply": "<the situation in which the agent should use it>",
  "derivation": "<which stated facts it follows from>"}}]}}
```
"""

TARGET_PRIOR_CHECK_PROMPT = """\
You are checking prior-knowledge entries written for a 2D physics puzzle from
its factual description alone. Judge each entry strictly and independently.

{env_block}

═══ LEVEL (factual description) ═══
{target_block}

═══ ENTRIES ═══
{entries}

For each entry answer:
- `follows_from_facts`: does every claim follow from the facts above plus
  standard physics, without assuming anything about this level that is not
  stated?
- `prescribes_default`: does it recommend a default, starting or preferred
  value or direction for any free parameter (x, y, radius), instead of
  describing trade-offs to test?
Output ONLY a JSON object in a fenced ```json block:
```json
{{"entries": [{{"index": 1, "follows_from_facts": true, "prescribes_default": false, "why": "<1 sentence>"}}]}}
```
"""

TARGET_PRIOR_CONFIDENCE = 0.6
TARGET_PRIOR_TITLE_TAG = "[From level description] "


def _env_block(prompt_template: str) -> str:
    env = re.search(r"\*\*Environment.*?(?=\n═══)", prompt_template, re.S)
    return env.group(0).strip() if env else ""


def _build_target_priors(target_level: str, target_block: str, env_block: str,
                         teacher_model: str, output_dir: Path,
                         allowed: set[float]) -> tuple[list[dict], dict]:
    raw = call_teacher(TARGET_PRIOR_PROMPT.format(
        env_block=env_block, target_level=target_level, target_block=target_block),
        model=teacher_model)
    (output_dir / f"_target_priors_raw_{target_level}.txt").write_text(raw or "")
    priors = (_parse_json_object(raw or "") or {}).get("priors", []) or []
    log: dict = {"proposed": priors, "dropped": []}
    if not priors:
        return [], {**log, "error": "no priors parsed"}

    entries = "\n\n".join(
        f"[{i}] {p.get('title', '')}\nprinciple: {p.get('principle', '')}\n"
        f"when_to_apply: {p.get('when_to_apply', '')}"
        for i, p in enumerate(priors, 1))
    raw_chk = call_teacher(TARGET_PRIOR_CHECK_PROMPT.format(
        env_block=env_block, target_block=target_block, entries=entries),
        model=teacher_model)
    verdicts = {v.get("index"): v for v in
                ((_parse_json_object(raw_chk or "") or {}).get("entries", []) or [])
                if isinstance(v, dict)}
    log["check"] = verdicts
    kept = []
    for i, p in enumerate(priors, 1):
        v = verdicts.get(i)
        bad_nums = set()
        for f in ("title", "principle", "when_to_apply"):
            bad_nums |= _numbers_in(str(p.get(f, ""))) - allowed
        reason = None
        if v is None:
            reason = "check output missing"
        elif not v.get("follows_from_facts"):
            reason = f"does not follow from facts: {v.get('why', '')}"
        elif v.get("prescribes_default"):
            reason = f"prescribes a default: {v.get('why', '')}"
        elif bad_nums:
            reason = "numbers not in env/target: " + ", ".join(f"{n:g}" for n in sorted(bad_nums))
        if reason:
            log["dropped"].append({"title": p.get("title", ""), "reason": reason})
        else:
            kept.append(p)
    return kept, log


# ── Detailed factual scene descriptions (--detailed-scenes) ────────────
# Same rule as the short blocks: facts only (objects, geometry, randomisation
# ranges, what is static/dynamic, success test), no approach implied. Taken
# from the interphyre level definitions. Used for both the source and the
# target side so the teacher can map entities by the role they play.

SOURCE_LEVEL_BLOCKS_DETAILED: dict[str, str] = {
    "down_to_earth": """\
**Scene (factual, from the level definition):**
- Box: static walls on all four edges of [-5, 5] × [-5, 5].
- Purple Ground: static horizontal bar spanning the full width just above the bottom wall (y = -4.9).
- Black Platform: one static horizontal bar, width random in [1, 7], left end random, height random in [-2, 2]. Open space on both sides of it.
- Green Ball: dynamic, radius 0.5. Starts at rest near the top of the box (y = 4.0), horizontally centred over the platform, and falls under gravity from t = 0 onto the platform.

**Success condition:** the green ball touches the purple ground continuously for at least 3 seconds.

**Action:** place ONE red ball at (x, y, radius), 0.1 ≤ r ≤ 2.0, fully inside the box, not overlapping anything at t = 0. It is dynamic, starts at rest and falls under gravity.
""",
}

TARGET_LEVEL_BLOCKS_DETAILED: dict[str, str] = {
    "falling_into_place": """\
**Scene (factual, from the level definition — no approach implied):**
- Box: static walls on all four edges of [-5, 5] × [-5, 5].
- Black Platform: two static horizontal bars at the same height (random in [-2, 1]). One runs from the left wall to the left edge of a gap, the other from the right edge of the gap to the right wall. The gap is 2.0 wide; its centre is random but close to x = 0.
- Green Ball: dynamic, radius 0.5, starts at rest on top of one of the two bars (left or right, random), at least one radius away from the gap edge.
- Blue Basket: dynamic cup, upside down (opening facing DOWN), about 1.4 wide. Starts at x = 0, y = 4.3, above the gap, and falls under gravity from t = 0.
- Bottom Ramp: static bar starting at the bottom-left corner and rising to the right at 10°, near the floor.

**Success condition:** the green ball touches the blue basket continuously for at least 3 seconds.

**Action:** place ONE red ball at (x, y, radius), 0.1 ≤ r ≤ 2.0, fully inside the box, not overlapping anything at t = 0. It is dynamic, starts at rest and falls under gravity.
""",
    "two_body_problem": """\
**Scene (factual, from the level definition — no approach implied):**
- Box: static walls on all four edges of [-5, 5] × [-5, 5]. There are no other static objects (no platforms).
- Green Ball: dynamic, radius random in [0.3, 0.6].
- Blue Ball: dynamic, radius random in [0.3, 0.6].
- Both balls start at rest with their bottoms at the same height (random in [-3, 3]). The green ball is always to the LEFT of the blue ball; the horizontal gap between their surfaces is at least the larger of the two radii. Both fall under gravity from t = 0 onto the bottom wall.

**Success condition:** the green ball touches the blue ball continuously for at least 3 seconds.

**Action:** place ONE red ball at (x, y, radius), 0.1 ≤ r ≤ 2.0, fully inside the box, not overlapping anything at t = 0. It is dynamic, starts at rest and falls under gravity.
""",
}


# ── Per-item transfer (--per-item) ──────────────────────────────────────
# A small teacher cannot build a whole bank under many rules in one call
# (observed: cited the right source skill but reversed its content, wrote
# numbers it was told not to). Per-item mode splits the job:
#   1. one role-mapping call: which target entity plays each source entity's role
#   2. one call per source entry: does its cause exist in the target? if so,
#      rewrite it by swapping entities, keeping every relation word
#   3. code checks per entry: no numbers outside env/target facts, no lost
#      direction/relation words -> one retry with the problem stated, then drop
# Citations are exact by construction (each output entry has one source).
# Everything here is target-agnostic: no level-specific examples.

# Factual roles, read off each level definition: which object the agent
# places, which object the success condition is about, which object it must
# touch, and what is static scenery. Mapping by these roles is mechanical,
# so it is done in code (a 7B teacher mis-mapped same-named objects).
SCENE_ROLES: dict[str, dict[str, str]] = {
    "down_to_earth": {"Red Ball": "agent ball", "Green Ball": "success ball",
                      "Purple Ground": "success partner",
                      "Black Platform": "static bar"},
    "falling_into_place": {"Red Ball": "agent ball", "Green Ball": "success ball",
                           "Blue Basket": "success partner",
                           "Black Platform": "static bar", "Bottom Ramp": "static bar"},
    "two_body_problem": {"Red Ball": "agent ball", "Green Ball": "success ball",
                         "Blue Ball": "success partner"},
}
ROLE_TEXT = {
    "agent ball": "the ball the agent places",
    "success ball": "the object the success condition is about",
    "success partner": "the object it must touch for success",
    "static bar": "static scenery (bar/platform)",
}


def _role_mapping(source_level: str, target_level: str) -> list[dict]:
    src, tgt = SCENE_ROLES.get(source_level, {}), SCENE_ROLES.get(target_level, {})
    out = []
    for name, role in src.items():
        same = [n for n, r in tgt.items() if r == role]
        if name in tgt and tgt[name] == role:
            same = [name]
        out.append({"source": name, "role": ROLE_TEXT[role],
                    "target": " / ".join(same) if same else "none"})
    return out


PER_ITEM_ENV = """\
**Environment (shared across all levels):** 2D physics simulation (Box2D), gravity = -9.8 m/s², \
world bounds [-5, 5] on both axes. The agent places ONE red ball at (x, y) with a chosen radius \
(0.1 ≤ r ≤ 2.0) at t = 0; the simulation then runs to completion."""

TRANSLATION_GUIDE = """\
HOW TO TRANSLATE A SOURCE ENTRY INTO THE TARGET LEVEL
1. Find the CAUSE: what physical interaction makes the source entry work —
   which object acts on which, in which direction, and why.
2. Map ROLES, not names: use the role mapping. Each source entity is replaced
   by the target entity that plays the same role.
3. Check the cause exists in the target: the same kind of interaction must be
   possible with the target's objects, motions and constraints as described.
   If the entry mentions an object that has no counterpart, the entry still
   applies when its cause does not depend on that object (restate it without
   the object, e.g. in terms of the direction the success object must move).
   It does NOT apply only when the cause itself needs the missing object.
4. Rewrite: keep the source sentence's structure and replace each entity by
   its counterpart. Keep every word that says where or which way (opposite,
   away from, toward, above, below, beside, behind, beyond, before, after,
   left/right of an object). Never drop or reverse such a word.
5. No numbers: where the source used a number, write the relation it
   expressed instead (e.g. "slightly", "just beyond contact distance"). Never
   write a new number.
6. Add nothing: do not add physics, strategy or facts that the source entry
   does not state."""

ROLE_MAP_PROMPT = """\
You are mapping the objects of one physics puzzle onto another.

{env}

=== SOURCE LEVEL: {source_level} ===
{source_scene}

=== TARGET LEVEL: {target_level} ===
{target_scene}

For EACH object in the source scene (including the red ball), state the role it
plays in reaching the source success condition (e.g. "object that must be
moved", "goal it must reach", "support it rests or lands on", "obstacle",
"agent's tool"), then name the target object that plays the SAME role, or
"none" if no target object does. Base this only on the two descriptions.

Output ONLY a JSON object in a fenced ```json block:
```json
{{"mapping": [{{"source": "<source object>", "role": "<role>", "target": "<target object or none>", "why": "<1 sentence>"}}]}}
```
"""

PER_ITEM_PROMPT = """\
You are transferring ONE entry from a source skill bank to a new target level.
No target-level trajectories exist; use only the descriptions below.

{env}

=== SOURCE LEVEL: {source_level} ===
{source_scene}

=== TARGET LEVEL: {target_level} ===
{target_scene}

=== ROLE MAPPING (source object -> target object) ===
{mapping}

{guide}

=== SOURCE ENTRY ({kind}, id {entry_id}) ===
{entry}
{feedback}
Rewrite the entry for the target following the steps above. Set "applies" to
false ONLY if the cause itself needs an object that has no counterpart.

Output ONLY a JSON object in a fenced ```json block:
```json
{fmt}
```
"""

PER_ITEM_SKILL_FMT = """\
{{"applies": true|false,
 "cause": "<1 sentence: the physical cause behind the source entry>",
 "reason": "<1 sentence: why that cause does or does not exist in the target>",
 "title": "<3-7 words>",
 "principle": "<the rewritten principle>",
 "when_to_apply": "<the rewritten condition>",
 "example": "<qualitative, no numbers, or empty>"}}"""

PER_ITEM_MISTAKE_FMT = """\
{{"applies": true|false,
 "cause": "<1 sentence: the physical cause behind the source entry>",
 "reason": "<1 sentence: why that cause does or does not exist in the target>",
 "description": "<the rewritten mistake>",
 "why_it_happens": "<the rewritten root cause>",
 "how_to_avoid": "<the rewritten fix>"}}"""

# Relation words whose loss or reversal changes what an entry tells the agent.
_RELATION_GROUPS: dict[str, tuple[str, ...]] = {
    "opposite": ("opposite",),
    "away": ("away",),
    "toward": ("toward", "towards"),
    "below": ("below", "beneath", "under", "lower than"),
    "above": ("above", "over", "higher than"),
    "side": ("left", "right"),
    "behind/beyond": ("behind", "beyond"),
    "order": ("before", "after"),
}
# Words naming parts of each SOURCE scene. A term the target description
# does not contain names something the target lacks and must not appear in a
# transferred entry (it would point the agent at a non-existent object).
SCENE_TERMS: dict[str, tuple[str, ...]] = {
    "down_to_earth": ("platform", "gap", "edge", "ground", "purple", "floor"),
}
_SEED_REF_RE = re.compile(
    r"\s*\(?\b[Ss]eeds?\s*#?\d+(?:\s*(?:,|and|/|&)\s*#?\d+)*\)?")
# snake_case words that are object ids, level names or error-message terms,
# not agent tools
_NON_TOOL_SUFFIXES = ("_ball", "_basket", "_bar", "_platform", "_ground", "_ramp", "_wall")
_LEVEL_NAMES = {"down_to_earth", "two_body_problem", "pass_the_parcel",
                "falling_into_place", "catapult", "cliffhanger"}
_NON_TOOL_TERMS = {"sum_of_radii"}
PER_ITEM_ATTEMPTS = 3
_TOOL_NAME_RE = re.compile(r"\b[a-z]{3,}(?:_[a-z]+)+\b")
PER_ITEM_MAX_SKILLS = 6
PER_ITEM_MAX_MISTAKES = 4
PER_ITEM_CONF_CAP = 0.7   # single-source entries; matches rule "0.7+ needs >=2 banks"


def _relation_groups(text: str) -> set[str]:
    t = f" {text.lower()} "
    return {g for g, words in _RELATION_GROUPS.items()
            if any(re.search(rf"\b{re.escape(w)}\b", t) for w in words)}


def _strip_seed_refs(text: str) -> str:
    """Drop evidence labels like 'Seed 4', '(seeds 1 and 9)' from source text."""
    return _SEED_REF_RE.sub("", text or "")


def _forbidden_terms(source_level: str, target_scene: str) -> set[str]:
    t = target_scene.lower()
    return {w for w in SCENE_TERMS.get(source_level, ())
            if not re.search(rf"\b{w}\b", t)}


def _per_item_problems(kind: str, src_core: str, out: dict,
                       allowed: set[float], forbidden: set[str] = frozenset()) -> list[str]:
    fields = (("title", "principle", "when_to_apply", "example") if kind == "skill"
              else ("description", "why_it_happens", "how_to_avoid"))
    text = " ".join(str(out.get(f, "")) for f in fields)
    problems = []
    bad = _numbers_in(text) - allowed
    if bad:
        problems.append("contains numbers not stated in the environment or target "
                        "description: " + ", ".join(f"{n:g}" for n in sorted(bad)))
    src_groups = _relation_groups(src_core)
    lost = src_groups - _relation_groups(text)
    # one group may go (a paraphrase that keeps the rest keeps the direction);
    # losing more, or the only one, means the direction was dropped or reversed
    if lost and len(src_groups) - len(lost) < max(1, len(src_groups) - 1):
        problems.append("dropped relation word(s) present in the source entry: "
                        + ", ".join(sorted(lost)))
    low = text.lower()
    terms = sorted(w for w in forbidden if re.search(rf"\b{w}s?\b", low))
    if terms:
        problems.append("names objects the target level does not have: "
                        + ", ".join(terms) + " (restate in terms of target objects)")
    tools = sorted(t for t in set(_TOOL_NAME_RE.findall(text))
                   if not t.endswith(_NON_TOOL_SUFFIXES) and t not in _LEVEL_NAMES
                   and t not in _NON_TOOL_TERMS)
    if tools:
        problems.append("names source tools/functions: " + ", ".join(tools)
                        + " (describe the action itself instead)")
    return problems


def _transfer_one(kind: str, entry, source_level: str, source_scene: str,
                  target_level: str, target_scene: str, mapping_txt: str,
                  allowed: set[float], teacher_model: str) -> dict:
    S = _strip_seed_refs
    if kind == "skill":
        entry_txt = (f"title: {S(entry.title)}\nprinciple: {S(entry.principle)}\n"
                     f"when_to_apply: {S(entry.when_to_apply)}\nexample: {S(entry.example)}")
        src_core = S(f"{entry.title} {entry.principle}")
        entry_id, fmt = entry.skill_id, PER_ITEM_SKILL_FMT
    else:
        entry_txt = (f"description: {S(entry.description)}\nwhy_it_happens: "
                     f"{S(entry.why_it_happens)}\nhow_to_avoid: {S(entry.how_to_avoid)}")
        src_core = S(f"{entry.description} {entry.how_to_avoid}")
        entry_id, fmt = entry.mistake_id, PER_ITEM_MISTAKE_FMT
    forbidden = _forbidden_terms(source_level, target_scene)
    # a relation word tied to a missing object (e.g. "the gap side") cannot be kept
    src_core_rel = src_core
    log = {"kind": kind, "source_id": entry_id, "attempts": []}
    feedback = ""
    for attempt in range(PER_ITEM_ATTEMPTS):
        prompt = PER_ITEM_PROMPT.format(
            env=PER_ITEM_ENV, source_level=source_level, source_scene=source_scene,
            target_level=target_level, target_scene=target_scene,
            mapping=mapping_txt, guide=TRANSLATION_GUIDE, kind=kind,
            entry_id=entry_id, entry=entry_txt, feedback=feedback, fmt=fmt)
        out = _parse_json_object(call_teacher(prompt, model=teacher_model) or "") or {}
        if not out:
            log["attempts"].append({"error": "unparseable"})
            continue
        if not out.get("applies"):
            log["attempts"].append({"applies": False, "reason": out.get("reason", "")})
            log["action"] = "not_applicable"
            log["reason"] = out.get("reason", "")
            return log
        problems = _per_item_problems(kind, src_core_rel, out, allowed, forbidden)
        log["attempts"].append({"output": out, "problems": problems})
        if not problems:
            log["action"] = "keep"
            log["output"] = out
            return log
        feedback = ("\n=== YOUR PREVIOUS REWRITE WAS REJECTED ===\n"
                    + json.dumps(out, indent=1) + "\nProblems:\n- "
                    + "\n- ".join(problems) + "\nFix these problems.\n")
    log["action"] = "drop"
    return log


def _run_per_item_transfer(target_level: str, target_scene: str,
                           loaded_sources: dict[str, SkillBank],
                           source_scenes: dict[str, str],
                           allowed: set[float], teacher_model: str,
                           output_dir: Path) -> tuple[dict, dict]:
    """Per-item transfer: returns (parsed-like dict, log)."""
    jobs, mappings = [], {}
    for lvl, bank in loaded_sources.items():
        rm = _role_mapping(lvl, target_level)
        mappings[lvl] = rm
        mapping_txt = "\n".join(
            f"- {m['source']} ({m['role']}) -> {m['target']}" for m in rm
        ) or "(no factual role table for this pair: map entities by role yourself)"
        print(f"  Role mapping {lvl} -> {target_level}:")
        for m in rm:
            print(f"    {m.get('source')} [{m.get('role')}] -> {m.get('target')}")
        for s in bank.level_skills.get(lvl, []):
            jobs.append(("skill", s, lvl, mapping_txt))
        for m in bank.level_mistakes.get(lvl, []):
            jobs.append(("mistake", m, lvl, mapping_txt))

    with ThreadPoolExecutor(max_workers=4) as ex:
        logs = list(ex.map(lambda j: _transfer_one(
            j[0], j[1], j[2], source_scenes[j[2]], target_level, target_scene,
            j[3], allowed, teacher_model), jobs))

    skills, mistakes = [], []
    for (kind, entry, lvl, _), lg in zip(jobs, logs):
        lg["source_level"] = lvl
        tag = {"keep": "KEEP", "drop": "DROP", "not_applicable": "N/A"}.get(lg.get("action"), "ERR")
        last = (lg["attempts"][-1].get("problems") if lg["attempts"] else None) or lg.get("reason", "")
        print(f"    [{tag}] {kind} {lg['source_id']}" + (f" ({str(last)[:110]})" if tag != "KEEP" else ""))
        if lg.get("action") != "keep":
            continue
        out = lg["output"]
        cite = [{"source_level": lvl, "skill_id": lg["source_id"]}]
        rationale = f"{out.get('cause', '')} {out.get('reason', '')}".strip()
        if kind == "skill":
            skills.append({
                "title": out.get("title", entry.title), "principle": out.get("principle", ""),
                "when_to_apply": out.get("when_to_apply", ""), "example": out.get("example", ""),
                "confidence": min(PER_ITEM_CONF_CAP, float(entry.confidence)),
                "source_skills": cite, "transfer_rationale": rationale})
        else:
            mistakes.append({
                "description": out.get("description", ""),
                "why_it_happens": out.get("why_it_happens", ""),
                "how_to_avoid": out.get("how_to_avoid", ""),
                "source_skills": cite, "transfer_rationale": rationale})
    skills.sort(key=lambda s: -s["confidence"])
    parsed = {"skills": skills[:PER_ITEM_MAX_SKILLS],
              "mistakes": mistakes[:PER_ITEM_MAX_MISTAKES],
              "synthesis_reasoning": (f"per-item transfer: {len(skills)} skill(s) and "
                                      f"{len(mistakes)} mistake(s) kept of {len(jobs)} source entries")}
    log = {"role_mappings": mappings, "items": logs,
           "capped": {"skills": max(0, len(skills) - PER_ITEM_MAX_SKILLS),
                      "mistakes": max(0, len(mistakes) - PER_ITEM_MAX_MISTAKES)}}
    (output_dir / f"_per_item_{target_level}.json").write_text(json.dumps(log, indent=2, default=str))
    return parsed, log


#: Prompt variants selectable by name. ``None`` / ``"auto"`` keeps the historic
#: teacher-model-prefix routing, so existing InterPhyre invocations are unchanged.
#: "full" and "slim" are the older names for "physics" and "physics-slim".
PROMPT_VARIANTS: dict[str, tuple[str, str]] = {
    "physics": (CROSS_LEVEL_SYNTHESIS_PROMPT, "physics (8-rule, with audit fields)"),
    "physics-slim": (CROSS_LEVEL_SYNTHESIS_PROMPT_QWEN, "physics-slim (3-rule, no audit fields)"),
    "full-neutral": (CROSS_LEVEL_SYNTHESIS_PROMPT_NEUTRAL,
                     "full-neutral (9-rule, audit fields, no hint examples)"),
    "text-game": (CROSS_TASK_SYNTHESIS_PROMPT, "text-game (9-rule, with audit fields)"),
}
PROMPT_VARIANTS["full"] = PROMPT_VARIANTS["physics"]
PROMPT_VARIANTS["slim"] = PROMPT_VARIANTS["physics-slim"]


def _select_prompt_template(
    teacher_model: str,
    prompt_variant: str | None = None,
) -> tuple[str, str]:
    """Return (prompt_template, label) for the given teacher.

    An explicit *prompt_variant* wins. Without one (``None`` or ``"auto"``),
    routing falls back to the historic behaviour: Claude models get the long
    8-rule physics prompt with audit fields, everything else gets the slim
    Qwen-friendly physics variant. full-neutral has no example
    mechanisms/placements/entities.

    That fallback is a trap for any non-Claude strong teacher — gpt-oss-120b
    would silently receive the small-model prompt and drop the audit fields the
    transfer inspection reads — so non-physics callers must name their variant.
    The label is used in logs and recorded in the audit.
    """
    if prompt_variant not in (None, "auto"):
        try:
            return PROMPT_VARIANTS[prompt_variant]
        except KeyError:
            raise ValueError(
                f"Unknown prompt_variant {prompt_variant!r}. "
                f"Known: {sorted(PROMPT_VARIANTS)}."
            ) from None
    if teacher_model.startswith("claude-"):
        return CROSS_LEVEL_SYNTHESIS_PROMPT, "claude (8-rule, with audit fields)"
    return CROSS_LEVEL_SYNTHESIS_PROMPT_QWEN, "qwen-slim (3-rule, no audit fields)"


# ── Source bank rendering for the prompt ───────────────────────────────

def format_source_bank_for_teacher(bank: SkillBank, level_name: str,
                                   show_evidence: bool = False) -> str:
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
            evidence = f", seeds={len(s.source_seeds)}" if show_evidence else ""
            lines.append(
                f"- `{s.skill_id}` (confidence={s.confidence:.2f}, gen={s.generation}{evidence}) "
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
    transfer_guard: bool = False,
    source_critic: bool = False,
    target_priors: bool = False,
    default_check: bool = False,
    prompt_variant: str | None = "auto",
    per_item: bool = False,
    detailed_scenes: bool = False,
    source_block_overrides: Mapping[str, str] | None = None,
    domain_block: str | None = None,
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
    transfer_guard : Add rules 9-10 (no source magnitudes, mechanism check)
        and validate the output in code: one repair round, then offending
        items are dropped. See TRANSFER_GUARD_BLOCK.
    source_critic : Add rule 11 (corroboration = same reason, evidence size)
        and audit every item with a separate critic call per item; keep only
        accepted sources, rewrite or drop, cap confidence by accepted
        evidence. See SOURCE_CRITIC_BLOCK.
    target_priors : Add 2-4 entries derived ONLY from the shared environment
        and the factual target description (no sources, no trajectories),
        labelled source_type="target_prior" and tagged in the title, checked
        by a separate call. See TARGET_PRIOR_PROMPT.
    default_check : Apply the "prescribes a default?" test to every kept
        transferred entry; rewrite offenders once, re-check, drop the rest.
        See DEFAULT_CHECK_PROMPT.
    prompt_variant : One of ``PROMPT_VARIANTS``, or ``"auto"``/``None`` for the
        historic teacher-prefix routing (see :func:`_select_prompt_template`).
    source_block_overrides : Optional {source_level: factual scene block},
        consulted before ``SOURCE_LEVEL_BLOCKS``. Required for any source
        outside InterPhyre, whose blocks are not in the module-level registry.
    domain_block : Shared-environment paragraph for the ``text-game`` variant,
        which parameterises it rather than hardcoding Box2D. Ignored by the
        physics variants, which bake their own in.
    dry_run : Build and dump the prompt, then stop without calling the teacher.
        Returns ``(None, audit)`` with ``audit["dry_run"] is True``.

    Returns
    -------
    (SkillBank | None, audit_dict)
    """
    if target_block_override is not None:
        target_block = target_block_override
    elif detailed_scenes:
        if target_level not in TARGET_LEVEL_BLOCKS_DETAILED:
            raise ValueError(f"No detailed scene block for target '{target_level}'.")
        target_block = TARGET_LEVEL_BLOCKS_DETAILED[target_level]
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
    source_scenes: dict[str, str] = {}
    overrides = dict(source_block_overrides or {})
    for source_level, bank_path in source_bank_paths.items():
        if source_level not in overrides and source_level not in SOURCE_LEVEL_BLOCKS:
            raise ValueError(
                f"No factual scene block registered for source level "
                f"'{source_level}'. Add it to SOURCE_LEVEL_BLOCKS or pass it "
                f"in source_block_overrides."
            )
        bank = SkillBank.load(bank_path)
        loaded_sources[source_level] = bank

        if source_level in overrides:
            scene = overrides[source_level]
        elif detailed_scenes and source_level in SOURCE_LEVEL_BLOCKS_DETAILED:
            scene = SOURCE_LEVEL_BLOCKS_DETAILED[source_level]
        else:
            scene = SOURCE_LEVEL_BLOCKS[source_level]
        source_scenes[source_level] = scene
        skills_md = format_source_bank_for_teacher(bank, source_level,
                                                   show_evidence=source_critic)
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
        transfer_guard_block=(TRANSFER_GUARD_BLOCK if transfer_guard else "")
                             + (SOURCE_CRITIC_BLOCK if source_critic else ""),
    )
    if source_critic:
        print("  Source critic: ON (rule 11 + per-item critic)")
    if transfer_guard:
        print("  Transfer guard: ON (rules 9-10 + numeric validator)")

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

    per_item_log: dict = {}
    if per_item:
        print(f"  Per-item transfer ({teacher_model}): role mapping + one call per source entry")
        allowed_pi = _numbers_in(PER_ITEM_ENV) | _numbers_in(target_block)
        parsed, per_item_log = _run_per_item_transfer(
            target_level, target_block, loaded_sources, source_scenes,
            allowed_pi, teacher_model, output_dir)
    else:
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

    guard_log: dict = {}
    if transfer_guard:
        allowed = _allowed_numbers(prompt_template, target_block)
        source_nums = _source_numbers(source_bank_paths)
        violations = _guard_violations(parsed, allowed, source_nums)
        guard_log = {"allowed_numbers": sorted(allowed),
                     "initial_violations": violations}
        print(f"  Guard: {len(violations)} item(s) in violation")
        for v in violations:
            print(f"    - {v['kind']}[{v['index']}] {v['title'][:50]!r}: {'; '.join(v['problems'])}")
        if violations:
            repair_prompt = GUARD_REPAIR_PROMPT.format(
                original_prompt=prompt,
                previous_json=json.dumps(parsed, indent=2),
                violations="\n".join(
                    f"- {v['kind']}[{v['index']}] \"{v['title']}\": {'; '.join(v['problems'])}"
                    for v in violations),
            )
            print("  Guard: asking teacher to repair...")
            repaired_raw = call_teacher(repair_prompt, model=teacher_model)
            (output_dir / f"_synthesis_repair_raw_{target_level}.txt").write_text(repaired_raw or "")
            repaired = _parse_json_object(repaired_raw) if repaired_raw else None
            if repaired and repaired.get("skills"):
                parsed = repaired
            else:
                print("  Guard: repair unparseable, keeping original and dropping offenders")
            violations = _guard_violations(parsed, allowed, source_nums)
            dropped = []
            for v in sorted(violations, key=lambda v: -v["index"]):
                dropped.append(v)
                parsed[v["kind"]].pop(v["index"])
            guard_log["repaired"] = bool(repaired and repaired.get("skills"))
            guard_log["dropped_after_repair"] = dropped
            print(f"  Guard: {len(dropped)} item(s) still in violation after repair -> dropped")
            for v in dropped:
                print(f"    - {v['kind']} {v['title'][:50]!r}: {'; '.join(v['problems'])}")

    critic_log: list[dict] = []
    if source_critic:
        print("  Source critic: auditing each item against its cited sources...")
        parsed, critic_log = _run_source_critic(
            parsed, loaded_sources, target_block, teacher_model,
            allowed=_allowed_numbers(prompt_template, target_block) if transfer_guard else None,
            source_nums=_source_numbers(source_bank_paths) if transfer_guard else None)
        for c in critic_log:
            acc = sum(v["holds_in_target"] and v["supports_entry"] for v in c.get("verdicts", []))
            tot = len(c.get("verdicts", []))
            extra = (f" conf {c['confidence_before']:.2f}->{min(c['confidence_before'], c['confidence_cap']):.2f}"
                     if "confidence_cap" in c else "")
            print(f"    - {c['kind'][:-1]} {c['title'][:50]!r}: {acc}/{tot} sources accepted"
                  f" -> {c['action'].upper()}{extra}"
                  + (f" ({c['reason']})" if c.get("reason") else ""))
        (output_dir / f"_source_critic_{target_level}.json").write_text(
            json.dumps(critic_log, indent=2))

    default_log: list[dict] = []
    if default_check:
        print("  Default check: testing transferred entries for parameter defaults...")
        parsed, default_log = _run_default_check(
            parsed, target_block, teacher_model,
            _allowed_numbers(prompt_template, target_block) if transfer_guard else None)
        for d in default_log:
            print(f"    - {d['kind'][:-1]} {d['title'][:50]!r}: {d['action'].upper()}"
                  + (f" ({d['reason'][:100]})" if d.get("reason") else ""))

    prior_entries: list[dict] = []
    prior_log: dict = {}
    if target_priors:
        print("  Target priors: deriving from the environment + target description only...")
        prior_entries, prior_log = _build_target_priors(
            target_level, target_block, _env_block(prompt_template), teacher_model,
            output_dir, _allowed_numbers(prompt_template, target_block))
        for d in prior_log.get("dropped", []):
            print(f"    - prior {d['title'][:50]!r}: DROP ({d['reason'][:120]})")

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
            "mechanism_check": raw.get("mechanism_check", ""),
        })
        n_sources_cited = len(raw.get("source_skills", []))
        print(
            f"    [SKILL {skill_id}] '{raw.get('title', '')[:60]}' "
            f"conf={confidence:.2f} cites={n_sources_cited}"
        )

    for i, raw in enumerate(prior_entries):
        skill_id = f"{prefix}_tp_{i:03d}"
        title = TARGET_PRIOR_TITLE_TAG + raw.get("title", "")
        bank.add_skill(Skill(
            skill_id=skill_id,
            title=title,
            principle=raw.get("principle", ""),
            when_to_apply=raw.get("when_to_apply", ""),
            example="",
            source_level=target_level,
            source_type="target_prior",
            source_seeds=[],
            generation=0,
            confidence=TARGET_PRIOR_CONFIDENCE,
        ))
        audit_skills.append({
            "skill_id": skill_id,
            "title": title,
            "confidence": TARGET_PRIOR_CONFIDENCE,
            "source_type": "target_prior",
            "source_skills": [],
            "derivation": raw.get("derivation", ""),
        })
        print(f"    [PRIOR {skill_id}] '{raw.get('title', '')[:60]}' conf={TARGET_PRIOR_CONFIDENCE:.2f}")

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
            "mechanism_check": raw.get("mechanism_check", ""),
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
        "prompt_template": ("per-item (role map + per-entry rewrite + code checks)"
                            if per_item else prompt_label),
        "detailed_scenes": detailed_scenes,
        "per_item": {k: v for k, v in per_item_log.items() if k != "items"} if per_item else False,
        "transfer_guard": guard_log if transfer_guard else False,
        "source_critic": critic_log if source_critic else False,
        "target_priors": prior_log if target_priors else False,
        "default_check": default_log if default_check else False,
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
        "--prompt-variant", choices=["auto", *sorted(PROMPT_VARIANTS)], default=None,
        help="Synthesis prompt template. auto (default): full 8-rule physics "
             "prompt for claude-* teachers, physics-slim otherwise — correct only "
             "for InterPhyre. physics/full and physics-slim/slim force one; "
             "full-neutral is the full structure with every hint example "
             "removed (no example mechanisms, placements or entities); "
             "text-game is the domain-neutral prompt for ORBIT-style tasks. "
             "Overrides --blocks-module's PROMPT_VARIANT.",
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
        "--dry-run", action="store_true",
        help="Build and dump the synthesis prompt, then stop without calling "
             "the teacher.",
    )
    parser.add_argument(
        "--per-item", action="store_true",
        help="Small-teacher mode: one role-mapping call, then one call per "
             "source entry (rewrite by swapping entities, keep relation words), "
             "with code checks for numbers and lost relation words (one retry, "
             "then drop). Replaces the single whole-bank call.",
    )
    parser.add_argument(
        "--detailed-scenes", action="store_true",
        help="Use the detailed factual scene descriptions (from the level "
             "definitions) for source and target instead of the short ones.",
    )
    parser.add_argument(
        "--structural-hint", default=None,
        help="Optional one-paragraph hint telling the teacher how the source "
             "and target levels are structurally related (e.g. shared "
             "mechanism) and what NOT to transfer. High-leverage for small "
             "teachers — prevents hallucinated mechanisms.",
    )
    parser.add_argument(
        "--transfer-guard", action="store_true",
        help="Forbid carrying source-calibrated numbers into the target bank, "
             "require a per-item mechanism check, and validate the output "
             "(one repair round, then offending items are dropped).",
    )
    parser.add_argument(
        "--source-critic", action="store_true",
        help="Rule 11 (corroboration = same reason; show seeds per source entry) "
             "plus a separate critic call per item that judges each cited "
             "source on its own stated reason; unsupported content is rewritten "
             "or dropped and confidence capped by accepted evidence.",
    )
    parser.add_argument(
        "--target-priors", action="store_true",
        help="Add 2-4 entries derived only from the shared environment and the "
             "factual target description (no source banks, no trajectories), "
             "labelled target_prior, checked by a separate call, no defaults.",
    )
    parser.add_argument(
        "--default-check", action="store_true",
        help="Test every kept transferred entry for a parameter default "
             "(rule 9); rewrite offenders once, re-check, drop the rest.",
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
        transfer_guard=args.transfer_guard,
        source_critic=args.source_critic,
        target_priors=args.target_priors,
        default_check=args.default_check,
        prompt_variant=prompt_variant,
        per_item=args.per_item,
        detailed_scenes=args.detailed_scenes,
        source_block_overrides=source_block_overrides,
        target_block_override=target_block_override,
        domain_block=domain_block,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    _main()
