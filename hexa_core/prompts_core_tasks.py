"""Task-neutral variants of HeXA's three teacher prompts, for CORE reasoning tasks.

HeXA's method is general; only its prompt wording is InterPhyre-specific. These
three templates are the neutral swap-ins, following the discipline of
``skillrl/distillation/teacher_prompts_catapult.py``:

* **same placeholder names** as the originals
  (:data:`~skillrl.distillation.teacher_prompts.CONTRASTIVE_DISTILLATION_PROMPT`,
  ``COMMON_MISTAKES_PROMPT``, and
  :data:`~skillrl.distillation.evolution_prompts.SKILL_BANK_EVOLUTION_PROMPT`), so
  ``distill_contrastive`` / ``distill_mistakes`` / ``evolve_skill_bank`` can format
  them without changing a line;
* **same output JSON schema**, so ``_parse_json_array`` / ``_parse_json_object``
  and every field read downstream (``source_seeds``, ``mistakes``,
  ``partial_skills``, ``confidence``, ``is_new``, ``removed_skill_titles``,
  ``reasoning``, ``example``) keep working untouched;
* Box2D wording replaced by the single-turn reasoning setting, iteration-count
  reward language replaced by the binary reward that
  :func:`hexa_core.hexa_bridge.binary_trajectory_reward` actually produces.

:data:`CORE_TASK_DESCRIPTIONS` fills the ``{level_description}`` slot. Each entry
describes the task and how it is scored, and deliberately leaks **no solution
strategy** — the same restraint the catapult variant applies, so that any strategy
the teacher banks is one it read out of the trajectories rather than out of the
prompt.
"""

from __future__ import annotations

# ── Task descriptions (facts and scoring rules only — no approach implied) ──
# Sources, all read from `core-reasoning/src/tasks/single_turn/`:
#   matchstick_task.py    TASK_INSTRUCTION l.24, OUTPUT_INSTRUCTION l.26,
#                         parse_answer l.190, evaluate l.196-204
#   mathgap_task.py       PROMPT_TEMPLATE_NO_GUIDELINES l.28 (the problem text and
#                         nothing else -- no task_instruction, and
#                         output_instruction is inherited as ""), parse_answer
#                         l.98-138, evaluate l.140-148, _prepare_data l.150-184
#   zebralogic_task.py    OUTPUT_INSTRUCTION l.148, _build_problem_prompt l.199,
#                         evaluate l.489-505
#   tower_of_hanoi_task.py STATE_TASK_INSTRUCTION l.50, evaluate l.703-714

CORE_TASK_DESCRIPTIONS: dict[str, str] = {
    "matchstick": (
        "Each problem shows one arithmetic equation written in Roman numerals with "
        "the operators +, - and =, where every numeral and operator is imagined to be "
        "built out of matchsticks (I is one vertical stick, X is two crossed diagonal "
        "sticks, = is two stacked horizontal sticks, and so on). The equation as "
        "printed is not a valid equality. The solver must move exactly one matchstick "
        "— relocating it, never adding or removing one — so that the equation becomes "
        "a valid Roman-numeral equality. The '=' may not be turned into '≠', and an "
        "angled stick is not interchangeable with a vertical stick. The answer is a "
        "single line, \"Correct version of the equation: <equation>\"; the scorer reads "
        "the last such line in the response and accepts it on an upper-cased, "
        "whitespace-collapsed exact match against a list of accepted equations, so a "
        "problem may have more than one acceptable answer."
    ),
    "mathgap": (
        "Each problem is an arithmetic word problem about how many of one object "
        "(oranges, bananas, ...) each of a cast of named people has, ending in a "
        "question about exactly one of them. The statements come in four forms: a "
        "plain count (\"Myles has 15 oranges\"), a comparison (\"Taylor has 5 more "
        "oranges than Adolfo\", sometimes compounded as \"10 more than X and 10 fewer "
        "than Y\"), a transfer (\"Robert gives 3 oranges to Myles\"), and a "
        "comparison-transfer that defines a new person indirectly (\"The number of "
        "oranges that Adolfo has more than Taylor is equal to the difference between "
        "the number of oranges that James and Myles have\"). Both sides of that last "
        "form are signed: it states (Adolfo - Taylor) = (James - Myles), keeping the "
        "order named on each side, and either side may be negative. Transfers take "
        "effect in the order stated, so a comparison that mentions a person refers to "
        "that person's count at that point in the text, not their starting count. "
        "Every quantity is a small non-negative integer. "
        "The statements are given in an order that need not match the order they have "
        "to be resolved in, and most problems are long: the majority state far more "
        "relationships than the question depends on, so chains that never reach the "
        "queried person are present by construction. The prompt carries no task "
        "instruction of any kind — it is the problem text plus one format line. The "
        "answer is \"The final answer is $\\boxed{#your answer}$.\"; the scorer takes "
        "whichever recognised answer pattern begins *latest* in the response, then "
        "accepts it on an upper-cased, whitespace-collapsed exact match against a "
        "single reference integer, so there is exactly one accepted answer and no "
        "partial credit."
    ),
    "zebralogic": (
        "Each problem is a constraint-satisfaction grid puzzle: N houses in a row "
        "numbered left to right, several attributes (name, drink, hobby, ...) each "
        "taking N distinct values, and a list of natural-language clues. Exactly one "
        "assignment of every value to a house satisfies all the clues, and the solver "
        "must produce it. This split is the 'xlarge' size group, the hardest of the "
        "four CORE defines: membership is by search-space size (houses factorial, "
        "raised to the number of attributes, at least 10^10), which here means 5 "
        "houses with 5 or 6 attributes, or 6 houses with 4, 5 or 6 attributes. The "
        "answer is a JSON object with "
        "top-level keys \"reasoning\" and \"solution\", where \"solution\" maps "
        "\"House k\" to that house's attribute-value pairs, filling in a blank schema "
        "printed in the prompt. Scoring canonicalises the whole grid and compares it "
        "to the reference: a single wrong cell fails the item, and there is no partial "
        "credit."
    ),
    "tower-of-hanoi": (
        "Each problem gives a three-peg Tower of Hanoi position: n disks numbered 1 "
        "(smallest) to n (largest) distributed across the pegs, each peg listed bottom "
        "to top, together with a goal configuration. Both the initial and the goal "
        "configuration are arbitrary legal positions, not necessarily the classical "
        "all-disks-on-one-peg form. Only one disk moves at a time, only the topmost "
        "disk of a peg may move, and a larger disk may never rest on a smaller one. "
        "The answer is \"moves = [[disk_id, from_peg, to_peg], ...]\" with 0-indexed "
        "pegs; the scorer reads the last parseable move list in the response, replays "
        "it from the initial configuration, and fails the item on any illegal move or "
        "if the final configuration differs from the goal. The move list does not have "
        "to be optimal. Training problems use 5 disks and evaluation problems use 6."
    ),
}


# ── Ablation: minimal task descriptions ────────────────────────────────
# `WEAKNESSES.md` §12 records the objection this exists to answer. CORE ships
# mathgap with **0 ch of task_instruction** (matchstick 1437, hanoi 1120,
# zebralogic 34 + schema), so for mathgap alone `CORE_TASK_DESCRIPTIONS` is the
# only source of task semantics anywhere in the system. A mathgap gain is
# therefore partly attributable to the description rather than to the learning
# loop, and the only way to size that is to run the loop again with the semantics
# removed.
#
# "Minimal" means: keep what the system needs to function at all -- what genre of
# problem this is, and the answer format -- and drop every statement about *what
# the problem text means*. Specifically removed relative to the full entry:
#   * the four statement forms (plain count / comparison / transfer /
#     comparison-transfer),
#   * the signed reading of the comparison-transfer form, which is the rule the
#     first run got wrong (`run0_signbug_ABANDONED`),
#   * the "transfers take effect in the order stated" rule,
#   * the note that most problems state far more relationships than the question
#     needs, so irrelevant chains are present by construction,
#   * the scorer's latest-match rule and the exact-match/no-partial-credit note.
#
# This is deliberately NOT the default. The env gate keeps every unablated run --
# including the three mathgap replicates and all of hanoi/zebralogic/matchstick --
# reading the byte-identical dict above.
CORE_TASK_DESCRIPTIONS_MINIMAL: dict[str, str] = {
    "mathgap": (
        "Each problem is an arithmetic word problem about how many of one object "
        "(oranges, bananas, ...) each of a cast of named people has, ending in a "
        "question about exactly one of them. The prompt carries no task instruction "
        "of any kind — it is the problem text plus one format line. The answer is a "
        "single integer, written as "
        "\"The final answer is $\\boxed{#your answer}$.\""
    ),
}


def _apply_minimal_description_override() -> tuple[str, ...]:
    """Swap in :data:`CORE_TASK_DESCRIPTIONS_MINIMAL` for tasks named in
    ``$HEXA_MINIMAL_DESC`` (comma-separated).

    Mutates :data:`CORE_TASK_DESCRIPTIONS` in place at import time so that every
    downstream reader -- including ``hexa_bridge``'s default-argument binding at
    module scope -- sees the same object. Returns the tasks actually overridden so
    the caller can stamp them into a run manifest; an ablation that is not recorded
    in provenance is indistinguishable from the real run six months later.

    Unknown task names raise rather than being ignored: a typo here would silently
    produce an unablated run wearing an ablation's output directory.
    """
    import os

    raw = os.environ.get("HEXA_MINIMAL_DESC", "").strip()
    if not raw:
        return ()
    names = tuple(n.strip() for n in raw.split(",") if n.strip())
    unknown = [n for n in names if n not in CORE_TASK_DESCRIPTIONS_MINIMAL]
    if unknown:
        raise ValueError(
            f"HEXA_MINIMAL_DESC names {unknown}, which have no entry in "
            f"CORE_TASK_DESCRIPTIONS_MINIMAL (available: "
            f"{sorted(CORE_TASK_DESCRIPTIONS_MINIMAL)})."
        )
    for name in names:
        CORE_TASK_DESCRIPTIONS[name] = CORE_TASK_DESCRIPTIONS_MINIMAL[name]
    return names


MINIMAL_DESCRIPTION_TASKS: tuple[str, ...] = _apply_minimal_description_override()


# ── Shared framing: what the actor can and cannot do ───────────────────
# Stated explicitly so the teacher never banks iterative-refinement advice ("try
# a placement, read the simulation, adjust") that a single-turn actor cannot act
# on, and never assumes a graded score that does not exist.

_CORE_SETTING = """\
**Setting**: a single-turn reasoning benchmark. For each problem the model receives \
one prompt and produces one response. There are no tools, no environment feedback \
and no retries: a deterministic program parses a final answer out of the response \
and compares it with the accepted answer(s), once. Anything useful — restating the \
constraints, enumerating cases, verifying the candidate answer — has to happen \
inside that single response.

**Reward** is printed in each attempt's header and is **baseline-relative**: it \
measures how much better the attempt did than this same model does on this same \
problem with no help at all.

- A success on a problem the model rarely solves unaided scores near +1.00. That is \
the informative kind: something in that attempt overcame a genuine difficulty.
- A success on a problem it almost always solves unaided scores near 0.00. It \
demonstrates nothing that needs a skill.
- A failure scores negatively, most negatively when the problem was easy.
- A reward of exactly +1.00 or -1.00 means no baseline was measured for that problem \
and the bare verdict is all the evidence there is.

So attempts are **not** equally informative, and a skill drawn from high-reward \
attempts is worth more than one drawn from attempts that merely went well. Weigh by \
the reward, not by verdict alone and not by length."""

_ANTI_MEMORISATION = """\
Write every skill as guidance for a FRESH, unseen problem of the same task: a \
reusable method, representation, ordering, or self-check. Never state a fact about \
one specific problem instance (a particular equation, a particular clue set, a \
particular move list) — the bank is injected into the prompts of *other* problems, \
where an instance-specific fact is useless at best and misleading at worst."""

_SOURCE_SEEDS = """\
- **source_seeds**: the problem identifiers of the attempts above that this skill \
was primarily derived from. Copy them verbatim (they are strings) from the \
`--- Problem <id>` headers; do not invent, renumber or abbreviate them. Include \
only problems whose behaviour directly demonstrates or motivates the skill."""


# ── Phase 1a: contrastive skill distillation ───────────────────────────

CONTRASTIVE_DISTILLATION_PROMPT_CORE = """\
You are an expert analyst of reasoning traces, distilling model behaviour into concise, actionable skills.

""" + _CORE_SETTING + """

**Task: {level_name}**
{level_description}

Below are ACCEPTED and REJECTED attempts at problems from this task. Each shows the model's reasoning (Thought), the answer it submitted (Action), the verifier's verdict (Observation), and the **Reward**.

=== ACCEPTED ATTEMPTS ({n_successes}) ===
{success_block}

=== REJECTED ATTEMPTS ({n_failures}) ===
{failure_block}

---

Your task: by CONTRASTING the accepted and the rejected attempts, extract the KEY SOLUTION-STRATEGY SKILLS that separate solving this task from failing it. What did the accepted attempts do — in how they represented the problem, ordered their work, pruned candidates, or checked themselves — that the rejected attempts did not? Let the attempts, not your prior expectations about the task, determine what works.

""" + _ANTI_MEMORISATION + """

For each skill, provide:
- **title**: Short name (3-7 words)
- **principle**: The reasoning insight (2-3 sentences). What was done, and why does it work?
- **when_to_apply**: Specific trigger condition (1 sentence)
""" + _SOURCE_SEEDS + """

Output a JSON array:
```json
[
  {{
    "title": "...",
    "principle": "...",
    "when_to_apply": "...",
    "source_seeds": ["<problem id>", "<problem id>"]
  }}
]
```

Extract 3-4 skills. Each should capture a DISTINCT insight from the accepted/rejected contrast. Avoid redundancy.
"""


# ── Phase 1b: common mistakes + partial skills ─────────────────────────

COMMON_MISTAKES_PROMPT_CORE = """\
You are an expert at analysing model failures and distilling them into avoidable mistake patterns.

""" + _CORE_SETTING + """

**Task: {level_name}**
{level_description}

Below are REJECTED attempts. Each shows the model's reasoning, the answer it submitted, and the verifier's verdict.

{failure_block}

---

Your task has TWO parts:

**Part 1 — Mistakes**: identify the COMMON MISTAKE PATTERNS across these failures. For each mistake, analyse:
1. What exactly the model got wrong — distinguish a reasoning error from a bookkeeping slip, from a misread constraint, from a correct solution reported in the wrong output format, from a response that ran out of room before committing to an answer.
2. WHY the model made this error (what mistaken belief or shortcut led to it)
3. A concrete actionable fix

**Part 2 — Partial insights**: even in rejected attempts, individual steps often show CORRECT reasoning or a useful move (for example the model derived a sound intermediate deduction and then contradicted it, or set up the problem well and then made an arithmetic slip). Extract 1-2 skills from these "good steps within bad attempts". They must be genuine reusable insights, not restatements of what went wrong.

""" + _ANTI_MEMORISATION + """

Format as a JSON object with two arrays:
```json
{{
  "mistakes": [
    {{
      "description": "What the mistake is (1 sentence)",
      "why_it_happens": "The mistaken belief or reasoning error that causes this (1 sentence)",
      "how_to_avoid": "Concrete actionable fix — what to do instead (1-2 sentences)"
    }}
  ],
  "partial_skills": [
    {{
      "title": "Short name (3-7 words)",
      "principle": "The reusable insight from the rejected attempt (2-3 sentences)",
      "when_to_apply": "Specific trigger condition (1 sentence)",
      "source_seeds": ["<problem id>", "<problem id>"]
    }}
  ]
}}
```

Extract 2-3 mistakes and 1-2 partial skills. For mistakes, group similar failures into one and focus on ROOT CAUSES. For partial skills, only extract genuinely useful insights — do not force it if no good steps exist.
"""


# ── Phase 2: skill bank evolution (round 2+) ───────────────────────────

#: Opt-in addendum, OFF by default. Injected in place of the ``{truncation_guidance}``
#: marker by :func:`evolution_prompt`.
#:
#: Off by default because every banked result -- ZebraLogic +0.247, Hanoi +0.257, both
#: pooled over 3 runs -- was produced *without* it, and the ablation programme needs the
#: default configuration to stay byte-identical to what those numbers came from. Turning
#: this on is a deliberate arm, not an improvement to the baseline.
#:
#: What it addresses (`WEAKNESSES.md` §11): when the evolver's failure set is mostly
#: attempts cut off by the generation limit, there is no reasoning error to contrast, so
#: it writes plausible-sounding verification checks -- which *add* per-candidate work and
#: make truncation worse. Measured on Matchstick: a bank of seven such restatements
#: nearly doubled truncation (29% -> 56% on identical problems) and cost 0.27 accuracy.
#: The guidance is written to be self-limiting ("do not invent this if truncation is
#: rare"), so it is safe to try on tasks where truncation is not the binding constraint.
TRUNCATION_AWARE_GUIDANCE = """\
**The task statement is not a source of skills.** Constraints the solver was already
told ("move exactly one X", "the result must be valid", "recompute and check the
arithmetic", "do not reorder the items") are already in its prompt. Restating one as a
skill costs context twice over: once for the tokens, and again because a restated rule
reads as an *obligation to verify on every candidate*, which multiplies the work of a
search that may already be running out of budget. If a skill's content could be derived
from the problem statement alone, it is not a skill.

SECOND, before writing any skill, check HOW the rejected attempts failed. Trajectory
headers mark attempts cut off by the generation limit ("truncated at the token limit").
Count them:
- **If truncation is a large share of the failures, the bank's job is to make the solver
  reach an answer in FEWER tokens.** A truncated attempt did not reason wrongly; it
  reasoned too long. Skills that add checks, verification passes, or exhaustive
  enumeration make this failure mode *worse*, and they are the most tempting thing to
  write because a truncated trace looks like sloppy reasoning.
- In that case bank at least one skill that shortens the search: how to prune the
  candidate space, how to commit once a candidate provably satisfies the goal instead of
  enumerating the rest, or simply to keep the reasoning within budget. A skill of this
  kind is worth more than a third constraint check.
- Do not invent this if truncation is rare in the attempts above. Look, then decide.

"""

#: The marker :func:`evolution_prompt` substitutes. A plain ``str.replace`` is used, not
#: ``str.format``: the template still holds the evolver's own ``{level_name}`` fields and
#: ``{{``-escaped JSON braces, and formatting it early would consume both.
_TRUNCATION_MARKER = "{truncation_guidance}"

SKILL_BANK_EVOLUTION_PROMPT_CORE_TEMPLATE = """\
You are a teacher curating a strategy bank for a model solving reasoning problems.

""" + _CORE_SETTING + """

**Task: {level_name}**
{level_description}

CURRENT SKILL BANK (from previous rounds):
{existing_skills_block}

NEW ATTEMPTS (from the latest round):
Accepted: {n_successes}
Rejected: {n_failures}

{new_trajectories_block}

YOUR TASK: evolve the bank using the evidence in the new attempts.

FIRST, before writing any skill, work out the CONTRAST:
- What did the higher-reward attempts do that the lower-reward ones did not?
- Look at the reasoning itself, not just the verdicts: where does a rejected attempt
  first go wrong, and what did an accepted attempt do at that same point instead?
- **If a behaviour appears in both accepted and rejected attempts, it does not explain
  the outcome.** Do not turn it into a skill, however sensible it sounds.

A skill earns its place only if it names something the attempts revealed. The bank is
NOT a summary of good practice for this task: a competent solver already knows the
rules and the standard method, and repeating those back costs context and teaches
nothing. Before including a skill, ask: "would a capable solver have said this before
seeing a single attempt?" If yes, drop it.

{truncation_guidance}RULES:
1. Output the COMPLETE FINAL bank (not a diff) — include both retained existing skills and any new ones.
2. Hard constraints:
   - Maximum {max_skills} total skills for this task.
   - Maximum {max_mistakes} total mistakes for this task.
3. For each skill you include:
   - If it is a RETAINED skill from the existing bank: set "is_new": false and keep its title and wording faithful (you may lightly tighten it).
   - If it is a NEW skill extracted from the new attempts: set "is_new": true.
   - Put a COUNTEREXAMPLE in "example", not a worked illustration: the failure mode is
     the reusable part, and quoting a mistake cannot leak a correct answer into the
     bank. (Two banks so far have carried a verbatim training answer in this field.)
   - Include "source_seeds" listing the problem identifiers (strings) where the skill was observed, copied verbatim from the `--- Problem <id>` headers above; this drives confidence calibration for new skills, and an identifier that does not appear above is silently ignored.
   - Include "confidence": a float in [0.1, 1.0]. For RETAINED skills this is the
     value that is kept. For NEW skills it is recomputed from the rewards of the
     attempts in "source_seeds", so supply your best estimate but do not agonise.
   - The "principle" must state the concrete error it prevents, in the form
     "<method>; without this, <the specific mistake seen in the rejected attempts>".
     If you cannot name an error from the attempts above, the skill is generic — drop it.
4. For retained skills, preserve their existing confidence values (they have already been validated).
5. For new skills, estimate confidence from:
   - The accept rate among the attempts the skill was drawn from.
   - Generality — a skill that applies across several different problems of this task is worth more than one that fits a single problem.
   - Clarity and actionability of the principle.
6. Do not include duplicate skills. If a new attempt confirms an existing skill, keep the existing one (optionally raising its confidence slightly).
6b. The bank is usually already at capacity, so adding means displacing. To add a new
   skill, name in "reasoning" the existing skill it displaces and why the evidence
   favours the newcomer. **If no existing skill is weaker than the candidate, add
   nothing.** Replacing a skill that is working, with one that merely sounds better,
   makes the bank worse.
7. Remove skills that are:
   - Redundant or subsumed by another skill.
   - Contradicted by the new attempts.
   - Too instance-specific to reuse on an unseen problem.
   - Low confidence (< 0.3) AND not observed in the new attempts.
8. Do not remove mistakes unless the new attempts show they are no longer common.

""" + _ANTI_MEMORISATION + """

OUTPUT JSON OBJECT:
{{
  "skills": [
    {{
      "title": "<short name of skill>",
      "principle": "<2-3 sentence reasoning insight>",
      "when_to_apply": "<condition for applicability>",
      "example": "<the WRONG pattern seen in a rejected attempt, stated generically -- NOT a worked answer>",
      "source_seeds": ["<problem id>", "<problem id>"],
      "confidence": <float in [0.1, 1.0]>,
      "is_new": <true|false>
    }}
  ],
  "mistakes": [
    {{
      "description": "<what the mistake is>",
      "why_it_happens": "<why models make this mistake>",
      "how_to_avoid": "<actionable fix>",
      "is_new": <true|false>
    }}
  ],
  "removed_skill_titles": ["<title of removed skill 1>", ...],
  "reasoning": "<brief explanation of key changes: what was removed, what was added, why>"
}}

Be concise but precise. Focus on transferable reasoning strategies grounded in the attempts.
"""


#: The default-configuration prompt: the template with the marker removed and nothing
#: else moved, so it is byte-identical to the prompt every banked result was produced
#: with. A module constant rather than a call, because `hexa_bridge`'s seam test asserts
#: the *identity* (``is``) of the bound prompt against this name -- see
#: :func:`evolution_prompt`, which returns this exact object on the default path.
SKILL_BANK_EVOLUTION_PROMPT_CORE = SKILL_BANK_EVOLUTION_PROMPT_CORE_TEMPLATE.replace(
    _TRUNCATION_MARKER, ""
)

#: The truncation-aware variant, also built once. Same reasoning: `patch` binds one of
#: these two objects, so re-patching cannot leave two equal-but-distinct prompts around.
SKILL_BANK_EVOLUTION_PROMPT_CORE_TRUNCATION_AWARE = (
    SKILL_BANK_EVOLUTION_PROMPT_CORE_TEMPLATE.replace(
        _TRUNCATION_MARKER, TRUNCATION_AWARE_GUIDANCE
    )
)


def evolution_prompt(*, truncation_aware: bool = False) -> str:
    """The evolution prompt, with the truncation addendum in or out.

    Returns one of two pre-built module constants, never a fresh string: the seam test
    asserts ``evolving_distill.SKILL_BANK_EVOLUTION_PROMPT is
    SKILL_BANK_EVOLUTION_PROMPT_CORE`` after a default `patch`, and that invariant is
    worth keeping -- it is what proves the default path binds *the* prompt rather than
    something merely equal to it.
    """
    return (
        SKILL_BANK_EVOLUTION_PROMPT_CORE_TRUNCATION_AWARE
        if truncation_aware
        else SKILL_BANK_EVOLUTION_PROMPT_CORE
    )
