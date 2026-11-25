"""Slim evolution prompt for small open-weight teachers (Qwen-2.5-3B).

The default ``SKILL_BANK_EVOLUTION_PROMPT`` is ~8.2k characters (~2k tokens) of
instructions, and it is sent on EVERY round from round 2 on, on top of the bank
and the trajectory block. A 3B teacher reliably loses the task in that much
preamble: it returns flat confidences, stops merging, and emits near-duplicate
skills until the cap.

This module keeps the SEMANTICS of evolution and cuts the prose:

  kept  : complete-bank output, hard caps, is_new / source_seeds, confidence
          moving in BOTH directions, numeric constants copied through,
          duplicate and dead-skill removal, mistakes held to the skill bar,
          mistakes contrastive rather than failure-only, self-consistency.
  cut   : worked bad/good examples, the per-rule rationales, and the optional
          audit fields (``confidence_changes``, ``reasoning``, and the
          invariant/citation extras), all of which ``evolving_distill`` reads
          with ``.get(..., default)`` and none of which are required.
          ``removed_skill_titles`` is KEPT -- the loop prints it per round.

Only the EVOLUTION prompt is replaced. The two round-1 distillation prompts are
~470 tokens each and fire once, so they are left alone.

Placeholders are identical to the default template, so it is a drop-in rebind.

Usage mirrors ``teacher_prompts_catapult``::

    import skillrl.distillation.teacher_prompts_qwen_slim as tpq
    tpq.patch()

``evolving_loop`` calls ``patch()`` automatically when the teacher model looks
like a 3B Qwen; see ``is_small_qwen_teacher``.
"""

from __future__ import annotations

import re

# Teachers this variant is auto-selected for: Qwen at the 3B scale.
_SMALL_QWEN_RE = re.compile(r"qwen.*?(?<![\d.])3b", re.IGNORECASE)


def is_small_qwen_teacher(model: str | None) -> bool:
    """True when `model` names a ~3B Qwen, e.g. ``openai:Qwen/Qwen2.5-3B-Instruct``."""
    return bool(model) and bool(_SMALL_QWEN_RE.search(model))


SKILL_BANK_EVOLUTION_PROMPT_QWEN_SLIM = """\
You are a physics teacher maintaining a skill bank for a puzzle-solving agent.

LEVEL: {level_name}
LEVEL DESCRIPTION: {level_description}

CURRENT SKILL BANK:
{existing_skills_block}

NEW TRAJECTORIES (latest round) — successes: {n_successes}, failures: {n_failures}

{new_trajectories_block}

TASK: return the COMPLETE new bank (not a diff), merging what the bank already
knows with what these trajectories show. At most {max_skills} skills and
{max_mistakes} mistakes. Returning fewer is correct when the rest would be vague.

SKILLS
- Each skill names a SPECIFIC action with values. Drop "unsolvable", "results
  vary", "calibrate per seed", and anything that gates when the agent may act.
- Copy concrete numbers through EXACTLY. Change one only when a trajectory THIS
  round shows a different value working, and cite that seed.
- Nothing about legality: limits, overlap, margins, or checking before
  simulating. The simulator already rejects and corrects those, so such a skill
  spends turns instead of solving.
- One fact, one skill. No two skills share an "example" or rest on the same
  seed, and no paraphrases. Returning fewer is correct.
- Remove skills that are contradicted, below 0.3 confidence and unobserved this
  round, or present in several failed episodes with no success.

CONFIDENCE (moves both ways)
- Followed and the episode SUCCEEDED: may rise slightly.
- Followed and the episode FAILED: MUST fall by at least 0.1.
- Not applicable this round: unchanged.
- Never raise a skill that appeared in a failing episode.
- A new skill drawn only from failures starts at 0.4 or below.

MISTAKES (same bar as skills; delete freely)
- Write each one by contrast: say what the SUCCESSFUL trajectories did
  differently at the same decision point, and put that in "how_to_avoid" with
  values. If no contrasting success exists, drop the mistake.
- "how_to_avoid" must be self-contained — never "apply the offset principle".
- Drop mistakes about action VALIDITY (out of bounds, overlap, malformed): the
  environment already reports those.
- Do not anchor every mistake on the same object. If the successes acted on a
  different object or order, say so.

CONSISTENCY
Before returning, check entries against each other. If two give opposing
guidance on the same quantity, keep the one these trajectories support and
delete or rewrite the other.

OUTPUT (JSON object only):
{{
  "skills": [
    {{"title": "...", "principle": "...", "when_to_apply": "...",
      "example": "...", "source_seeds": [<ints>],
      "confidence": <0.1-1.0>, "is_new": <true|false>}}
  ],
  "mistakes": [
    {{"description": "...", "why_it_happens": "...", "how_to_avoid": "...",
      "is_new": <true|false>}}
  ],
  "removed_skill_titles": ["..."]
}}
"""


def patch() -> None:
    """Rebind the evolution prompt to the slim variant.

    Call BEFORE evolve_skill_bank(). Patches the module-level binding in place,
    which is safe because the function resolves the name at call time.

    Composes with ``teacher_prompts_catapult.patch()``: that one sets the two
    round-1 distillation prompts plus the evolution prompt, so call it FIRST and
    this second to keep its factual scene block on round 1 while the slim
    template drives rounds 2+.
    """
    import skillrl.distillation.evolving_distill as _evolving_distill

    _evolving_distill.SKILL_BANK_EVOLUTION_PROMPT = SKILL_BANK_EVOLUTION_PROMPT_QWEN_SLIM
