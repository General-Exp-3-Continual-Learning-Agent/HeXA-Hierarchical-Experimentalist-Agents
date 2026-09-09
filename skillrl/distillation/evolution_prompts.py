"""Teacher prompts for skill bank evolution (v2 variant).

This module extends the original distillation approach. Instead of extracting
skills from trajectories alone, the teacher receives:
  1. The PREVIOUS skill bank (what we already know)
  2. NEW trajectories (what we just learned)

The teacher's job is to EVOLVE the bank:
  - Keep existing skills that are still valid
  - Add novel skills discovered in the new trajectories
  - Remove redundant or low-value skills
  - Respect a hard cap on total skills/mistakes
"""

from __future__ import annotations

from skillrl.core.skill_bank import Skill, Mistake, SkillBank


def format_skill_bank_for_teacher(bank: SkillBank, level_name: str) -> str:
    """Format existing skill bank for the teacher prompt.

    Returns a readable text block showing:
      - Level-specific skills with title, principle, confidence, when_to_apply
      - Level-specific mistakes with description and how_to_avoid
    """
    lines = []

    level_skills = bank.level_skills.get(level_name, [])
    if level_skills:
        lines.append("EXISTING LEVEL-SPECIFIC SKILLS:")
        for skill in level_skills:
            conf_pct = f"{skill.confidence * 100:.0f}%"
            # NOTE: the principle is shown in FULL. Truncating it (previously [:100])
            # hid the concrete numeric values, which live at the end of most
            # principles, so the teacher silently reinvented them each round and the
            # constants random-walked away from validated values.
            lines.append(f"  - [{conf_pct}] {skill.title}")
            lines.append(f"      principle: {skill.principle}")
            if skill.when_to_apply:
                lines.append(f"      apply when: {skill.when_to_apply}")
            if getattr(skill, "example", ""):
                lines.append(f"      example: {skill.example}")
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


SKILL_BANK_EVOLUTION_PROMPT = """\
You are a physics teacher evolving a skill bank for puzzle-solving agents.

LEVEL: {level_name}
LEVEL DESCRIPTION: {level_description}

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
   - Maximum {max_skills} total skills for this level
   - Maximum {max_mistakes} total mistakes for this level
3. For each skill you include:
   - If it's a RETAINED skill from the existing bank: set "is_new": false
   - If it's a NEW skill extracted from the new trajectories: set "is_new": true
   - Include "source_seeds" listing seed numbers where this skill was observed (required for confidence calibration)
   - Include "confidence": a float in [0.1, 1.0] representing your confidence in this skill
4. CONFIDENCE MOVES IN BOTH DIRECTIONS. For a retained skill:
   - If the agent FOLLOWED it and the episode SUCCEEDED: you may raise confidence slightly.
   - If the agent FOLLOWED it and the episode FAILED: you MUST lower its confidence
     (by at least 0.1). A skill the agent obeyed on the way to a failure is
     implicated by that failure, not neutral to it.
   - If it was not applicable this round: leave confidence unchanged.
   Never raise the confidence of a skill that appeared in a failing episode.
5. For new skills, estimate confidence based on:
   - Success rate among source trajectories (high success = high confidence)
   - Universality (applies to multiple seed conditions = higher confidence)
   - Clarity and actionability of the principle
   A skill extracted only from FAILED trajectories starts at confidence <= 0.4.
6. Do not include duplicate skills. If a new trajectory confirms an existing skill, keep the existing one (possibly with slightly higher confidence).
7. PRESERVE VALIDATED NUMERIC VALUES. Concrete constants (positions, radii, drop
   heights, thresholds) inside a retained skill are evidence, not prose. Copy them
   through EXACTLY. You may only change a number when a NEW trajectory in this round
   directly demonstrates a different value working better, and you must cite the seed
   that shows it. Do not widen, narrow, round, or re-derive a number you were not
   shown evidence against. If you weaken or drop a concrete value, name the seed that
   justified it in "reasoning".
8. Remove skills that are:
   - Redundant or subsumed by other skills.
   - Contradicted by the new trajectories
   - Too specific or rarely applicable
   - Low confidence (< 0.3) and not directly observed in new trajectories
   - Present in the retrieved set for MULTIPLE failed episodes while contributing no
     successes since being added.
9. DO NOT create skills that rationalise failure or that cannot be acted on. Reject
   any skill whose content is essentially "this seed may be unsolvable", "results
   vary per seed", "behaviour here is chaotic/unpredictable", or "calibrate per
   seed". These are unfalsifiable: they can never be contradicted by evidence, so
   they accumulate confidence forever and crowd out actionable skills. Every skill
   must tell the agent a SPECIFIC THING TO DO, with values.
10. DO NOT create skills that gate or restrict when the agent may commit an action
   (e.g. "only call simulate_action when within X units"). The agent has a bounded
   iteration budget; a gate that never opens burns the entire budget and guarantees
   failure. Prefer skills that say what to try next over skills that forbid acting.
11. Do not remove mistakes unless the new trajectories show they're no longer common.

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
  "confidence_changes": [
    {{"title": "<skill title>", "from": <old conf>, "to": <new conf>, "why": "<followed in failing seed N / confirmed by seed M>"}}
  ],
  "reasoning": "<brief explanation of key changes: what was removed, what was added, why. If any numeric constant changed, cite the seed that justified it.>"
}}

Be concise but precise. Focus on physics insights that directly help puzzle-solving.
"""
