"""Distill + evolve passes for the **skills-only** HeXA ablation.

Two ways this differs from the standard pipeline in `distill.py` / `evolving_distill.py`:

1. Inputs are filtered to **successful trajectories only** — failures are
   dropped before any teacher call.
2. The teacher's output schema contains **only `skills`** — there is no
   `mistakes` block and the returned SkillBank therefore has zero Mistake
   entries for the level.

Public entry points:
- `run_skills_only_distillation(...)` — round-1 seeding (success-only).
- `evolve_skill_bank_skills_only(...)`  — rounds-2+ evolution (success-only,
  no mistakes in the output).

The round-1 confidence calibration reuses `_compute_initial_confidence` from
`distill.py` so the confidence scale stays consistent with HeXA.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from skillrl.core.config import LEVEL_DESCRIPTIONS, TEACHER_MODEL
from skillrl.core.skill_bank import Skill, SkillBank
from skillrl.distillation.distill import (
    _compute_initial_confidence,
    _parse_json_array,
    _parse_json_object,
    call_teacher,
    load_trajectories,
)
from skillrl.distillation.teacher_prompts import format_trajectories_block
from skillrl.distillation.skills_only_prompts import (
    SUCCESS_ONLY_DISTILLATION_PROMPT,
    SKILL_BANK_EVOLUTION_SKILLS_ONLY_PROMPT,
    format_skill_bank_for_teacher_skills_only,
)


# ─────────────────────────────────────────────────────────────────────────
# Round 1: seed the bank from initial successful trajectories
# ─────────────────────────────────────────────────────────────────────────

def run_skills_only_distillation(
    level_name: str,
    initial_traj_dir: Path,
    output_path: Path,
    teacher_model: str = TEACHER_MODEL,
    max_success_trajs: int = 5,
) -> SkillBank:
    """Distill an initial SkillBank from the SUCCESSFUL trajectories only.

    Failures in `initial_traj_dir` are loaded for reporting but never sent to
    the teacher. The resulting bank has no Mistake entries for the level.
    """
    successes, failures = load_trajectories(initial_traj_dir, level_name)
    print(
        f"\n  Skills-only distillation: {len(successes)} successes available "
        f"({len(failures)} failures dropped — not sent to teacher)"
    )

    bank = SkillBank()
    if not successes:
        print("  [Warning] No successful trajectories — emitting empty bank.")
        bank.save(output_path)
        return bank

    level_desc = LEVEL_DESCRIPTIONS.get(level_name, "No description available.")
    success_block = format_trajectories_block(successes, max_trajs=max_success_trajs)

    prompt = SUCCESS_ONLY_DISTILLATION_PROMPT.format(
        level_name=level_name,
        level_description=level_desc,
        n_successes=len(successes),
        success_block=success_block,
    )

    print(f"  Calling teacher ({teacher_model}) for success-only distillation...")
    response = call_teacher(prompt, model=teacher_model)
    raw_skills = _parse_json_array(response)

    if not raw_skills:
        print("  [Warning] Teacher returned no skills — emitting empty bank.")
        bank.save(output_path)
        return bank

    prefix = level_name[:3]
    for i, raw in enumerate(raw_skills):
        seeds = raw.get("source_seeds", [])
        if not seeds:
            seeds = [t.get("seed", -1) for t in successes[:3]]
        # Confidence comes from the rewards of the cited source successes.
        confidence = _compute_initial_confidence(seeds, successes, fallback_confidence=0.6)

        bank.add_skill(Skill(
            skill_id=f"{prefix}_sk_{i:03d}",
            title=raw.get("title", ""),
            principle=raw.get("principle", ""),
            when_to_apply=raw.get("when_to_apply", ""),
            example=raw.get("example", ""),
            source_level=level_name,
            source_type="skills_only_distilled",
            source_seeds=seeds,
            generation=0,
            confidence=confidence,
        ))
        print(f"    Skill '{raw.get('title', '')}': seeds={seeds} → conf={confidence:.3f}")

    bank.save(output_path)
    print(f"\n  Skill bank seeded: {output_path}")
    print(f"  {bank}")
    return bank


# ─────────────────────────────────────────────────────────────────────────
# Rounds 2+: evolve the bank with new successful trajectories
# ─────────────────────────────────────────────────────────────────────────

def evolve_skill_bank_skills_only(
    level_name: str,
    prev_bank: SkillBank,
    new_trajs_dir: Path,
    output_path: Path,
    max_skills: int = 10,
    teacher_model: str = TEACHER_MODEL,
) -> SkillBank:
    """Evolve the bank using prev_bank + new SUCCESSFUL trajectories only.

    The resulting bank has zero Mistake entries for the level (any pre-existing
    mistakes in `prev_bank` for `level_name` are dropped — they would not
    survive a true skills-only ablation anyway).
    """
    successes, failures = load_trajectories(new_trajs_dir, level_name)
    print(
        f"\n  Skills-only evolve: {len(successes)} successes available "
        f"({len(failures)} failures dropped — not sent to teacher)"
    )

    if not successes:
        # No new evidence — carry the previous (skills-only) bank forward.
        new_bank = _strip_mistakes_for_level(prev_bank, level_name)
        new_bank.save(output_path)
        print(f"  No new successes — saved previous bank (mistakes stripped) to {output_path}")
        return new_bank

    existing_skills_block = format_skill_bank_for_teacher_skills_only(prev_bank, level_name)
    new_trajs_block = format_trajectories_block(successes, max_trajs=5)
    level_desc = LEVEL_DESCRIPTIONS.get(level_name, "No description available.")

    prompt = SKILL_BANK_EVOLUTION_SKILLS_ONLY_PROMPT.format(
        level_name=level_name,
        level_description=level_desc,
        existing_skills_block=existing_skills_block,
        n_successes=len(successes),
        new_trajectories_block=new_trajs_block,
        max_skills=max_skills,
    )

    print(f"  Calling teacher ({teacher_model}) for skills-only evolution...")
    response = call_teacher(prompt, model=teacher_model)
    parsed = _parse_json_object(response)

    if not parsed:
        print("  [Warning] Teacher response could not be parsed. Carrying previous bank forward.")
        new_bank = _strip_mistakes_for_level(prev_bank, level_name)
        new_bank.save(output_path)
        return new_bank

    # Previous generation, by inspecting only skills for the target level
    prev_gen = 0
    for skill in prev_bank._all_skills():
        if skill.source_level == level_name and skill.generation > prev_gen:
            prev_gen = skill.generation

    # Build the new bank.
    new_bank = SkillBank()
    # Carry over general skills + non-target level skills (untouched).
    for skill in prev_bank.general_skills:
        new_bank.add_skill(skill)
    for lvl, skills in prev_bank.level_skills.items():
        if lvl != level_name:
            for skill in skills:
                new_bank.add_skill(skill)
    # ABLATION INVARIANT: do NOT carry over any mistakes for the target level.
    # For other levels and common-mistakes, we leave them unchanged.
    for mistake in prev_bank.common_mistakes:
        new_bank.add_mistake(mistake)
    for lvl, mistakes in prev_bank.level_mistakes.items():
        if lvl != level_name:
            for mistake in mistakes:
                new_bank.add_mistake(mistake)

    # Look up previous skills (target level) by title so we can reuse skill_ids
    prev_skills_by_title = {
        s.title: s for s in prev_bank.level_skills.get(level_name, [])
    }

    raw_skills = parsed.get("skills", [])
    for i, raw in enumerate(raw_skills):
        is_new = bool(raw.get("is_new", False))
        seeds = raw.get("source_seeds", [])

        if is_new:
            # Confidence calibrated from the new successes' rewards.
            confidence = _compute_initial_confidence(seeds, successes, fallback_confidence=0.6)
            generation = prev_gen + 1
            prefix = level_name[:3]
            skill_id = f"{prefix}_so_{generation}_{i:03d}"
        else:
            # Retained skill: preserve its confidence and skill_id where possible.
            confidence = float(raw.get("confidence", 0.5))
            confidence = max(0.1, min(1.0, confidence))
            generation = prev_gen
            original = prev_skills_by_title.get(raw.get("title", ""))
            if original is not None:
                skill_id = original.skill_id
                # Keep the original generation if older — only refresh if new bank says so.
                generation = original.generation
            else:
                prefix = level_name[:3]
                skill_id = f"{prefix}_so_{generation}_{i:03d}"

        new_bank.add_skill(Skill(
            skill_id=skill_id,
            title=raw.get("title", ""),
            principle=raw.get("principle", ""),
            when_to_apply=raw.get("when_to_apply", ""),
            example=raw.get("example", ""),
            source_level=level_name,
            source_type="skills_only_evolved",
            source_seeds=seeds,
            generation=generation,
            confidence=confidence,
        ))
        tag = "NEW" if is_new else "KEPT"
        print(f"    [{tag}] {raw.get('title', '')}: conf={confidence:.3f}, gen={generation}")

    removed = parsed.get("removed_skill_titles", []) or []
    if removed:
        print(f"\n  Removed: {removed}")
    reasoning = parsed.get("reasoning", "")
    if reasoning:
        print(f"  Teacher reasoning: {reasoning}")

    new_bank.save(output_path)
    print(f"\n  Skills-only evolved bank saved: {output_path}")
    print(f"  {new_bank}")
    return new_bank


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _strip_mistakes_for_level(bank: SkillBank, level_name: str) -> SkillBank:
    """Return a copy of `bank` with all level_mistakes for `level_name` removed.

    Used when there's nothing new to evolve from — we still want to enforce
    the skills-only invariant on the carry-forward bank.
    """
    out = SkillBank()
    for s in bank.general_skills:
        out.add_skill(s)
    for lvl, skills in bank.level_skills.items():
        for s in skills:
            out.add_skill(s)
    for m in bank.common_mistakes:
        out.add_mistake(m)
    for lvl, mistakes in bank.level_mistakes.items():
        if lvl == level_name:
            continue
        for m in mistakes:
            out.add_mistake(m)
    return out
