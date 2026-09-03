"""Phase 2: Evolve skill bank using previous bank + new trajectories.

This is the v2 variant of skill distillation. Instead of extracting skills
from trajectories alone, the teacher receives:
  1. Previous skill bank (what we know)
  2. New trajectories (what we just observed)

The teacher outputs an evolved skill bank with proper balancing of old/new content.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from skillrl.core.config import LEVEL_DESCRIPTIONS, TEACHER_MODEL
from skillrl.distillation.distill import (
    _compute_initial_confidence,
    _parse_json_object,
    call_teacher,
    load_trajectories,
)
from skillrl.distillation.evolution_prompts import (
    SKILL_BANK_EVOLUTION_PROMPT,
    format_skill_bank_for_teacher,
)
from skillrl.core.skill_bank import Skill, Mistake, SkillBank
from skillrl.distillation.teacher_prompts import format_trajectories_block


def evolve_skill_bank(
    level_name: str,
    prev_bank: SkillBank,
    new_trajs_dir: Path,
    output_path: Path,
    max_skills: int = 10,
    max_mistakes: int = 5,
    teacher_model: str = TEACHER_MODEL,
) -> SkillBank:
    """Evolve a skill bank using previous bank + new trajectories.

    Parameters
    ----------
    level_name : Level being trained.
    prev_bank : Previous SkillBank to evolve from.
    new_trajs_dir : Directory with new trajectory JSON files.
    output_path : Where to save the evolved bank.
    max_skills : Maximum total skills for this level.
    max_mistakes : Maximum total mistakes for this level.
    teacher_model : Teacher model to use (default: claude-sonnet-4-6).

    Returns
    -------
    SkillBank
        The evolved bank with proper generation tracking and confidence values.
    """
    # Load new trajectories
    successes, failures = load_trajectories(new_trajs_dir, level_name)
    print(f"\n  Evolving skill bank from {len(successes)} successes, {len(failures)} failures")

    # Format existing bank for teacher
    existing_skills_block = format_skill_bank_for_teacher(prev_bank, level_name)

    # Format new trajectories (reduced to 5 to avoid token limits)
    new_trajs_block = format_trajectories_block(successes + failures, max_trajs=5)

    # Level description
    level_desc = LEVEL_DESCRIPTIONS.get(level_name, "No description available.")

    # Build evolution prompt
    prompt = SKILL_BANK_EVOLUTION_PROMPT.format(
        level_name=level_name,
        level_description=level_desc,
        existing_skills_block=existing_skills_block,
        n_successes=len(successes),
        n_failures=len(failures),
        new_trajectories_block=new_trajs_block,
        max_skills=max_skills,
        max_mistakes=max_mistakes,
    )

    print(f"  Calling teacher to evolve skill bank...")
    response = call_teacher(prompt, model=teacher_model)
    parsed = _parse_json_object(response)

    if not parsed:
        print(f"  [Warning] Teacher response could not be parsed. Returning previous bank unchanged.")
        prev_bank.save(output_path)
        print(f"  Saved previous bank (unchanged) to: {output_path}")
        return prev_bank

    # Extract previous bank's generation for reference
    prev_gen = 0
    for skill in prev_bank._all_skills():
        if skill.generation > prev_gen:
            prev_gen = skill.generation

    # Get existing skills and mistakes by ID for quick lookup
    prev_skill_ids = {skill.skill_id: skill for skill in prev_bank._all_skills() if skill.source_level == level_name}
    prev_mistake_ids = {mistake.mistake_id: mistake for mistake in prev_bank.level_mistakes.get(level_name, [])}

    # Build new bank
    new_bank = SkillBank()

    # Copy general skills unchanged (evolution is level-specific)
    for skill in prev_bank.general_skills:
        new_bank.add_skill(skill)

    # Copy non-target level skills unchanged
    for lvl, skills in prev_bank.level_skills.items():
        if lvl != level_name:
            for skill in skills:
                new_bank.add_skill(skill)
    for mistake in prev_bank.common_mistakes:
        new_bank.add_mistake(mistake)
    for lvl, mistakes in prev_bank.level_mistakes.items():
        if lvl != level_name:
            for mistake in mistakes:
                new_bank.add_mistake(mistake)

    # Process evolved skills
    all_trajs = successes + failures
    raw_skills = parsed.get("skills", [])
    added_skills = 0
    removed_skills = parsed.get("removed_skill_titles", [])

    for i, raw in enumerate(raw_skills):
        is_new = raw.get("is_new", False)
        skill_seeds = raw.get("source_seeds", [])

        # Compute or preserve confidence
        if is_new:
            # New skill: compute confidence from source trajectories
            confidence = _compute_initial_confidence(skill_seeds, all_trajs, fallback_confidence=0.5)
            generation = prev_gen + 1
        else:
            # Retained skill: use teacher's confidence value (mirrors old bank)
            confidence = raw.get("confidence", 0.5)
            generation = prev_gen  # Retained skills keep their generation
            # Clamp just in case
            confidence = max(0.1, min(1.0, confidence))

        # Assign skill ID
        # New skills get a fresh ID, retained skills keep their ID
        if is_new:
            prefix = level_name[:3]
            skill_id = f"{prefix}_ev_{generation}_{i:03d}"
        else:
            # Try to find the original ID by matching title
            original_skill = None
            for old_skill in prev_skill_ids.values():
                if old_skill.title == raw.get("title", ""):
                    original_skill = old_skill
                    skill_id = old_skill.skill_id
                    break
            if not original_skill:
                # Fallback: generate new ID
                prefix = level_name[:3]
                skill_id = f"{prefix}_ev_{generation}_{i:03d}"

        skill = Skill(
            skill_id=skill_id,
            title=raw.get("title", ""),
            principle=raw.get("principle", ""),
            when_to_apply=raw.get("when_to_apply", ""),
            example=raw.get("example", ""),
            source_level=level_name,
            source_type="evolved",
            source_seeds=skill_seeds,
            generation=generation,
            confidence=confidence,
        )
        new_bank.add_skill(skill)
        added_skills += 1
        status = "NEW" if is_new else "KEPT"
        print(f"    [{status}] {raw.get('title', '')}: confidence={confidence:.3f}, gen={generation}")

    # Process evolved mistakes
    raw_mistakes = parsed.get("mistakes", [])
    for i, raw in enumerate(raw_mistakes):
        is_new = raw.get("is_new", False)
        generation = prev_gen + 1 if is_new else prev_gen

        if is_new:
            # New mistake: assign fresh ID
            prefix = level_name[:3]
            mistake_id = f"{prefix}_evo_{i:03d}"
        else:
            # Retained mistake: try to find original ID by matching description
            mistake_id = None
            for old_id, old_mistake in prev_mistake_ids.items():
                if old_mistake.description == raw.get("description", ""):
                    mistake_id = old_id
                    break
            if not mistake_id:
                # Fallback: generate new ID
                prefix = level_name[:3]
                mistake_id = f"{prefix}_err_{i:03d}"

        mistake = Mistake(
            mistake_id=mistake_id,
            description=raw.get("description", ""),
            why_it_happens=raw.get("why_it_happens", ""),
            how_to_avoid=raw.get("how_to_avoid", ""),
            source_level=level_name,
            source_seeds=raw.get("source_seeds", []),
            generation=generation,
        )
        new_bank.add_mistake(mistake)
        status = "NEW" if is_new else "KEPT"
        print(f"    [{status}] Mistake: {raw.get('description', '')[:50]}...")

    # Log summary
    reasoning = parsed.get("reasoning", "")
    if removed_skills:
        print(f"\n  Removed skills: {', '.join(removed_skills)}")
    if reasoning:
        print(f"\n  Teacher reasoning: {reasoning}")

    # Save new bank
    new_bank.save(output_path)
    print(f"\n  Evolved skill bank saved: {output_path}")
    print(f"  {new_bank}")

    return new_bank
