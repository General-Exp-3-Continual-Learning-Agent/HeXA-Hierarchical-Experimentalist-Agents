"""Round-2+ evolution for the *contrastive-only* SkillRL variant.

Mirrors evolving_distill.py but with no reward weighting and no confidence
preservation:
  - Existing skills are shown to the teacher without confidence percentages.
  - New trajectories are shown without reward labels and without reward sort.
  - Retained / new skills both carry the fixed NEUTRAL_CONFIDENCE; the field is
    kept only so the SkillBank schema and existing retrieval code keep working.
"""

from __future__ import annotations

from pathlib import Path

from skillrl.core.config import LEVEL_DESCRIPTIONS, TEACHER_MODEL
from skillrl.core.skill_bank import Skill, Mistake, SkillBank
from skillrl.distillation.distill import (
    _parse_json_object,
    call_teacher,
    load_trajectories,
)
from skillrl.distillation.contrastive_only_distill import NEUTRAL_CONFIDENCE
from skillrl.distillation.teacher_prompts_contrastive_only import (
    SKILL_BANK_EVOLUTION_PROMPT_NO_REWARD,
    format_skill_bank_for_teacher_no_conf,
    format_trajectories_block_no_reward,
)


def evolve_skill_bank_contrastive_only(
    level_name: str,
    prev_bank: SkillBank,
    new_trajs_dir: Path,
    output_path: Path,
    max_skills: int = 10,
    max_mistakes: int = 5,
    teacher_model: str = TEACHER_MODEL,
) -> SkillBank:
    """Evolve a contrastive-only skill bank with new trajectories.

    Parameters
    ----------
    level_name : Level being trained.
    prev_bank : Previous SkillBank to evolve from.
    new_trajs_dir : Directory with new trajectory JSON files.
    output_path : Where to save the evolved bank.
    max_skills : Maximum total skills for this level.
    max_mistakes : Maximum total mistakes for this level.
    teacher_model : Teacher model to use.
    """
    successes, failures = load_trajectories(new_trajs_dir, level_name)
    print(
        f"\n  Evolving (contrastive-only) skill bank from "
        f"{len(successes)} successes, {len(failures)} failures"
    )

    existing_skills_block = format_skill_bank_for_teacher_no_conf(prev_bank, level_name)
    new_trajs_block = format_trajectories_block_no_reward(
        successes + failures, max_trajs=5
    )
    level_desc = LEVEL_DESCRIPTIONS.get(level_name, "No description available.")

    prompt = SKILL_BANK_EVOLUTION_PROMPT_NO_REWARD.format(
        level_name=level_name,
        level_description=level_desc,
        existing_skills_block=existing_skills_block,
        n_successes=len(successes),
        n_failures=len(failures),
        new_trajectories_block=new_trajs_block,
        max_skills=max_skills,
        max_mistakes=max_mistakes,
    )

    print(f"  Calling teacher to evolve skill bank (no rewards/confidence)...")
    response = call_teacher(prompt, model=teacher_model)
    parsed = _parse_json_object(response)

    if not parsed:
        print(
            f"  [Warning] Teacher response could not be parsed. "
            f"Returning previous bank unchanged."
        )
        prev_bank.save(output_path)
        return prev_bank

    # Highest generation number seen in the previous bank — new items will be
    # tagged one above that.
    prev_gen = 0
    for skill in prev_bank._all_skills():
        if skill.generation > prev_gen:
            prev_gen = skill.generation

    prev_skill_by_title = {
        skill.title: skill
        for skill in prev_bank._all_skills()
        if skill.source_level == level_name
    }
    prev_mistake_by_desc = {
        m.description: m for m in prev_bank.level_mistakes.get(level_name, [])
    }

    new_bank = SkillBank()

    # Carry over everything that isn't being evolved this round (general skills
    # plus other levels' skills/mistakes).
    for skill in prev_bank.general_skills:
        new_bank.add_skill(skill)
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

    # Evolved skills for the target level.
    raw_skills = parsed.get("skills", [])
    prefix = level_name[:3]
    for i, raw in enumerate(raw_skills):
        is_new = bool(raw.get("is_new", False))
        title = raw.get("title", "")
        generation = prev_gen + 1 if is_new else prev_gen

        if not is_new and title in prev_skill_by_title:
            skill_id = prev_skill_by_title[title].skill_id
        else:
            skill_id = f"{prefix}_co_ev_{generation}_{i:03d}"

        skill = Skill(
            skill_id=skill_id,
            title=title,
            principle=raw.get("principle", ""),
            when_to_apply=raw.get("when_to_apply", ""),
            example=raw.get("example", ""),
            source_level=level_name,
            source_type="contrastive_only_evolved",
            source_seeds=raw.get("source_seeds", []),
            generation=generation,
            confidence=NEUTRAL_CONFIDENCE,
        )
        new_bank.add_skill(skill)
        status = "NEW" if is_new else "KEPT"
        print(f"    [{status}] {title} (gen={generation})")

    # Evolved mistakes for the target level.
    raw_mistakes = parsed.get("mistakes", [])
    for i, raw in enumerate(raw_mistakes):
        is_new = bool(raw.get("is_new", False))
        description = raw.get("description", "")
        generation = prev_gen + 1 if is_new else prev_gen

        if not is_new and description in prev_mistake_by_desc:
            mistake_id = prev_mistake_by_desc[description].mistake_id
        else:
            mistake_id = f"{prefix}_co_err_{generation}_{i:03d}"

        new_bank.add_mistake(Mistake(
            mistake_id=mistake_id,
            description=description,
            why_it_happens=raw.get("why_it_happens", ""),
            how_to_avoid=raw.get("how_to_avoid", ""),
            source_level=level_name,
            source_seeds=raw.get("source_seeds", []),
            generation=generation,
        ))
        status = "NEW" if is_new else "KEPT"
        print(f"    [{status}] Mistake: {description[:60]}")

    removed = parsed.get("removed_skill_titles", [])
    reasoning = parsed.get("reasoning", "")
    if removed:
        print(f"\n  Removed skills: {', '.join(removed)}")
    if reasoning:
        print(f"\n  Teacher reasoning: {reasoning}")

    new_bank.save(output_path)
    print(f"\n  Evolved skill bank saved: {output_path}")
    print(f"  {new_bank}")
    return new_bank
