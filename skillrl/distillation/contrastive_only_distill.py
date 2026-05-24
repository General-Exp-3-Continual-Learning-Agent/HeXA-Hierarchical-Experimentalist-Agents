"""Round-1 distillation for the *contrastive-only* SkillRL variant.

This is the analogue of distill.py but with no reward computation and no
confidence calibration:
  - Trajectories are presented to the teacher without reward labels.
  - Trajectories are NOT sorted by reward — they are taken in directory order.
  - Extracted skills carry a fixed neutral confidence (NEUTRAL_CONFIDENCE) so
    the existing SkillBank schema and confidence-sorted retrieval continue to
    work, but no quality signal is encoded.

Public entry point: ``run_distillation_contrastive_only``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from skillrl.core.config import (
    ALL_LEVELS,
    EXISTING_RESULTS,
    LEVEL_DESCRIPTIONS,
    SKILL_BANK_PATH,
    TEACHER_MODEL,
)
from skillrl.core.skill_bank import Skill, Mistake, SkillBank
from skillrl.distillation.distill import (
    _parse_json_array,
    _parse_json_object,
    call_teacher,
    load_trajectories,
)
from skillrl.distillation.teacher_prompts_contrastive_only import (
    CONTRASTIVE_DISTILLATION_PROMPT_NO_REWARD,
    COMMON_MISTAKES_PROMPT_NO_REWARD,
    format_trajectories_block_no_reward,
)


# Fixed neutral confidence for every skill in this variant. Kept on the Skill
# dataclass purely so existing retrieval/serialization code still works; it
# carries no quality signal.
NEUTRAL_CONFIDENCE: float = 0.5


# ── Pass 1: contrastive skill extraction (no reward) ─────────────────────────

def distill_contrastive_no_reward(
    level_name: str,
    successes: list[dict],
    failures: list[dict],
    teacher_model: str = TEACHER_MODEL,
    max_success_trajs: int = 5,
    max_failure_trajs: int = 5,
) -> list[Skill]:
    """Send successes + failures together to teacher (no reward weighting)."""
    if not successes and not failures:
        return []

    success_block = (
        format_trajectories_block_no_reward(successes, max_trajs=max_success_trajs)
        if successes else "(no successes)"
    )
    failure_block = (
        format_trajectories_block_no_reward(failures, max_trajs=max_failure_trajs)
        if failures else "(no failures)"
    )
    level_desc = LEVEL_DESCRIPTIONS.get(level_name, "No description available.")

    prompt = CONTRASTIVE_DISTILLATION_PROMPT_NO_REWARD.format(
        level_name=level_name,
        level_description=level_desc,
        n_successes=len(successes),
        n_failures=len(failures),
        success_block=success_block,
        failure_block=failure_block,
    )

    sent_succ = successes[:max_success_trajs]
    sent_fail = failures[:max_failure_trajs]
    print(
        f"  Contrastive (no-reward) distillation: "
        f"{len(sent_succ)}/{len(successes)} successes + "
        f"{len(sent_fail)}/{len(failures)} failures sent to teacher"
    )

    response = call_teacher(prompt, model=teacher_model)
    raw_skills = _parse_json_array(response)

    skills: list[Skill] = []
    prefix = level_name[:3]
    for i, raw in enumerate(raw_skills):
        skill_seeds = raw.get("source_seeds", [])
        skills.append(Skill(
            skill_id=f"{prefix}_co_{i:03d}",
            title=raw.get("title", ""),
            principle=raw.get("principle", ""),
            when_to_apply=raw.get("when_to_apply", ""),
            example=raw.get("example", ""),
            source_level=level_name,
            source_type="contrastive_only",
            source_seeds=skill_seeds,
            generation=0,
            confidence=NEUTRAL_CONFIDENCE,
        ))
        print(f"    Skill '{raw.get('title', '')}' (seeds={skill_seeds})")
    return skills


# ── Pass 2: mistakes + partial skills (no reward) ────────────────────────────

def distill_mistakes_no_reward(
    level_name: str,
    failures: list[dict],
    teacher_model: str = TEACHER_MODEL,
    max_failure_trajs: int = 8,
) -> tuple[list[Mistake], list[Skill]]:
    """Failure-only pass: extract mistake patterns and partial skills."""
    if not failures:
        return [], []

    failure_block = format_trajectories_block_no_reward(
        failures, max_trajs=max_failure_trajs
    )
    level_desc = LEVEL_DESCRIPTIONS.get(level_name, "No description available.")

    prompt = COMMON_MISTAKES_PROMPT_NO_REWARD.format(
        level_name=level_name,
        level_description=level_desc,
        failure_block=failure_block,
    )

    print(f"  Extracting mistakes + partial skills from {len(failures)} failures...")
    response = call_teacher(prompt, model=teacher_model)
    parsed = _parse_json_object(response)

    prefix = level_name[:3]

    mistakes: list[Mistake] = []
    for i, raw in enumerate(parsed.get("mistakes", [])):
        mistakes.append(Mistake(
            mistake_id=f"{prefix}_err_{i:03d}",
            description=raw.get("description", ""),
            why_it_happens=raw.get("why_it_happens", ""),
            how_to_avoid=raw.get("how_to_avoid", ""),
            source_level=level_name,
            source_seeds=[t.get("seed", -1) for t in failures[:5]],
            generation=0,
        ))

    partial_skills: list[Skill] = []
    for i, raw in enumerate(parsed.get("partial_skills", [])):
        skill_seeds = raw.get("source_seeds", [])
        partial_skills.append(Skill(
            skill_id=f"{prefix}_cofsk_{i:03d}",
            title=raw.get("title", ""),
            principle=raw.get("principle", ""),
            when_to_apply=raw.get("when_to_apply", ""),
            example=raw.get("example", ""),
            source_level=level_name,
            source_type="failure_partial",
            source_seeds=skill_seeds,
            generation=0,
            confidence=NEUTRAL_CONFIDENCE,
        ))
        print(f"    Partial skill '{raw.get('title', '')}' (seeds={skill_seeds})")

    return mistakes, partial_skills


# ── Full pipeline ────────────────────────────────────────────────────────────

def run_distillation_contrastive_only(
    traj_dirs: Optional[dict[str, Path]] = None,
    output_path: Optional[Path] = None,
    levels: Optional[list[str]] = None,
    teacher_model: str = TEACHER_MODEL,
    max_success_trajs: int = 5,
    max_failure_trajs: int = 5,
) -> SkillBank:
    """Round-1 contrastive-only distillation pipeline.

    For each level:
      1. Contrastive distillation (successes + failures together → skills)
      2. Mistakes pass (failures → mistakes + partial skills)

    No cross-level generalization pass (kept simple for the variant).
    """
    if traj_dirs is None:
        traj_dirs = EXISTING_RESULTS
    if output_path is None:
        output_path = SKILL_BANK_PATH
    if levels is None:
        levels = ALL_LEVELS

    bank = SkillBank()

    for level_name in levels:
        traj_dir = traj_dirs.get(level_name)
        if traj_dir is None:
            print(f"  [Skip] No trajectory directory for {level_name}")
            continue

        successes, failures = load_trajectories(Path(traj_dir))
        print(f"\n{'=' * 50}")
        print(f"  {level_name}: {len(successes)} successes, {len(failures)} failures")

        skills = distill_contrastive_no_reward(
            level_name, successes, failures, teacher_model,
            max_success_trajs=max_success_trajs,
            max_failure_trajs=max_failure_trajs,
        )
        for skill in skills:
            bank.add_skill(skill)

        mistakes, partial_skills = distill_mistakes_no_reward(
            level_name, failures, teacher_model,
            max_failure_trajs=max_failure_trajs,
        )
        for mistake in mistakes:
            bank.add_mistake(mistake)
        for skill in partial_skills:
            bank.add_skill(skill)

        print(
            f"  → {len(skills)} contrastive + "
            f"{len(partial_skills)} failure-derived skills + "
            f"{len(mistakes)} mistakes"
        )

    bank.save(output_path)
    print(f"\nSkill bank saved to {output_path}")
    print(f"  {bank}")
    return bank
