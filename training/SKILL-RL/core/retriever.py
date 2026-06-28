"""Phase 2: Rule-based skill retrieval by level type."""

from __future__ import annotations

from skillrl.core.skill_bank import Skill, SkillBank

# Level similarity map for cross-level skill transfer.
# If a level has fewer than MIN_SPECIFIC_SKILLS, skills from related levels
# are included as supplementary context.
RELATED_LEVELS: dict[str, list[str]] = {
    "two_body_problem": ["pass_the_parcel"],       # both: push green toward blue
    "pass_the_parcel": ["two_body_problem", "catapult"],  # push + mechanism
    "down_to_earth": ["basket_case"],              # both: push green off/out
    "basket_case": ["down_to_earth"],
    "catapult": ["pass_the_parcel"],               # both involve indirect launch
    "falling_into_place": [],                      # unique interception mechanics
}

MIN_SPECIFIC_SKILLS = 3


def retrieve_skills(
    skill_bank: SkillBank,
    level_name: str,
    max_general: int = 2,
    max_specific: int = 3,
) -> list[Skill]:
    """Retrieve relevant skills for a level.

    Strategy:
    1. All level-specific skills (sorted by confidence, capped at max_specific)
    2. Top-confidence general skills (capped at max_general)
    3. If level has < MIN_SPECIFIC_SKILLS, include skills from related levels
    """
    # Level-specific skills
    specific = sorted(
        skill_bank.level_skills.get(level_name, []),
        key=lambda s: s.confidence,
        reverse=True,
    )[:max_specific]

    # Supplement from related levels if sparse
    if len(specific) < MIN_SPECIFIC_SKILLS:
        related_names = RELATED_LEVELS.get(level_name, [])
        for rel in related_names:
            rel_skills = skill_bank.level_skills.get(rel, [])
            for s in sorted(rel_skills, key=lambda s: s.confidence, reverse=True):
                if len(specific) >= max_specific:
                    break
                # Avoid duplicates by skill_id
                if s.skill_id not in {sk.skill_id for sk in specific}:
                    specific.append(s)

    # General skills
    general = sorted(
        skill_bank.general_skills,
        key=lambda s: s.confidence,
        reverse=True,
    )[:max_general]

    return general + specific
