"""Skill data model and SkillBank with JSON persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Skill:
    """A single distilled physics skill (strategy or contrastive insight)."""

    skill_id: str           # e.g. "general_001" or "catapult_003"
    title: str              # Short name: "Lever angle vs energy separation"
    principle: str          # 2-3 sentence physics insight
    when_to_apply: str      # Condition for applicability
    example: str = ""       # Optional concrete example with coordinates
    source_level: str = ""  # "general" or specific level name
    source_type: str = ""   # "success", "failure", or "contrastive"
    source_seeds: list[int] = field(default_factory=list)
    generation: int = 0     # 0 = initial distillation, 1+ = evolution round
    confidence: float = 0.5 # 0-1, updated empirically


@dataclass
class Mistake:
    """A common mistake pattern with root-cause analysis."""

    mistake_id: str         # e.g. "cat_err_001"
    description: str        # What the mistake is (1 sentence)
    why_it_happens: str     # Why agents make this mistake (1 sentence)
    how_to_avoid: str       # Concrete actionable fix (1-2 sentences)
    source_level: str = ""  # "general" or specific level name
    source_seeds: list[int] = field(default_factory=list)
    generation: int = 0


class SkillBank:
    """Hierarchical skill library: general skills + per-level skills + mistakes."""

    def __init__(self) -> None:
        self.general_skills: list[Skill] = []
        self.level_skills: dict[str, list[Skill]] = {}
        self.common_mistakes: list[Mistake] = []
        self.level_mistakes: dict[str, list[Mistake]] = {}

    # ── Persistence ────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "general_skills": [asdict(s) for s in self.general_skills],
            "level_skills": {
                lvl: [asdict(s) for s in skills]
                for lvl, skills in self.level_skills.items()
            },
            "common_mistakes": [asdict(m) for m in self.common_mistakes],
            "level_mistakes": {
                lvl: [asdict(m) for m in mistakes]
                for lvl, mistakes in self.level_mistakes.items()
            },
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "SkillBank":
        path = Path(path)
        data = json.loads(path.read_text())
        bank = cls()
        bank.general_skills = [Skill(**s) for s in data.get("general_skills", [])]
        bank.level_skills = {
            lvl: [Skill(**s) for s in skills]
            for lvl, skills in data.get("level_skills", {}).items()
        }
        bank.common_mistakes = [Mistake(**m) for m in data.get("common_mistakes", [])]
        bank.level_mistakes = {
            lvl: [Mistake(**m) for m in mistakes]
            for lvl, mistakes in data.get("level_mistakes", {}).items()
        }
        return bank

    # ── Mutations ──────────────────────────────────────────────────────

    def add_skill(self, skill: Skill) -> None:
        if skill.source_level == "general":
            self.general_skills.append(skill)
        else:
            self.level_skills.setdefault(skill.source_level, []).append(skill)

    def add_mistake(self, mistake: Mistake) -> None:
        if mistake.source_level == "general":
            self.common_mistakes.append(mistake)
        else:
            self.level_mistakes.setdefault(mistake.source_level, []).append(mistake)

    def remove_skill(self, skill_id: str) -> None:
        self.general_skills = [s for s in self.general_skills if s.skill_id != skill_id]
        for lvl in self.level_skills:
            self.level_skills[lvl] = [
                s for s in self.level_skills[lvl] if s.skill_id != skill_id
            ]

    def update_confidence(self, skill_id: str, new_confidence: float) -> None:
        for skill in self._all_skills():
            if skill.skill_id == skill_id:
                skill.confidence = max(0.0, min(1.0, new_confidence))
                return

    # ── Retrieval ──────────────────────────────────────────────────────

    def get_skills_for_level(self, level_name: str) -> list[Skill]:
        """Return general skills + level-specific skills, sorted by confidence."""
        general = sorted(self.general_skills, key=lambda s: s.confidence, reverse=True)
        specific = sorted(
            self.level_skills.get(level_name, []),
            key=lambda s: s.confidence,
            reverse=True,
        )
        return general + specific

    def get_mistakes_for_level(self, level_name: str) -> list[Mistake]:
        """Return common mistakes + level-specific mistakes."""
        return self.common_mistakes + self.level_mistakes.get(level_name, [])

    def format_skills_as_prompt(
        self,
        level_name: str,
        skills: Optional[list[Skill]] = None,
        mistakes: Optional[list[Mistake]] = None,
        max_mistakes: Optional[int] = None,
    ) -> str:
        """Render skills + mistakes into a structured text block for prompt injection."""
        if skills is None:
            skills = self.get_skills_for_level(level_name)
        if mistakes is None:
            mistakes = self.get_mistakes_for_level(level_name)
        if max_mistakes is not None:
            mistakes = mistakes[:max_mistakes]

        if not skills and not mistakes:
            return ""

        general = [s for s in skills if s.source_level == "general"]
        specific = [s for s in skills if s.source_level != "general"]

        lines = ["=== LEARNED PHYSICS SKILLS ===", ""]

        if general:
            lines.append("## General Principles")
            for i, s in enumerate(general, 1):
                lines.append(f"{i}. [{s.title}] {s.principle}")
                lines.append(f"   Apply when: {s.when_to_apply}")
                if s.example:
                    lines.append(f"   Example: {s.example}")
                lines.append("")

        if specific:
            lines.append(f"## {level_name.replace('_', ' ').title()}-Specific Skills")
            for i, s in enumerate(specific, 1):
                tag = "Lesson" if s.source_type == "failure" else "Strategy"
                lines.append(f"{i}. [{tag}: {s.title}] {s.principle}")
                lines.append(f"   Apply when: {s.when_to_apply}")
                if s.example:
                    lines.append(f"   Example: {s.example}")
                lines.append("")

        if mistakes:
            lines.append("## Common Mistakes to Avoid")
            for i, m in enumerate(mistakes, 1):
                lines.append(f"{i}. **{m.description}**")
                lines.append(f"   Why it happens: {m.why_it_happens}")
                lines.append(f"   How to avoid: {m.how_to_avoid}")
                lines.append("")

        lines.append("=== END SKILLS ===")
        return "\n".join(lines)

    # ── Helpers ────────────────────────────────────────────────────────

    def _all_skills(self) -> list[Skill]:
        all_skills = list(self.general_skills)
        for skills in self.level_skills.values():
            all_skills.extend(skills)
        return all_skills

    def skill_count(self) -> dict[str, int]:
        counts = {"general": len(self.general_skills)}
        for lvl, skills in self.level_skills.items():
            counts[lvl] = len(skills)
        return counts

    def mistake_count(self) -> dict[str, int]:
        counts = {"general": len(self.common_mistakes)}
        for lvl, mistakes in self.level_mistakes.items():
            counts[lvl] = len(mistakes)
        return counts

    def __repr__(self) -> str:
        s_counts = self.skill_count()
        m_counts = self.mistake_count()
        total_s = sum(s_counts.values())
        total_m = sum(m_counts.values())
        return f"SkillBank({total_s} skills, {total_m} mistakes: skills={s_counts}, mistakes={m_counts})"
