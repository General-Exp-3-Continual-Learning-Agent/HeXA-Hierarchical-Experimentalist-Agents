"""Shared helpers for loading a skill bank and formatting it into a prompt block.

Used by both the data-generation script (bake-in path, static-skilled variant)
and verl/utils/dataset/rl_dataset.py (runtime injection path, evolving variant).
"""
from __future__ import annotations

import json
from typing import Any


def load_skill_bank(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def format_skills_block(level_name: str, skill_bank: dict[str, Any]) -> str:
    """Format skills + mistakes from a skill bank JSON into a prompt block."""
    skills = skill_bank.get("level_skills", {}).get(level_name, [])
    mistakes = skill_bank.get("level_mistakes", {}).get(level_name, [])

    if not skills and not mistakes:
        return ""

    lines = ["## Learned Physics Skills"]

    if skills:
        title = level_name.replace("_", " ").title()
        lines.append(f"\n### {title} Skills")
        for i, s in enumerate(skills, 1):
            lines.append(f"{i}. **{s['title']}** — {s['principle']}")
            lines.append(f"   When to apply: {s['when_to_apply']}")
            lines.append(f"   Example: {s['example']}")

    if mistakes:
        lines.append("\n### Common Mistakes to Avoid")
        for m in mistakes:
            lines.append(f"- {m['description']}")
            lines.append(f"  Why it happens: {m['why_it_happens']}")
            lines.append(f"  How to avoid: {m['how_to_avoid']}")

    return "\n".join(lines)
