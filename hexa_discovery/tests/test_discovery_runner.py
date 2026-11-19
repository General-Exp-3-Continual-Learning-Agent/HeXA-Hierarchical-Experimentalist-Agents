"""Unit tests for skill retrieval/rendering + injection-metadata bookkeeping
in discovery_runner. No LLM / scienceagent needed (only touches SkillBank)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # HeXA (hexa_discovery, skillrl)

from hexa_discovery import _paths
_paths.bootstrap()  # puts HeXA root (skillrl) on sys.path before we import it

from skillrl.core.skill_bank import Skill, Mistake, SkillBank
from hexa_discovery.discovery_runner import (
    build_skills_block,
    retrieve_skills_and_mistakes,
    render_skills_block,
)
from hexa_discovery.config import DISCOVERY_LEVEL


def _make_bank():
    bank = SkillBank()
    bank.add_skill(Skill(
        skill_id="dis_sk_000", title="Distance-Sweep to Identify Force Exponent",
        principle="Sweep r and read log-log slope.", when_to_apply="At the start.",
        source_level=DISCOVERY_LEVEL, source_type="contrastive",
        source_seeds=[0], confidence=0.9,
    ))
    bank.add_skill(Skill(
        skill_id="dis_sk_001", title="Isolate p1 and p2 Dependences Separately",
        principle="Vary one property at a time.", when_to_apply="After radial scaling.",
        source_level=DISCOVERY_LEVEL, source_type="contrastive",
        source_seeds=[0], confidence=0.8,
    ))
    bank.add_mistake(Mistake(
        mistake_id="dis_err_000", description="Assumed a static law.",
        why_it_happens="Short-time snapshots hid time-dependence.",
        how_to_avoid="Extend the observation window before submitting.",
        source_level=DISCOVERY_LEVEL, source_seeds=[0],
    ))
    return bank


def test_build_skills_block_matches_retrieve_plus_render():
    bank = _make_bank()
    block_a = build_skills_block(bank, max_skills=8, max_mistakes=4)
    skills, mistakes = retrieve_skills_and_mistakes(bank, max_skills=8, max_mistakes=4)
    block_b = render_skills_block(bank, skills, mistakes)
    assert block_a == block_b
    assert "Distance-Sweep to Identify Force Exponent" in block_a
    assert "Cross-World Experimentation Skills" in block_a
    assert "Discovery-Specific Skills" not in block_a  # header must be relabeled


def test_retrieve_skills_and_mistakes_returns_ids_and_titles():
    bank = _make_bank()
    skills, mistakes = retrieve_skills_and_mistakes(bank, max_skills=8, max_mistakes=4)
    ids = {s.skill_id for s in skills}
    titles = {s.title for s in skills}
    assert ids == {"dis_sk_000", "dis_sk_001"}
    assert "Isolate p1 and p2 Dependences Separately" in titles
    assert [m.mistake_id for m in mistakes] == ["dis_err_000"]


def test_bank_none_short_circuits_with_no_injection():
    assert build_skills_block(None) == ""
    skills, mistakes = retrieve_skills_and_mistakes(None)
    assert skills == [] and mistakes == []


def test_max_skills_cap_is_respected():
    bank = _make_bank()
    skills, _ = retrieve_skills_and_mistakes(bank, max_skills=1, max_mistakes=4)
    assert len(skills) == 1


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
