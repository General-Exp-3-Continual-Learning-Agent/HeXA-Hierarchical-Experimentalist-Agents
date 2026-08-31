"""Design C Phase 0a: cross-task synthesis on ORBIT.

The synthesis machinery was built for InterPhyre and hardcoded its domain in
three places — the source-block registry, the prompt body's Box2D paragraph, and
a teacher-prefix router that would have handed gpt-oss-120b the small-model
prompt. These tests pin the generalisation and the one automatic guard the
design admits: cross-task synthesis sees no target trajectories, so the only
checkable part of the teacher's audit trail is that the source ids it cites
actually exist.
"""

from __future__ import annotations

import json
import math

import pytest

from hexa_orbit import cross_task_blocks as blocks
from skillrl.core.skill_bank import Mistake, Skill, SkillBank
from skillrl.distillation import cross_level_synthesis as xl


# ── Blocks ─────────────────────────────────────────────────────────────

def test_targets_are_never_sources():
    """The held-out split must stay held out: no task is both."""
    assert not set(blocks.SOURCE_BLOCKS) & set(blocks.TARGET_BLOCKS)


def test_target_blocks_carry_no_source_task_entities():
    """A target block naming a source's entities would pre-do the re-grounding."""
    forbidden = ("mine", "flag", "letter", "secret word", "green", "yellow")
    for name, block in blocks.TARGET_BLOCKS.items():
        lowered = block.lower()
        for word in forbidden:
            assert word not in lowered, f"{name} block leaks source entity {word!r}"


def test_blocks_module_exposes_the_contract_the_cli_reads():
    assert blocks.PROMPT_VARIANT in xl.PROMPT_VARIANTS
    assert blocks.DOMAIN_BLOCK.strip()
    assert set(blocks.TARGET_BLOCKS) == {"maze", "mastermind"}


# ── Prompt template selection ──────────────────────────────────────────

def test_explicit_variant_beats_teacher_prefix():
    """The prefix router is InterPhyre-only; gpt-oss must not silently get slim."""
    template, label = xl._select_prompt_template("gpt-oss-120b", "text-game")
    assert template is xl.CROSS_TASK_SYNTHESIS_PROMPT
    assert "text-game" in label


def test_default_routing_is_unchanged():
    assert xl._select_prompt_template("claude-sonnet-4-6")[0] is xl.CROSS_LEVEL_SYNTHESIS_PROMPT
    assert xl._select_prompt_template("qwen3-8b")[0] is xl.CROSS_LEVEL_SYNTHESIS_PROMPT_QWEN


def test_unknown_variant_raises():
    with pytest.raises(ValueError, match="Unknown prompt_variant"):
        xl._select_prompt_template("claude-sonnet-4-6", "nonexistent")


# ── Citation verification ──────────────────────────────────────────────

def _source_bank(level: str, skill_id: str) -> SkillBank:
    bank = SkillBank()
    bank.add_skill(Skill(
        skill_id=skill_id, title="t", principle="p", when_to_apply="w",
        example="e", source_level=level, source_type="evolved",
        source_seeds=[], generation=1, confidence=0.8,
    ))
    bank.add_mistake(Mistake(
        mistake_id=f"{skill_id}_err", description="d", why_it_happens="y",
        how_to_avoid="a", source_level=level, source_seeds=[], generation=1,
    ))
    return bank


def test_verify_citations_accepts_real_ids():
    sources = {"wordle": _source_bank("wordle", "wd_000")}
    report = xl.verify_citations(
        [{"skill_id": "mas_xl_000",
          "source_skills": [{"source_level": "wordle", "skill_id": "wd_000"},
                            {"source_level": "wordle", "skill_id": "wd_000_err"}]}],
        sources,
    )
    assert report == {
        "n_citations": 2, "n_invalid": 0,
        "invalid_citations": [], "uncited_entries": [],
    }


def test_verify_citations_flags_fabrication():
    sources = {"wordle": _source_bank("wordle", "wd_000")}
    report = xl.verify_citations(
        [
            {"skill_id": "a", "source_skills": [{"source_level": "wordle", "skill_id": "wd_999"}]},
            {"skill_id": "b", "source_skills": [{"source_level": "blackjack", "skill_id": "bj_000"}]},
            {"mistake_id": "c", "source_skills": []},
        ],
        sources,
    )
    assert report["n_invalid"] == 2
    assert report["uncited_entries"] == ["c"]
    whys = {e["why"] for e in report["invalid_citations"]}
    assert whys == {"skill_id not in source bank", "unknown source_level"}


# ── Runner ─────────────────────────────────────────────────────────────

@pytest.fixture
def orbit_sources(tmp_path):
    paths = {}
    for level, sid in (("minesweeper", "ms_000"), ("wordle", "wd_000")):
        path = tmp_path / f"{level}.json"
        _source_bank(level, sid).save(path)
        paths[level] = path
    return paths


def _synth_kwargs(orbit_sources, out_dir):
    return dict(
        target_level="mastermind",
        source_bank_paths=orbit_sources,
        output_dir=out_dir,
        teacher_model="gpt-oss-120b",
        source_block_overrides=blocks.SOURCE_BLOCKS,
        target_block_override=blocks.TARGET_BLOCKS["mastermind"],
        domain_block=blocks.DOMAIN_BLOCK,
        prompt_variant=blocks.PROMPT_VARIANT,
    )


def test_dry_run_builds_the_prompt_without_a_teacher(orbit_sources, tmp_path, monkeypatch):
    def _boom(*a, **k):  # pragma: no cover - the point is that it never runs
        raise AssertionError("teacher called during dry run")

    monkeypatch.setattr(xl, "call_teacher", _boom)
    out = tmp_path / "out"
    bank, audit = xl.synthesize_target_bank(dry_run=True, **_synth_kwargs(orbit_sources, out))

    assert bank is None and audit["dry_run"] is True
    prompt = (out / "_synthesis_prompt_mastermind.txt").read_text()
    # Domain paragraph parameterised in, both sources rendered, no Box2D leakage.
    assert "turn-based text games" in prompt
    assert "ms_000" in prompt and "wd_000" in prompt
    assert "Box2D" not in prompt and "red ball" not in prompt


def test_unregistered_source_without_override_raises(orbit_sources, tmp_path):
    kwargs = _synth_kwargs(orbit_sources, tmp_path / "out")
    kwargs["source_block_overrides"] = {"wordle": blocks.SOURCE_BLOCKS["wordle"]}
    with pytest.raises(ValueError, match="minesweeper"):
        xl.synthesize_target_bank(dry_run=True, **kwargs)


def test_text_game_variant_demands_a_domain_block(orbit_sources, tmp_path):
    kwargs = _synth_kwargs(orbit_sources, tmp_path / "out")
    kwargs["domain_block"] = None
    with pytest.raises(ValueError, match="domain_block"):
        xl.synthesize_target_bank(dry_run=True, **kwargs)


TEACHER_RESPONSE = json.dumps({
    "skills": [{
        "title": "Vary one position at a time",
        "principle": "A guess differing in one position isolates that position.",
        "when_to_apply": "When a prior guess returned a nonzero black-peg count.",
        "confidence": 0.8,
        "source_skills": [{"source_level": "wordle", "skill_id": "wd_000"}],
        "transfer_rationale": "Both score per-position matches.",
    }, {
        "title": "Fabricated citation",
        "principle": "x", "when_to_apply": "y", "confidence": 0.4,
        "source_skills": [{"source_level": "wordle", "skill_id": "wd_404"}],
        "transfer_rationale": "z",
    }],
    "mistakes": [{
        "description": "Repeating a guess already scored.",
        "why_it_happens": "w", "how_to_avoid": "h",
        "source_skills": [{"source_level": "minesweeper", "skill_id": "ms_000"}],
        "transfer_rationale": "r",
    }],
    "synthesis_reasoning": "wordle carried the load.",
})


def test_end_to_end_writes_bank_and_audits_citations(orbit_sources, tmp_path, monkeypatch):
    monkeypatch.setattr(xl, "call_teacher", lambda prompt, model=None: f"```json\n{TEACHER_RESPONSE}\n```")
    out = tmp_path / "out"
    bank, audit = xl.synthesize_target_bank(**_synth_kwargs(orbit_sources, out))

    assert len(bank.level_skills["mastermind"]) == 2
    assert (out / "skill_bank_xl_mastermind.json").exists()

    saved = json.loads((out / "skill_bank_xl_mastermind_audit.json").read_text())
    assert saved["prompt_variant"] == audit["prompt_variant"] == "text-game (9-rule, with audit fields)"

    check = saved["citation_check"]
    assert check["n_citations"] == 3
    assert check["n_invalid"] == 1
    assert check["invalid_citations"][0]["cite"]["skill_id"] == "wd_404"
    assert check["uncited_entries"] == []


# ── target blocks must describe the task the eval actually runs ──────────────
#
# Regression guard for a real confound: MASTERMIND_BLOCK once said "4-digit
# code, digits 1-8, 1680 codes" while TASK_CONFIG was code_length=3,
# num_numbers=6 (120 codes). Nothing caught it, so three banks were synthesised
# against a task that does not exist and the whole Mastermind arm of Phase 4 had
# to be rerun. A wrong description is invisible downstream -- the synthesis
# succeeds, the gates pass, and the bank is confidently wrong -- so it is pinned
# here instead.

def test_mastermind_block_matches_task_config():
    from hexa_orbit.cross_task_blocks import TARGET_BLOCKS
    from hexa_orbit.mastermind import TASK_CONFIG

    block = TARGET_BLOCKS["mastermind"]
    length = TASK_CONFIG["code_length"]
    numbers = TASK_CONFIG["num_numbers"]
    codes = math.perm(numbers, length) if not TASK_CONFIG["duplicate_numbers"] \
        else numbers ** length

    assert f"{length}-digit code" in block
    assert f"digits are drawn from 1-{numbers}" in block.lower()
    assert f"{codes} possible codes" in block
    assert r"\boxed{" + " ".join(["d"] * length) + "}" in block
    assert f"{length} black pegs" in block


def test_maze_block_matches_task_config():
    from hexa_orbit.adapter import MazeTask
    from hexa_orbit.cross_task_blocks import TARGET_BLOCKS

    block = TARGET_BLOCKS["maze"]
    cfg = MazeTask().task_cfg
    # The maze env reports each of the four neighbours as wall/path/goal and
    # nothing further; the block must not promise information the actor cannot
    # see, which is how an inverted premise gets in.
    for token in ("`wall`", "`path`", "`goal`"):
        assert token in block
    assert f"{cfg['max_turns_per_episode']} turns per episode" in block
    assert (f"{cfg['shortest_path_min_length']} or "
            f"{cfg['shortest_path_max_length']} moves") in block
