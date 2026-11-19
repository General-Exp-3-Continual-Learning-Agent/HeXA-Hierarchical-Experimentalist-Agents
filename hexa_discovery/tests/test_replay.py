"""Cross-world contrast buffer, provenance labels, and bank re-grounding."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # HeXA (hexa_discovery, skillrl)

import pytest

from hexa_discovery import reward as _reward
from hexa_discovery import teacher_prompts_discovery as _prompts
from hexa_discovery.bank_reground import DECAY_FLOOR, reground_bank
from hexa_discovery.replay import HeldOutLeakage, ReplayBuffer, describe


def T(world, seed, reward, success=None):
    return {"world": world, "seed": seed, "reward": reward,
            "success": (reward > 0.5) if success is None else success,
            "trajectory": [], "norm_mse": 0.01, "expl_score": 0.8}


def _rw(t):
    return t["reward"]


# ── the core problem: a unanimous round gets the missing class ─────────

def test_failure_only_round_gets_successes_from_other_worlds():
    """The measured case: coulomb_easy 0 successes / 3 failures."""
    buf = ReplayBuffer()
    buf.add([T("gravity", 0, 1.00), T("yukawa", 1, 0.99), T("fractional", 2, 0.69, False)])
    new = [T("coulomb_easy", 100, -0.30, False), T("coulomb_easy", 101, -0.12, False),
           T("coulomb_easy", 102, -0.05, False)]

    batch = buf.select_contrast(new, max_trajs=6, success_slots=3, reward_fn=_rw)
    # Only 2 successes exist in the buffer, so a 6-slot batch yields 5 — the
    # point is that the round now HAS both classes, not that slots are filled.
    assert len(batch) == 5
    assert sum(1 for t in batch if t["success"]) == 2, "both bank successes pulled in"
    assert sum(1 for t in batch if not t["success"]) == 3
    assert {t["world"] for t in batch if t["success"]} == {"gravity", "yukawa"}
    assert all(t["_replay"] is False for t in batch[:3]), "new episodes kept, unmarked"
    assert all(t["_replay"] for t in batch[3:]), "backfill marked as replay"


def test_success_only_round_gets_failures():
    """The other measured case: yukawa 3 successes / 0 failures."""
    buf = ReplayBuffer()
    buf.add([T("three_species", 0, -0.16, False), T("dark_matter", 1, -0.35, False)])
    new = [T("yukawa", 100, 0.90), T("yukawa", 101, 0.95), T("yukawa", 102, 0.99)]

    batch = buf.select_contrast(new, max_trajs=6, success_slots=3, reward_fn=_rw)
    assert sum(1 for t in batch if not t["success"]) == 2


def test_backfill_prefers_unseen_worlds():
    buf = ReplayBuffer()
    buf.add([T("gravity", 0, 1.00), T("gravity", 1, 0.99), T("yukawa", 2, 0.95)])
    new = [T("circle", 100, -0.2, False)]
    batch = buf.select_contrast(new, max_trajs=3, success_slots=2, reward_fn=_rw)
    worlds = {t["world"] for t in batch if t.get("_replay")}
    assert worlds == {"gravity", "yukawa"}, "breadth beats taking two from one world"


def test_worst_failures_are_chosen_as_contrast():
    buf = ReplayBuffer()
    buf.add([T("a", 0, -0.40, False), T("b", 1, -0.05, False)])
    new = [T("yukawa", 100, 0.99)]
    batch = buf.select_contrast(new, max_trajs=2, success_slots=1, reward_fn=_rw)
    assert [t["world"] for t in batch if t.get("_replay")] == ["a"]


def test_no_backfill_when_the_round_is_already_balanced():
    buf = ReplayBuffer()
    buf.add([T("gravity", 0, 1.0)])
    new = [T("x", 100, 0.9), T("x", 101, -0.3, False)]
    batch = buf.select_contrast(new, max_trajs=2, success_slots=1, reward_fn=_rw)
    assert len(batch) == 2 and not any(t.get("_replay") for t in batch)


def test_originals_are_not_mutated_by_replay_stamping():
    buf = ReplayBuffer()
    orig = T("gravity", 7, 1.0)
    buf.add([orig])
    buf.select_contrast([T("x", 100, -0.3, False)], max_trajs=2, success_slots=1, reward_fn=_rw)
    assert orig["seed"] == 7 and "_replay" not in orig


def test_replayed_seeds_do_not_collide_with_the_rounds_own():
    buf = ReplayBuffer()
    buf.add([T("gravity", 100, 1.0), T("yukawa", 101, 0.99)])
    new = [T("circle", 100, -0.3, False), T("circle", 101, -0.2, False)]
    batch = buf.select_contrast(new, max_trajs=4, success_slots=2, reward_fn=_rw)
    seeds = [t["seed"] for t in batch]
    assert len(set(seeds)) == len(seeds), "a collision makes source_seeds ambiguous"


# ── leakage ────────────────────────────────────────────────────────────

def test_buffer_refuses_held_out_worlds():
    buf = ReplayBuffer(held_out_worlds=["ether", "hubble"])
    with pytest.raises(HeldOutLeakage, match="ether"):
        buf.add([T("ether", 0, 0.5)])


def test_buffer_accepts_train_worlds():
    buf = ReplayBuffer(held_out_worlds=["ether"])
    assert buf.add([T("gravity", 0, 1.0)]) == 1


# ── materialisation ────────────────────────────────────────────────────

def test_evolve_dir_is_readable_by_skillrl(tmp_path):
    from skillrl.distillation.distill import load_trajectories

    buf = ReplayBuffer()
    buf.add([T("gravity", 0, 1.0)])
    batch = buf.select_contrast([T("circle", 100, -0.3, False)],
                                max_trajs=2, success_slots=1, reward_fn=_rw)
    d = buf.write_evolve_dir(batch, tmp_path / "ev")
    s, f = load_trajectories(d, "discovery")
    assert len(s) == 1 and len(f) == 1


def test_describe_names_the_replay_worlds():
    buf = ReplayBuffer()
    buf.add([T("gravity", 0, 1.0)])
    batch = buf.select_contrast([T("circle", 100, -0.3, False)],
                                max_trajs=2, success_slots=1, reward_fn=_rw)
    assert "gravity" in describe(batch) and "new 0S/1F" in describe(batch)


# ── provenance labels ──────────────────────────────────────────────────

def test_formatter_labels_new_vs_reference():
    _reward.patch()
    fmt = _prompts.format_trajectory_for_teacher_discovery
    assert "[NEW — this round]" in fmt({**T("x", 1, 0.9), "_replay": False})
    assert "[REFERENCE — earlier world]" in fmt({**T("x", 1, 0.9), "_replay": True})


def test_formatter_unlabelled_when_no_replay_key():
    """Round-1 distillation output must stay byte-identical."""
    _reward.patch()
    out = _prompts.format_trajectory_for_teacher_discovery(T("x", 1, 0.9))
    assert "[NEW" not in out and "[REFERENCE" not in out


def test_evolver_block_keeps_the_whole_configured_batch():
    """The call site hardcodes max_trajs=5; a 6-item batch must not be cut."""
    _reward.patch()
    batch = [{**T("w%d" % i, i, 0.9 if i < 3 else -0.3, i < 3), "_replay": i >= 3}
             for i in range(6)]
    block = _prompts.make_evolver_block(max_trajs=6)(batch, max_trajs=5)
    assert block.count("--- ") == 6


# ── re-grounding ───────────────────────────────────────────────────────

class _S:
    def __init__(self, sid, title, conf, gen, seeds=None):
        self.skill_id, self.title, self.confidence = sid, title, conf
        self.generation, self.source_seeds = gen, seeds or []


class _B:
    def __init__(self, skills): self.level_skills = {"discovery": skills}


def test_uncited_retained_skill_keeps_previous_confidence_decayed():
    """Measured: 6/12 skills had empty source_seeds yet confidence 0.95-0.98."""
    prev = _B([_S("s1", "T", 0.80, 2)])
    new = _B([_S("s1", "T", 0.98, 7)])          # teacher asserted 0.98, cited nothing
    reground_bank(new, prev, [], "discovery", decay=0.95, verbose=False)
    s = new.level_skills["discovery"][0]
    assert abs(s.confidence - 0.76) < 1e-9, "teacher's 0.98 must not stand"
    assert s.generation == 2, "lineage restored, not prev_gen"


def test_cited_retained_skill_is_recomputed_from_rewards():
    prev = _B([_S("s1", "T", 0.80, 1)])
    new = _B([_S("s1", "T", 0.99, 5, seeds=[100])])
    trajs = [T("w", 100, -0.5, False)]           # floor-reward evidence
    reground_bank(new, prev, trajs, "discovery", verbose=False)
    assert new.level_skills["discovery"][0].confidence < 0.2


def test_genuinely_new_skills_are_left_alone():
    prev = _B([])
    new = _B([_S("s9", "New", 0.42, 3)])
    reground_bank(new, prev, [], "discovery", verbose=False)
    s = new.level_skills["discovery"][0]
    assert s.confidence == 0.42 and s.generation == 3


def test_reworded_title_still_matches_by_id():
    prev = _B([_S("s1", "Old title", 0.90, 4)])
    new = _B([_S("s1", "Reworded title", 0.99, 8)])
    reground_bank(new, prev, [], "discovery", verbose=False)
    assert new.level_skills["discovery"][0].generation == 4


def test_reissued_id_still_matches_by_title():
    prev = _B([_S("s1", "Same title", 0.90, 4)])
    new = _B([_S("dis_ev_8_003", "Same title", 0.99, 8)])
    reground_bank(new, prev, [], "discovery", verbose=False)
    assert new.level_skills["discovery"][0].generation == 4


def test_decay_has_a_floor():
    prev = _B([_S("s1", "T", 0.16, 1)])
    new = _B([_S("s1", "T", 0.99, 5)])
    reground_bank(new, prev, [], "discovery", decay=0.5, verbose=False)
    assert new.level_skills["discovery"][0].confidence == DECAY_FLOOR


# ── resume correctness ─────────────────────────────────────────────────

def test_reuse_or_loads_an_existing_bank(tmp_path):
    """The teacher is stochastic: rebuilding on resume mixes two banks in one arm."""
    from skillrl.core.skill_bank import Skill, SkillBank
    from hexa_discovery.cross_world_loop import _reuse_or

    b = SkillBank()
    b.add_skill(Skill(skill_id="s1", title="T", principle="P",
                      when_to_apply="W", source_level="discovery"))
    path = tmp_path / "bank.json"
    b.save(path)

    got = _reuse_or(path)
    assert got is not None
    assert [s.skill_id for s in got.level_skills["discovery"]] == ["s1"]


def test_reuse_or_returns_none_when_absent(tmp_path):
    from hexa_discovery.cross_world_loop import _reuse_or
    assert _reuse_or(tmp_path / "missing.json") is None


def test_reuse_or_rebuilds_rather_than_dying_on_corruption(tmp_path):
    from hexa_discovery.cross_world_loop import _reuse_or
    bad = tmp_path / "bad.json"
    bad.write_text("{ this is not json")
    assert _reuse_or(bad) is None
