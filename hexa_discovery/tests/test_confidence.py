"""Unit tests for the rescaled confidence map and the balanced evolver block.

No LLM / scienceagent needed, but skillrl must be importable (the patches
rebind its module globals).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # HeXA (hexa_discovery, skillrl)

from hexa_discovery import confidence as _confidence
from hexa_discovery import reward as _reward
from hexa_discovery import teacher_prompts_discovery as _prompts
from hexa_discovery.confidence import (
    CONFIDENCE_MIN,
    compute_initial_confidence,
    confidence_from_reward,
)


def _traj(seed, reward, success=None):
    return {
        "seed": seed,
        "reward": reward,
        "success": (reward > 0.0) if success is None else success,
        "trajectory": [],
    }


# ── the map ────────────────────────────────────────────────────────────

def test_reward_floor_reaches_confidence_floor():
    # The stock map sent our -0.5 floor to 0.25; it must now reach 0.1.
    assert abs(confidence_from_reward(-0.5) - CONFIDENCE_MIN) < 1e-9


def test_reward_ceiling_reaches_one():
    assert abs(confidence_from_reward(1.0) - 1.0) < 1e-9


def test_map_is_monotonic():
    xs = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    confs = [confidence_from_reward(x) for x in xs]
    assert confs == sorted(confs)


def test_pruning_threshold_is_reachable():
    """The evolution prompt drops skills under 0.3; that must be a real band.

    Under the stock map it needed reward < -0.4 (floor is -0.5), i.e. almost
    nothing qualified. It should now fall well inside the range.
    """
    assert confidence_from_reward(-0.5) < 0.3      # total miss: prunable
    assert confidence_from_reward(-0.3) < 0.3      # clearly-failing law: prunable
    assert confidence_from_reward(0.0) > 0.3       # at the pass line: kept


def test_out_of_range_clamps_rather_than_raising():
    assert confidence_from_reward(-99.0) == CONFIDENCE_MIN
    assert confidence_from_reward(99.0) == 1.0


# ── the drop-in ────────────────────────────────────────────────────────

def test_matches_cited_seeds_only():
    trajs = [_traj(0, 1.0), _traj(1, -0.5)]
    hi = compute_initial_confidence([0], trajs, reward_fn=lambda t: t["reward"])
    lo = compute_initial_confidence([1], trajs, reward_fn=lambda t: t["reward"])
    assert hi > lo
    assert abs(hi - 1.0) < 1e-9
    assert abs(lo - CONFIDENCE_MIN) < 1e-9


def test_pools_every_trajectory_sharing_a_seed():
    """Seeds restart per world, so `pooled` mode collides them across worlds.

    The stock function keeps the FIRST occurrence per seed and discards the
    rest; both trajectories must count here.
    """
    trajs = [_traj(0, 1.0), _traj(0, -0.5)]
    conf = compute_initial_confidence([0], trajs, reward_fn=lambda t: t["reward"])
    assert abs(conf - confidence_from_reward(0.25)) < 1e-9  # mean of 1.0 and -0.5


def test_unmatched_citation_falls_back_to_level_average():
    trajs = [_traj(0, 1.0), _traj(1, 0.0)]
    conf = compute_initial_confidence([99], trajs, reward_fn=lambda t: t["reward"])
    assert abs(conf - confidence_from_reward(0.5)) < 1e-9


def test_empty_inputs_use_the_fallback():
    assert compute_initial_confidence([], [{"seed": 0}], fallback_confidence=0.42) == 0.42
    assert compute_initial_confidence([0], [], fallback_confidence=0.42) == 0.42


# ── patch installation ─────────────────────────────────────────────────

def test_patch_rebinds_both_modules():
    import skillrl.distillation.distill as _distill
    import skillrl.distillation.evolving_distill as _evolving

    _reward.patch()
    _confidence.patch()
    assert _distill._compute_initial_confidence is compute_initial_confidence
    # evolving_distill imported it BY VALUE, so it needs its own rebind.
    assert _evolving._compute_initial_confidence is compute_initial_confidence


# ── balanced evolver block ─────────────────────────────────────────────

def test_evolver_block_reserves_slots_for_failures():
    """Successes outrank failures by reward, so top-N alone would drop them."""
    _reward.patch()
    trajs = [_traj(i, 0.9) for i in range(8)] + [_traj(100 + i, -0.4) for i in range(4)]
    block = _prompts.make_evolver_block()(trajs, max_trajs=5)
    # 5 slots, 60% success -> 3 successes + 2 failures.
    assert block.count("Success: True") == 3
    assert block.count("Success: False") == 2


def test_evolver_block_backfills_when_a_class_is_empty():
    _reward.patch()
    trajs = [_traj(i, 0.9) for i in range(8)]
    block = _prompts.make_evolver_block()(trajs, max_trajs=5)
    assert block.count("Success: True") == 5      # full block, not truncated to 3


def test_evolver_block_is_a_noop_at_the_default_seed_count():
    """--seeds-per-world 3 yields <= 3 trajectories; all of them get sent."""
    _reward.patch()
    trajs = [_traj(0, 0.9), _traj(1, 0.9), _traj(2, -0.4)]
    block = _prompts.make_evolver_block()(trajs, max_trajs=5)
    assert block.count("--- Seed") == 3


def test_evolver_block_picks_the_worst_failures():
    _reward.patch()
    trajs = [_traj(0, 0.9), _traj(1, -0.1), _traj(2, -0.5)]
    block = _prompts.make_evolver_block()(trajs, max_trajs=2)
    # 2 slots at 60% -> 1 success + 1 failure; the failure must be the worst.
    assert "Seed 2" in block and "Seed 1" not in block


def test_patch_installs_the_block_on_the_evolver_only():
    import skillrl.distillation.distill as _distill
    import skillrl.distillation.evolving_distill as _evolving
    import skillrl.distillation.teacher_prompts as _tp

    stock = _tp.format_trajectories_block
    _reward.patch()
    _prompts.patch()
    assert _evolving.format_trajectories_block is not stock
    # Round-1 distillation already uses separate per-class caps; leave it alone.
    assert _distill.format_trajectories_block is stock


def test_invalid_config_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        _prompts.make_evolver_block(max_trajs=0)
    with pytest.raises(ValueError):
        _prompts.make_evolver_block(success_fraction=1.0)


# ── graded transfer metric ─────────────────────────────────────────────

def test_summarise_reports_reward_where_pass_rate_is_flat():
    """The measured failure mode: pass-rate 0 both arms, reward clearly apart."""
    from hexa_discovery.cross_world_loop import summarise

    worse = summarise([{"success": False, "reward": -0.16, "norm_mse": 0.15, "expl_score": 0.2},
                       {"success": False, "reward": -0.11, "norm_mse": 0.15, "expl_score": 0.3}])
    better = summarise([{"success": False, "reward": 0.70, "norm_mse": 3.3e-5, "expl_score": 0.4},
                        {"success": False, "reward": 0.75, "norm_mse": 5.3e-4, "expl_score": 0.5}])
    assert worse["pass_rate"] == better["pass_rate"] == 0.0
    assert better["mean_reward"] > worse["mean_reward"]
    assert better["mean_expl_score"] > worse["mean_expl_score"]


def test_summarise_handles_missing_and_empty():
    from hexa_discovery.cross_world_loop import summarise

    empty = summarise([])
    assert empty["n"] == 0 and empty["mean_reward"] is None and empty["pass_rate"] == 0.0
    partial = summarise([{"success": True, "reward": 1.0},
                         {"success": True, "reward": None}])
    assert partial["mean_reward"] == 1.0          # None dropped, not counted as 0
    assert partial["mean_norm_mse"] is None


def test_summarise_drops_nan_norm_mse():
    from hexa_discovery.cross_world_loop import summarise

    out = summarise([{"success": False, "reward": -0.5, "norm_mse": float("nan")},
                     {"success": False, "reward": -0.5, "norm_mse": 0.2}])
    assert out["mean_norm_mse"] == 0.2


# ── teacher over stdin ─────────────────────────────────────────────────

def test_teacher_sends_prompt_on_stdin_not_argv():
    """Measured round-1 prompt is 139,023 bytes; the argv limit is 131,072."""
    import json as _json, types
    from hexa_discovery import teacher_cli

    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        seen["input"] = kw.get("input")
        return types.SimpleNamespace(
            returncode=0, stdout=_json.dumps({"result": "SKILLS"}), stderr="")

    orig, teacher_cli.subprocess.run = teacher_cli.subprocess.run, fake_run
    try:
        big = "x" * 150_000
        out = teacher_cli.call_teacher(big, model="claude-sonnet-4-6")
    finally:
        teacher_cli.subprocess.run = orig

    assert out == "SKILLS"
    assert seen["input"] == big, "prompt must go to stdin"
    assert not any(len(a) > teacher_cli.ARGV_LIMIT for a in seen["cmd"]), \
        "no argv entry may exceed the single-argument limit"


def test_teacher_returns_empty_string_on_failure():
    """The empty-bank guard depends on '' meaning 'teacher produced nothing'."""
    import types
    from hexa_discovery import teacher_cli

    orig = teacher_cli.subprocess.run
    teacher_cli.subprocess.run = lambda cmd, **kw: types.SimpleNamespace(
        returncode=1, stdout="", stderr="boom")
    try:
        assert teacher_cli.call_teacher("x", model="claude-sonnet-4-6") == ""
    finally:
        teacher_cli.subprocess.run = orig


def test_teacher_rejects_non_claude_model():
    import pytest
    from hexa_discovery import teacher_cli

    with pytest.raises(ValueError, match="claude-"):
        teacher_cli.call_teacher("x", model="sonnet")


def test_teacher_patch_rebinds_both_modules():
    import skillrl.distillation.distill as _distill
    import skillrl.distillation.evolving_distill as _evolving
    from hexa_discovery import teacher_cli

    teacher_cli.patch()
    assert _distill.call_teacher is teacher_cli.call_teacher
    # evolving_distill imported it by value AND sends the larger prompt.
    assert _evolving.call_teacher is teacher_cli.call_teacher


def test_actor_prompt_goes_on_stdin_not_argv():
    """A long episode's conversation exceeds the 128KiB single-argument cap."""
    import json as _json, types
    from hexa_discovery import cli_client

    seen = {}
    def fake_run(cmd, **kw):
        seen["cmd"], seen["input"] = cmd, kw.get("input")
        return types.SimpleNamespace(
            returncode=0, stdout=_json.dumps({"result": "ok"}), stderr="")
    orig, cli_client.subprocess.run = cli_client.subprocess.run, fake_run
    try:
        big = "x" * 150_000
        out = cli_client.cli_complete("sonnet", [{"role": "user", "content": big}])
    finally:
        cli_client.subprocess.run = orig

    assert out == "ok"
    assert big in seen["input"], "prompt must reach the CLI via stdin"
    assert not any(len(a) > 128 * 1024 for a in seen["cmd"]), \
        "no argv entry may exceed MAX_ARG_STRLEN"


# ── parameter-refit signal ─────────────────────────────────────────────

def test_refit_penalty_is_surfaced_to_the_teacher():
    """Measured: refit was worse in 10/16 episodes, median 150x, worst 7e5x.
    Only the post-refit number was stored, so the teacher could not learn it."""
    from hexa_discovery import reward as _reward, teacher_prompts_discovery as _p
    _reward.patch()
    out = _p.format_trajectory_for_teacher_discovery(
        {"seed": 0, "world": "circle", "success": False, "redacted": True,
         "norm_mse": 1.79e-03, "norm_mse_own": 5.79e-07, "expl_score": 0.5,
         "trajectory": []})
    assert "[parameter refit]" in out
    assert "3091x worse" in out or "worse" in out
    assert "already-measured constant" in out


def test_refit_improvement_is_reported_too():
    """Not one-sided: the teacher must see when declaring params HELPED."""
    from hexa_discovery import reward as _reward, teacher_prompts_discovery as _p
    _reward.patch()
    out = _p.format_trajectory_for_teacher_discovery(
        {"seed": 0, "world": "circle", "success": True, "redacted": True,
         "norm_mse": 1.48e-06, "norm_mse_own": 1.1e-05, "expl_score": 0.9,
         "trajectory": []})
    assert "improved this episode" in out


def test_no_refit_line_when_the_gap_is_small_or_absent():
    from hexa_discovery import reward as _reward, teacher_prompts_discovery as _p
    _reward.patch()
    base = {"seed": 0, "world": "circle", "success": True, "redacted": True,
            "norm_mse": 1e-05, "expl_score": 0.9, "trajectory": []}
    assert "[parameter refit]" not in _p.format_trajectory_for_teacher_discovery(base)
    assert "[parameter refit]" not in _p.format_trajectory_for_teacher_discovery(
        {**base, "norm_mse_own": 9e-06})     # ratio ~1.1, not worth a line
