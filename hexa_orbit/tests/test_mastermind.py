"""Mastermind adapter: episode boundaries, code persistence, pool disjointness."""

from __future__ import annotations

import pytest

from hexa_orbit.orbit_bridge import gem_available

pytestmark = pytest.mark.skipif(
    not gem_available(),
    reason="GEM not importable (ships in ORBIT/third_party/gem; needs reasoning-gym)",
)


@pytest.fixture(scope="module")
def task():
    from hexa_orbit.mastermind import MastermindTask
    return MastermindTask()


def test_the_turn_cap_actually_binds(task):
    """`max_turns_per_episode` is NOT a termination condition in MultiEpisodeEnv.

    It appears only in `_get_max_turns`, which backfills metrics for unplayed
    episodes; episodes end solely on the inner env's `done`. On the `-random`
    variant the inner cap is drawn from [(2,4,10),(4,6,20),(4,8,30)] and cannot be
    set piecemeal -- `_is_random` is true when ANY of code_length/num_numbers/
    max_turns is None, and `reset()` then overwrites all three. So ORBIT's
    published config (cap 3, total 9) yields ONE episode, never three. Pinning all
    three is what makes the cap stick.
    """
    from hexa_orbit.mastermind import TASK_CONFIG

    assert TASK_CONFIG["max_turns"] == 3
    for key in ("code_length", "num_numbers", "max_turns"):
        assert TASK_CONFIG[key] is not None, key

    spec = task.eval_specs(1)[0]
    assert spec.max_turns == 3, "randomisation was not pinned"


def test_three_episodes_on_one_persistent_code(task):
    """The code must survive episode resets, or there is nothing to learn across."""
    inst = task.instance(task.eval_specs(1)[0])
    inst.reset()
    codes = [tuple(inst._env._env.game_code)]
    boundaries = 0
    for _ in range(9):
        _, _, done, info = inst.step("\\boxed{1 2 3 4}")
        codes.append(tuple(inst._env._env.game_code))
        boundaries += bool(info["episode_done"])
        if done:
            break
    assert len(inst.episodes) == 3, inst.episodes
    assert boundaries == 3
    assert all(c == codes[0] for c in codes), "code changed between episodes"


def test_pools_are_disjoint_by_construction(task):
    """Independent draws cannot work here: the code space is far smaller than Maze's.

    perm(6, 3) = 120 codes total at ORBIT's published config, so two independent
    streams would overlap almost entirely. Eval takes the front of one stream and
    train the remainder. A shared *code* is the answer itself -- unlike a shared
    maze layout, which is tolerated and reported by a leakage split.
    """
    from hexa_orbit.mastermind import assert_disjoint

    ev = task.eval_specs(40)
    tr = task.train_specs(60)
    assert len({s.code_hash for s in ev}) == 40
    assert len({s.code_hash for s in tr}) == 60
    assert_disjoint(tr, ev)


def test_eval_pool_does_not_move_when_the_train_size_changes(task):
    """Eval is a stable prefix, as `MazeTask.eval_specs` is."""
    ev = [s.instance_id for s in task.eval_specs(8)]
    task.train_specs(32)
    assert [s.instance_id for s in task.eval_specs(8)] == ev


def test_instance_identity_is_the_code(task):
    from hexa_orbit.mastermind import code_hash

    spec = task.eval_specs(1)[0]
    assert spec.instance_id == code_hash(spec.code)[:16]
    assert code_hash((1, 2, 3, 4)) != code_hash((4, 3, 2, 1))


def test_every_prompt_placeholder_is_supplied_by_its_caller():
    """A drifted placeholder raises KeyError only when the teacher is first called.

    The mistakes prompt carried an `{n_failures}` that `distill_mistakes` does not
    supply, and the run died four minutes in -- after a full round of model calls --
    rather than at import. These sets are what the three `skillrl` call sites pass.
    """
    import re

    from hexa_orbit import mastermind_teacher as mt

    supplied = {
        "CONTRASTIVE_DISTILLATION_PROMPT_MASTERMIND": {
            "level_name", "level_description", "n_successes", "n_failures",
            "success_block", "failure_block"},
        "COMMON_MISTAKES_PROMPT_MASTERMIND": {
            "level_name", "level_description", "failure_block"},
        "SKILL_BANK_EVOLUTION_PROMPT_MASTERMIND": {
            "level_name", "level_description", "existing_skills_block",
            "n_successes", "n_failures", "new_trajectories_block",
            "max_skills", "max_mistakes"},
    }
    for name, keys in supplied.items():
        used = set(re.findall(r"{(\w+)}", getattr(mt, name)))
        assert not (used - keys), f"{name} needs {used - keys}, caller supplies {keys}"


@pytest.fixture
def restore_maze_bindings():
    """Snapshot and restore every binding `install()` mutates.

    `install` rewrites process-wide state: the loop's level, three prompts, the
    trajectory formatter, and -- via `hexa_bridge.patch` -- the trajectory loader
    and the confidence wrapper in the shared distillation modules. Re-calling
    Maze's `install` afterwards is not enough to undo it, because `patch` does not
    restore what a later `patch` overwrote. Two Maze tests failed that way before
    this snapshotted the attributes directly.
    """
    import skillrl.distillation.distill as _distill
    import skillrl.distillation.evolving_distill as _evolving
    import skillrl.distillation.teacher_prompts as _tp

    from hexa_orbit import loop, mastermind_teacher, teacher

    watched = [
        (_tp, "format_trajectory_for_teacher"),
        (_tp, "format_trajectories_block"),
        (_distill, "CONTRASTIVE_DISTILLATION_PROMPT"),
        (_distill, "COMMON_MISTAKES_PROMPT"),
        (_distill, "load_trajectories"),
        (_distill, "format_trajectories_block"),
        (_distill, "compute_trajectory_reward"),
        # `patch(confidence_fn=...)` rewrites this, and Mastermind's install does
        # not pass one -- so it reverts to stock and stays there, because Maze's
        # `install` early-returns when already installed and never re-patches.
        (_distill, "_compute_initial_confidence"),
        (_evolving, "_compute_initial_confidence"),
        (_evolving, "SKILL_BANK_EVOLUTION_PROMPT"),
        (_evolving, "load_trajectories"),
        (_evolving, "format_trajectories_block"),
    ]
    from hexa_orbit import reward as _reward

    saved = [(m, a, getattr(m, a, None)) for m, a in watched]
    saved_level = loop.level()
    # Mastermind's install turns graded failures on, which changes the reward a
    # never-solved instance gets and so the text the Maze renderer prints.
    saved_graded = _reward.GRADED_FAILURES[0]
    yield
    _reward.set_graded_failures(saved_graded)
    for module, attr, value in saved:
        if value is not None:
            setattr(module, attr, value)
    loop.set_level(saved_level)
    mastermind_teacher._INSTALLED.clear()
    teacher._INSTALLED.clear()


def test_install_points_the_shared_code_at_mastermind(restore_maze_bindings):
    """`run_hexa_loop` used to install the Maze teacher unconditionally.

    Because `hexa_bridge.patch` rebinds `load_trajectories` in both distillation
    consumers, that silently replaced the Mastermind loader and the evolver read a
    six-run directory as "0 successes, 0 failures".
    """
    from hexa_orbit import loop, mastermind_teacher
    import skillrl.distillation.evolving_distill as ev
    import hexa_orbit.mastermind_trajectory as mmt

    mastermind_teacher.install()
    assert loop.level() == "mastermind"
    assert ev.load_trajectories is mmt.load_instance_trajectories
    assert "Mastermind" in ev.SKILL_BANK_EVOLUTION_PROMPT
