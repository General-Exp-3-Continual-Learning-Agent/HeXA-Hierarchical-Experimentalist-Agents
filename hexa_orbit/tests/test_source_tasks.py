"""Phase 1 tests for Design C's source tasks: identity, determinism, T=3, labels.

Same job as ``test_adapter``/``test_mastermind`` for Maze and Mastermind. Two
tests here are not bookkeeping and would each invalidate Phase 2 if they failed:
``test_pinned_opening_holds_the_layout_across_episodes`` (D-C1 — without it the
task is not multi-episode at all) and the two success-labelling tests (the
terminal reward does not identify success in either env, which is what forced the
Mastermind rescoring pass).
"""

from __future__ import annotations

import pytest

from hexa_orbit.source_tasks import (
    PINNED_OPENING,
    MinesweeperInstance,
    MinesweeperTask,
    SourceInstanceSpec,
    WordleTask,
    assert_disjoint,
)


@pytest.fixture(scope="module")
def ms_task() -> MinesweeperTask:
    return MinesweeperTask()


@pytest.fixture(scope="module")
def wd_task() -> WordleTask:
    return WordleTask()


@pytest.fixture(scope="module")
def ms_specs(ms_task):
    return ms_task.train_specs(6)


@pytest.fixture(scope="module")
def wd_specs(wd_task):
    return wd_task.train_specs(6)


# ── pools ──────────────────────────────────────────────────────────────

def test_pools_are_disjoint_by_content(ms_task, wd_task):
    for task, n_eval, n_train in ((ms_task, 8, 8), (wd_task, 8, 8)):
        train = task._stream(n_eval + n_train)[n_eval:]
        eval_ = task._stream(n_eval)
        assert_disjoint(train, eval_)
        assert len({s.content for s in train} & {s.content for s in eval_}) == 0


def test_pool_draws_are_deterministic(ms_task, ms_specs):
    again = MinesweeperTask().train_specs(6)
    assert [s.content for s in again] == [s.content for s in ms_specs]
    assert [s.dataset_seed for s in again] == [s.dataset_seed for s in ms_specs]


def test_stream_deduplicates_by_content(wd_task):
    specs = wd_task._stream(40)
    assert len({s.content for s in specs}) == 40


def test_wordle_capacity_is_the_vocabulary_and_is_enforced(wd_task):
    """D-C3: 192 five-letter candidates is the whole space, so the pool has a
    ceiling and asking past it must fail loudly rather than spin."""
    assert wd_task.capacity() == 192
    with pytest.raises(ValueError, match="only 192 distinct"):
        wd_task._stream(200)


def test_specs_carry_the_record_schema_the_actor_reads(ms_specs, wd_specs):
    for spec in (ms_specs[0], wd_specs[0]):
        assert spec.resolved_seed == spec.dataset_seed
        assert isinstance(spec.shortest_path_len, int)
        assert spec.instance_id == spec.content_hash[:16]
        assert spec.to_dict()["instance_id"] == spec.instance_id


def test_identity_is_content_not_seed(ms_specs):
    """Two specs with the same content are the same instance however seeded."""
    a = ms_specs[0]
    b = SourceInstanceSpec(level=a.level, dataset_seed=a.dataset_seed + 1,
                           content=a.content, max_turns=a.max_turns)
    assert a.content_hash == b.content_hash
    with pytest.raises(AssertionError, match="1 instance"):
        assert_disjoint([a], [b])


# ── D-C1 ───────────────────────────────────────────────────────────────

def _play_minesweeper(inst, moves_per_episode=1):
    """Advance with safe reveals, recording the mine layout at each turn."""
    inst.reset()
    layouts = []
    while not inst.done:
        inner = inst._env._env
        layouts.append(frozenset(
            (r, c) for r in range(inner.rows) for c in range(inner.cols)
            if inner.grid[r][c] == -1
        ))
        safe = [(r, c) for r in range(inner.rows) for c in range(inner.cols)
                if inner.grid[r][c] != -1 and not inner.revealed[r][c]]
        if not safe:
            break
        safe.sort(key=lambda rc: inner.grid[rc[0]][rc[1]])
        inst.step(f"\\boxed{{reveal {safe[0][0]} {safe[0][1]}}}")
    return layouts


def test_pinned_opening_holds_the_layout_across_episodes(ms_task, ms_specs):
    """The premise ORBIT's system prompt asserts, which the env does not give
    for free: without the pin, a seed carries ~12.8 layouts (D-C1)."""
    for spec in ms_specs[:3]:
        layouts = _play_minesweeper(ms_task.instance(spec))
        assert len(set(layouts)) == 1
        assert set(layouts[0]) == {tuple(m) for m in spec.detail["mines"]}


def test_opening_is_played_by_the_instance_not_the_agent(ms_task, ms_specs):
    inst = ms_task.instance(ms_specs[0])
    obs = inst.reset()
    inner = inst._env._env
    assert inst._episode_step == 1                       # the opening already ran
    assert inner.revealed[PINNED_OPENING[0]][PINNED_OPENING[1]]
    assert obs.startswith("[Episode 1] New episode begins.")


@pytest.mark.parametrize("episode", (0, 1, 2))
def test_every_episode_opens_with_the_action_format(ms_task, ms_specs, episode):
    """The reset text is the only statement of `\boxed{reveal R C}`; the per-step
    suffix carries the board and nothing else. Dropping it cost the first
    Minesweeper arm 760 of 768 episodes to format errors on the agent's first
    move, at 0.4% success -- which read as a hard task, not as a bug."""
    inst = ms_task.instance(ms_specs[0])
    obs = inst.reset()
    mine = tuple(ms_specs[0].detail["mines"][0])
    for _ in range(episode):
        obs, *_ = inst.step(f"\\boxed{{reveal {mine[0]} {mine[1]}}}")
    assert "\\boxed{reveal" in obs
    assert f"[Episode {episode + 1}] New episode begins." in obs
    # and the board the opening produced is there too
    assert "Here is the current board layout" in obs


def test_min_turns_counts_the_safe_cells_the_opening_left(ms_specs):
    spec = ms_specs[0]
    assert 0 <= spec.shortest_path_len <= 25 - 5


# ── success labelling ──────────────────────────────────────────────────

def test_minesweeper_success_comes_from_the_observation(ms_task, ms_specs):
    """Hitting a mine returns ``fail_reward = 0.0`` and a clear returns a
    fraction, so the terminal scalar cannot separate them."""
    spec = ms_specs[0]
    inst = ms_task.instance(spec)
    inst.reset()
    mine = tuple(spec.detail["mines"][0])
    obs, reward, _done, _info = inst.step(f"\\boxed{{reveal {mine[0]} {mine[1]}}}")
    assert "hit a mine" in obs.lower()
    assert reward >= 0.0                                  # the trap
    assert inst.episodes[0].success is False
    assert inst.episodes[0].progress < 1.0


def test_minesweeper_clear_is_a_success(ms_task, ms_specs):
    inst = ms_task.instance(ms_specs[0])
    _play_minesweeper(inst)
    assert inst.episodes[0].success is True
    assert inst.episodes[0].progress == pytest.approx(1.0)


def test_wordle_secret_is_fixed_across_episodes_and_solve_is_labelled(wd_task, wd_specs):
    spec = wd_specs[0]
    inst = wd_task.instance(spec)
    inst.reset()
    secrets = []
    while not inst.done:
        secrets.append(inst._env._env.secret_word)
        inst.step(f"\\boxed{{{spec.content}}}")
    assert set(secrets) == {spec.content}
    assert [e.success for e in inst.episodes] == [True, True, True]
    assert [e.progress for e in inst.episodes] == [1.0, 1.0, 1.0]


def test_wordle_progress_tracks_greens_not_reward(wd_task, wd_specs):
    """A near-miss earns milestone reward without solving; progress must reflect
    the greens, and success must stay False."""
    spec = wd_specs[0]
    near = spec.content[:-1] + ("A" if spec.content[-1] != "A" else "E")
    inst = wd_task.instance(spec)
    inst.reset()
    inst.step(f"\\boxed{{{near}}}")
    assert inst.episodes == []                            # episode is not over
    for _ in range(9):
        if inst.done:
            break
        inst.step("\\boxed{ZZZZZ}")
    assert inst.episodes[0].success is False
    assert inst.episodes[0].progress == pytest.approx(0.8)


# ── T = 3 ──────────────────────────────────────────────────────────────

def test_exactly_three_episodes_even_when_they_end_early(ms_task, ms_specs, wd_task, wd_specs):
    """D2. Three fast episodes leave ~9 of Minesweeper's 24 steps unspent; under
    ORBIT's step-cap rule a fourth would begin."""
    inst = ms_task.instance(ms_specs[0])
    _play_minesweeper(inst)
    assert inst.done and len(inst.episodes) == 3
    assert inst._total_steps < 24

    winst = wd_task.instance(wd_specs[0])
    winst.reset()
    while not winst.done:
        winst.step(f"\\boxed{{{wd_specs[0].content}}}")
    assert len(winst.episodes) == 3 and winst._total_steps == 3


def test_episode_boundary_concatenates_and_headers(ms_task, ms_specs):
    inst = ms_task.instance(ms_specs[0])
    inst.reset()
    mine = tuple(ms_specs[0].detail["mines"][0])
    obs, _r, _d, info = inst.step(f"\\boxed{{reveal {mine[0]} {mine[1]}}}")
    assert info["episode_done"] is True
    assert "[Episode 2] New episode begins." in obs      # ORBIT's stream shape


def test_stepping_a_finished_instance_raises(wd_task, wd_specs):
    inst = wd_task.instance(wd_specs[0])
    inst.reset()
    while not inst.done:
        inst.step(f"\\boxed{{{wd_specs[0].content}}}")
    with pytest.raises(RuntimeError, match="finished"):
        inst.step("\\boxed{ABOUT}")


# ── manifest ───────────────────────────────────────────────────────────

def test_manifests_record_the_decisions(ms_task, wd_task):
    assert ms_task.manifest()["pinned_opening"] == list(PINNED_OPENING)       # D-C1
    wd = wd_task.manifest()
    assert wd["nltk_version"] == "3.9.1" and wd["vocabulary_size"] == 192     # D-C2/3
