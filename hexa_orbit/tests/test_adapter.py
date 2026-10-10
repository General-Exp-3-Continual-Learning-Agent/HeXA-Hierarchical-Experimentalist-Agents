"""Phase 1 adapter tests: identity, determinism, the T=3 cap, and ORBIT parity.

The load-bearing test here is :func:`test_observation_stream_matches_multi_episode_env`.
Everything else checks our own bookkeeping; that one checks we have not quietly
diverged from ORBIT's protocol, which is the claim Phase 2 rests on.
"""

from __future__ import annotations

import numpy as np
import pytest

from hexa_orbit import oracle, pools
from hexa_orbit.adapter import (
    EPISODES_PER_INSTANCE,
    ORBIT_SYSTEM_PROMPT_MULTI,
    MazeTask,
)
from hexa_orbit.orbit_bridge import install_orbit_on_path

WALL_BANG = oracle.boxed("up")


@pytest.fixture(scope="module")
def task() -> MazeTask:
    return MazeTask()


@pytest.fixture(scope="module")
def eval_specs(task: MazeTask):
    return task.eval_specs(8)


# ── the bridge (D4) ───────────────────────────────────────────────────────

def test_envs_package_skip_does_not_break_maze_detection():
    """ORBIT's maze tracking must stay live despite our GEM-free envs package.

    ``MultiEpisodeEnv._is_maze_inner_env`` imports ``envs.maze_env_adapter``
    inside a bare ``except Exception: return False``, so a broken ``envs``
    package would disable ``maze/unique_visited_states`` *silently* rather than
    raising. This is the assertion promised in ``orbit_bridge._register_envs_package``.
    """
    install_orbit_on_path()
    from envs.maze_env_adapter import MazeEnvAdapter
    from envs.multi_episode_env import MultiEpisodeEnv

    env = MultiEpisodeEnv(
        inner_env_class=MazeEnvAdapter,
        inner_env_kwargs=_inner_kwargs(MazeTask().task_cfg),
        total_step_cap=27,
    )
    assert env._is_maze_inner_env() is True


def _inner_kwargs(cfg: dict) -> dict:
    return {
        "env_id": "maze",
        "env_kwargs": {
            "shapes": [list(s) for s in cfg["shapes"]],
            "max_turns": int(cfg["max_turns_per_episode"]),
            "shortest_path_min_length": int(cfg["shortest_path_min_length"]),
            "shortest_path_max_length": int(cfg["shortest_path_max_length"]),
        },
    }


# ── pools: derivation, identity, disjointness (D6) ────────────────────────

def test_seed_derivation_regression(task: MazeTask):
    """Pin ORBIT's derived pool seeds. A change here means the scheme drifted."""
    assert pools.task_specific_seed(task.task_cfg) == 921231998
    assert pools.task_specific_seed(task.task_cfg) * 2 == 1842463996


def test_published_config_is_unmodified(task: MazeTask):
    """The hash input must stay ORBIT's config verbatim, cap included (RECON §2).

    D3 runs at 27 steps, but mutating ``total_step_cap`` here would redraw every
    seed in both pools and silently replace ORBIT's published eval set.
    """
    assert task.task_cfg["total_step_cap"] == 45
    assert task.task_cfg["max_turns_per_episode"] == 9
    assert task.task_cfg["test_size"] == 256


def test_resolve_seed_matches_adapter(task: MazeTask, eval_specs):
    """Our seed walk must reproduce the adapter's own map, not just its length.

    ``pools.resolve_seed`` reimplements ``MazeEnvAdapter.reset``'s filter loop
    because the adapter discards the walked seed. This is the equivalence check
    that reimplementation is only safe with.
    """
    install_orbit_on_path()
    from envs.maze_env_adapter import MazeEnvAdapter, MazeGenerator

    cfg = task.task_cfg
    generator = MazeGenerator([tuple(s) for s in cfg["shapes"]])
    env = MazeEnvAdapter(env_id="maze", env_kwargs=_inner_kwargs(cfg)["env_kwargs"])

    for spec in eval_specs:
        env.reset(seed=spec.dataset_seed)
        _, maze, init, path_len = pools.resolve_seed(
            generator,
            spec.dataset_seed,
            min_len=int(cfg["shortest_path_min_length"]),
            max_len=int(cfg["shortest_path_max_length"]),
        )
        assert np.array_equal(maze, env.map)
        assert tuple(init) == tuple(env.init_position)
        assert path_len == env.shortest_path_len == spec.shortest_path_len
        assert pools.maze_hash(env.map, env.init_position) == spec.maze_hash


def test_pools_are_seed_disjoint_and_leakage_is_measured(task: MazeTask):
    """Seeds disjoint (must be exact); layouts overlap (expected, measured)."""
    train = pools.build_pool(task.task_cfg, 512, is_train=True)
    eval_ = pools.build_pool(task.task_cfg, 256, is_train=False)
    pools.assert_disjoint(train, eval_)  # raises on any shared seed

    summary = pools.pool_summary(train, eval_)
    assert summary["distinct_eval_mazes"] == 243
    assert summary["distinct_train_mazes"] == 459
    assert summary["eval_seeds_walked"] == 181
    assert summary["n_eval_leaked"] == 49
    assert summary["n_eval_clean"] == 207

    split = pools.leakage_split(eval_, train)
    assert len(split["leaked"]) + len(split["clean"]) == len(eval_)
    assert not {s.maze_hash for s in split["clean"]} & {s.maze_hash for s in train}


def test_disjointness_check_catches_contamination(task: MazeTask):
    """The guard must actually fire -- a silent pass would be worse than none."""
    pool = pools.build_pool(task.task_cfg, 4, is_train=False)
    with pytest.raises(pools.PoolContamination):
        pools.assert_disjoint(pool, pool)


def test_eval_pool_truncation_is_a_prefix(task: MazeTask):
    """A 50-instance pilot must be a subset of the full run, for comparability."""
    assert [s.instance_id for s in task.eval_specs(5)] == [
        s.instance_id for s in task.eval_specs(20)
    ][:5]


# ── instances: same hidden maze across episodes ───────────────────────────

def test_hidden_maze_is_identical_across_episodes(task: MazeTask, eval_specs):
    """Three episodes, one map and one start -- Maze has no visible randomness.

    The plan's Phase 1 requirement. Contrast Blackjack, where deck order *should*
    differ between episodes; Maze must not.
    """
    instance = task.instance(eval_specs[0])
    maps, starts = [], []
    for _ in range(EPISODES_PER_INSTANCE):
        instance.new_episode()
        maps.append(instance._env.map.copy())
        starts.append(tuple(instance._env.init_position))
    assert all(np.array_equal(maps[0], m) for m in maps[1:])
    assert len(set(starts)) == 1


def test_instance_ids_are_layout_based_not_seed_based(task: MazeTask):
    """D6: identity is the layout hash, and duplicate layouts must collide."""
    eval_ = pools.build_pool(task.task_cfg, 256, is_train=False)
    assert len({s.instance_id for s in eval_}) == 243 < len(eval_)
    walked = [s for s in eval_ if s.seed_was_walked]
    assert walked and all(s.resolved_seed != s.dataset_seed for s in walked)


# ── D2: the hard three-episode cap ────────────────────────────────────────

def test_failed_episodes_stop_at_three(task: MazeTask, eval_specs):
    instance = task.instance(eval_specs[0])
    instance.reset()
    steps = 0
    while not instance.done:
        instance.step(WALL_BANG)
        steps += 1
    assert len(instance.episodes) == EPISODES_PER_INSTANCE
    assert steps == 3 * instance.max_steps == 27
    assert instance.success_count == 0


def test_fast_successes_still_stop_at_three(task: MazeTask, eval_specs):
    """The case ORBIT's step cap gets wrong (RECON §0d).

    Solving optimally uses 3 x 7-8 = 21-24 steps, leaving budget under either
    cap -- so ``MultiEpisodeEnv`` would open a fourth episode while reporting
    only three. Counting episodes cannot.
    """
    instance = task.instance(eval_specs[0])
    instance.reset()
    for _ in range(EPISODES_PER_INSTANCE):
        for action in oracle.instance_optimal_actions(instance):
            _, _, done, _ = instance.step(oracle.boxed(action))
            if done:
                break
    assert instance.done
    assert len(instance.episodes) == EPISODES_PER_INSTANCE
    assert instance.success_count == EPISODES_PER_INSTANCE
    assert instance._total_steps < 27  # budget left over, yet it stopped
    assert all(e.steps == instance.spec.shortest_path_len for e in instance.episodes)


def test_multi_episode_env_would_overrun_without_the_cap(task: MazeTask, eval_specs):
    """Pin the behaviour D2 exists to prevent, so the deviation stays justified."""
    install_orbit_on_path()
    from envs.maze_env_adapter import MazeEnvAdapter
    from envs.multi_episode_env import MultiEpisodeEnv

    cfg = task.task_cfg
    env = MultiEpisodeEnv(
        inner_env_class=MazeEnvAdapter,
        inner_env_kwargs=_inner_kwargs(cfg),
        total_step_cap=int(cfg["total_step_cap"]),  # ORBIT's own 45
    )
    spec = eval_specs[0]
    env.reset(seed=spec.dataset_seed, task={"seed": spec.dataset_seed})
    actions = oracle.optimal_actions(env.inner_env.map, env.inner_env.init_position)
    done = False
    while not done:
        for action in actions:
            _, _, done, _ = env.step(oracle.boxed(action))
            if done:
                break
    successes = env.get_trajectory_info()["episode_successes"]
    assert len(successes) > EPISODES_PER_INSTANCE, (
        "expected ORBIT's step cap to allow a 4th+ episode on fast successes"
    )
    assert env.get_metrics()["episode/success_count"] > EPISODES_PER_INSTANCE


# ── fidelity: the observation stream ─────────────────────────────────────

def test_observation_stream_matches_multi_episode_env(task: MazeTask, eval_specs):
    """Byte-identical observations vs real ``MultiEpisodeEnv`` over 3 episodes.

    Driven with wall-banging actions so both run exactly 27 steps and three
    episodes, making the streams directly comparable. This covers the episode
    header, the terminal+opening concatenation at each boundary, and the
    no-reset-on-final-episode behaviour.
    """
    install_orbit_on_path()
    from envs.maze_env_adapter import MazeEnvAdapter
    from envs.multi_episode_env import MultiEpisodeEnv

    cfg = task.task_cfg
    spec = eval_specs[0]

    theirs = MultiEpisodeEnv(
        inner_env_class=MazeEnvAdapter,
        inner_env_kwargs=_inner_kwargs(cfg),
        total_step_cap=3 * int(cfg["max_turns_per_episode"]),  # 27: exactly 3 episodes
    )
    their_obs, _ = theirs.reset(seed=spec.dataset_seed, task={"seed": spec.dataset_seed})

    instance = task.instance(spec)
    our_obs = instance.reset()
    assert our_obs == their_obs

    their_stream, our_stream = [], []
    their_done = False
    while not instance.done:
        assert not their_done
        obs_t, _, their_done, _ = theirs.step(WALL_BANG)
        obs_o, _, _, _ = instance.step(WALL_BANG)
        their_stream.append(obs_t)
        our_stream.append(obs_o)

    assert their_done, "ORBIT's env should finish on the same step ours does"
    assert our_stream == their_stream
    assert theirs.get_trajectory_info()["episode_successes"] == [
        e.success for e in instance.episodes
    ]
    assert theirs.get_trajectory_info()["episode_lengths"] == [
        e.steps for e in instance.episodes
    ]


def test_system_prompt_is_orbit_verbatim():
    """Byte-identical to ``get_default_system_prompt("multi")`` (RECON §3)."""
    assert ORBIT_SYSTEM_PROMPT_MULTI == (
        "You are solving the same task across multiple episodes with a fixed total step budget. "
        "Each episode resets the environment but keeps the task identical. "
        "Leverage information gathered from earlier episodes to succeed faster. "
        "Think briefly and respond with actions inside \\boxed{} each turn. "
        "Overlong responses will be penalized."
    )


# ── metrics and oracle ───────────────────────────────────────────────────

def test_metrics_have_no_sentinels(task: MazeTask, eval_specs):
    """Under a hard T=3 every episode is real, so no ``-1`` filtering (RECON §5)."""
    instance = task.instance(eval_specs[0])
    instance.reset()
    while not instance.done:
        instance.step(WALL_BANG)
    metrics = instance.metrics()
    assert all(v >= 0 for v in metrics.values() if isinstance(v, (int, float)))
    assert {f"episode_{i}/steps" for i in (1, 2, 3)} <= set(metrics)
    assert "episode_4/steps" not in metrics


def test_oracle_matches_generator_shortest_path(task: MazeTask, eval_specs):
    for spec in eval_specs:
        instance = task.instance(spec)
        instance.reset()
        assert len(oracle.instance_optimal_actions(instance)) == spec.shortest_path_len


def test_regret_is_zero_on_optimal_play(task: MazeTask, eval_specs):
    instance = task.instance(eval_specs[0])
    instance.reset()
    for _ in range(EPISODES_PER_INSTANCE):
        for action in oracle.instance_optimal_actions(instance):
            _, _, done, _ = instance.step(oracle.boxed(action))
            if done:
                break
    report = oracle.regret(instance)
    assert report["n_failed"] == 0
    assert report["mean_excess_on_success"] == 0


# ── D5: concurrency must not change a single bit ──────────────────────────

def test_concurrent_runs_are_bit_identical_to_serial(task: MazeTask):
    """The claim `--concurrency` rests on, tested rather than argued.

    ORBIT's generator seeds the **global** RNG (``random.seed`` /
    ``np.random.seed`` at ``envs/maze_env_adapter.py:42-43``), so interleaved
    ``generate()`` calls would hand threads mazes neither asked for. The reset lock
    is supposed to make that impossible while leaving the expensive part -- the
    model call -- concurrent. This drives 8 instances both ways and requires the
    serialised records to match exactly.
    """
    import json

    from hexa_orbit import loop, trajectory
    from hexa_orbit.actor import MazeActor
    from hexa_orbit.tests.test_actor import gen

    specs = task.eval_specs(8)

    class LazyOracleStub:
        """Resolves the route at generate time, not factory time.

        ``play_batch`` builds the instance and hands it to the factory *before*
        anything resets it, so the map is not readable yet -- reading it eagerly
        raises. Deferring is what the CLIs' ``OracleStub`` does for the same reason.
        """

        def __init__(self, instance):
            self.instance = instance
            self.queue: list[str] = []

        def generate_structured(self, *, messages, **_):
            if not self.queue:
                self.queue = oracle.instance_optimal_actions(self.instance)
            return gen(oracle.boxed(self.queue.pop(0)))

    def factory(guidelines, instance=None):
        # Deterministic per instance: replay its own optimal route, so any maze
        # mix-up between threads shows up as a changed step count or outcome.
        return MazeActor(client=LazyOracleStub(instance), guidelines=guidelines)

    serial, _ = loop.play_batch(task, specs, factory, None, concurrency=1)
    parallel, _ = loop.play_batch(task, specs, factory, None, concurrency=4)

    # Input order is preserved, so this is a positional comparison.
    assert [r.instance_id for r in parallel] == [r.instance_id for r in serial]
    assert [json.dumps(r.to_dict(), sort_keys=True) for r in parallel] == [
        json.dumps(r.to_dict(), sort_keys=True) for r in serial
    ]
    assert all(r.success_count == EPISODES_PER_INSTANCE for r in serial)


def test_maze_generation_stays_correct_under_thread_pressure(task: MazeTask):
    """Resolve the same specs from many threads; every maze must match its hash.

    A tighter probe than the run-level test: it hammers `reset` specifically,
    which is the only RNG-touching call, and checks the resulting layout against
    the identity computed single-threaded at pool-build time.
    """
    from concurrent.futures import ThreadPoolExecutor

    from hexa_orbit.pools import maze_hash

    # Sized to make the race near-certain rather than merely possible. Measured
    # against a build with the lock removed, 12 specs x 8 workers caught the
    # corruption only ~1 run in 6 -- a guard that weak is barely a guard. 24 specs
    # x 6 repeats x 16 workers puts ~144 concurrent `generate()` calls in flight,
    # each of which reseeds the global RNG and then makes hundreds of draws from
    # it. Measured detection with the lock removed: 5 runs in 6, up from 1 in 6.
    # A race test is a probabilistic detector and this one is not an exception --
    # treat a single green run as weak evidence, not proof.
    specs = task.eval_specs(24)

    def check(spec):
        instance = task.instance(spec)
        instance.reset()
        env = instance._env
        return maze_hash(env.map, env.init_position) == spec.maze_hash

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(check, specs * 6))
    assert all(results), f"{results.count(False)}/{len(results)} mazes were corrupted"


def test_default_maze_config_is_unchanged():
    """The historical pool must not move. Every Maze result recorded before
    2026-09-04 is keyed on these instance ids."""
    task = MazeTask()
    assert task.config_name == pools.DEFAULT_MAZE_CONFIG == "eval_config.yaml"
    assert task.task_cfg["total_step_cap"] == 45
    assert len({s.instance_id for s in task.eval_specs()}) == 243


def test_paper_qwen_config_is_a_separate_disjoint_pool():
    """ORBIT's ``benchmark.sh`` -- which produced the paper's Qwen3-8B Maze
    numbers -- runs ``total_step_cap: 27``, and that value is inside
    ``task_specific_seed``'s hash, so it selects a different 256 mazes."""
    paper = MazeTask(config_name=pools.PAPER_QWEN_CONFIG)
    assert paper.task_cfg["total_step_cap"] == 27
    assert paper.manifest()["maze_config"] == pools.PAPER_QWEN_CONFIG

    default_ids = {s.instance_id for s in MazeTask().eval_specs()}
    paper_ids = {s.instance_id for s in paper.eval_specs()}
    # Not disjoint by construction -- distinct seeds can land on the same layout
    # -- but overlapping only at the background layout-collision rate, far from
    # the same pool. Guards against a config change silently becoming a no-op.
    assert len(paper_ids & default_ids) / len(paper_ids) < 0.20

    # Everything except the cap is identical, so this really is one knob.
    differing = {
        k for k in paper.task_cfg
        if paper.task_cfg[k] != MazeTask().task_cfg.get(k)
    }
    assert differing == {"total_step_cap"}
