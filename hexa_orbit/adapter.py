"""Task-neutral view over ORBIT's Maze, with HExA's instance/episode structure.

One :class:`MazeInstance` is one fixed hidden maze played for exactly
:data:`EPISODES_PER_INSTANCE` episodes -- the Design B unit the evolver reflects
on. The maze rules themselves (generation, wall collision, the shortest-path
filter, termination) are never reimplemented: ``MazeEnvAdapter`` is imported in
place and every episode is a real ``reset(seed=...)`` on it.

What *is* ours is the outer multi-episode loop, and it exists for one reason:
D2's hard 3-episode cap. ORBIT's ``MultiEpisodeEnv`` terminates on
``total_steps >= total_step_cap`` with no episode counter, so an instance whose
episodes end early silently gets a fourth and fifth episode while
``get_metrics`` reports only three (RECON §0d). :class:`MazeInstance` counts
episodes instead of steps.

**Fidelity strategy.** Reimplementing an outer loop is exactly the kind of thing
that invalidates a comparison quietly, so the two decisions inside it that carry
semantics are not reimplemented either:

* The success rule is ``MultiEpisodeEnv._is_episode_success`` itself, imported
  and called (:attr:`_is_episode_success`), not a transcription of it. On Maze it
  reduces to "reached the goal", because ``MazeEnvAdapter`` hardcodes
  ``truncated=False`` (``envs/maze_env_adapter.py:507``) and gives reward only at
  the goal -- but relying on that reduction would silently break if either
  changed.
* The observation stream is byte-identical to ``MultiEpisodeEnv``'s, including
  its episode header and its concatenation of a finished episode's terminal
  observation with the next episode's opening one into a *single* observation
  (``envs/multi_episode_env.py:243-262``). That layout determines how many user
  messages the actor sees, so Phase 2's "same protocol as ORBIT" claim depends on
  it. ``tests/test_adapter.py`` asserts the equality against a real
  ``MultiEpisodeEnv`` over a scripted action sequence rather than trusting this
  docstring.

Step budgets: ``max_turns=9`` per episode is enforced by the inner env; the outer
``total_step_cap`` is deliberately absent, since under a hard T=3 it can never
bind (3 x 9 = 27 <= 45) and carrying it would only re-introduce the ambiguity D3
removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence

from .orbit_bridge import install_orbit_on_path
from .pools import (
    DEFAULT_GLOBAL_SEED,
    EVAL_POOL_SIZE,
    MazeInstanceSpec,
    build_pool,
    published_maze_config,
)

#: D2: episodes per instance, counted rather than derived from a step budget.
EPISODES_PER_INSTANCE = 3

#: ``MultiEpisodeEnv``'s default, and what ``scripts/eval/openai.py`` passes
#: explicitly in ``env_args`` (RECON §3).
EPISODE_HEADER = "New episode begins."

#: ORBIT's multi-episode system prompt, verbatim from
#: ``scripts/eval/openai.py:get_default_system_prompt`` (:385-401), ``env_mode="multi"``.
#: Transcribed rather than imported because that module imports rllm's engine at
#: module scope. Byte-identical: the plan requires it so added-context
#: comparisons stay meaningful.
ORBIT_SYSTEM_PROMPT_MULTI = (
    "You are solving the same task across multiple episodes with a fixed total step budget. "
    "Each episode resets the environment but keeps the task identical. "
    "Leverage information gathered from earlier episodes to succeed faster. "
    "Think briefly and respond with actions inside \\boxed{} each turn. "
    "Overlong responses will be penalized."
)


def _is_episode_success(done: bool, info: dict, reward: float) -> bool:
    """ORBIT's own success rule, called rather than copied."""
    install_orbit_on_path()
    from envs.multi_episode_env import MultiEpisodeEnv  # noqa: PLC0415

    return MultiEpisodeEnv._is_episode_success(done=done, info=info, reward=reward)


def format_observation(observation: str, episode_index: int) -> str:
    """Prepend ORBIT's episode header. Mirrors ``_format_observation`` (:567-580)."""
    return f"[Episode {episode_index + 1}] {EPISODE_HEADER}".strip() + f"\n{observation}"


@dataclass
class EpisodeRecord:
    """Outcome of one finished episode."""

    index: int
    success: bool
    steps: int
    reached_goal: bool
    #: Distinct positions visited in this episode. The per-episode analogue of
    #: ORBIT's trajectory-wide ``maze/unique_visited_states`` (RECON §5), kept
    #: per episode because the reward mapping's "exploratory failure" vs "no
    #: adaptation across own episodes" bins compare episodes *to each other*.
    visited: set[tuple[int, int]] = field(default_factory=set)


class MazeInstance:
    """One fixed maze, played for exactly three episodes.

    Drive it like ORBIT's env: :meth:`reset` once, then :meth:`step` per action,
    until ``done``. The hidden maze is fixed by ``spec.dataset_seed`` and every
    episode re-derives it through a real ``MazeEnvAdapter.reset``, so all three
    episodes share one map *and* one start position -- Maze has no per-episode
    visible randomness, unlike Blackjack's deck order.
    """

    task = "maze"

    def __init__(
        self,
        spec: MazeInstanceSpec,
        task_cfg: dict[str, Any],
        *,
        num_episodes: int = EPISODES_PER_INSTANCE,
    ) -> None:
        install_orbit_on_path()
        from envs.maze_env_adapter import MazeEnvAdapter  # noqa: PLC0415

        self.spec = spec
        self.num_episodes = int(num_episodes)
        self.max_steps = int(task_cfg["max_turns_per_episode"])
        self._env = MazeEnvAdapter(
            env_id="maze",
            env_kwargs={
                "shapes": [list(s) for s in task_cfg["shapes"]],
                "max_turns": self.max_steps,
                "shortest_path_min_length": int(task_cfg["shortest_path_min_length"]),
                "shortest_path_max_length": int(task_cfg["shortest_path_max_length"]),
            },
        )
        self.episodes: list[EpisodeRecord] = []
        self._episode_index = 0
        self._episode_step = 0
        self._total_steps = 0
        self._visited: set[tuple[int, int]] = set()
        self._done = True

    # ── identity ──────────────────────────────────────────────────────────

    @property
    def id(self) -> str:
        return self.spec.instance_id

    @property
    def done(self) -> bool:
        return self._done

    @property
    def success_count(self) -> int:
        """R(M;tau) in ORBIT's terms -- bounded to [0, 3] by the hard cap (D2)."""
        return sum(1 for e in self.episodes if e.success)

    # ── driving ───────────────────────────────────────────────────────────

    def _reset_inner(self) -> str:
        obs, _ = self._env.reset(seed=self.spec.dataset_seed)
        self._episode_step = 0
        self._track()
        return obs

    def _track(self) -> None:
        try:
            self._visited.add(tuple(self._env.get_state_id()))
        except Exception:  # never let bookkeeping break the env loop
            pass

    def reset(self) -> str:
        """Start the instance. Returns the first observation, header included."""
        self.episodes = []
        self._episode_index = 0
        self._total_steps = 0
        self._visited = set()
        self._done = False
        return format_observation(self._reset_inner(), self._episode_index)

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Advance one turn. Returns ``(observation, reward, instance_done, info)``.

        On an episode boundary the returned observation is the finished episode's
        terminal observation followed by the next episode's header and opening
        observation, in one string -- ORBIT's layout (:243-262). On the *final*
        episode boundary there is no next episode, so only the terminal
        observation is returned and ``instance_done`` is True.
        """
        if self._done:
            raise RuntimeError("instance is done; call reset() to replay it")

        obs, reward, episode_done, info = self._env.step(action)
        self._episode_step += 1
        self._total_steps += 1
        self._track()

        if not episode_done:
            return obs, float(reward), False, self._info(info, episode_done=False)

        success = _is_episode_success(done=True, info=info, reward=float(reward))
        self.episodes.append(
            EpisodeRecord(
                index=self._episode_index,
                success=success,
                steps=self._episode_step,
                reached_goal=bool(info.get("is_correct", success)),
                visited=set(self._visited),
            )
        )

        # D2: count episodes, not steps. This is the whole point of the class.
        if len(self.episodes) >= self.num_episodes:
            self._done = True
            return obs, float(reward), True, self._info(info, episode_done=True)

        terminal_obs = obs
        self._episode_index += 1
        next_obs = format_observation(self._reset_inner(), self._episode_index)
        return (
            f"{terminal_obs}\n\n{next_obs}",
            float(reward),
            False,
            self._info(info, episode_done=True),
        )

    def _info(self, base: dict, *, episode_done: bool) -> dict:
        info = dict(base or {})
        info.update(
            {
                "instance_id": self.id,
                "episode_index": self._episode_index,
                "episode_step": self._episode_step,
                "total_step": self._total_steps,
                "episode_done": episode_done,
                "episode_successes": [e.success for e in self.episodes],
                "episode_lengths": [e.steps for e in self.episodes],
                "instance_done": self._done,
            }
        )
        return info

    # ── metrics ───────────────────────────────────────────────────────────

    def metrics(self) -> dict[str, Any]:
        """Per-instance metrics in ORBIT's key format, plus what D-plan needs.

        Deliberately *not* ORBIT's ``get_metrics``: that emits ``-1`` sentinels
        for missing episodes and reports ``total_step_cap`` rather than real
        steps on truncation (RECON §5). Under a hard T=3 every instance has
        exactly three episodes, so no sentinel is needed and every value is real.
        """
        out: dict[str, Any] = {
            "episode/success_rate": 1.0 if self.success_count else 0.0,
            "episode/success_count": self.success_count,
            "episode/num_episodes": len(self.episodes),
            "episode/total_steps": self._total_steps,
            "maze/unique_visited_states": float(len(self._visited)),
            "instance/shortest_path_len": self.spec.shortest_path_len,
        }
        for record in self.episodes:
            out[f"episode_{record.index + 1}/success_rate"] = 1.0 if record.success else 0.0
            out[f"episode_{record.index + 1}/steps"] = record.steps
        return out

    # ── the plan's episode-level view ─────────────────────────────────────

    def new_episode(self) -> "Episode":
        """A standalone episode on this instance's hidden maze.

        The plan's Phase 1 ``Episode`` shape. This bypasses the multi-episode
        bookkeeping above and is for tests and oracle work -- code that wants one
        isolated episode without a cross-episode transcript. The actor should
        drive :meth:`reset`/:meth:`step` instead, so the observation layout stays
        faithful to ORBIT's.
        """
        obs = self._reset_inner()

        def step(action: str) -> tuple[str, float, bool]:
            observation, reward, done, _ = self._env.step(action)
            self._track()
            return observation, float(reward), bool(done)

        return Episode(obs_0=obs, step=step, max_steps=self.max_steps)


@dataclass
class Episode:
    """One episode as the plan's Phase 1 protocol describes it."""

    obs_0: str
    step: Callable[[str], tuple[str, float, bool]]
    max_steps: int


class MazeTask:
    """The Maze task: two disjoint instance pools and a factory over them."""

    name = "maze"

    def __init__(
        self,
        *,
        global_seed: int = DEFAULT_GLOBAL_SEED,
        num_episodes: int = EPISODES_PER_INSTANCE,
    ) -> None:
        self.task_cfg = published_maze_config()
        self.global_seed = global_seed
        self.num_episodes = num_episodes

    def train_specs(self, n: int, seed: int | None = None) -> list[MazeInstanceSpec]:
        return build_pool(
            self.task_cfg, n, is_train=True, global_seed=seed or self.global_seed
        )[:n]

    def eval_specs(self, n: int = EVAL_POOL_SIZE) -> list[MazeInstanceSpec]:
        """ORBIT's published eval pool, optionally truncated to a prefix.

        Truncation takes a prefix rather than a sample so that a 50-instance
        pilot is a subset of the 100-instance run is a subset of the full 256 --
        making the pilot's numbers directly comparable to the full run's.
        """
        return build_pool(
            self.task_cfg, EVAL_POOL_SIZE, is_train=False, global_seed=self.global_seed
        )[:n]

    def instance(self, spec: MazeInstanceSpec) -> MazeInstance:
        return MazeInstance(spec, self.task_cfg, num_episodes=self.num_episodes)

    def instances(self, specs: Sequence[MazeInstanceSpec]) -> Iterator[MazeInstance]:
        """Instances one at a time -- serial by construction, per D5."""
        for spec in specs:
            yield self.instance(spec)

    def manifest(self) -> dict[str, Any]:
        from .orbit_bridge import manifest as bridge_manifest  # noqa: PLC0415

        return {
            "task": self.name,
            "task_cfg": dict(self.task_cfg),
            "global_seed": self.global_seed,
            "episodes_per_instance": self.num_episodes,
            "max_turns_per_episode": int(self.task_cfg["max_turns_per_episode"]),
            "system_prompt": ORBIT_SYSTEM_PROMPT_MULTI,
            **bridge_manifest(),
        }
