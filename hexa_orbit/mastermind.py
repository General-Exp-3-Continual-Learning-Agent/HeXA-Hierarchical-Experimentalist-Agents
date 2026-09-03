"""Mastermind as a Design-B task: same hidden code, three episodes.

Why this task exists alongside Maze. The maze result was a strong negative --
injecting a distilled bank cost gpt-oss-120b 15.2 points of episode-3 accuracy
against no bank at all (p=0.0002), and the mechanism traced to substitution: the
agent replaced a well-tuned implicit navigation policy with a crude verbalised
one. That predicts the bank should *help* where the model's implicit policy is
weak and the useful rule is compactly stateable. Mastermind is the cleanest
available test of that prediction: the winning rule ("only guess codes consistent
with every peg result so far") is one sentence, and language models routinely
violate it by re-guessing codes their own feedback has already excluded.

**The episode-boundary trap, and why the config below looks unusual.**
``MultiEpisodeEnv`` never uses ``max_turns_per_episode`` as a termination
condition -- it appears only in ``_get_max_turns``, which backfills metrics for
unplayed episodes. Episodes end *solely* when the inner env reports ``done``. Maze
works because ``MazeEnvAdapter`` is separately constructed with ``max_turns=9``.
For Mastermind the inner cap comes from GEM, and on the ``-random`` variant it is
drawn from ``[(2,4,10), (4,6,20), (4,8,30)]`` and cannot be overridden piecemeal:
``_is_random`` is set when *any* of ``code_length``/``num_numbers``/``max_turns``
is ``None``, and ``reset()`` then overwrites all three. Passing ``max_turns=3``
alone silently does nothing. Against ``total_step_cap: 9`` an inner cap of 10-30
means the outer loop always ends first, so **ORBIT's published Mastermind config
yields exactly one episode, never three**. Setting all three pins the tier and
makes the cap stick; verified in ``test_mastermind``.

**Why 3-of-6, and the pool sizes that forces.** These are ORBIT's published eval
parameters, read off ``configs/eval_config.yaml`` in the ORBIT checkout::

    - env_id: "game:Mastermind-v0-random"
      max_turns_per_episode: 3
      total_step_cap: 9
      code_length: 3
      num_numbers: 6
      duplicate_numbers: False

``MultiEpisodeEnv.from_dict`` forwards all three of ``code_length``/
``num_numbers``/``max_turns`` into the GEM constructor, so ``_is_random`` stays
False and the tier list above is never consulted. The prompt this produces is
character-identical to Appendix A of the paper.

This config was previously 4-of-8 (1680 codes) so that a 512-train + 256-eval
draw could be disjoint. That deviation made the task 14x harder than the one
ORBIT reports on: Qwen3-8B considers a median of 13 candidate codes per step,
which is 11% of a 120-code space but 0.77% of 1680, and it floored at ~0.08
episode-3 accuracy against ORBIT's published base-model 0.27.

Matching the paper costs pool size. Only ``perm(6, 3) = 120`` codes exist, and
instance identity *is* the code, so eval and train partition those 120 between
them -- see :attr:`MastermindTask.EVAL_POOL_SIZE`. ORBIT itself draws 256 test
instances from this space and therefore cannot have distinct codes across its
test set; that is harmless for a trained policy and fatal here, where a bank
built on a training code could meet the same code at eval.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Sequence

from .adapter import EpisodeRecord, format_observation
from .orbit_bridge import gem_available, install_orbit_on_path

#: D2 again: episodes are counted here, not derived from a step budget. The outer
#: loop must not depend on `total_step_cap`, or "episode 3" would mean different
#: things on instances that finish early.
EPISODES_PER_INSTANCE = 3

#: Pinned so `_is_random` stays False and the turn cap actually applies. See the
#: module docstring for what happens otherwise.
TASK_CONFIG: dict[str, Any] = {
    "env_id": "game:Mastermind-v0-random",
    "code_length": 3,
    "num_numbers": 6,
    "duplicate_numbers": False,
    "max_turns": EPISODES_PER_INSTANCE and 3,
}

_SEED_HI = 1_000_000


class GemUnavailable(RuntimeError):
    """GEM could not be imported; Mastermind cannot be constructed."""


def code_hash(code: Sequence[int]) -> str:
    """Content identity of an instance -- the secret code itself.

    The analogue of ``pools.maze_hash``: two draws that produce the same code are
    the same instance however they were seeded, and the pools deduplicate on this
    rather than on the seed (the maze run tripped `streaming_leak` after ten rounds
    because 512 seeds held only 459 distinct layouts).
    """
    return hashlib.sha256(",".join(str(int(c)) for c in code).encode()).hexdigest()


@dataclass(frozen=True)
class MastermindInstanceSpec:
    dataset_seed: int
    code: tuple[int, ...]
    code_length: int
    num_numbers: int
    max_turns: int

    @property
    def resolved_seed(self) -> int:
        """Same as ``dataset_seed``: no seed-walking is needed here.

        Maze rejects seeds whose shortest path falls outside the configured band
        and walks to the next one, so its two seeds differ. Every Mastermind seed
        yields a usable code, so the pair is trivially equal. Present because
        `MazeActor.run_instance` reads it off the spec, and forking the actor for a
        field that is definitionally equal would be worse than mapping it.
        """
        return self.dataset_seed

    @property
    def shortest_path_len(self) -> int:
        """Code length, as the analogue of Maze's optimal-solution length.

        Both name "the least the agent could get away with": Maze's shortest route,
        Mastermind's number of symbols that must be pinned down. Reported for
        parity with the Maze record schema, not used as a reward term here.
        """
        return self.code_length

    @property
    def code_hash(self) -> str:
        return code_hash(self.code)

    @property
    def instance_id(self) -> str:
        return self.code_hash[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "dataset_seed": self.dataset_seed,
            "code_length": self.code_length,
            "num_numbers": self.num_numbers,
            "max_turns": self.max_turns,
        }


def _adapter(task_cfg: dict[str, Any]) -> Any:
    if not gem_available():
        raise GemUnavailable(
            "GEM is not importable. It ships inside ORBIT at third_party/gem and "
            "its `envs/__init__` imports reasoning_gym at module level, so run "
            "`pip install reasoning-gym` even for tasks that do not use it."
        )
    install_orbit_on_path()
    from envs.gem_env_adapter import GEMEnvAdapter  # noqa: PLC0415

    kwargs = {k: v for k, v in task_cfg.items() if k != "env_id"}
    return GEMEnvAdapter(env_id=task_cfg["env_id"], env_kwargs=kwargs)


class MastermindInstance:
    """One hidden code, played for exactly :data:`EPISODES_PER_INSTANCE` episodes.

    The inner env is reset with the *same* seed at each episode boundary, which is
    what keeps the code fixed across episodes -- the property that makes this a
    multi-episode task rather than three unrelated games. Verified rather than
    assumed: ``test_mastermind`` asserts the code is identical in all three.
    """

    task = "mastermind"

    def __init__(
        self,
        spec: MastermindInstanceSpec,
        task_cfg: Optional[dict[str, Any]] = None,
        *,
        num_episodes: int = EPISODES_PER_INSTANCE,
    ) -> None:
        self.spec = spec
        self.task_cfg = dict(task_cfg or TASK_CONFIG)
        self.num_episodes = int(num_episodes)
        self.max_steps = int(self.task_cfg["max_turns"])
        self._env = _adapter(self.task_cfg)
        self.episodes: list[EpisodeRecord] = []
        self._episode_index = 0
        self._episode_step = 0
        self._total_steps = 0
        self._guesses: list[str] = []
        self._done = True

    @property
    def id(self) -> str:
        return self.spec.instance_id

    @property
    def done(self) -> bool:
        return self._done

    @property
    def success_count(self) -> int:
        return sum(1 for e in self.episodes if e.success)

    def _reset_inner(self) -> str:
        obs, _ = self._env.reset(seed=self.spec.dataset_seed)
        self._episode_step = 0
        return obs

    def reset(self) -> str:
        self.episodes = []
        self._episode_index = 0
        self._total_steps = 0
        self._guesses = []
        self._done = False
        return format_observation(self._reset_inner(), self._episode_index)

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Advance one guess. Returns ``(observation, reward, instance_done, info)``."""
        if self._done:
            raise RuntimeError("instance is finished; call reset()")
        obs, reward, inner_done, info = self._env.step(action)
        self._episode_step += 1
        self._total_steps += 1
        self._guesses.append(str(action))

        if not inner_done:
            return obs, float(reward), False, self._info(info, episode_done=False)

        # ORBIT's own rule, called rather than reimplemented -- the same seam Maze
        # uses. The line this replaces was
        #     bool(info.get("is_correct", reward > 0)) or reward > 0
        # and GEM's Mastermind never puts `is_correct` in `info`, so it collapsed to
        # `reward > 0`. GEM scores black_pegs/4, so ONE peg in the right place on the
        # final guess counted as a solved episode: 3.51 flagged successes per code
        # actually named, and 35% of round-0 episodes labelled wrong in the block the
        # evolver reads. Measured on the live env: a genuine solve returns
        # reward=1.0, terminated=True, truncated=False, while an exhausted episode
        # returns reward=0.25, terminated=True, truncated=True -- so ORBIT's rule
        # ("truncated is never a success") is exactly right here and was simply
        # being bypassed.
        # GEM scores Mastermind as black_pegs/4, so 1.0 -- and only 1.0 -- means all
        # four pegs are placed, i.e. the code was named. ORBIT's own rule is not
        # enough here: it rejects `truncated` episodes correctly but still admits
        # `terminated and reward > 0`, which passes a partial-peg finish. Measured on
        # the live env: solve -> reward 1.0/terminated/not-truncated; exhausted ->
        # reward 0.25/terminated/truncated; and a terminated, non-truncated episode
        # can still end on reward 0.5. Threshold on the reward itself, which is the
        # only signal that separates them cleanly.
        success = float(reward) >= 1.0
        self.episodes.append(EpisodeRecord(
            index=self._episode_index,
            success=success,
            steps=self._episode_step,
            reached_goal=success,
            visited=set(),
            # black pegs / 4 -- a real partial-credit signal. Under `1.0 if success
            # else 0.0` every failure scored 0.0, so `graded_failure_reward` mapped
            # all of them to its floor and the failure bin carried no ranking signal
            # at all. That is the exact degeneracy the graded reward exists to remove.
            progress=max(0.0, min(1.0, float(reward))),
        ))

        if len(self.episodes) >= self.num_episodes:
            self._done = True
            return obs, float(reward), True, self._info(info, episode_done=True)

        terminal = obs
        self._episode_index += 1
        nxt = format_observation(self._reset_inner(), self._episode_index)
        return (
            f"{terminal}\n\n{nxt}",
            float(reward),
            False,
            self._info(info, episode_done=True),
        )

    def _info(self, base: dict, *, episode_done: bool) -> dict:
        info = dict(base or {})
        info.update({
            "instance_id": self.id,
            "episode_index": self._episode_index,
            "episode_step": self._episode_step,
            "total_step": self._total_steps,
            "episode_done": episode_done,
            "episode_successes": [e.success for e in self.episodes],
            "episode_lengths": [e.steps for e in self.episodes],
            "instance_done": self._done,
        })
        return info

    def metrics(self) -> dict[str, Any]:
        n = len(self.episodes) or 1
        return {
            "episode/success_rate": self.success_count / n,
            "episode/success_count": self.success_count,
            "episode/num_episodes": len(self.episodes),
            "episode/total_steps": self._total_steps,
            "instance/code_length": self.spec.code_length,
            "instance/num_numbers": self.spec.num_numbers,
        }


def _draw_stream(
    n: int, *, global_seed: int, task_cfg: dict[str, Any]
) -> list[MastermindInstanceSpec]:
    """``n`` distinct instances in draw order, from one RNG stream.

    One adapter is reset repeatedly rather than constructed per draw: the code is a
    deterministic function of the seed, so the pool is read off the generator with
    no model calls. Draws continue until ``n`` *distinct* codes are found, so a
    collision costs an extra draw rather than silently shrinking the pool -- the
    maze pool's 512-seeds-to-459-layouts shrinkage is the failure this avoids.
    """
    import numpy as np  # noqa: PLC0415

    rng = np.random.default_rng(global_seed)
    env = _adapter(task_cfg)
    inner = env._env
    out: list[MastermindInstanceSpec] = []
    seen: set[str] = set()
    attempts = 0
    while len(out) < n:
        attempts += 1
        if attempts > 500 * n:
            raise RuntimeError(
                f"only {len(out)} distinct codes in {attempts} draws; the code space "
                f"for {task_cfg} is too small for a pool of {n}"
            )
        seed = int(rng.integers(0, _SEED_HI))
        env.reset(seed=seed)
        code = tuple(int(c) for c in inner.game_code)
        h = code_hash(code)
        if h in seen:
            continue
        seen.add(h)
        out.append(MastermindInstanceSpec(
            dataset_seed=seed, code=code,
            code_length=int(inner.code_length),
            num_numbers=int(inner.num_numbers),
            max_turns=int(inner.max_turns),
        ))
    return out


class MastermindTask:
    """Pools and instances, mirroring :class:`hexa_orbit.adapter.MazeTask`."""

    task = "mastermind"

    def __init__(
        self,
        task_cfg: Optional[dict[str, Any]] = None,
        *,
        global_seed: int = 42,
        num_episodes: int = EPISODES_PER_INSTANCE,
    ) -> None:
        self.task_cfg = dict(task_cfg or TASK_CONFIG)
        self.global_seed = int(global_seed)
        self.num_episodes = int(num_episodes)

    #: Eval takes the front of the stream so its membership never depends on the
    #: training size, the way `MazeTask.eval_specs` is prefix-truncatable.
    #:
    #: 40 of the 120 available codes, leaving 80 for training. ORBIT's own 256 is
    #: not reachable: `perm(6, 3) = 120` is the whole space, and a code shared
    #: between the pools hands the bank the answer. The split trades eval power
    #: for that guarantee -- 80 train codes cap a run at 8 rounds of batch 10.
    EVAL_POOL_SIZE = 40

    #: `perm(num_numbers, code_length)`. `_draw_stream` raises rather than
    #: silently shrinking a pool, so a train+eval request above this fails loudly.
    CODE_SPACE_SIZE = 120

    def _stream(self, n: int) -> list[MastermindInstanceSpec]:
        return _draw_stream(n, global_seed=self.global_seed, task_cfg=self.task_cfg)

    def train_specs(self, n: int) -> list[MastermindInstanceSpec]:
        """Instances after the eval prefix -- disjoint from eval by construction.

        Two independent draws cannot be used here as they are for Maze. Maze seeds
        come from a 1e6 space; codes come from a space of
        ``perm(num_numbers, code_length)`` = 120 at the pinned config, so two
        independent draws would overlap almost entirely. Partitioning one stream
        makes disjointness structural rather than probabilistic, which matters more
        here than for Maze: a repeated *layout* is tolerated and reported by a
        leakage split, but a repeated *code* is the answer itself.
        """
        return self._stream(self.EVAL_POOL_SIZE + n)[self.EVAL_POOL_SIZE:]

    def eval_specs(self, n: int = EVAL_POOL_SIZE) -> list[MastermindInstanceSpec]:
        return self._stream(self.EVAL_POOL_SIZE)[:n]

    def instance(self, spec: MastermindInstanceSpec) -> MastermindInstance:
        return MastermindInstance(spec, self.task_cfg, num_episodes=self.num_episodes)

    def instances(
        self, specs: Sequence[MastermindInstanceSpec]
    ) -> Iterator[MastermindInstance]:
        for spec in specs:
            yield self.instance(spec)

    def manifest(self) -> dict[str, Any]:
        from .orbit_bridge import manifest as bridge_manifest  # noqa: PLC0415

        return {"task": self.task, "task_cfg": dict(self.task_cfg),
                "num_episodes": self.num_episodes, **bridge_manifest()}


def assert_disjoint(
    train: Sequence[MastermindInstanceSpec], eval_: Sequence[MastermindInstanceSpec]
) -> None:
    """Assert the pools share no code. Unlike Maze there is no tolerated overlap.

    A repeated maze *layout* between pools is expected and handled by a leakage
    split; a repeated *code* is not, because the code is the entire instance -- an
    agent that has seen it has seen the answer.
    """
    shared = {s.code_hash for s in train} & {s.code_hash for s in eval_}
    if shared:
        raise AssertionError(
            f"{len(shared)} code(s) appear in both pools, expected 0"
        )


# ── reward signal ────────────────────────────────────────────────────────

#: ``"This guess receives 0 black peg(s) and 2 white peg(s)."``
def _peg_re():
    import re as _re  # noqa: PLC0415

    return _re.compile(
        r"receives\s+(\d+)\s+black peg\(s\)\s+and\s+(\d+)\s+white peg\(s\)",
        _re.IGNORECASE,
    )


def episode_progress(observations: Sequence[str], code_length: int) -> float:
    """Best fraction of the code pinned during an episode, in ``[0, 1]``.

    Black pegs only. A black peg is a symbol correct *and* in position, so
    ``black / code_length`` is 1.0 exactly when the code is solved -- the same
    property Maze's progress has, which keeps the two tasks' graded-failure scales
    comparable. White pegs say a symbol is present somewhere and are deliberately
    excluded: they would make a guess with every symbol right and every position
    wrong score as near-solved, when in information terms it is not.

    Read from the feedback text the agent itself saw, so this is observable rather
    than privileged -- unlike Maze, where progress needed the known map.
    """
    pattern = _peg_re()
    best = 0
    for observation in observations:
        for black, _white in pattern.findall(str(observation or "")):
            best = max(best, int(black))
    return min(1.0, best / code_length) if code_length else 0.0


def adapted_across_episodes(guesses_by_episode: Sequence[Sequence[str]]) -> bool:
    """Whether any two episodes differed in their guess sequence.

    The Mastermind analogue of Maze's identical-move-sequence check, and the
    discriminator behind the ``-1.00`` reward bin: an agent that replays the same
    guesses at a code its own feedback already excluded has learned nothing from
    its attempts. On Maze that bin never once fired; here it is a live possibility,
    because repeating an excluded guess is the specific failure the consistency
    rule addresses.
    """
    sequences = [tuple(str(g).strip() for g in seq) for seq in guesses_by_episode if seq]
    return len(set(sequences)) > 1 if len(sequences) > 1 else True
