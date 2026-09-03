"""Design C source tasks: Minesweeper and Wordle, as Design-B-shaped instances.

Both are GEM environments reached through ORBIT's own ``GEMEnvAdapter``, exactly
as Mastermind is, so this module is :mod:`hexa_orbit.mastermind` with the
task-specific parts swapped: what identifies an instance, what counts as a
success, and what partial credit means. The outer three-episode loop, the
observation stream, and the record schema are the same ones
``tests/test_adapter.py`` pins against a real ``MultiEpisodeEnv``.

These are **source** tasks. Nothing measured on them is a headline: their banks
exist to be transferred to Maze and Mastermind (Phase 3), and their own held-out
accuracy is a gate on whether the bank is worth transferring, not a claim.

Two decisions from ``DESIGN_C_STATUS.md`` are implemented here rather than
described:

**D-C1 — Minesweeper's opening is pinned to (2, 2).** ``_setup_mines`` excludes
the 3x3 safe zone around the first reveal (``minesweeper.py:172-186``), so the
mine layout is chosen by where the agent opens: measured over 100 seeds, only
8.3% of the 25 possible openings reproduce a given seed's (2,2) layout, and a
seed carries 12.8 distinct layouts on average. Left alone, episode 2 is a
different board than episode 1 more than 90% of the time -- while ORBIT's system
prompt tells the agent the task is identical across episodes and to use what it
learned. :class:`MinesweeperInstance` plays the pinned opening itself as turn 1
of every episode, in every arm, which restores the premise the protocol asserts.
It costs the agent little: the safe zone guarantees the opening has no adjacent
mines, so it always cascades (18 of 25 cells) and can never lose.

**D-C3 — Wordle pools are partitioned by secret word.** The candidate list is
``words.words("en-basic")`` filtered to five letters: 192 words, total. A 512-seed
train draw holds 187 of them and shares 96.5% of its secrets with a 256-seed eval
draw -- D1's Mastermind leak, same number, same cause. Both pools therefore come
from one deduplicated stream (:class:`WordleTask`), which makes disjointness
structural, and the whole task is capped at 192 instances.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Sequence

from .adapter import EpisodeRecord, format_observation
from .mastermind import GemUnavailable, _adapter as _gem_adapter

#: D2 again: episodes are counted, never derived from a step budget.
EPISODES_PER_INSTANCE = 3

_SEED_HI = 1_000_000

#: ORBIT's train-split configs, verbatim from
#: ``configs/multi_task_multi_episode_config.yaml:28-32`` and ``:40-44``, with
#: ``max_turns`` set to the config's ``max_turns_per_episode`` because that is
#: what actually reaches the inner env (``multi_episode_env.py:733-742``); the
#: registered defaults of 25 never run.
MINESWEEPER_CONFIG: dict[str, Any] = {
    "env_id": "game:Minesweeper-v0-only-reveal",
    "rows": 5,
    "cols": 5,
    "num_mines": 5,
    "max_turns": 8,
}

WORDLE_CONFIG: dict[str, Any] = {
    "env_id": "game:Wordle-v0-hard",
    "word_length": 5,
    "only_real_words": True,
    "max_turns": 10,
}

#: D-C1. The board centre: its 3x3 safe zone is fully interior, so it excludes the
#: largest possible region and yields the most informative opening cascade.
PINNED_OPENING: tuple[int, int] = (2, 2)


#: Whether :func:`ensure_wordle_corpus` has already run.
_CORPUS_READY = False


def ensure_wordle_corpus() -> None:
    """Make ``WordleEnv`` constructible offline, and once rather than every time.

    ``WordleEnv.__init__`` calls ``nltk.download("words")`` (``wordle.py:37``), so
    every instance construction is a network round-trip -- hundreds per round at
    the loop's concurrency, and a hard failure on a compute node with no route
    out. This loads the corpus once, then no-ops the downloader for the rest of
    the process. Raises if the corpus is genuinely absent, which is the D-C2
    setup step, not something to paper over at run time.
    """
    global _CORPUS_READY
    if _CORPUS_READY:
        return
    import nltk  # noqa: PLC0415
    from nltk.corpus import words  # noqa: PLC0415

    try:
        words.words("en-basic")
    except LookupError as exc:  # pragma: no cover - environment setup
        raise RuntimeError(
            "nltk's `words` corpus is missing. Run `python -c \"import nltk; "
            "nltk.download('words')\"` on a node with network access (D-C2)."
        ) from exc
    nltk.download = lambda *a, **k: True
    _CORPUS_READY = True


def _boxed_reveal(row: int, col: int) -> str:
    return f"\\boxed{{reveal {int(row)} {int(col)}}}"


@dataclass(frozen=True)
class SourceInstanceSpec:
    """One source-task instance, identified by content rather than by seed.

    ``content`` is the mine layout under the pinned opening (Minesweeper) or the
    secret word (Wordle) -- the same move ``pools.maze_hash`` and
    ``mastermind.code_hash`` make, and for the same reason: two draws that produce
    the same content are the same instance however they were seeded.
    """

    level: str
    dataset_seed: int
    content: str
    max_turns: int
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_seed(self) -> int:
        """Equal to ``dataset_seed``: neither task walks its seed.

        Present because ``MazeActor.run_instance`` reads it off the spec
        (``actor.py:701``); Mastermind maps it the same way.
        """
        return self.dataset_seed

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(f"{self.level}:{self.content}".encode()).hexdigest()

    @property
    def instance_id(self) -> str:
        return self.content_hash[:16]

    @property
    def shortest_path_len(self) -> int:
        """The record schema's "least the agent could get away with".

        Minesweeper: the safe cells still hidden after the pinned opening, i.e.
        the fewest further reveals a perfect player needs. Wordle: 1, since a
        single lucky guess ends it. Reported for parity with the Maze record, not
        used as a reward term.
        """
        return int(self.detail.get("min_turns", 1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "dataset_seed": self.dataset_seed,
            "level": self.level,
            "max_turns": self.max_turns,
            **self.detail,
        }


class _SourceInstance:
    """Shared three-episode loop. Subclasses supply the task-specific parts.

    Deliberately a near-copy of :class:`hexa_orbit.mastermind.MastermindInstance`
    rather than a refactor of it: that class is pinned by
    ``test_mastermind``/``test_adapter`` against ORBIT's own ``MultiEpisodeEnv``,
    and rewriting it to host two more tasks would put a live result's protocol
    fidelity at risk to save a few dozen lines.
    """

    task = ""
    config: dict[str, Any] = {}

    def __init__(
        self,
        spec: SourceInstanceSpec,
        task_cfg: Optional[dict[str, Any]] = None,
        *,
        num_episodes: int = EPISODES_PER_INSTANCE,
    ) -> None:
        self.spec = spec
        self.task_cfg = dict(task_cfg or self.config)
        self.num_episodes = int(num_episodes)
        self.max_steps = int(self.task_cfg["max_turns"])
        self._env = _gem_adapter(self.task_cfg)
        self.episodes: list[EpisodeRecord] = []
        self._episode_index = 0
        self._episode_step = 0
        self._total_steps = 0
        self._episode_observations: list[str] = []
        self._done = True

    # ── task-specific seams ───────────────────────────────────────────

    def _opening(self) -> Optional[str]:
        """An action the instance plays itself before the agent's first turn."""
        return None

    def _is_success(self, observation: str, reward: float, info: dict) -> bool:
        raise NotImplementedError

    def _progress(self) -> float:
        """Partial credit for the episode just finished, in [0, 1]."""
        raise NotImplementedError

    def _instance_metrics(self) -> dict[str, Any]:
        return {}

    # ── loop ──────────────────────────────────────────────────────────

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
        self._episode_observations = []
        opening = self._opening()
        if opening is not None:
            # Counts as a turn against the inner env's budget, in every arm.
            instructions = str(obs)
            obs, _reward, _done, _info = self._env.step(opening)
            self._episode_step += 1
            self._episode_observations.append(str(obs))
            # Keep the reset text. It is the ONLY statement of the action format
            # ("\boxed{reveal R C}"), and the inner env does not repeat it: the
            # per-step suffix is just the board and "Enter your guess.". Returning
            # the post-opening observation alone cost the first Minesweeper arm
            # every episode -- the agent answered `\boxed{(0,1)}`, which fails the
            # env's `([a-zA-Z]+)\s(\d+)\s(\d+)` pattern and *terminates* the
            # episode as a format error. 760 of 768 episodes died on the agent's
            # first move, at a measured 0.4% success rate.
            # Concatenated into one message, the same way `step` joins a finished
            # episode's terminal observation to the next episode's opening
            # (`multi_episode_env.py:243-262`), so the actor still sees exactly one
            # user turn per step.
            obs = f"{instructions}\n{obs}"
        return obs

    def reset(self) -> str:
        self.episodes = []
        self._episode_index = 0
        self._total_steps = 0
        self._done = False
        return format_observation(self._reset_inner(), self._episode_index)

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Advance one agent turn. Returns ``(observation, reward, done, info)``."""
        if self._done:
            raise RuntimeError("instance is finished; call reset()")
        obs, reward, inner_done, info = self._env.step(action)
        self._episode_step += 1
        self._total_steps += 1
        self._episode_observations.append(str(obs))

        if not inner_done:
            return obs, float(reward), False, self._info(info, episode_done=False)

        self.episodes.append(EpisodeRecord(
            index=self._episode_index,
            success=self._is_success(str(obs), float(reward), info or {}),
            steps=self._episode_step,
            reached_goal=self._is_success(str(obs), float(reward), info or {}),
            visited=set(),
            progress=self._progress(),
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
            **self._instance_metrics(),
        }


# ── Minesweeper ────────────────────────────────────────────────────────

_MS_SOLVED = "successfully cleared the minesweeper board"


class MinesweeperInstance(_SourceInstance):
    """5x5, 5 mines, 8 turns, opening pinned at :data:`PINNED_OPENING` (D-C1)."""

    task = "minesweeper"
    config = MINESWEEPER_CONFIG

    def _opening(self) -> Optional[str]:
        return _boxed_reveal(*PINNED_OPENING)

    def _is_success(self, observation: str, reward: float, info: dict) -> bool:
        """The observation string, never the reward.

        A cleared board returns ``new_revealed / safe_cells``
        (``minesweeper.py:138-147``) and hitting a mine returns
        ``fail_reward = 0.0``, so the terminal scalar does not separate them --
        the same trap that cost the Mastermind runs a rescoring pass
        (SESSION_NOTES §2). The env announces the clear in words; that is ground
        truth, and ``_is_solved`` on the live env agrees with it.
        """
        return _MS_SOLVED in observation.lower()

    def _progress(self) -> float:
        """Fraction of the safe board revealed when the episode ended.

        Read off the live env rather than the transcript. Privileged, like Maze's
        shortest-path progress and for the same reason: it is a grading signal for
        the reward binning and is never shown to the agent.
        """
        inner = self._env._env
        safe = inner.rows * inner.cols - inner.num_mines
        revealed = sum(
            1
            for r in range(inner.rows)
            for c in range(inner.cols)
            if inner.revealed[r][c] and inner.grid[r][c] != -1
        )
        return min(1.0, revealed / safe) if safe else 0.0

    def _instance_metrics(self) -> dict[str, Any]:
        return {
            "instance/num_mines": int(self.task_cfg["num_mines"]),
            "instance/pinned_opening": list(PINNED_OPENING),
        }


# ── Wordle ─────────────────────────────────────────────────────────────

_WORDLE_SOLVED = "you guessed the secret word"
_FEEDBACK_RE = re.compile(r"^([GYX]{2,})$", re.MULTILINE)


class WordleInstance(_SourceInstance):
    """One 5-letter secret, fixed across all three episodes, 10 turns each."""

    task = "wordle"
    config = WORDLE_CONFIG

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ensure_wordle_corpus()
        super().__init__(*args, **kwargs)

    def _is_success(self, observation: str, reward: float, info: dict) -> bool:
        """The observation string again. A solve is worth 0.5-1.0 and a
        milestone-only episode can reach 0.5 (``wordle.py:112,130``), so the
        scalar does not separate them either."""
        return _WORDLE_SOLVED in observation.lower()

    def _progress(self) -> float:
        """Best fraction of the word turned green during the episode.

        The env's own milestone vector (``wordle.py:126-134``) is exactly this and
        is reset per episode, so it needs no parsing. Unlike Minesweeper's, this
        signal is fully observable to the agent -- every G it earned is in its
        transcript.
        """
        inner = self._env._env
        milestones = list(getattr(inner, "milestones", []) or [])
        return sum(milestones) / len(milestones) if milestones else 0.0

    def _instance_metrics(self) -> dict[str, Any]:
        return {"instance/word_length": int(self.task_cfg["word_length"])}


# ── Pools ──────────────────────────────────────────────────────────────

class _SourceTask:
    """Pools and instances, mirroring :class:`hexa_orbit.mastermind.MastermindTask`.

    Both pools come from one stream with an eval prefix, so disjointness is
    structural rather than probabilistic. For Wordle that is the only thing that
    works at all (D-C3); for Minesweeper it is merely the safer default.
    """

    task = ""
    instance_cls: type[_SourceInstance] = _SourceInstance
    config: dict[str, Any] = {}
    EVAL_POOL_SIZE = 256

    def __init__(
        self,
        task_cfg: Optional[dict[str, Any]] = None,
        *,
        global_seed: int = 42,
        num_episodes: int = EPISODES_PER_INSTANCE,
    ) -> None:
        self.task_cfg = dict(task_cfg or self.config)
        self.global_seed = int(global_seed)
        self.num_episodes = int(num_episodes)

    # ── task-specific ─────────────────────────────────────────────────

    def _content(self, env: Any) -> tuple[str, dict[str, Any]]:
        """``(content key, detail)`` for the instance the adapter currently holds."""
        raise NotImplementedError

    def capacity(self) -> Optional[int]:
        """Distinct instances the task can supply, or ``None`` if effectively
        unbounded. Wordle's is its vocabulary; Minesweeper's is not enumerable."""
        return None

    # ── stream ────────────────────────────────────────────────────────

    def _stream(self, n: int) -> list[SourceInstanceSpec]:
        import numpy as np  # noqa: PLC0415

        cap = self.capacity()
        if cap is not None and n > cap:
            raise ValueError(
                f"{self.task} has only {cap} distinct instances; asked for {n}. "
                f"Reduce the pool sizes (eval prefix is {self.EVAL_POOL_SIZE})."
            )

        rng = np.random.default_rng(self.global_seed)
        env = _gem_adapter(self.task_cfg)
        out: list[SourceInstanceSpec] = []
        seen: set[str] = set()
        attempts = 0
        while len(out) < n:
            attempts += 1
            if attempts > 500 * n:
                raise RuntimeError(
                    f"only {len(out)} distinct {self.task} instances in {attempts} "
                    f"draws; the space is too small for a pool of {n}"
                )
            seed = int(rng.integers(0, _SEED_HI))
            env.reset(seed=seed)
            content, detail = self._content(env)
            if content in seen:
                continue
            seen.add(content)
            out.append(SourceInstanceSpec(
                level=self.task, dataset_seed=seed, content=content,
                max_turns=int(self.task_cfg["max_turns"]), detail=detail,
            ))
        return out

    def train_specs(self, n: int) -> list[SourceInstanceSpec]:
        return self._stream(self.EVAL_POOL_SIZE + n)[self.EVAL_POOL_SIZE:]

    def eval_specs(self, n: Optional[int] = None) -> list[SourceInstanceSpec]:
        n = self.EVAL_POOL_SIZE if n is None else n
        return self._stream(self.EVAL_POOL_SIZE)[:n]

    def instance(self, spec: SourceInstanceSpec) -> _SourceInstance:
        return self.instance_cls(spec, self.task_cfg, num_episodes=self.num_episodes)

    def instances(self, specs: Sequence[SourceInstanceSpec]) -> Iterator[_SourceInstance]:
        for spec in specs:
            yield self.instance(spec)

    def manifest(self) -> dict[str, Any]:
        from .orbit_bridge import manifest as bridge_manifest  # noqa: PLC0415

        out = {
            "task": self.task,
            "task_cfg": dict(self.task_cfg),
            "num_episodes": self.num_episodes,
            "eval_pool_size": self.EVAL_POOL_SIZE,
            **bridge_manifest(),
        }
        if self.task == "minesweeper":
            out["pinned_opening"] = list(PINNED_OPENING)  # D-C1
        if self.task == "wordle":
            import nltk  # noqa: PLC0415

            # D-C3: the candidate list *is* the corpus, so its version is part of
            # instance identity in a way no other ORBIT task's is.
            out["nltk_version"] = nltk.__version__
            out["vocabulary_size"] = self.capacity()
        return out


class MinesweeperTask(_SourceTask):
    task = "minesweeper"
    instance_cls = MinesweeperInstance
    config = MINESWEEPER_CONFIG
    EVAL_POOL_SIZE = 256

    def _content(self, env: Any) -> tuple[str, dict[str, Any]]:
        """The mine layout the pinned opening produces (D-C1).

        The opening has to be *played* to learn it -- mines do not exist until the
        first reveal -- so the pool build plays one move per draw. No model calls;
        the env is the generator.
        """
        env.step(_boxed_reveal(*PINNED_OPENING))
        inner = env._env
        mines = sorted(
            (r, c)
            for r in range(inner.rows)
            for c in range(inner.cols)
            if inner.grid[r][c] == -1
        )
        hidden_safe = sum(
            1
            for r in range(inner.rows)
            for c in range(inner.cols)
            if inner.grid[r][c] != -1 and not inner.revealed[r][c]
        )
        content = ";".join(f"{r},{c}" for r, c in mines)
        return content, {"mines": [list(m) for m in mines], "min_turns": hidden_safe}


class WordleTask(_SourceTask):
    task = "wordle"
    instance_cls = WordleInstance
    config = WORDLE_CONFIG

    #: D-C3. 192 five-letter candidates total, so the eval prefix cannot be 256
    #: and a 512-instance train pool does not exist. 60 leaves 132 for training.
    EVAL_POOL_SIZE = 60

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ensure_wordle_corpus()
        super().__init__(*args, **kwargs)

    def _content(self, env: Any) -> tuple[str, dict[str, Any]]:
        secret = str(env._env.secret_word)
        return secret, {"secret_length": len(secret), "min_turns": 1}

    def capacity(self) -> Optional[int]:
        inner = _gem_adapter(self.task_cfg)._env
        inner.reset(seed=0)
        return len({w for w in inner.available_words})


def assert_disjoint(
    train: Sequence[SourceInstanceSpec], eval_: Sequence[SourceInstanceSpec]
) -> None:
    """Assert the pools share no instance content. Zero tolerance, as Mastermind.

    A repeated Wordle secret between pools is the answer itself, and a repeated
    Minesweeper layout is the mine map -- neither is the tolerated
    same-layout-different-seed case Maze reports through a leakage split.
    """
    shared = {s.content_hash for s in train} & {s.content_hash for s in eval_}
    if shared:
        raise AssertionError(f"{len(shared)} instance(s) appear in both pools, expected 0")


#: Run scripts select a source by name.
SOURCE_TASKS: dict[str, type[_SourceTask]] = {
    "minesweeper": MinesweeperTask,
    "wordle": WordleTask,
}

__all__ = [
    "EPISODES_PER_INSTANCE", "PINNED_OPENING", "MINESWEEPER_CONFIG", "WORDLE_CONFIG",
    "SourceInstanceSpec", "MinesweeperInstance", "WordleInstance", "ensure_wordle_corpus",
    "MinesweeperTask", "WordleTask", "SOURCE_TASKS", "assert_disjoint",
    "GemUnavailable",
]
