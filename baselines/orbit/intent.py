"""Retrieval keys for ORBIT instances, built only from observable information.

Both methods key memory on an *intent*: the text that gets embedded and compared
against past instances.  On InterPhyre that is the scene geometry, which the
agent can read off the toolkit.  ORBIT gives no such thing.  What distinguishes
one instance from another is the maze layout or the secret code, and the agent
is never shown either -- discovering it is the task.  So the intent is built
from the **episode-1 opening observation** and nothing else: exactly what the
agent has in hand at the moment retrieval happens.

Keying on privileged state instead would not be a stronger baseline, it would be
a different experiment.  A memory indexed by maze layout retrieves the route for
a map the agent has not seen, and one indexed by the secret code retrieves the
answer.  Neither says anything about whether episodic memory helps.

Two modes:

`observation` (default)
    The whole opening observation with ORBIT's ``[Episode 1] New episode
    begins.`` header removed.  Literally what the agent reads.

`salient`
    Only the instance-varying part, the analogue of InterPhyre's `geometry`
    mode.  On Maze that is the start cell and the four adjacent cells; the rest
    of the prompt is the same prose on every instance and contributes no
    retrieval signal, only cosine inflation.

**What this yields, measured, and why it is reported rather than fixed.**

*Maze* keys carry a weak but real signal: the start position and its four
neighbours differ across instances.  They describe one cell of a 6x6-to-7x7
maze, so two instances with the same key can still have entirely different
layouts -- retrieval here is genuinely low-information, and that is a property
of the task under an observable-only key, not a defect of the port.

*Mastermind* keys are **identical across every instance**.  The opening prompt
is fixed text, and instances differ only in a hidden code.  Cosine similarity is
therefore 1.0 for every pair, and the consequences are exact and worth stating
before any run: Episodic RAG falls back to retrieving an arbitrary past success
and an arbitrary past failure, and MemRL's Phase A admits the whole store, so
its z-scored similarity term is constant and selection collapses to pure
Q-ranking.  That last case is not a degradation -- it is MemRL's value-driven
retrieval with the similarity channel carrying no information, which is a clean
and reportable measurement of what the value channel alone buys.

`ORBIT's one apparent exception is removed deliberately.`  The Mastermind prompt
ends with a randomly drawn example, ``for example, \\boxed{4 3 2}``, which does
differ per instance.  It is stripped from both modes.  Measured over the
40-instance eval pool it matches the secret code 0/40 times and shares 0.425
positions with it against 0.5 expected by chance, so it is uncorrelated with the
answer -- but it is a deterministic function of the dataset seed, which makes it
an *identity fingerprint*.  Leaving it in would give the embedder something to
discriminate on while telling it nothing about which past instance is relevant,
turning a visibly degenerate key into an invisibly arbitrary one.  Silent
nonsense is worse than loud degeneracy.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MODES = ("observation", "salient")
DEFAULT_MODE = "observation"

#: Which mode each task defaults to, and why they differ.
#:
#: Measured over 100 Maze instances (60 train + 40 eval) with the hashing key:
#:
#:     mode          mean    std     min     distinct keys
#:     observation   0.982   0.0095  0.923   59 / 100
#:     salient       0.945   0.0294  0.758   59 / 100
#:
#: Same information, three times the spread.  The maze prompt is ~90% fixed
#: prose, and embedding it drags every pair towards 1.0, which is the regime
#: where MemRL's Phase A is all-or-nothing and its similarity z-score is noise.
#: This is the same result `baselines/memory/intent.py` records for InterPhyre's
#: `full` versus `geometry`, and it lands the same way: strip the boilerplate.
#:
#: Mastermind keeps `observation` because its `salient` key is the empty string.
#: Both are degenerate -- there is nothing observable to key on -- but a constant
#: non-empty key embeds to a constant non-zero vector and pairwise cosine 1.0,
#: whereas an empty one embeds to zeros and every similarity is 0.0.  The first
#: is legible as "no discrimination"; the second looks like a broken embedder.
#:
#: Note the distinct-key count is 59/100 either way: 41% of Maze instances share
#: their episode-1 observation with another instance, because the key is one
#: cell plus its four neighbours and a 6x6-to-7x7 maze has few such patterns.
#: That is a real limit on what observable-only retrieval can resolve here, and
#: it is not something a better embedder fixes.
DEFAULT_MODE_BY_TASK = {"maze": "salient", "mastermind": "observation"}


def default_mode(task: str) -> str:
    return DEFAULT_MODE_BY_TASK.get(task, DEFAULT_MODE)

#: ORBIT's per-episode header (`adapter.format_observation`). Constant, so it is
#: dropped from every key.
_HEADER_RE = re.compile(r"^\[Episode \d+\][^\n]*\n?", re.MULTILINE)

#: The randomly drawn example guess in Mastermind's prompt. See the module
#: docstring: a seed fingerprint, uncorrelated with the code, removed on purpose.
_MASTERMIND_EXAMPLE_RE = re.compile(
    r"for example,\s*\\boxed\{[^}]*\}", re.IGNORECASE
)

#: The instance-varying clause of a maze observation: where the agent stands and
#: what the four neighbouring cells are. Everything else in the prompt is fixed.
_MAZE_SALIENT_RE = re.compile(
    r"(you are at position \(\d+,\s*\d+\).*?)(?:\s*Output your next move|\Z)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class Intent:
    task: str
    instance_id: str
    text: str
    mode: str = DEFAULT_MODE


def strip_boilerplate(observation: str, task: str, mode: str) -> str:
    """Reduce a raw opening observation to its retrieval key."""
    text = _HEADER_RE.sub("", str(observation or "")).strip()
    if task == "mastermind":
        text = _MASTERMIND_EXAMPLE_RE.sub("for example, \\\\boxed{...}", text)
    if mode == "salient":
        if task == "maze":
            match = _MAZE_SALIENT_RE.search(text)
            text = match.group(1).strip() if match else text
        else:
            # Nothing in a Mastermind prompt varies once the example is masked.
            # An empty key is the honest rendering of that.
            text = ""
    return text.strip()


def build_intent(orbit_task, spec, mode: str = DEFAULT_MODE) -> Intent:
    """Construct the intent for one instance by opening it and reading turn 1.

    Costs one environment reset and no model call.  For Maze this generates the
    maze -- deterministic in `spec.dataset_seed`, and serialised by the adapter's
    generation lock -- and then throws it away; the instance the run plays is
    constructed fresh.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown intent mode {mode!r}; expected one of {MODES}")
    observation = orbit_task.instance(spec).reset()
    return Intent(
        task=orbit_task.name,
        instance_id=spec.instance_id,
        text=strip_boilerplate(observation, orbit_task.name, mode),
        mode=mode,
    )


def seed_key(instance_id: str) -> int:
    """A stable int for an instance, for `MemoryRecord.seed`.

    `MemoryRecord.seed` is typed `int` and `exclude_seeds` filters on it, but an
    ORBIT instance is identified by a hex digest.  The same truncation
    `run_orbit_maze_hexa.paired_seed` uses gives a stable, collision-free-in
    -practice int over pools of a few hundred; the real id travels in
    `MemoryRecord.meta["instance_id"]` and is what every report keys on.
    """
    return int(str(instance_id)[:8], 16)


class IntentCache:
    """Disk-backed intent texts, keyed by `task:mode:instance_id`.

    Worth caching: a Maze intent costs a full maze generation, and the same
    pools are opened by the calibrator, the memory-building pass and the eval
    pass of every run.
    """

    def __init__(self, path: Optional[Path] = None, mode: str = DEFAULT_MODE):
        if mode not in MODES:
            raise ValueError(f"Unknown intent mode {mode!r}; expected one of {MODES}")
        self.mode = mode
        self.path = Path(path) if path else None
        self._mem: dict[str, str] = {}
        if self.path and self.path.exists():
            try:
                self._mem = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self._mem = {}

    def get(self, orbit_task, spec) -> Intent:
        key = f"{orbit_task.name}:{self.mode}:{spec.instance_id}"
        if key not in self._mem:
            self._mem[key] = build_intent(orbit_task, spec, self.mode).text
            self._flush()
        return Intent(
            task=orbit_task.name,
            instance_id=spec.instance_id,
            text=self._mem[key],
            mode=self.mode,
        )

    def warm(self, orbit_task, specs) -> list[Intent]:
        """Build every intent up front.

        The eval pass plays instances concurrently, and building an intent means
        resetting an environment.  On Maze that touches the global RNG, so doing
        it lazily inside a worker would interleave with another instance's
        generation -- the exact corruption `adapter._GENERATION_LOCK` exists to
        prevent, reintroduced one layer up.  Warm the cache serially instead.
        """
        return [self.get(orbit_task, spec) for spec in specs]

    def _flush(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._mem, indent=2))


def cache_path(cache_dir: Path, task: str, mode: str) -> Path:
    return Path(cache_dir) / f"orbit_intents_{task}_{mode}.json"
