"""The retrieval key for a DiscoverPhysics episode, and its one honest caveat.

Episodic RAG indexes rollouts "according to their problem embedding".  A
DiscoverPhysics problem is a *world*: the observable setup plus the controls the
agent may set.  `config.world_facts_with_controls` is exactly that text, it is
answer-free by construction (it is what HeXA already shows its teacher as the
level description), and it names no seed -- so it is the right key and it is the
same string HeXA's own machinery keys on.

**Within one world every episode therefore has an identical key.**  That is a
property of the benchmark, not of the embedder: a seed reshuffles observation
noise and leaves the hidden law alone, so there is nothing for a problem
embedding to separate.  Retrieval degenerates to "some past success and some
past failure from this world", and the tie-break in `baselines/discovery/store.py`
is what decides which.  Every run records `n_distinct_keys` and the similarity
spread so a reader can see this rather than infer it.

Across worlds -- the hubble transfer arm -- the key is doing real work: nine
training worlds have nine genuinely different setup descriptions.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DISCOVERY = Path(__file__).resolve().parents[2] / "discovery-agents"
if str(_DISCOVERY) not in sys.path:
    sys.path.insert(0, str(_DISCOVERY))

from hexa_discovery.config import world_facts_with_controls  # noqa: E402


def intent_for(world: str) -> str:
    """The embedded text for any episode of `world`.

    `include_base_prompt=False`: the base-prompt addendum is a note to the
    *teacher* about what the agent already knows, not part of the problem
    statement, and including it would make the key differ from the problem.
    """
    return world_facts_with_controls(world, include_base_prompt=False)


from hexa_discovery.config import BENCHMARK_WORLDS  # noqa: E402

#: Stable anonymous label per world, fixed by `BENCHMARK_WORLDS` order.
#:
#: A retrieved block must never print the world's real name. Several of them ARE
#: the answer -- "oscillator" states the time-dependence a within-world agent is
#: meant to discover, and "hubble", "yukawa", "ether" and "dark_matter" name
#: their own hidden laws. The agent's own system prompt never names its world,
#: so printing it in an injected memory would hand over the one thing
#: `traj_adapter`'s law-form redaction exists to withhold.
#:
#: HeXA solves this with `teacher_prompts_discovery.world_label`, which assigns
#: W1, W2, ... in first-seen order. That is per-process, and a memory store is
#: written once and read back in later runs, so the mapping is pinned to a fixed
#: world order here instead.
_WORLD_LABELS = {w: f"W{i}" for i, w in enumerate(BENCHMARK_WORLDS, 1)}


def world_label(world: str) -> str:
    """Anonymous, stable, answer-free display label for a world."""
    return _WORLD_LABELS.get(world, "W?")
