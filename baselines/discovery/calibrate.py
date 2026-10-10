"""MemRL's dataset-dependent constants, measured on an arm's own intent set.

MemRL's Phase A admits memories with ``sim > delta`` and Phase B z-scores the
similarity with *precomputed dataset* statistics. The paper sets delta from the
data:

    "The similarity threshold delta is adaptive to the dataset density;
    specifically, we determine delta by calculating the pairwise cosine
    similarity ... and taking the top-20% quantile."

That prescription breaks on a within-world DiscoverPhysics arm, and it breaks
loudly rather than subtly. Every episode of one world has an identical intent,
so every pairwise cosine is exactly 1.0: the 80th-percentile delta is 1.0,
``sim > delta`` is false for every record, and MemRL would silently inject
nothing and become the no-bank arm. The same degeneracy makes the similarity
z-score a division by zero.

Both are handled the way `baselines/orbit/calibrate.py` handles them, for the
same reason and with the same constants: when the spread is degenerate, delta
drops below the observed minimum so Phase A admits the store, and the z-score
std falls back to 1.0 (MemRL's own guard). Which branch ran is recorded in the
arm's manifest, because "MemRL scored the baseline" and "MemRL never injected
anything" look identical in a results table.

No rollouts, no model calls: intents are pure text from `config.WORLD_FACTS`.
"""

from __future__ import annotations

import itertools
import statistics
from typing import Sequence

from baselines.discovery.intent import intent_for
from baselines.memory.store import cosine

#: Margin below the observed minimum when the pool is degenerate, so Phase A's
#: strict ``sim > delta`` admits rather than rejecting everything.
DEGENERATE_DELTA_MARGIN = 0.01

#: Spread below which the similarity channel carries no information.
DEGENERATE_STD = 1e-6


def quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[idx]


def calibrate(worlds: Sequence[str], embedder, *, eval_world: str) -> dict:
    """Constants for an arm whose memory holds `worlds` and is queried by `eval_world`.

    `worlds` is the multiset of worlds the memory will contain, so a
    within-world arm passes one world repeated: that is what makes the
    degeneracy visible here instead of at retrieval time.

    **delta is set from the QUERY-to-memory similarities, not memory-to-memory.**
    The paper takes the top-20% quantile of "the pairwise cosine similarity",
    which on its own benchmarks is the same distribution either way, because a
    held-out instance is drawn from the same task as the memories. A
    leave-one-group-out DiscoverPhysics arm breaks that equivalence and does so
    silently: hubble's memory spans nine other worlds whose mutual similarity
    reaches 1.0 (yukawa and fractional share a world description verbatim), so
    the memory-to-memory 80th percentile is 0.987 -- while a hubble query's best
    match to any memory is 0.83. Phase A tests ``sim > delta``, so that delta
    admits nothing, every eval episode gets an empty block, and MemRL reports as
    the no-bank arm while looking like it ran. Delta is a recall threshold for
    queries, so it is calibrated on the similarities Phase A will actually see.
    Recorded as a deviation, with both distributions kept below.
    """
    texts = [intent_for(w) for w in worlds]
    vectors = [embedder.embed_one(t) for t in texts]
    query = embedder.embed_one(intent_for(eval_world))
    sims = [cosine(query, v) for v in vectors]
    if len(vectors) < 2:
        internal = [1.0]
    else:
        internal = [cosine(vectors[i], vectors[j])
                    for i, j in itertools.combinations(range(len(vectors)), 2)]
    order = sorted(sims)
    mean = statistics.fmean(sims)
    std = statistics.pstdev(sims) if len(sims) > 1 else 0.0
    delta = quantile(order, 0.80)

    degenerate = std < DEGENERATE_STD
    if degenerate:
        delta = order[0] - DEGENERATE_DELTA_MARGIN

    iorder = sorted(internal)
    return {
        "worlds": list(worlds),
        "eval_world": eval_world,
        # Kept alongside so a reader can see the two distributions differ, which
        # is the whole reason delta is taken from the query one.
        "internal_sim_mean": statistics.fmean(internal),
        "internal_sim_std": statistics.pstdev(internal) if len(internal) > 1 else 0.0,
        "internal_delta_p80": quantile(iorder, 0.80),
        "num_distinct_intents": len({t for t in texts}),
        "num_pairs": len(sims),
        "sim_mean": mean,
        "sim_std": std,
        "sim_min": order[0],
        "sim_max": order[-1],
        "degenerate": degenerate,
        # What merge-on-write will do to the bank before a rollout is spent.
        # A threshold most pairs already exceed means the store stops growing.
        # Merge-on-write acts on memory-to-memory similarity, so it is read off
        # the internal distribution, not the query one.
        "frac_pairs_above": {
            f"{t:.2f}": sum(1 for v in internal if v >= t) / len(internal)
            for t in (0.85, 0.90, 0.95, 0.99)
        },
        "n_admitted_at_delta": sum(1 for v in sims if v > delta),
        "recommended": {
            "delta": delta,
            "sim_norm_mean": mean,
            "sim_norm_std": std if std > DEGENERATE_STD else 1.0,
        },
    }


def report(cal: dict) -> None:
    print(f"  intents: {cal['num_distinct_intents']} distinct over "
          f"{len(cal['worlds'])} episodes, {cal['num_pairs']} pairs")
    print(f"  query->memory cosine : mean {cal['sim_mean']:.4f}  "
          f"std {cal['sim_std']:.4f}  min {cal['sim_min']:.4f}  "
          f"max {cal['sim_max']:.4f}")
    print(f"  memory<->memory      : mean {cal['internal_sim_mean']:.4f}  "
          f"std {cal['internal_sim_std']:.4f}  "
          f"(paper delta would be {cal['internal_delta_p80']:.4f})")
    print(f"  pairs >= 0.99 (merge-on-write): "
          f"{cal['frac_pairs_above']['0.99']:.0%}")
    r = cal["recommended"]
    print(f"  -> delta {r['delta']:.4f}  sim_norm_mean {r['sim_norm_mean']:.4f}  "
          f"sim_norm_std {r['sim_norm_std']:.4f}")
    if cal["degenerate"]:
        print("  ! DEGENERATE: every intent identical, so the similarity "
              "channel carries no\n    information. delta lowered below the "
              "minimum so Phase A admits the store\n    rather than injecting "
              "nothing; selection is by Q alone.")
    print(f"  Phase A admits {cal['n_admitted_at_delta']}/{len(cal['worlds'])} "
          f"memories for an {cal['eval_world']} query")
    if cal["n_admitted_at_delta"] == 0:
        print("  ! delta admits NOTHING: MemRL would inject an empty block on "
              "every episode\n    and silently report as the no-bank arm.")
    if cal["frac_pairs_above"]["0.99"] > 0.5:
        print("  ! most pairs exceed the merge threshold: the bank will "
              "collapse toward one\n    record per outcome class.")
