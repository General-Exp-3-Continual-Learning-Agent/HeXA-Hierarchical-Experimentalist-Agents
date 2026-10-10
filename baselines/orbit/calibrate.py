"""MemRL's similarity constants, measured on an ORBIT instance pool. Run first.

MemRL's Phase-A threshold is dataset-dependent.  From the paper's appendix:

    "The similarity threshold delta is adaptive to the dataset density;
    specifically, we determine delta by calculating the pairwise cosine
    similarity distribution of task descriptions within each benchmark and
    selecting the threshold at the top 20% quantile."

and its similarity z-normalization uses precomputed dataset statistics, not
per-pool ones.  All three constants therefore have to be measured on ORBIT
intents.  This is `baselines/calibrate.py` for ORBIT keys; it reports the same
spread diagnostic, which is the thing to read before trusting a MemRL run.

    python -m baselines.orbit.calibrate --task maze
    python -m baselines.orbit.calibrate --task mastermind

**The degenerate branch is not a formality here.**  Mastermind intents are
identical across instances (see `intent.py`), so every pairwise cosine is 1.0
and the top-20% quantile is 1.0 as well.  Phase A tests ``sim > delta``
strictly, so a delta of 1.0 admits *nothing* and MemRL would inject empty blocks
on every instance -- a run that completes, reports numbers, and is silently just
the no-bank arm.  When the spread is degenerate this lowers delta below the
observed floor so Phase A admits the store, and says so loudly.  The similarity
channel then carries no information and Phase B reduces to Q-ranking, which is
the honest reading of what MemRL can do on a task whose instances are
indistinguishable until played.

There is no `numeric` key mode, unlike InterPhyre.  A numeric key there is the
scene geometry, which the agent can see; the ORBIT equivalent would be the maze
layout or the secret code, which it cannot.  See `intent.py`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

from baselines.calibrate import quantile
from baselines.memory.store import cosine
from baselines.orbit.embedding import (
    HASHING,
    WORKING_DEFAULT,
    build_embedder,
    embedder_note,
)
from baselines.orbit.intent import (
    MODES,
    IntentCache,
    cache_path,
    default_mode,
)
from baselines.orbit.tasks import TASKS, build_task

#: Margin below the observed minimum used when the pool is degenerate, so
#: Phase A's strict ``sim > delta`` admits rather than rejecting everything.
DEGENERATE_DELTA_MARGIN = 0.01

#: Spread below which the similarity channel is treated as carrying no signal.
DEGENERATE_STD = 1e-6


def calibration_path(cache_dir: Path, task: str, mode: str, embedding_model: str) -> Path:
    slug = embedding_model.replace("/", "__").replace(":", "_")
    return Path(cache_dir) / f"orbit_calibration_{task}_{mode}_{slug}.json"


def calibrate(
    task_name: str,
    *,
    train_size: int,
    eval_size: int,
    embedding_model: str = WORKING_DEFAULT,
    cache_dir: Path,
    intent_mode: str | None = None,
) -> dict:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    intent_mode = intent_mode or default_mode(task_name)

    orbit_task = build_task(task_name)
    # Both pools: MemRL calibrates "within each benchmark", and the benchmark is
    # every instance the method will ever meet, not just the ones it learns from.
    train_specs = orbit_task.train_specs(train_size)
    eval_specs = orbit_task.eval_specs(eval_size)
    orbit_task.assert_disjoint(train_specs, eval_specs)
    specs = list(train_specs) + list(eval_specs)

    intents = IntentCache(
        cache_path(cache_dir, task_name, intent_mode), intent_mode
    )
    print(f"  Building {len(specs)} intents ({intent_mode}) ...")
    texts = [i.text for i in intents.warm(orbit_task, specs)]
    distinct = len(set(texts))

    print(f"  Embedding {len(texts)} intents with {embedding_model} ...")
    print(f"    {embedder_note(embedding_model)}")
    embedder = build_embedder(embedding_model, cache_dir=cache_dir)
    vectors = embedder.embed(texts)

    sims = [
        cosine(vectors[i], vectors[j])
        for i, j in itertools.combinations(range(len(vectors)), 2)
    ]
    sims_sorted = sorted(sims)
    mean = statistics.fmean(sims)
    std = statistics.pstdev(sims) if len(sims) > 1 else 0.0
    delta = quantile(sims_sorted, 0.80)

    degenerate = std < DEGENERATE_STD
    if degenerate:
        # See the module docstring: the paper's delta would reject the whole
        # store, turning MemRL into the no-bank arm without saying so.
        delta = sims_sorted[0] - DEGENERATE_DELTA_MARGIN

    result = {
        "task": task_name,
        "embedding_model": embedding_model,
        "intent_mode": intent_mode,
        "n_train": len(train_specs),
        "n_eval": len(eval_specs),
        "num_instances": len(specs),
        "num_distinct_intents": distinct,
        "num_pairs": len(sims),
        "intent_chars_mean": statistics.fmean(len(t) for t in texts),
        "sim_mean": mean,
        "sim_std": std,
        "sim_min": sims_sorted[0],
        "sim_max": sims_sorted[-1],
        "quantiles": {
            f"p{int(q * 100)}": quantile(sims_sorted, q)
            for q in (0.05, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95)
        },
        "degenerate": degenerate,
        # What MemRL's merge-on-write will do to the bank, before a single
        # rollout is spent. `add_memories` folds a new memory into an existing
        # one whose intent is this similar, so a threshold most pairs already
        # exceed means the store stops growing and every instance retrieves the
        # same handful of records. The InterPhyre port found 99-100% of pairs
        # above their configured 0.90; check this rather than rediscovering it
        # from a finished run's store size.
        "frac_pairs_above": {
            f"{t:.2f}": sum(1 for v in sims if v >= t) / len(sims)
            for t in (0.85, 0.90, 0.95, 0.99)
        },
        "recommended": {
            "delta": delta,
            "sim_norm_mean": mean,
            # A zero std would make the z-score a division by zero; MemRL's own
            # normalizer guards with 1.0, and recording that keeps the manifest
            # honest about which branch ran.
            "sim_norm_std": std if std > DEGENERATE_STD else 1.0,
        },
    }

    out_path = calibration_path(cache_dir, task_name, intent_mode, embedding_model)
    out_path.write_text(json.dumps(result, indent=2))

    print(f"\n  {distinct} distinct intents over {len(specs)} instances")
    if distinct < len(specs):
        print(
            f"    ({len(specs) - distinct} instances share an episode-1 key with "
            f"another — retrieval cannot tell those apart)"
        )
    print(f"  Pairwise cosine similarity over {len(sims)} pairs:")
    print(f"    mean {mean:.4f}   std {std:.4f}   "
          f"min {sims_sorted[0]:.4f}   max {sims_sorted[-1]:.4f}")
    for name, value in result["quantiles"].items():
        print(f"    {name}: {value:.4f}")

    if degenerate:
        print(
            "\n  [!] DEGENERATE: every instance embeds to the same key, so the "
            "similarity\n      channel carries no information. delta has been "
            f"lowered to {delta:.4f} so Phase A\n      admits the store instead "
            "of rejecting it; MemRL then selects on Q alone, and\n      Episodic "
            "RAG retrieves an arbitrary past success and failure. Expected on\n"
            "      Mastermind -- the instances differ only in a hidden code. "
            "Report it; do not\n      try to fix it with a different embedder."
        )
    elif std < 0.01:
        print(
            "\n  [!] Similarities are nearly degenerate (std < 0.01): Phase A "
            "will be close to\n      all-or-nothing and the z-normalized "
            "similarity term is mostly noise."
        )

    print("\n  Pairs at or above MemRL's merge-on-write thresholds:")
    for threshold, fraction in result["frac_pairs_above"].items():
        print(f"    >= {threshold}: {fraction:.1%}")
    above_default = result["frac_pairs_above"]["0.99"]
    if above_default > 0.5:
        print(
            f"\n  [!] {above_default:.0%} of pairs are at or above 0.99, the "
            f"merge-on-write default.\n      MemRL will fold most writes into "
            f"existing records and the bank will barely grow.\n      Pass "
            f"--add-similarity-threshold 1.01 to disable merging, or accept it "
            f"and report\n      the store size."
        )

    print(f"\n  Recommended MemRL constants for {task_name} ({intent_mode}):")
    print(f"    --delta {result['recommended']['delta']:.4f}")
    print(f"    --sim-norm-mean {result['recommended']['sim_norm_mean']:.4f}")
    print(f"    --sim-norm-std {result['recommended']['sim_norm_std']:.4f}")
    print(f"\n  Written to {out_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate MemRL similarity constants on ORBIT intents"
    )
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--train-size", type=int, default=80)
    parser.add_argument("--eval-size", type=int, default=40)
    parser.add_argument(
        "--embedding-model", default=WORKING_DEFAULT,
        help=f"text embedder, or {HASHING!r} for a dependency-free character "
             f"n-gram key. See baselines/orbit/embedding.py for why the CORE "
             f"paper's jina-v2 is not the default here",
    )
    parser.add_argument(
        "--intent-mode", default=None, choices=MODES,
        help="default is per task: salient for maze, observation for "
             "mastermind. See baselines/orbit/intent.py:DEFAULT_MODE_BY_TASK",
    )
    parser.add_argument("--cache-dir", default="baselines/data/cache")
    args = parser.parse_args()

    calibrate(
        args.task,
        train_size=args.train_size,
        eval_size=args.eval_size,
        embedding_model=args.embedding_model,
        cache_dir=Path(args.cache_dir),
        intent_mode=args.intent_mode,
    )


if __name__ == "__main__":
    main()
