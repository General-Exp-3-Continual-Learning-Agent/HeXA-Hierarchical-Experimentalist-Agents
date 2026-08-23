#!/usr/bin/env python
"""Materialise the Maze instance pools and their leakage partition.

Writes an auditable record of *which instances* a run used, before any model is
called. The leaked/clean partition has to exist ahead of the Phase 2 baseline
rather than being retrofitted (DECISIONS.md, the section following D1), so this
script is a prerequisite for Phase 2 and not a reporting convenience.

    python scripts/build_orbit_maze_pools.py --out results/orbit_maze_pools

Writes ``pools.json`` (every instance, both splits, with identity and leakage
flags) and ``summary.json`` (the manifest plus aggregate counts).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hexa_orbit import pools  # noqa: E402
from hexa_orbit.adapter import MazeTask  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--eval-size", type=int, default=pools.EVAL_POOL_SIZE)
    parser.add_argument("--global-seed", type=int, default=pools.DEFAULT_GLOBAL_SEED)
    parser.add_argument("--out", type=Path, default=Path("results/orbit_maze_pools"))
    args = parser.parse_args()

    task = MazeTask(global_seed=args.global_seed)
    train = task.train_specs(args.train_size)
    eval_ = task.eval_specs(args.eval_size)

    # Fails loudly rather than reporting a contaminated pool as clean.
    pools.assert_disjoint(train, eval_)

    split = pools.leakage_split(eval_, train)
    leaked_ids = {s.instance_id for s in split["leaked"]}
    summary = pools.pool_summary(train, eval_)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pools.json").write_text(
        json.dumps(
            {
                "train": [s.to_dict() for s in train],
                "eval": [
                    {**s.to_dict(), "leaked": s.instance_id in leaked_ids}
                    for s in eval_
                ],
            },
            indent=2,
        )
    )
    (args.out / "summary.json").write_text(
        json.dumps({"manifest": task.manifest(), "pools": summary}, indent=2)
    )

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}/pools.json and {args.out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
