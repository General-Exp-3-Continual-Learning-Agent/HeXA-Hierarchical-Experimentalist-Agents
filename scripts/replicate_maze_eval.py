#!/usr/bin/env python
"""Re-run the Maze frozen eval against an already-trained bank.

The Maze arms are n=1 per arm: one bank, one eval. The Mastermind experience is
the warning -- a single eval pair there showed +0.175 on episode 3, and three
replicates removed it entirely. Maze's n=256 makes each run far tighter than
Mastermind's n=40, but the *bank* is still a single draw and the eval is still a
single sample of an unseeded actor.

`ActorConfig.seed` defaults to `None` and `VLLMClient` omits `seed` from the
request body when unset, so re-running the same 256 layouts draws new
trajectories rather than replaying the old ones.

This calls `loop.frozen_eval` -- the same function `run_orbit_maze_hexa.py` calls
after training -- with the same task, eval specs, actor config and bank, writing
to a fresh directory. Defaults reproduce `orbit_maze_hexa_v3`'s recorded actor
config; pass `--max-response-tokens` etc. only to deliberately change the arm.

Usage
-----
    python scripts/replicate_maze_eval.py \\
        --bank results/orbit_maze_hexa_v3/round9/bank.json \\
        --out results/orbit_maze_hexa_v3/eval_rep1 \\
        --model Qwen/Qwen3-8B --concurrency 40
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hexa_orbit import loop, pools, report  # noqa: E402
from hexa_orbit.actor import ActorConfig, MazeActor  # noqa: E402
from hexa_orbit.adapter import MazeTask  # noqa: E402
from skillrl.core.skill_bank import SkillBank  # noqa: E402

logger = logging.getLogger("maze_replicate")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True,
                    help="Frozen bank.json from the final training round.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Fresh output directory; must not be an existing eval.")
    ap.add_argument("--train-size", type=int, default=512,
                    help="Must match the training run. Eval and train are drawn "
                         "independently for Maze, but `assert_disjoint` is still "
                         "checked, and the bank's provenance refers to this pool.")
    ap.add_argument("--eval-size", type=int, default=pools.EVAL_POOL_SIZE)
    ap.add_argument("--model", default=ActorConfig.model)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--concurrency", type=int, default=40)
    # Defaults below mirror `orbit_maze_hexa_v3/run.json`'s actor block. Changing
    # one makes this a different arm, not a replicate of that one.
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-response-tokens", type=int, default=2048)
    ap.add_argument("--reasoning-effort", default="low",
                    help="'' for models with no reasoning channel; becomes None.")
    ap.add_argument("--transcript-mode", default="final")
    ap.add_argument("--truncation-scoring", default="exclude")
    ap.add_argument("--injection-budget-tokens", type=int,
                    default=loop.INJECTION_BUDGET_TOKENS)
    ap.add_argument("--per-episode-injection", action="store_true")
    ap.add_argument("--inject-top-k", type=int, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = ActorConfig(
        model=args.model, base_url=args.base_url, temperature=args.temperature,
        top_p=args.top_p, max_response_tokens=args.max_response_tokens,
        reasoning_effort=args.reasoning_effort or None,
        transcript_mode=args.transcript_mode,
        truncation_scoring=args.truncation_scoring,
    )
    task = MazeTask()
    train_specs = task.train_specs(args.train_size)
    eval_specs = task.eval_specs(args.eval_size)
    pools.assert_disjoint(train_specs, eval_specs)

    bank = SkillBank.load(args.bank)
    logger.info("bank %s: %s", args.bank, bank)
    logger.info("actor %s", config.to_dict())

    def actor_factory(guidelines, instance=None):
        return MazeActor(config, guidelines=guidelines)

    started = time.monotonic()
    summary = loop.frozen_eval(
        task, eval_specs, actor_factory, bank, args.out,
        train_specs=train_specs,
        injection_budget_tokens=args.injection_budget_tokens,
        concurrency=args.concurrency,
        per_episode_injection=args.per_episode_injection,
        inject_top_k=args.inject_top_k,
    )
    print()
    print(report.format_summary(summary["report"],
                                title=f"Maze frozen bank, replicate ({config.model})"))
    print(f"\nwall clock {time.monotonic() - started:.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
