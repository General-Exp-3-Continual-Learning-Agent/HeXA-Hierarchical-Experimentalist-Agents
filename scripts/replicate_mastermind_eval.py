#!/usr/bin/env python
"""Re-run the Mastermind frozen eval against an already-trained bank.

Exists because the eval is underpowered by construction, not by accident.
`perm(6, 3) = 120` codes is the whole space at ORBIT's published config, and the
train/eval partition leaves 40 held-out instances. The first paired comparison
came out +0.175 on episode 3 with p = 0.19 (14 discordant pairs to 7) -- the right
direction, but a three-instance swing would erase it.

More instances are not available. Replicates are: `ActorConfig.seed` defaults to
`None` and `VLLMClient` omits `seed` from the request body when unset, so the
actor samples freshly at temperature 0.6 / top_p 0.95 on every call. Re-running
the same 40 codes therefore draws new trajectories rather than replaying the old
ones, and k replicates give 40k paired observations plus an honest estimate of
within-instance variance.

This calls `loop.frozen_eval` -- the exact function `run_orbit_mastermind_hexa.py`
calls at the end of training -- with the same task, eval specs, actor config and
bank, writing to a fresh directory. Nothing about the arm changes but the output
path.

Usage
-----
    python scripts/replicate_mastermind_eval.py \\
        --bank results/orbit_mastermind_hexa_qwen8/round7/bank.json \\
        --out results/orbit_mastermind_hexa_qwen8/eval_rep1 \\
        --model Qwen/Qwen3-8B --max-response-tokens 16384 --concurrency 40
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hexa_orbit import loop, report  # noqa: E402
from hexa_orbit.actor import ActorConfig, MazeActor  # noqa: E402
from hexa_orbit.mastermind import MastermindTask, assert_disjoint  # noqa: E402
from hexa_orbit import mastermind_teacher  # noqa: E402
from skillrl.core.skill_bank import SkillBank  # noqa: E402

logger = logging.getLogger("mastermind_replicate")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True,
                    help="Frozen bank.json from the final training round.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Fresh output directory; must not be an existing eval.")
    ap.add_argument("--train-size", type=int, default=80,
                    help="Must match the training run -- eval specs are the front "
                         "of the same stream, so a different value would re-partition "
                         "the pool and change which codes are held out.")
    ap.add_argument("--eval-size", type=int, default=40)
    ap.add_argument("--model", default=ActorConfig.model)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--concurrency", type=int, default=40)
    ap.add_argument("--max-response-tokens", type=int, default=16384)
    ap.add_argument("--injection-budget-tokens", type=int, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # The teacher is never called here, but `install()` also registers the
    # Mastermind trajectory loader that `frozen_eval` uses to render blocks.
    mastermind_teacher.install()

    config = ActorConfig(model=args.model, base_url=args.base_url,
                         max_response_tokens=args.max_response_tokens)
    task = MastermindTask()
    train_specs = task.train_specs(args.train_size)
    eval_specs = task.eval_specs(args.eval_size)
    assert_disjoint(train_specs, eval_specs)

    bank = SkillBank.load(args.bank)
    logger.info("bank %s: %s", args.bank, bank)

    def actor_factory(guidelines, instance=None):
        return MazeActor(config, guidelines=guidelines)

    started = time.monotonic()
    kwargs = {}
    if args.injection_budget_tokens is not None:
        kwargs["injection_budget_tokens"] = args.injection_budget_tokens
    summary = loop.frozen_eval(
        task, eval_specs, actor_factory, bank, args.out,
        train_specs=train_specs, concurrency=args.concurrency, **kwargs,
    )
    print()
    print(report.format_summary(summary["report"],
                                title=f"Mastermind frozen bank, replicate ({config.model})"))
    print(f"\nwall clock {time.monotonic() - started:.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
