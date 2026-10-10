"""Frozen-eval an arbitrary bank on a source task's held-out pool.

`run_orbit_source_hexa.py` can only evaluate the bank its own run produced, which
makes one question unanswerable: *does the actor follow an injected bank at all?*
Qwen3-8B on Minesweeper reads the board correctly, deduces correctly, and then
spends the turn on a move that cannot win -- flagging, or re-revealing a cell the
board already shows as revealed. That is a goal mismatch, not a comprehension
failure, so it is the kind of thing a bank could in principle state away.

This runs any bank.json against the same eval pool the no-bank baseline used, so a
hand-written probe bank can measure the ceiling: if an explicit "flags never win,
only reveal cells showing '.'" does not move the flag rate, no teacher-written
bank will either, and the level is closed for this actor.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skillrl.core.skill_bank import SkillBank  # noqa: E402
from hexa_orbit import loop, report, source_teacher  # noqa: E402
from hexa_orbit.actor import ActorConfig, MazeActor  # noqa: E402
from hexa_orbit.source_tasks import SOURCE_TASKS  # noqa: E402

RUN_SEED_STRIDE = 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(SOURCE_TASKS))
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-index", type=int, default=0, choices=(0, 1, 2))
    ap.add_argument("--eval-size", type=int, default=None)
    ap.add_argument("--model", default=ActorConfig.model)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-response-tokens", type=int, default=16384)
    ap.add_argument("--injection-budget-tokens", type=int, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    source_teacher.install(args.task)

    config = ActorConfig(model=args.model, base_url=args.base_url,
                         max_response_tokens=args.max_response_tokens,
                         seed=RUN_SEED_STRIDE * args.run_index)
    task_cls = SOURCE_TASKS[args.task]
    task = task_cls(dict(task_cls.config))
    eval_specs = task.eval_specs(args.eval_size)

    bank = SkillBank.load(args.bank)
    print(f"bank: {args.bank}\n  {bank}")

    kwargs = {}
    if args.injection_budget_tokens is not None:
        kwargs["injection_budget_tokens"] = args.injection_budget_tokens
    summary = loop.frozen_eval(
        task, eval_specs, lambda guidelines, instance=None: MazeActor(config, guidelines=guidelines),
        bank, args.out, concurrency=args.concurrency, **kwargs)
    print()
    print(report.format_summary(
        summary["report"], title=f"{args.task} frozen bank {args.bank} ({config.model})"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
