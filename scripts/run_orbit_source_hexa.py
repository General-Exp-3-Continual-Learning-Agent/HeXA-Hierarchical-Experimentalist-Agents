#!/usr/bin/env python
"""Phase 2: evolve a source skill bank on Minesweeper or Wordle, then freeze it.

The bank this produces is **input to Phase 3's cross-task synthesis**, not a
result. Its own held-out accuracy is reported only as a gate -- "is this bank
worth transferring?" -- and must not be quoted as a within-task claim, which is
Design B's territory (`Design_C_Orbit_Plan.md`, "Do not").

Mirrors `run_orbit_mastermind_hexa.py`; the loop is task-agnostic and everything
task-specific is installed by `source_teacher.install(level)`.

**Three runs, by design.** `--run-index` selects one of three replicates: it seeds
actor sampling and reshuffles the training order, leaving the pools identical so
the replicates are paired. The project has run n=1 per arm throughout; this is
the arm where that stops.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hexa_orbit import loop, report, source_teacher  # noqa: E402
from hexa_orbit.actor import ActorConfig, MazeActor  # noqa: E402
from hexa_orbit.source_tasks import SOURCE_TASKS, assert_disjoint  # noqa: E402

logger = logging.getLogger("source_hexa")

RUN_SEED_STRIDE = 100_000

#: Wordle's vocabulary is 192 words and 60 are reserved for eval (D-C3), so its
#: train pool cannot be Mastermind's 512. Minesweeper has no such ceiling.
DEFAULT_TRAIN_SIZE = {"minesweeper": 400, "wordle": 132}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(SOURCE_TASKS))
    ap.add_argument("--run-index", type=int, default=0, choices=(0, 1, 2))
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--train-size", type=int, default=None)
    ap.add_argument("--eval-size", type=int, default=None)
    ap.add_argument("--max-turns", type=int, default=None,
                    help="override the task's per-episode turn limit. Wordle at "
                         "ORBIT's 10 is saturated (any=1.00, zero failures in every "
                         "round), so its evolver never sees a contrast; 5 restores "
                         "both headroom and a failure class. Does not change the "
                         "pools -- the secret is drawn before max_turns is read.")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--model", default=ActorConfig.model)
    ap.add_argument("--teacher-model", default=ActorConfig.model,
                    help="actor and teacher are the same model; HExA is "
                         "self-improvement, not distillation from a stronger one")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-response-tokens", type=int, default=4096)
    ap.add_argument("--injection-budget-tokens", type=int,
                    default=loop.INJECTION_BUDGET_TOKENS)
    ap.add_argument("--max-trajs", type=int, default=12)
    ap.add_argument("--no-reward-bin", action="store_true",
                    help="fall back to the stock `paired` selector. It cannot work "
                         "here -- an instance is played once per round, so no "
                         "same-instance pair ever forms -- and starved the "
                         "Mastermind v2 evolver of every slow success it had.")
    ap.add_argument("--admission", action="store_true")
    ap.add_argument("--admission-trials", type=int, default=3)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    reward_bin = not args.no_reward_bin

    def install():
        source_teacher.install(args.task, max_trajs=args.max_trajs,
                               reward_bin_selection=reward_bin)

    install()

    config = ActorConfig(model=args.model, base_url=args.base_url,
                         max_response_tokens=args.max_response_tokens,
                         seed=RUN_SEED_STRIDE * args.run_index)
    task_cls = SOURCE_TASKS[args.task]
    task_cfg = dict(task_cls.config)
    if args.max_turns:
        task_cfg["max_turns"] = args.max_turns
    task = task_cls(task_cfg)
    train_size = args.train_size or DEFAULT_TRAIN_SIZE[args.task]
    train_specs = task.train_specs(train_size)
    eval_specs = task.eval_specs(args.eval_size)
    assert_disjoint(train_specs, eval_specs)

    # Replicates differ in the order training instances arrive, so each round's
    # batch is a different draw, while the pool is identical.
    random.Random(args.run_index).shuffle(train_specs)

    needed = args.rounds * args.batch_size
    if needed > len(train_specs):
        logger.warning("--rounds x --batch-size = %d exceeds the %d-instance "
                       "training pool; the run will stop early", needed, len(train_specs))

    out = args.out or Path(f"results/orbit_{args.task}_hexa/run{args.run_index}")

    def actor_factory(guidelines, instance=None):
        return MazeActor(config, guidelines=guidelines)

    started = time.monotonic()
    result = loop.run_hexa_loop(
        task, train_specs, actor_factory, out,
        rounds=args.rounds, batch_size=args.batch_size,
        teacher_model=args.teacher_model,
        admission_enabled=args.admission,
        admission_trials=args.admission_trials,
        injection_budget_tokens=args.injection_budget_tokens,
        concurrency=args.concurrency,
        evolver_v2=True,
        teacher_install=install,
        resume=not args.no_resume,
        manifest={"arm": "hexa", "run_index": args.run_index,
                  "actor": config.to_dict(), "teacher_model": args.teacher_model,
                  "loop": {"rounds": args.rounds, "batch_size": args.batch_size,
                           "train_size": train_size, "max_trajs": args.max_trajs,
                           "reward_bin": reward_bin},
                  **task.manifest()},
    )

    print()
    for record in result.rounds:
        print(f"round {record.round_index}: skills {record.n_skills_before}"
              f"->{record.n_skills_after}  bins {record.reward_bins}"
              f"  injected {record.injection['tokens']}tok")

    # A truncated evolution response makes `evolve_skill_bank` fall back to the
    # previous bank with a printed warning and no error, which in the round line
    # is indistinguishable from convergence -- Mastermind v3 legitimately left its
    # bank unchanged in 17 of 20 rounds. The first Design C Phase 2 lost every
    # Minesweeper round and about half of Wordle's that way, unnoticed until the
    # logs were read by hand. So count it here and say so loudly.
    unchanged = sum(1 for r in result.rounds
                    if r.n_skills_before == r.n_skills_after)
    if result.rounds and unchanged == len(result.rounds):
        print(f"\nWARNING: all {unchanged} rounds left the bank unchanged. Check the "
              f"log for '[Teacher TRUNCATED]' -- at HEXA_TEACHER_MAX_TOKENS="
              f"{os.getenv('HEXA_TEACHER_MAX_TOKENS', 'unset')} the evolution "
              f"response may not fit, in which case this arm learned nothing.")
    elif unchanged > len(result.rounds) // 2:
        print(f"\nNOTE: {unchanged}/{len(result.rounds)} rounds left the bank "
              f"unchanged; check the log for '[Teacher TRUNCATED]'.")

    if args.skip_eval:
        print(f"\ntrained only; wrote {out}")
        return 0

    _next_round, bank, _ = loop.resume_state(out)
    if bank is None:
        print("\nno bank produced; nothing to evaluate")
        return 1

    summary = loop.frozen_eval(
        task, eval_specs, actor_factory, bank, out / "eval_frozen",
        train_specs=train_specs,
        injection_budget_tokens=args.injection_budget_tokens,
        concurrency=args.concurrency,
    )
    print()
    print(report.format_summary(
        summary["report"],
        title=f"{args.task} HExA gate, frozen bank, run{args.run_index} ({config.model})"))
    print(f"\nwall clock {time.monotonic() - started:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
