#!/usr/bin/env python
"""HExA Design B on Mastermind: distil a skill bank, then evaluate it frozen.

Mirrors `run_orbit_maze_hexa.py`. The loop itself is task-agnostic -- it drives
whatever `task` object it is handed -- so what differs is the level name, the
teacher prompts, and the trajectory loader, all installed by
`mastermind_teacher.install()`.

The number this arm has to beat is measured, but read the scoring note first.
ORBIT's `_is_episode_success` is `terminated and reward > 0`, and GEM's Mastermind
reward is black_pegs/4, so an episode that exhausts its three guesses with one peg
correct is flagged a success. That inflates the flagged rate 3.51x over the rate at
which the code is actually cracked. Score with `scripts/score_mastermind.py`.

On true solves (reward 4/4), the gpt-oss-120b no-bank baseline is ep3 0.270 /
any 0.273 -- not the 0.508 / 0.723 the flagged metric reports. A single hand-written
consistency rule reaches ep3 0.320 (+5.1, paired n=256, p=0.19) while losing 1.09
turns per instance to a 2048-token cap it no longer runs under. Whether a *distilled*
bank finds that in transcripts is the whole question.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hexa_orbit import loop, report  # noqa: E402
from hexa_orbit.actor import ActorConfig, MazeActor  # noqa: E402
from hexa_orbit.mastermind import MastermindTask, assert_disjoint  # noqa: E402
from hexa_orbit import mastermind_teacher  # noqa: E402

logger = logging.getLogger("mastermind_hexa")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--train-size", type=int, default=80)
    ap.add_argument("--eval-size", type=int, default=40)
    ap.add_argument("--out", type=Path,
                    default=Path("results/orbit_mastermind_hexa"))
    ap.add_argument("--model", default=ActorConfig.model)
    ap.add_argument("--teacher-model", default=ActorConfig.model)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--concurrency", type=int, default=16)
    # Maze used 70-109 of 2048 tokens a turn and never truncated. Mastermind
    # deduction is far longer -- 654 tokens a turn unaided, 947 with a rule
    # injected -- and hits the 2048 cap on 6.1% / 15.3% of turns. A truncated turn
    # emits no parseable guess, so it burns one of only three guesses in the
    # episode. The measured +14.5 ep3 gain from a single injected rule was achieved
    # while losing 15% of its turns that way.
    ap.add_argument("--max-response-tokens", type=int, default=4096)
    ap.add_argument("--injection-budget-tokens", type=int,
                    default=loop.INJECTION_BUDGET_TOKENS)
    ap.add_argument("--max-trajs", type=int, default=6)
    ap.add_argument("--reward-bin", action="store_true",
                    help="select trajectories by outcome class instead of the stock "
                         "`paired` selector. `paired` cannot work here -- an instance "
                         "is played once, so no same-instance pair ever forms and it "
                         "degrades to sorting by reward, which sent an identical "
                         "profile every round (4 episode-1 solves, 2 total failures) "
                         "and never once showed the evolver an instance that failed "
                         "episode 1 and then solved it.")
    ap.add_argument("--admission", action="store_true")
    ap.add_argument("--admission-trials", type=int, default=3)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    mastermind_teacher.install(max_trajs=args.max_trajs,
                               reward_bin_selection=args.reward_bin)

    config = ActorConfig(model=args.model, base_url=args.base_url,
                         max_response_tokens=args.max_response_tokens)
    task = MastermindTask()
    train_specs = task.train_specs(args.train_size)
    eval_specs = task.eval_specs(args.eval_size)
    assert_disjoint(train_specs, eval_specs)

    needed = args.rounds * args.batch_size
    if needed > len(train_specs):
        logger.warning("--rounds x --batch-size = %d exceeds the %d-instance "
                       "training pool; the run will stop early",
                       needed, len(train_specs))

    def actor_factory(guidelines, instance=None):
        return MazeActor(config, guidelines=guidelines)

    started = time.monotonic()
    result = loop.run_hexa_loop(
        task, train_specs, actor_factory, args.out,
        rounds=args.rounds, batch_size=args.batch_size,
        teacher_model=args.teacher_model,
        admission_enabled=args.admission,
        admission_trials=args.admission_trials,
        injection_budget_tokens=args.injection_budget_tokens,
        concurrency=args.concurrency,
        evolver_v2=True,
        teacher_install=lambda: mastermind_teacher.install(max_trajs=args.max_trajs,
                               reward_bin_selection=args.reward_bin),
        resume=not args.no_resume,
        manifest={"actor": config.to_dict(), "teacher_model": args.teacher_model,
                  **task.manifest()},
    )

    print()
    for record in result.rounds:
        print(f"round {record.round_index}: skills {record.n_skills_before}"
              f"->{record.n_skills_after}  bins {record.reward_bins}"
              f"  injected {record.injection['tokens']}tok")

    if args.skip_eval:
        print(f"\ntrained only; wrote {args.out}")
        return 0

    next_round, bank, _ = loop.resume_state(args.out)
    if bank is None:
        print("\nno bank produced; nothing to evaluate")
        return 1

    summary = loop.frozen_eval(
        task, eval_specs, actor_factory, bank, args.out / "eval_frozen",
        train_specs=train_specs,
        injection_budget_tokens=args.injection_budget_tokens,
        concurrency=args.concurrency,
    )
    print()
    print(report.format_summary(summary["report"],
                                title=f"Mastermind HExA, frozen bank ({config.model})"))
    print(f"\nwall clock {time.monotonic() - started:.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
