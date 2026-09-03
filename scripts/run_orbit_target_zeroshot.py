#!/usr/bin/env python
"""Phase 4: evaluate a synthesised bank on its target task, zero-shot.

The bank is frozen and was built from source-task banks alone -- the target's
instances have never been played by anything that fed it. This script never
constructs the target's *training* pool, so no target instance can reach the bank
even accidentally.

**Episode 1 is the number.** ORBIT's meta-trained policy generalises through
weights but still opens every unseen instance where a base model would; a bank
that lifts episode-1 success on a task the agent has never played is a claim their
architecture cannot make about itself. Episodes 2 and 3 are reported, but a gain
that appears only there is within-instance adaptation, which Design B already
measures.

    python scripts/run_orbit_target_zeroshot.py --target maze --run-index 0 \
        --bank results/design_c/maze_from_minesweeper/run0/skill_bank_xl_maze.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hexa_orbit import loop, report  # noqa: E402
from hexa_orbit.actor import ActorConfig, MazeActor  # noqa: E402
from skillrl.core.skill_bank import SkillBank  # noqa: E402

RUN_SEED_STRIDE = 100_000


def _task(target: str, eval_size: int | None):
    if target == "maze":
        from hexa_orbit.adapter import MazeTask  # noqa: PLC0415

        task = MazeTask()
        return task, task.eval_specs(eval_size) if eval_size else task.eval_specs()
    from hexa_orbit.mastermind import MastermindTask  # noqa: PLC0415

    task = MastermindTask()
    return task, task.eval_specs(eval_size or MastermindTask.EVAL_POOL_SIZE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=("maze", "mastermind"))
    ap.add_argument("--bank", type=Path, required=True,
                    help="synthesised bank; pass --bank none for the no-bank arm")
    ap.add_argument("--run-index", type=int, default=0, choices=(0, 1, 2))
    ap.add_argument("--eval-size", type=int, default=None)
    ap.add_argument("--bank-level", default=None,
                    help="the level key retrieval reads out of the bank. Defaults "
                         "to --target, which is right for a synthesised bank. Set "
                         "it to the SOURCE level to run the raw-dump ablation: the "
                         "source bank injected verbatim, with no relevance "
                         "filtering, no stripping of source-specific bindings and "
                         "no re-grounding. If that matches the synthesised bank, "
                         "Phase 3's whole contribution is unearned -- which is a "
                         "result to report, not to bury.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=ActorConfig.model)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-response-tokens", type=int, default=4096)
    ap.add_argument("--injection-budget-tokens", type=int,
                    default=loop.INJECTION_BUDGET_TOKENS)
    args = ap.parse_args()

    task, eval_specs = _task(args.target, args.eval_size)
    bank = None if str(args.bank).lower() == "none" else SkillBank.load(args.bank)

    # `render_block` retrieves by `loop.level()`, so the key must name whichever
    # bank is being injected -- the target for a synthesised bank, the source for
    # the raw dump.
    loop.set_level(args.bank_level or args.target)

    cfg = ActorConfig(model=args.model, base_url=args.base_url,
                      max_response_tokens=args.max_response_tokens,
                      seed=RUN_SEED_STRIDE * args.run_index)

    def actor_factory(guidelines, instance=None):
        return MazeActor(cfg, guidelines=guidelines)

    started = time.monotonic()
    summary = loop.frozen_eval(
        task, eval_specs, actor_factory, bank, args.out,
        # Deliberately empty: there is no target training pool in Design C, and
        # passing one would invent a leakage split that does not exist.
        train_specs=(),
        injection_budget_tokens=args.injection_budget_tokens,
        concurrency=args.concurrency,
    )
    (args.out / "manifest.json").write_text(json.dumps({
        "arm": ("no-bank" if bank is None
                else "raw-dump" if args.bank_level and args.bank_level != args.target
                else "zero-shot"),
        "bank_level": args.bank_level or args.target,
        "target": args.target, "run_index": args.run_index,
        "bank": None if bank is None else str(args.bank),
        "target_trajectories_in_bank": 0,
        "actor": cfg.to_dict(), **task.manifest(),
    }, indent=2))

    print()
    print(report.format_summary(
        summary["report"],
        title=f"{args.target} zero-shot, run{args.run_index} "
              f"({'synthesised bank' if bank else 'no bank'})"))
    print(f"\nwall clock {time.monotonic() - started:.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
