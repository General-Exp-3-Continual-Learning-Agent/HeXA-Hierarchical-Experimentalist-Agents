#!/usr/bin/env python
"""No-bank baseline on a Design C source task -- Phase 2's gate, not a headline.

Mirrors `run_orbit_mastermind_baseline.py`. Same actor, same record schema, no
guidelines injected, so the system prompt is byte-identical to ORBIT's.

**Three runs, by design.** Every ORBIT arm in this project so far is n=1, which
leaves no across-run variance to compare an effect against. `--run-index` selects
one of three independent replicates: it seeds the actor's sampling and shuffles
the order instances are played in, while the eval pool itself stays fixed so the
three runs -- and the HExA arm's three -- are paired on the same instances. Run
all three before reading anything into a difference.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hexa_orbit.actor import ActorConfig, MazeActor  # noqa: E402
from hexa_orbit.source_tasks import SOURCE_TASKS  # noqa: E402

#: Actor sampling seed for replicate *i*. Spaced so two runs never share a
#: per-instance seed via the offset the actor adds internally.
RUN_SEED_STRIDE = 100_000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(SOURCE_TASKS))
    ap.add_argument("--n", type=int, default=None,
                    help="eval instances; defaults to the task's full eval pool "
                         "(minesweeper 256, wordle 60 -- see D-C3)")
    ap.add_argument("--run-index", type=int, default=0, choices=(0, 1, 2))
    ap.add_argument("--max-turns", type=int, default=None,
                    help="override the task's per-episode turn limit. Wordle at "
                         "ORBIT's 10 is saturated (any=1.00, zero failures in every "
                         "round), so its evolver never sees a contrast; 5 restores "
                         "both headroom and a failure class. Does not change the "
                         "pools -- the secret is drawn before max_turns is read.")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--model", default=ActorConfig.model)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-response-tokens", type=int, default=4096)
    args = ap.parse_args()

    task_cls = SOURCE_TASKS[args.task]
    task_cfg = dict(task_cls.config)
    if args.max_turns:
        task_cfg["max_turns"] = args.max_turns
    task = task_cls(task_cfg)
    specs = task.eval_specs(args.n)
    out = args.out or Path(f"results/orbit_{args.task}_baseline/run{args.run_index}")
    out.mkdir(parents=True, exist_ok=True)
    partial = out / "runs.partial.jsonl"
    partial.write_text("")

    # Play order varies per replicate; membership does not.
    order = list(specs)
    random.Random(args.run_index).shuffle(order)

    cfg = ActorConfig(model=args.model, base_url=args.base_url,
                      max_response_tokens=args.max_response_tokens,
                      seed=RUN_SEED_STRIDE * args.run_index)
    started = time.monotonic()
    done = [0]
    lock_out = partial.open("a")

    def play(spec):
        instance = task.instance(spec)
        run = MazeActor(cfg, guidelines=None).run_instance(instance, leaked=False)
        rec = run.to_dict()
        done[0] += 1
        lock_out.write(json.dumps(rec) + "\n")
        lock_out.flush()
        print(f"[{done[0]}/{len(order)}] {spec.instance_id} R={run.success_count}/3",
              flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        records = list(pool.map(play, order))
    lock_out.close()

    (out / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (out / "manifest.json").write_text(json.dumps({
        "arm": "baseline", "run_index": args.run_index,
        "actor": cfg.to_dict(), **task.manifest(),
    }, indent=2))

    n = len(records)
    def rate(i):
        return sum(1 for r in records
                   if len(r["episode_successes"]) > i and r["episode_successes"][i]) / n
    anyv = sum(1 for r in records if any(r["episode_successes"])) / n
    print(f"\n{args.task} baseline run{args.run_index}  n={n}  "
          f"ep1 {rate(0):.3f} / ep2 {rate(1):.3f} / ep3 {rate(2):.3f} / any {anyv:.3f}")
    print(f"wall clock {time.monotonic() - started:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
