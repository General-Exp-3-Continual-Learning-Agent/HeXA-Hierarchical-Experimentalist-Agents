#!/usr/bin/env python
"""No-bank baseline on Mastermind -- the arm every later claim is paired against.

Mirrors `run_orbit_maze_baseline.py`: the same actor, the same record schema, no
guidelines injected, so the system prompt is byte-identical to ORBIT's.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hexa_orbit.actor import ActorConfig, MazeActor  # noqa: E402
from hexa_orbit.mastermind import MastermindTask  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--model", default=ActorConfig.model)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--out", type=Path, default=Path("results/orbit_mastermind_baseline"))
    ap.add_argument("--max-response-tokens", type=int, default=4096,
                    help="Mastermind reasoning is long: 654 tok/turn unaided and "
                         "947 with a rule injected, hitting the old 2048 cap on "
                         "6.1%% / 15.3%% of turns. A truncated turn emits no guess "
                         "and burns one of three.")
    ap.add_argument("--guidelines-file", type=Path, default=None,
                    help="inject this text as the skill block. A go/no-go probe: it "
                         "tests whether ANY injected rule beats no-bank on this task, "
                         "before building the training loop that would produce one.")
    args = ap.parse_args()

    task = MastermindTask()
    specs = task.eval_specs(args.n)
    args.out.mkdir(parents=True, exist_ok=True)
    partial = args.out / "runs.partial.jsonl"
    partial.write_text("")

    guidelines = (args.guidelines_file.read_text().strip()
                  if args.guidelines_file else None)
    if guidelines:
        print(f"injecting {len(guidelines)} chars of guidelines")

    cfg = ActorConfig(model=args.model, base_url=args.base_url,
                      max_response_tokens=args.max_response_tokens)
    started = time.monotonic()
    done = [0]

    def play(spec):
        instance = task.instance(spec)
        run = MazeActor(cfg, guidelines=guidelines).run_instance(instance, leaked=False)
        rec = run.to_dict()
        done[0] += 1
        print(f"[{done[0]}/{len(specs)}] {spec.instance_id} R={run.success_count}/3",
              flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        records = list(pool.map(play, specs))

    with partial.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    (args.out / "runs.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )

    n = len(records)
    def rate(i):
        return sum(1 for r in records
                   if len(r["episode_successes"]) > i and r["episode_successes"][i]) / n
    anyv = sum(1 for r in records if any(r["episode_successes"])) / n
    print(f"\nn={n}  ep1={rate(0):.3f}  ep2={rate(1):.3f}  ep3={rate(2):.3f}  any={anyv:.3f}")
    print(f"mean R = {sum(r['success_count'] for r in records)/n:.2f}/3")
    print(f"wall clock {time.monotonic() - started:.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
