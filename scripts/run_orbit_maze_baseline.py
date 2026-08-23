#!/usr/bin/env python
"""Phase 2: the no-bank Maze baseline.

ORBIT's own protocol -- cumulative cross-episode chat within an instance, no
skill bank ever -- run on our model (gpt-oss-120b, D7). This is the same-model
reference line the HExA arm is diffed against; ORBIT's Table 2 stays a cited
cross-model reference and is never differenced directly (RECON §2, D3).

    # server up first:  sbatch scripts/serve_gpt_oss_vllm.sh
    python scripts/run_orbit_maze_baseline.py --n 50 --out results/orbit_maze_baseline

    # plumbing check, no server, oracle plays optimally:
    python scripts/run_orbit_maze_baseline.py --n 3 --stub --out /tmp/smoke

Serial by construction (D5): ORBIT's env seeding is global, so concurrent
instances are not independently reproducible. Writes ``runs.jsonl`` (one full
transcript per instance) and ``summary.json`` (metrics, leaked/clean split, run
manifest).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hexa_core.tokens import TOKENIZER_NOTE  # noqa: E402
from hexa_orbit import oracle, pools, report  # noqa: E402
from hexa_orbit.actor import ActorConfig, GenerationFailed, InstanceRun, MazeActor  # noqa: E402
from hexa_orbit.adapter import MazeTask  # noqa: E402


class OracleStub:
    """Plays the known-map optimum. Exercises the whole path without a server.

    Not a baseline and not a bound anyone should quote -- it knows the map, which
    no real agent does on episode 1 (see :mod:`hexa_orbit.oracle`). It exists to
    prove the plumbing end to end before GPU time is spent.
    """

    def __init__(self, instance):
        self.instance = instance
        self.queue: list[str] = []

    def generate_structured(self, *, messages, **_):
        from hexa_core.llm.base import Generation

        if not self.queue:
            self.queue = oracle.instance_optimal_actions(self.instance)
        action = self.queue.pop(0)
        return Generation.from_parts("", oracle.boxed(action), finish_reason="stop")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50, help="eval instances (prefix of the pool)")
    parser.add_argument("--out", type=Path, default=Path("results/orbit_maze_baseline"))
    parser.add_argument("--train-size", type=int, default=512, help="pool used for the leakage split")
    parser.add_argument("--model", default=ActorConfig.model)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=ActorConfig.temperature)
    parser.add_argument("--top-p", type=float, default=ActorConfig.top_p)
    parser.add_argument("--max-response-tokens", type=int, default=ActorConfig.max_response_tokens)
    parser.add_argument("--reasoning-effort", default=ActorConfig.reasoning_effort)
    parser.add_argument("--seed", type=int, default=None, help="per-call sampling seed")
    parser.add_argument("--stub", action="store_true", help="oracle stub; no server needed")
    parser.add_argument(
        "--abort-after-errors",
        type=int,
        default=5,
        help="stop the run once this many instances error (0 = never abort)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ActorConfig(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        top_p=args.top_p,
        max_response_tokens=args.max_response_tokens,
        reasoning_effort=args.reasoning_effort,
        seed=args.seed,
    )

    task = MazeTask()
    eval_specs = task.eval_specs(args.n)
    train_specs = task.train_specs(args.train_size)
    pools.assert_disjoint(train_specs, eval_specs)
    leaked_ids = {s.instance_id for s in pools.leakage_split(eval_specs, train_specs)["leaked"]}
    print(
        f"{len(eval_specs)} eval instances; "
        f"{len(leaked_ids & {s.instance_id for s in eval_specs})} share a layout with the "
        f"{len(train_specs)}-instance train pool"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    runs_path = args.out / "runs.jsonl"
    runs: list[InstanceRun] = []
    errored: list[InstanceRun] = []
    started = time.monotonic()

    with runs_path.open("w", encoding="utf-8") as handle:
        for position, spec in enumerate(eval_specs, start=1):
            instance = task.instance(spec)
            leaked = spec.instance_id in leaked_ids
            actor = MazeActor(
                config,
                client=OracleStub(instance) if args.stub else None,
            )
            try:
                run = actor.run_instance(instance, leaked=leaked)
            except GenerationFailed as exc:
                run = InstanceRun(
                    instance_id=spec.instance_id,
                    dataset_seed=spec.dataset_seed,
                    resolved_seed=spec.resolved_seed,
                    shortest_path_len=spec.shortest_path_len,
                    leaked=leaked,
                    steps=[],
                    messages=[],
                    metrics={},
                    episode_successes=[],
                    episode_lengths=[],
                    success_count=0,
                    added_tokens=0,
                    token_counter="",
                    error=f"{type(exc).__name__}: {exc}",
                )
                errored.append(run)
                print(f"[{position}/{len(eval_specs)}] {spec.instance_id} ERROR: {exc}")
                if args.abort_after_errors and len(errored) >= args.abort_after_errors:
                    print(f"aborting: {len(errored)} errors reached --abort-after-errors")
                    handle.write(json.dumps(run.to_dict()) + "\n")
                    break
            else:
                runs.append(run)
                print(
                    f"[{position}/{len(eval_specs)}] {spec.instance_id} "
                    f"R={run.success_count}/3 eps={run.episode_successes} "
                    f"steps={len(run.steps)} spl={spec.shortest_path_len}"
                    f"{' LEAKED' if leaked else ''}"
                )
            handle.write(json.dumps(run.to_dict()) + "\n")
            handle.flush()

    aggregated = report.aggregate(runs, errored=errored)
    title = f"Maze no-bank baseline ({'ORACLE STUB' if args.stub else config.model})"
    summary = {
        "arm": "no_bank_baseline_stub" if args.stub else "no_bank_baseline",
        "actor": config.to_dict(),
        "stub": args.stub,
        "n_requested": args.n,
        "wall_clock_s": round(time.monotonic() - started, 1),
        "report": aggregated,
        "manifest": task.manifest(),
        "token_note": TOKENIZER_NOTE,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print(report.format_summary(aggregated, title=title))
    print(f"\nwrote {runs_path} and {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
