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
from hexa_orbit import loop, oracle, pools, report, trajectory  # noqa: E402
from hexa_orbit.actor import ActorConfig, InstanceRun, MazeActor  # noqa: E402
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
    parser.add_argument(
        "--reasoning-effort", default=ActorConfig.reasoning_effort,
        help="pass '' for models without a reasoning channel (Qwen2.5, Llama, ...). "
             "Sending this to a model that does not understand it makes vLLM reject "
             "the request or ignore it silently, depending on version.",
    )
    parser.add_argument("--seed", type=int, default=None, help="per-call sampling seed")
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="instances in flight at once. Maze generation is serialised by a lock "
             "(the only place ORBIT touches the global RNG), so results are "
             "bit-identical to --concurrency 1; see DECISIONS.md D5. With Qwen3 "
             "thinking enabled an instance takes ~3.5 min, so 8-16 is the difference "
             "between a 15-hour and a 1-2 hour 256-instance arm.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="skip instances already present in runs.partial.jsonl. A 256-instance "
             "arm is ~2h, long enough that a walltime kill or a server death is a "
             "normal event rather than an exception.",
    )
    parser.add_argument(
        "--transcript-mode", choices=["final", "raw"], default="final",
        help="what the assistant turn carries. 'final' = answer channel only "
             "(~15 tokens/turn); 'raw' = reasoning + answer, which is what ORBIT's "
             "own agent stores (~800 tokens/turn with thinking on). See "
             "ActorConfig.transcript_mode -- this materially changes the numbers.",
    )
    parser.add_argument(
        "--trajectory-token-budget", type=int, default=None,
        help="total completion tokens per instance. ORBIT uses 31744 as a "
             "TRAJECTORY budget (data.max_response_length); ours is per-call and "
             "cannot bind. Set 31744 to impose theirs.",
    )
    parser.add_argument(
        "--truncation-scoring", choices=["exclude", "orbit"], default="exclude",
        help="how to score episodes an instance never reached. 'exclude' omits them; "
             "'orbit' counts them as failures, reproducing "
             "MultiEpisodeEnv.get_metrics's early-termination branch.",
    )
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
        # '' from the CLI means 'this model has no reasoning channel'. It must
        # become None, not stay an empty string: VLLMClient omits the argument
        # only when it is None, and would otherwise send reasoning_effort="".
        reasoning_effort=args.reasoning_effort or None,
        transcript_mode=args.transcript_mode,
        trajectory_token_budget=args.trajectory_token_budget,
        truncation_scoring=args.truncation_scoring,
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
    partial_path = args.out / "runs.partial.jsonl"

    # Resume: the partial file is the record of what already ran. Completed
    # instances are dropped from the work list and their runs carried forward, so a
    # resumed arm is the union of both passes rather than a second partial one.
    done_runs: list[InstanceRun] = []
    if args.resume and partial_path.exists():
        done_runs = [r for r in trajectory.read_runs(partial_path) if r.error is None]
        done_ids = {r.instance_id for r in done_runs}
        before = len(eval_specs)
        eval_specs = [s for s in eval_specs if s.instance_id not in done_ids]
        print(f"resuming: {len(done_runs)} instances already done, "
              f"{len(eval_specs)} of {before} remaining")
        # Re-seed the partial file with what we are keeping, since play_batch
        # truncates it. Without this, a second interruption would lose pass one.
        trajectory.write_runs(partial_path, done_runs)
        _carry = partial_path.read_text()
    else:
        _carry = ""
    runs: list[InstanceRun] = []
    errored: list[InstanceRun] = []
    started = time.monotonic()

    # Delegate to the shared play_batch so the baseline and HExA arms cannot drift
    # in how they drive instances -- and so both inherit the reset-lock discipline
    # that makes --concurrency safe (D5).
    def actor_factory(guidelines, instance=None):
        actor = MazeActor(config, guidelines=guidelines)
        if args.stub and instance is not None:
            actor._client = OracleStub(instance)
        return actor

    def progress(position: int, run: InstanceRun) -> None:
        if run.error:
            print(f"[{position}/{len(eval_specs)}] {run.instance_id} ERROR: {run.error}")
        else:
            print(
                f"[{position}/{len(eval_specs)}] {run.instance_id} "
                f"R={run.success_count}/3 eps={run.episode_successes} "
                f"steps={len(run.steps)} spl={run.shortest_path_len}"
                f"{' LEAKED' if run.leaked else ''}"
            )

    runs, errored = loop.play_batch(
        task, eval_specs, actor_factory, None,
        leaked_ids=leaked_ids, on_instance=progress, concurrency=args.concurrency,
        # Crash safety: appended per instance as it completes. A 256-instance arm is
        # hours long, so losing it to a walltime kill is not acceptable.
        partial_path=partial_path,
        carry_over=_carry,
    )
    runs = done_runs + runs
    trajectory.write_runs(runs_path, runs + errored)

    aggregated = report.aggregate(runs, errored=errored)
    title = f"Maze no-bank baseline ({'ORACLE STUB' if args.stub else config.model})"
    summary = {
        "arm": "no_bank_baseline_stub" if args.stub else "no_bank_baseline",
        "actor": config.to_dict(),
        "stub": args.stub,
        "n_requested": args.n,
        "wall_clock_s": round(time.monotonic() - started, 1),
        "concurrency": args.concurrency,
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
