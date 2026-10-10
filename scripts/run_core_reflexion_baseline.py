#!/usr/bin/env python3
"""Reflexion (K trials/item) on one CORE task -- no build phase, eval only.

Neither CORE's own repo nor the CORE paper defines a Reflexion baseline for
these four tasks; this is a new baseline for this project, analogous to
InterPhyre's own Reflexion (`docs/BASELINES.md`, K=2 trials) and to ORBIT's
episode-reset arm (`baselines/orbit/reflexion.py`), adapted to CORE's
single-turn structure -- see `baselines/core/reflexion.py`'s module docstring
for exactly how, and why "matched budget" does not mean 900 rollouts here:
Reflexion has no training phase or persistent memory, so its entire cost is
K trials x 100 eval items, run n=3 times (independent draws, matching Episodic
RAG/MemRL's eval replicate count).

Usage
-----
    python scripts/run_core_reflexion_baseline.py \\
        --task matchstick --k-trials 3 \\
        --base-url http://gpu017:8000/v1 \\
        --output-dir results_core_baselines/reflexion/matchstick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from baselines.core.reflexion_runner import run_reflexion_eval  # noqa: E402
from hexa_core.llm.vllm_client import VLLMClient  # noqa: E402
from hexa_core.tasks import TASK_NAMES, make_task  # noqa: E402

DEFAULT_N_TRAIN = 60  # eval split is identical regardless of n_train; kept for parity with the memory baselines' task construction
DEFAULT_K_TRIALS = 3
DEFAULT_EVAL_RUNS = 3


def endpoint_from_json(path: Path) -> tuple[str, dict]:
    data = json.loads(path.read_text())
    return data["base_url"], data


def assert_endpoint_serves(base_url: str, model: str) -> None:
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            listing = json.loads(response.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ERROR: cannot reach {url} ({type(exc).__name__}: {exc}).")
    served = [entry.get("id") for entry in listing.get("data", [])]
    if model not in served:
        raise SystemExit(f"ERROR: {base_url} is NOT serving {model!r}. It advertises: {served}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=TASK_NAMES)
    p.add_argument("--k-trials", type=int, default=DEFAULT_K_TRIALS)
    p.add_argument("--eval-runs", type=int, default=DEFAULT_EVAL_RUNS)
    p.add_argument("--output-dir", required=True, type=Path)

    p.add_argument("--model", default="openai/gpt-oss-120b")
    p.add_argument("--base-url", default=None)
    p.add_argument("--endpoint-json", default="logs/vllm_endpoint_latest.json")
    p.add_argument("--max-tokens", type=int, default=60000)
    p.add_argument("--reasoning-effort", default="high")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--reflection-max-tokens", type=int, default=8192)
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing output dir's reflexion_run*.jsonl")
    return p


def main() -> int:
    args = build_parser().parse_args()

    base_url = args.base_url
    if base_url is None:
        path = Path(args.endpoint_json)
        if not path.is_file():
            print(f"ERROR: no endpoint JSON at {path}. Start the server or pass --base-url.")
            return 2
        base_url, _ = endpoint_from_json(path)
    assert_endpoint_serves(base_url, args.model)

    out_dir = Path(args.output_dir)
    existing = list(out_dir.glob("reflexion_run*.jsonl")) if out_dir.exists() else []
    if existing and not args.force:
        raise SystemExit(
            f"ERROR: {out_dir} already has {[p.name for p in existing]}. "
            f"Reflexion has no build phase to reuse (unlike --skip-build on the "
            f"memory baselines) -- there is nothing partial to recover, so this "
            f"guards against silently overwriting a completed run. Remove the "
            f"files or pass --force."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    client = VLLMClient(
        model=args.model, base_url=base_url,
        max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort,
    )
    print("=" * 78)
    print(f"  reflexion (K={args.k_trials}) on CORE/{args.task}")
    print(f"  endpoint    : {base_url}   model={args.model}")
    print(f"  eval runs   : {args.eval_runs}")
    print(f"  concurrency : {args.concurrency}")
    print(f"  output      : {out_dir}")
    print("=" * 78)

    print("\n[probe] verifying the server produces comparable output ...")
    evidence = client.probe(strict=False, differential=True)
    print(f"  reasoning_parser={evidence.get('reasoning_parser')} "
          f"effort_verified={evidence.get('effort_verified')}")
    if not evidence.get("reasoning_parser"):
        print("ERROR: no reasoning channel; raw_text would drop the chain of thought. Aborting.")
        return 3
    (out_dir / "probe.json").write_text(json.dumps(evidence, indent=2, default=str))

    task = make_task(args.task, DEFAULT_N_TRAIN)
    (out_dir / "task_manifest.json").write_text(json.dumps(task.manifest(), indent=2, default=str))

    def generate_structured(**kwargs):
        kwargs.setdefault("max_new_tokens", args.reflection_max_tokens)
        return client.generate_structured(**kwargs)

    manifest = {
        "method": "reflexion",
        "task": args.task,
        "k_trials": args.k_trials,
        "eval_runs": args.eval_runs,
        "agent_model": args.model,
        "reflection_max_tokens": args.reflection_max_tokens,
        "deviations": [
            f"actor is {args.model}; neither CORE nor its paper defines a "
            f"Reflexion baseline for these tasks -- this is a new baseline for "
            f"this project, adapted from InterPhyre's own Reflexion (K=2) and "
            f"ORBIT's episode-reset arm to CORE's single-turn structure "
            f"(baselines/core/reflexion.py).",
            "no training/build phase and no persistent memory across items "
            "(the method has none); 'matched budget' therefore refers to "
            "K trials/eval item, not HeXA/RAG/MemRL's 900 training rollouts.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    started = time.time()
    eval_summaries = []
    for run_index in range(args.eval_runs):
        print(f"\n[reflexion run {run_index}] K={args.k_trials} trials x 100 held-out items ...")
        summary = run_reflexion_eval(
            task, client.generate, generate_structured,
            k_trials=args.k_trials, run_index=run_index,
            concurrency=args.concurrency, output_dir=out_dir,
        )
        eval_summaries.append(summary)
        flag = "" if summary["reliable"] else f"  *** UNRELIABLE: {summary['n_empty_actor_completions']} empty completions ***"
        print(
            f"  trial1={summary['trial1_acc']:.3f}  "
            + "  ".join(f"trial{k}={summary[f'trial{k}_acc']:.3f}" for k in range(2, args.k_trials + 1))
            + f"  any={summary['any_acc']:.3f}  mean_trials={summary['mean_trials_used']:.2f}"
              f"  empty_reflections={summary['n_empty_reflections']}{flag}"
        )

    reliable = [s for s in eval_summaries if s["reliable"]]
    unreliable = [s for s in eval_summaries if not s["reliable"]]

    def agg(key: str) -> tuple[float, float]:
        vals = [s[key] for s in reliable]
        if not vals:
            return 0.0, 0.0
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1) if len(vals) > 1 else 0.0
        return mean, (var ** 0.5) / (len(vals) ** 0.5)

    metrics = ["trial1_acc"] + [f"trial{k}_acc" for k in range(2, args.k_trials + 1)] + ["any_acc"]
    aggregated = {m: dict(zip(("mean", "sem"), agg(m))) for m in metrics}

    run_summary = {
        "manifest": manifest,
        "eval_runs": eval_summaries,
        "aggregated": aggregated,
        "n_reliable_eval_runs": len(reliable),
        "n_unreliable_eval_runs": len(unreliable),
        "elapsed_s": time.time() - started,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2, default=str))

    print("\n" + "=" * 78)
    if unreliable:
        print(f"  WARNING: {len(unreliable)}/{len(eval_summaries)} eval run(s) excluded "
              f"from the aggregate: run_index {[s['run_index'] for s in unreliable]}.")
    print(f"  reflexion / {args.task}: "
          + "  ".join(f"{m}={aggregated[m]['mean']:.3f}+/-{aggregated[m]['sem']:.3f}" for m in metrics)
          + f"  (n={len(reliable)} reliable eval runs / {len(eval_summaries)} requested)")
    print(f"  wall clock: {(time.time() - started) / 3600:.2f} h -> {out_dir}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
