#!/usr/bin/env python3
"""Episodic RAG / MemRL on one CORE task, budget-matched to HeXA.

Neither baseline exists in CORE's released repo -- `hexa_core/CORE_ENV_RESULTS.md`
§6 reads their numbers off the paper, at 2050-3000 rollouts, on NVIDIA NIM/Cerebras.
This script is the local, provider-matched, budget-matched rerun: same task
adapter (`hexa_core.adapter.Problem`/`hexa_core.tasks.make_task`), same local
vLLM `gpt-oss-120b`, same 60-item train split and 100-item frozen eval split HeXA
uses, and the same total 900-rollout budget (`baselines/core/runner.py`'s module
docstring has the accounting: 900 = 15 visits x 60 train items, since neither
method has an analogue of HeXA's 300-call `b_q` baseline pass).

One build (n=1), three frozen eval passes against it (n=3, run-index 0/1/2),
matching the "n=1 build, n=3 eval" protocol requested for this comparison --
NOT HeXA's own n=3, which rebuilds the bank three times. `eval_run{0,1,2}.jsonl`
draw the identical seeds `passive_loop._frozen_eval` draws for HeXA's own
with-memory arm, so `results_core/*/run*/eval_no_memory.jsonl` is a valid,
already-computed paired no-memory control for every row this script produces --
nothing needs to be re-run to get the "vs no memory" comparison.

Usage
-----
    # 1. bring up the server exactly as for HeXA (scripts/serve_gpt_oss_vllm.sh)

    # 2. Episodic RAG on Matchstick
    python scripts/run_core_memory_baseline.py \\
        --task matchstick --method episodic_rag \\
        --base-url http://gpu013:8000/v1 \\
        --output-dir results_core_baselines/episodic_rag/matchstick

    # 3. MemRL on the same task (auto-calibrates delta/sim-norm on the train pool
    #    unless --delta etc. are all given explicitly)
    python scripts/run_core_memory_baseline.py \\
        --task matchstick --method memrl \\
        --base-url http://gpu013:8000/v1 \\
        --output-dir results_core_baselines/memrl/matchstick
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

from baselines.core.calibrate import calibrate_core_task  # noqa: E402
from baselines.core.intent import retrieval_text  # noqa: E402
from baselines.core.runner import CoreStreamConfig, build_memory, frozen_eval  # noqa: E402
from baselines.core.summarize import make_core_summarizer  # noqa: E402
from baselines.orbit.embedding import (  # noqa: E402
    HASHING as HASHING_EMBED_MODEL,
    PAPER_DEFAULT as PAPER_EMBED_MODEL,
    WORKING_DEFAULT as DEFAULT_EMBED_MODEL,
    build_embedder,
    embedder_note,
)
from baselines.memory.memrl import MemRL, MemRLConfig  # noqa: E402
from baselines.memory.rag import EpisodicRAG, RAGConfig  # noqa: E402
from baselines.memory.store import MemoryStore  # noqa: E402
from hexa_core.llm.vllm_client import VLLMClient  # noqa: E402
from hexa_core.tasks import TASK_NAMES, make_task  # noqa: E402

#: See --embedding-model's help text for the measurement behind the
#: tower-of-hanoi override.
DEFAULT_EMBEDDING_BY_TASK = {"tower-of-hanoi": HASHING_EMBED_MODEL}
#: See --add-similarity-threshold's / --novelty-threshold's help text.
DEFAULT_ADD_SIM_THRESHOLD_BY_TASK = {"tower-of-hanoi": 0.995}
DEFAULT_NOVELTY_THRESHOLD_BY_TASK = {"tower-of-hanoi": 0.99}

DEFAULT_N_TRAIN = 60
#: 900 total, matched to HeXA's own accounting (`hexa_core/CORE_ENV_RESULTS.md` §3):
#: 600 training + 300 b_q. Neither baseline here has a b_q analogue, so the whole
#: 900 goes to the build phase; see `baselines/core/runner.py`'s module docstring.
DEFAULT_BUDGET = 900
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
    p.add_argument("--method", required=True, choices=("episodic_rag", "memrl"))
    p.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN)
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                   help="total build-phase rollouts; visits/item = budget // n_train")
    p.add_argument("--eval-runs", type=int, default=DEFAULT_EVAL_RUNS)
    p.add_argument("--output-dir", required=True, type=Path)

    # -- serving, matched to HeXA's CORE config (CORE_ENV_RESULTS.md §1) --
    p.add_argument("--model", default="openai/gpt-oss-120b")
    p.add_argument("--base-url", default=None)
    p.add_argument("--endpoint-json", default="logs/vllm_endpoint_latest.json")
    p.add_argument("--max-tokens", type=int, default=60000)
    p.add_argument("--reasoning-effort", default="high")
    p.add_argument("--concurrency", type=int, default=40)

    # -- retrieval --
    p.add_argument(
        "--embedding-model", default=None,
        help=f"retrieval key embedder. Default is per-task (see "
             f"DEFAULT_EMBEDDING_BY_TASK): {DEFAULT_EMBED_MODEL} for tasks whose "
             f"problem text is itself distinctive, '{HASHING_EMBED_MODEL}' for "
             f"tower-of-hanoi, whose prompt is ~90%% fixed boilerplate -- MiniLM "
             f"gave a pairwise-cosine std of 0.0012 on its 60 train items even "
             f"after stripping the boilerplate (baselines/core/intent.py), i.e. "
             f"near-total retrieval collapse; hashing on the same stripped text "
             f"gets to 0.0175. The CORE paper's own choice ({PAPER_EMBED_MODEL}) "
             f"needs trust_remote_code that imports transformers.onnx, removed in "
             f"transformers>=5 -- see baselines/orbit/embedding.py")
    p.add_argument("--cache-dir", default="baselines/data/cache")
    p.add_argument("--max-obs-chars", type=int, default=3000,
                   help="cap on a stored rollout's raw reasoning+answer text; "
                        "the paper stores rollouts verbatim, but gpt-oss's chain "
                        "of thought can run to tens of thousands of characters")

    # -- Episodic RAG (CORE paper spec: 1 success + 1 failure) --
    p.add_argument("--k-success", type=int, default=1)
    p.add_argument("--k-failure", type=int, default=1)

    # -- MemRL (defaults = configs/rl_alf_config.yaml, matching run_memrl.py) --
    p.add_argument("--k1", type=int, default=5)
    p.add_argument("--k2", type=int, default=3)
    p.add_argument("--delta", type=float, default=None)
    p.add_argument("--sim-norm-mean", type=float, default=None)
    p.add_argument("--sim-norm-std", type=float, default=None)
    p.add_argument("--lambda-q", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--q-init", type=float, default=0.0)
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--q-min-threshold", type=float, default=-10.0)
    p.add_argument("--success-reward", type=float, default=1.0)
    p.add_argument("--failure-reward", type=float, default=-1.0)
    p.add_argument("--add-similarity-threshold", type=float, default=None,
                   help="merge-on-write threshold. Default is per-task (see "
                        "DEFAULT_ADD_SIM_THRESHOLD_BY_TASK): MemRL's own ALFWorld "
                        "value 0.90 for tasks whose pairwise similarity is "
                        "well-spread, 0.995 for tower-of-hanoi -- measured "
                        "sim_min=0.895 even after the intent-extraction + "
                        "hashing fix for degenerate retrieval (baselines/core/"
                        "intent.py), so 0.90 merged nearly every new record on "
                        "write and collapsed the store to single digits after "
                        "multiple visits (job 64634494 read memory=3 after 120 "
                        "rollouts). 0.995 sits between the 99th and 99.9th "
                        "percentile of the measured distribution, where the "
                        "curve plateaus -- only genuinely near-duplicate states "
                        "merge")
    p.add_argument("--novelty-threshold", type=float, default=None,
                   help="reward-attribution threshold. Default is per-task (see "
                        "DEFAULT_NOVELTY_THRESHOLD_BY_TASK): 0.85 normally, 0.99 "
                        "for tower-of-hanoi for the same reason as "
                        "--add-similarity-threshold -- its measured sim_min "
                        "(0.895) already clears 0.85, so every new record would "
                        "attribute reward to its nearest neighbour regardless of "
                        "true relevance")
    p.add_argument("--reward-merge-gain", type=float, default=0.1)
    p.add_argument("--summarizer-max-tokens", type=int, default=1024)

    p.add_argument("--verbose", action="store_true")
    p.add_argument("--skip-build", action="store_true",
                   help="load an existing memory_store.json and only (re)run the "
                        "eval passes -- for recovering from a server dying mid-eval "
                        "without redoing the (much more expensive) build phase")
    p.add_argument("--force", action="store_true",
                   help="proceed even if --output-dir already has a memory_store.json "
                        "(builds on top of it -- see the stale-store error for why this "
                        "is dangerous)")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.embedding_model is None:
        args.embedding_model = DEFAULT_EMBEDDING_BY_TASK.get(args.task, DEFAULT_EMBED_MODEL)
    if args.add_similarity_threshold is None:
        args.add_similarity_threshold = DEFAULT_ADD_SIM_THRESHOLD_BY_TASK.get(args.task, 0.90)
    if args.novelty_threshold is None:
        args.novelty_threshold = DEFAULT_NOVELTY_THRESHOLD_BY_TASK.get(args.task, 0.85)

    base_url = args.base_url
    if base_url is None:
        path = Path(args.endpoint_json)
        if not path.is_file():
            print(f"ERROR: no endpoint JSON at {path}. Start the server or pass --base-url.")
            return 2
        base_url, _ = endpoint_from_json(path)
    assert_endpoint_serves(base_url, args.model)

    out_dir = Path(args.output_dir)
    stale_store = out_dir / "memory_store.json"
    if args.skip_build:
        if not stale_store.exists():
            raise SystemExit(
                f"ERROR: --skip-build requires an existing {stale_store}; none found."
            )
    elif stale_store.exists() and not args.force:
        raise SystemExit(
            f"ERROR: {stale_store} already exists. build_memory() is NOT "
            f"resumable (baselines/core/runner.py's module docstring) -- "
            f"MemoryStore would load these records and build on top of them, "
            f"silently double-counting every visit already run into this "
            f"directory. This is exactly what corrupted an earlier zebralogic "
            f"run restarted into its own output dir after a cancelled attempt "
            f"had already written partial visits. Remove the directory to "
            f"start clean, or pass --force if you have verified it is safe, "
            f"or --skip-build to only redo the eval passes against it (e.g. "
            f"to recover from a server dying mid-eval, which does not touch "
            f"the store)."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    client = VLLMClient(
        model=args.model, base_url=base_url,
        max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort,
    )
    print("=" * 78)
    print(f"  {args.method} on CORE/{args.task}, budget-matched to HeXA")
    print(f"  endpoint    : {base_url}   model={args.model}")
    print(f"  n_train     : {args.n_train}")
    visits = args.budget // args.n_train
    print(f"  build budget: {args.budget} rollouts -> {visits} visits/item "
          f"({visits * args.n_train} actual)")
    print(f"  eval runs   : {args.eval_runs} (frozen, HeXA-paired seeds)")
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

    task = make_task(args.task, args.n_train)
    (out_dir / "task_manifest.json").write_text(json.dumps(task.manifest(), indent=2, default=str))

    embedder = build_embedder(args.embedding_model, cache_dir=Path(args.cache_dir))
    store = MemoryStore(out_dir / "memory_store.json")

    if args.method == "episodic_rag":
        memory = EpisodicRAG(
            store,
            RAGConfig(k_success=args.k_success, k_failure=args.k_failure,
                      max_obs_chars=args.max_obs_chars, max_steps=None),
        )
        deviations = [
            f"actor is {args.model} (matches HeXA's CORE config); "
            f"the CORE paper's own Episodic RAG baseline also used gpt-oss-120b, "
            f"but on NVIDIA NIM/Cerebras, not local vLLM.",
            f"stored rollouts are truncated to their last {args.max_obs_chars} "
            f"characters (head+tail); the paper stores them verbatim.",
            embedder_note(args.embedding_model),
        ]
        summarizer_info = None
    else:
        train_texts = [retrieval_text(args.task, p.problem_text) for p in task.train_problems()]
        if args.delta is None or args.sim_norm_mean is None or args.sim_norm_std is None:
            calib = calibrate_core_task(
                args.task, train_texts, embedding_model=args.embedding_model,
                cache_dir=Path(args.cache_dir), embedder=embedder,
            )
            delta = args.delta if args.delta is not None else calib["recommended"]["delta"]
            sim_mean = args.sim_norm_mean if args.sim_norm_mean is not None else calib["recommended"]["sim_norm_mean"]
            sim_std = args.sim_norm_std if args.sim_norm_std is not None else calib["recommended"]["sim_norm_std"]
        else:
            delta, sim_mean, sim_std = args.delta, args.sim_norm_mean, args.sim_norm_std
        print(f"  MemRL calibration: delta={delta:.4f} sim_norm_mean={sim_mean:.4f} sim_norm_std={sim_std:.4f}")

        cfg = MemRLConfig(
            k1=args.k1, k2=args.k2, delta=delta,
            sim_norm_mean=sim_mean, sim_norm_std=sim_std,
            weight_q=args.lambda_q, weight_sim=1.0 - args.lambda_q,
            epsilon=args.epsilon, q_min_threshold=args.q_min_threshold,
            alpha=args.alpha, gamma=args.gamma, q_init=args.q_init,
            success_reward=args.success_reward, failure_reward=args.failure_reward,
            add_similarity_threshold=args.add_similarity_threshold,
            novelty_threshold=args.novelty_threshold, reward_merge_gain=args.reward_merge_gain,
            max_traj_steps=None, max_obs_chars=args.max_obs_chars,
        )
        summarizer = make_core_summarizer(
            client.generate, max_new_tokens=args.summarizer_max_tokens, verbose=args.verbose,
        )
        memory = MemRL(store, cfg, summarizer=summarizer)
        deviations = [
            f"actor is {args.model} (matches HeXA's CORE config); MemRL's own "
            f"release specifies ALFWorld as the closest analogue task family, "
            f"not any CORE task.",
            "memory-building (script/reflection) calls go through the SAME "
            "gpt-oss-120b vLLM endpoint as the actor, per HeXA's actor/teacher "
            "parity convention -- not the official release's separate LLM.",
            embedder_note(args.embedding_model),
        ]
        summarizer_info = {"backend": "core-vllm", "model": args.model}

    manifest = {
        "method": args.method,
        "task": args.task,
        "n_train": args.n_train,
        "build_budget_requested": args.budget,
        "visits_per_problem": visits,
        "build_budget_actual": visits * args.n_train,
        "eval_runs": args.eval_runs,
        "agent_model": args.model,
        "embedding_model": args.embedding_model,
        "max_obs_chars": args.max_obs_chars,
        "summarizer": summarizer_info,
        "deviations": deviations,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    stream_config = CoreStreamConfig(
        task_name=args.task, method=args.method, visits_per_problem=visits,
        output_dir=out_dir, concurrency=args.concurrency, verbose=args.verbose,
    )

    started = time.time()
    if args.skip_build:
        print(f"\n[build] SKIPPED -- reusing {stale_store} "
              f"({len(store.records)} records already in memory)")
        prior = out_dir / "run_summary.json"
        build_summary = (
            json.loads(prior.read_text())["build"] if prior.is_file()
            else {"skipped": True, "memory_stats": store.stats()}
        )
    else:
        print(f"\n[build] {visits} visits x {args.n_train} items = "
              f"{visits * args.n_train} rollouts ...")
        build_summary = build_memory(task, memory, store, embedder, client.generate, stream_config)
        print(f"  build accuracy: {build_summary['build_accuracy']:.3f} "
              f"({build_summary['n_correct']}/{build_summary['n_rollouts']})  "
              f"memory: {build_summary['memory_stats']}")

    eval_summaries = []
    for run_index in range(args.eval_runs):
        print(f"\n[eval run {run_index}] frozen memory, 100 held-out items ...")
        summary = frozen_eval(task, memory, embedder, client.generate, stream_config, run_index=run_index)
        eval_summaries.append(summary)
        flag = "" if summary["reliable"] else f"  *** UNRELIABLE: {summary['n_empty']} empty completions ***"
        print(f"  accuracy={summary['accuracy']:.3f} ({summary['correct']}/{summary['n']})  "
              f"mean_added_tokens={summary['mean_added_tokens']}{flag}")

    # A run with any empty completion (`n_empty > 0`) is not a real accuracy --
    # job 64581745's mid-eval SLURM timeout produced a 0.000 that was 100
    # connection errors, not 100 wrong answers, and averaging it in silently
    # would have reported episodic_rag/zebralogic as far worse than it is.
    reliable = [s for s in eval_summaries if s["reliable"]]
    unreliable = [s for s in eval_summaries if not s["reliable"]]
    accuracies = [s["accuracy"] for s in reliable]
    mean_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
    sem = (
        (sum((a - mean_acc) ** 2 for a in accuracies) / (len(accuracies) - 1)) ** 0.5
        / (len(accuracies) ** 0.5)
        if len(accuracies) > 1 else 0.0
    )
    run_summary = {
        "manifest": manifest,
        "build": build_summary,
        "eval_runs": eval_summaries,
        "mean_accuracy": mean_acc,
        "sem_accuracy": sem,
        "n_reliable_eval_runs": len(reliable),
        "n_unreliable_eval_runs": len(unreliable),
        "elapsed_s": time.time() - started,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2, default=str))

    print("\n" + "=" * 78)
    if unreliable:
        print(f"  WARNING: {len(unreliable)}/{len(eval_summaries)} eval run(s) excluded "
              f"from the mean (empty completions, likely a dead endpoint mid-eval): "
              f"run_index {[s['run_index'] for s in unreliable]}. Re-run with "
              f"--skip-build --eval-runs {args.eval_runs} to redo them against the "
              f"same frozen memory once the endpoint is back.")
    print(f"  {args.method} / {args.task}: {mean_acc:.3f} +/- {sem:.3f} "
          f"(n={len(accuracies)} reliable eval runs / {len(eval_summaries)} requested, 1 build)")
    print(f"  wall clock: {(time.time() - started) / 3600:.2f} h -> {out_dir}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
