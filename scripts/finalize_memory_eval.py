"""Rebuild `runs.jsonl` + summaries for a frozen memory eval that died after its
rollouts but before its write-out.

`stream.frozen_eval` has no resume: `play_batch` replays every instance, and the
summary needs the per-instance `Retrieval` records, which live only in the
process that ran it. So a run whose `runs.partial.jsonl` is complete but whose
`runs.jsonl` was never written looks unrecoverable and costs a full re-eval.

It is recoverable, because retrieval is a pure function of the frozen store, the
embedder, the intent cache and the spec -- no model call. This replays only that
half and joins it to the rollouts already on disk. Nothing here re-plays an
episode; if `runs.partial.jsonl` is short of the eval pool, this refuses.

Validate before trusting: `--validate-against` recomputes an already-finalised
run and diffs the result against its recorded summary. `r1` of the Mastermind
MemRL set reproduces exactly, which is what licenses using it on `r3`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hexa_orbit import report as report_mod  # noqa: E402
from hexa_orbit import trajectory as trajectory_mod  # noqa: E402

from baselines.memory.memrl import MemRL, MemRLConfig  # noqa: E402
from baselines.memory.rag import EpisodicRAG, RAGConfig  # noqa: E402
from baselines.memory.store import MemoryStore  # noqa: E402
from baselines.orbit import blocks, stream  # noqa: E402
from baselines.orbit.embedding import build_embedder  # noqa: E402
from baselines.orbit.intent import IntentCache, cache_path  # noqa: E402
from baselines.orbit.tasks import build_task  # noqa: E402


def rebuild(out_dir: Path, eval_dir: str, cache_dir: Path) -> dict:
    man = json.loads((out_dir / "manifest.json").read_text())
    task_name, method = man["task"], man["method"]
    cfg = man["config"]

    orbit_task = build_task(task_name)

    store = MemoryStore(out_dir / "memory_store.json")
    embedder = build_embedder(man["embedding_model"], cache_dir=cache_dir)
    intents = IntentCache(cache_path(cache_dir, task_name, man["intent_mode"]),
                          man["intent_mode"])

    if method == "episodic_rag":
        memory = EpisodicRAG(store, RAGConfig(**{
            k: cfg[k] for k in ("k_success", "k_failure", "max_obs_chars", "max_steps")
            if k in cfg}))
        block_formatter = blocks.format_rag_block
    else:
        memory = MemRL(store, MemRLConfig(**{
            k: v for k, v in cfg.items() if k in MemRLConfig.__dataclass_fields__}))
        block_formatter = blocks.format_memrl_block

    runs = trajectory_mod.read_runs(out_dir / eval_dir / "runs.partial.jsonl")
    completed = [r for r in runs if not getattr(r, "error", None)]
    errored = [r for r in runs if getattr(r, "error", None)]
    by_id = {r.instance_id: r for r in runs}

    eval_specs = orbit_task.eval_specs(None)
    missing = [s.instance_id for s in eval_specs if s.instance_id not in by_id]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(eval_specs)} eval instances are absent from "
            f"runs.partial.jsonl -- this run needs re-evaluating, not finalising")

    # Canonical order is eval-spec order; the partial file is in completion order.
    ordered = [by_id[s.instance_id] for s in eval_specs]

    intents.warm(orbit_task, eval_specs)
    per_instance, injected = [], []
    for spec in eval_specs:
        _, record = stream.retrieve_for(
            memory, embedder, intents, orbit_task, spec, block_formatter)
        run = by_id[spec.instance_id]
        result = stream.to_result(orbit_task, run, memory_block=record.block)
        per_instance.append({
            "instance_id": run.instance_id,
            "leaked": run.leaked,
            "success_count": run.success_count,
            "episode_successes": list(run.episode_successes),
            "reward": result.get("reward"),
            "retrieved_memory_ids": record.memory_ids,
            "retrieval_diagnostics": record.diagnostics,
            "added_context_tokens": record.tokens,
            "actor_added_tokens": run.added_tokens,
        })
        injected.append(record.tokens or 0)

    return {
        "arm": f"frozen_memory_{orbit_task.name}",
        "report": report_mod.aggregate(completed, errored=errored),
        "n_eval": len(eval_specs),
        "memory_stats": store.stats(),
        "retrieval": {
            "n_with_memory": sum(1 for p in per_instance if p["retrieved_memory_ids"]),
            "n_empty_block": sum(1 for p in per_instance if not p["retrieved_memory_ids"]),
            "mean_added_context_tokens": sum(injected) / len(injected) if injected else 0.0,
            "max_added_context_tokens": max(injected) if injected else 0,
        },
        "per_instance": per_instance,
    }, ordered, man


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="the run directory")
    ap.add_argument("--eval-dir", default="eval_frozen")
    ap.add_argument("--cache-dir", type=Path, default=Path("baselines/data/cache"))
    ap.add_argument("--validate-against", action="store_true",
                    help="recompute a finalised run and diff, writing nothing")
    args = ap.parse_args()

    summary, ordered, man = rebuild(args.out, args.eval_dir, args.cache_dir)

    if args.validate_against:
        truth = json.loads((args.out / args.eval_dir / "summary.json").read_text())
        bad = []
        for key in ("arm", "n_eval", "memory_stats", "retrieval", "report"):
            if json.dumps(summary[key], sort_keys=True, default=str) != \
               json.dumps(truth[key], sort_keys=True, default=str):
                bad.append(key)
        t = {p["instance_id"]: p for p in truth["per_instance"]}
        for p in summary["per_instance"]:
            if json.dumps(p, sort_keys=True, default=str) != \
               json.dumps(t.get(p["instance_id"]), sort_keys=True, default=str):
                bad.append(f"per_instance[{p['instance_id']}]")
        print("MISMATCH: " + ", ".join(bad[:8]) if bad else
              "EXACT MATCH on arm, n_eval, memory_stats, retrieval, report and all "
              f"{len(summary['per_instance'])} per-instance records")
        return 1 if bad else 0

    trajectory_mod.write_runs(args.out / args.eval_dir / "runs.jsonl", ordered)
    (args.out / args.eval_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    top = dict(summary)
    top["manifest"] = man
    top["finalized_from_partial"] = (
        "rollouts read from runs.partial.jsonl; retrieval recomputed from the "
        "frozen store. wall_clock_s is absent because this process did not run "
        "the eval.")
    (args.out / f"summary_{man['method']}_{args.eval_dir}.json").write_text(
        json.dumps(top, indent=2, default=str))
    print(report_mod.format_summary(
        summary["report"], title=f"{man['task']} — {man['method']}, finalised from partial"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
