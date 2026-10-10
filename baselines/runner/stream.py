"""The online seed-stream loop shared by Episodic RAG and MemRL.

Protocol (matched to HeXA's, see docs/BASELINES.md):

  * one level at a time; seeds run **in order** over a fixed grid (default
    6-55, the 50-seed grid the ReAct/Reflexion/Direct baselines used);
  * an optional offline init preloads memory from existing trajectory JSONs
    (HeXA's `Initial_trajectories/<level>`, seeds 1-5), so both methods get the
    same Off2On head start HeXA gets;
  * per seed: retrieve -> solve -> update utilities -> write back;
  * results are grouped into rounds of `--seeds-per-round` (default 3) so the
    success-rate curve overlays HeXA's per-round curve directly;
  * `--freeze-after` switches memory to read-only from that seed onward, which
    gives MemRL's "Transferring" setting (frozen bank, held-out instances).

Per-seed trajectories are written immediately and skipped on re-run, so an
interrupted stream resumes — same contract as
`skillrl/runner/run_skill_agent.py:run_batch`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from baselines.memory.blocks import count_tokens
from baselines.memory.embedder import Embedder
from baselines.memory.intent import IntentCache
from baselines.memory.store import MemoryStore
from baselines.runner.memory_runner import run_memory_augmented


@dataclass
class StreamConfig:
    level: str
    seeds: Sequence[int]
    output_dir: Path
    method: str                      # "episodic_rag" | "memrl"
    seeds_per_round: int = 3
    max_iterations: int = 25
    temperature: float = 0.3
    max_new_tokens: int = 800
    is_oss: bool = False
    verbose: bool = False
    freeze_after: Optional[int] = None   # seed index (0-based) to freeze memory
    exclude_own_seed: bool = True


def load_offline_trajectories(traj_dir: Optional[Path]) -> list[dict]:
    """Load HeXA-format seed trajectories (`trajectory_seed*.json`)."""
    if traj_dir is None:
        return []
    traj_dir = Path(traj_dir)
    if not traj_dir.exists():
        raise FileNotFoundError(f"Offline trajectory dir not found: {traj_dir}")

    out = []
    for path in sorted(traj_dir.glob("trajectory_seed*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"  [warn] unparseable trajectory, skipping: {path}")
            continue
        if data.get("seed") is None:
            # Recover the seed from the filename, e.g. trajectory_seed3_raw.json
            stem = path.stem.split("trajectory_seed")[-1]
            digits = ""
            for ch in stem:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if not digits:
                # Without a seed we cannot build the intent this rollout belongs
                # to, and guessing would attach it to the wrong geometry.
                print(f"  [warn] no seed in payload or filename, skipping: {path}")
                continue
            data["seed"] = int(digits)
        if not data.get("trajectory"):
            print(f"  [warn] empty trajectory, skipping: {path}")
            continue
        out.append(data)
    return out


def run_stream(
    *,
    config: StreamConfig,
    memory,                              # EpisodicRAG | MemRL
    store: MemoryStore,
    embedder: Embedder,
    intents: IntentCache,
    model_fn: Callable,
    block_formatter: Callable[[Sequence[Any]], str],
    offline_trajectories: Sequence[dict] = (),
    manifest: Optional[dict] = None,
) -> dict:
    """Run the seed stream and return a summary dict."""
    out_dir = Path(config.output_dir)
    level_dir = out_dir / config.level
    level_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = out_dir / "episodes.jsonl"

    # ── offline init ────────────────────────────────────────────────────
    preloaded_seeds = {r.seed for r in store.records if r.source == "offline"}
    for traj in offline_trajectories:
        seed = traj.get("seed")
        if seed is None or seed in preloaded_seeds:
            continue
        intent = intents.get(config.level, int(seed))
        memory.add_episode(
            result=traj,
            level=config.level,
            seed=int(seed),
            intent=intent.text,
            embedding=embedder.embed_one(intent.text),
            step_index=-1,
            source="offline",
        )
        preloaded_seeds.add(seed)
    if offline_trajectories:
        store.save()
        print(
            f"  Offline init: {len(preloaded_seeds)} trajectories in memory "
            f"({store.stats()['num_success']} success / "
            f"{store.stats()['num_failure']} failure)"
        )

    # ── stream ──────────────────────────────────────────────────────────
    records: list[dict] = []
    for step_index, seed in enumerate(config.seeds):
        traj_path = level_dir / f"trajectory_seed{seed}_{config.method}.json"
        if traj_path.exists():
            try:
                cached = json.loads(traj_path.read_text())
                records.append(cached)
                print(
                    f"\n  Seed {seed}: SKIPPING (exists — "
                    f"{'SUCCESS' if cached.get('success') else 'FAILURE'})"
                )
                continue
            except (json.JSONDecodeError, OSError):
                pass

        frozen = (
            config.freeze_after is not None and step_index >= config.freeze_after
        )

        print(f"\n{'#' * 60}")
        print(
            f"  [{config.method}] {config.level}  seed {seed}  "
            f"({step_index + 1}/{len(config.seeds)})"
            f"{'  [memory frozen]' if frozen else ''}"
        )
        print(f"{'#' * 60}")

        intent = intents.get(config.level, seed)
        query_embedding = embedder.embed_one(intent.text)

        exclude = [seed] if config.exclude_own_seed else []
        retrieved = memory.retrieve(
            query_embedding, level=config.level, exclude_seeds=exclude
        )
        block = block_formatter(retrieved.records)

        if config.verbose or retrieved.records:
            print(
                f"  Retrieved {len(retrieved.records)} memories "
                f"({count_tokens(block)} tokens): "
                f"{[r.memory_id for r in retrieved.records]}"
            )
            if config.verbose:
                print(f"  Diagnostics: {retrieved.diagnostics}")

        started = time.time()
        result = run_memory_augmented(
            model_fn,
            config.level,
            seed,
            block,
            max_iterations=config.max_iterations,
            verbose=config.verbose,
            temperature=config.temperature,
            max_new_tokens=config.max_new_tokens,
            is_oss=config.is_oss,
            retrieval_diagnostics=retrieved.diagnostics,
        )
        result["method"] = config.method
        result["stream_index"] = step_index
        result["memory_frozen"] = frozen
        result["retrieved_memory_ids"] = [r.memory_id for r in retrieved.records]
        result["memory_size_before"] = len(store.records)

        if not frozen:
            update_info = memory.update_utilities(retrieved, result)
            result["utility_updates"] = update_info
            memory.add_episode(
                result=result,
                level=config.level,
                seed=seed,
                intent=intent.text,
                embedding=query_embedding,
                step_index=step_index,
            )
            store.save()
        result["memory_size_after"] = len(store.records)
        result["memory_stats"] = store.stats()

        traj_path.write_text(json.dumps(result, indent=2, default=str))
        with episodes_path.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "method": config.method,
                        "level": config.level,
                        "seed": seed,
                        "stream_index": step_index,
                        "success": bool(result.get("success")),
                        "iterations": result.get("iterations"),
                        "simulator_calls": result.get("simulator_calls"),
                        "added_context_tokens": result.get("added_context_tokens"),
                        "retrieved_memory_ids": result["retrieved_memory_ids"],
                        "retrieval_diagnostics": retrieved.diagnostics,
                        "utility_updates": result.get("utility_updates"),
                        "memory_size_after": result["memory_size_after"],
                        "elapsed_time": result.get("elapsed_time"),
                    },
                    default=str,
                )
                + "\n"
            )
        records.append(result)

        print(
            f"  -> {'SUCCESS' if result.get('success') else 'FAILURE'} in "
            f"{result.get('iterations', '?')} iters, "
            f"{result.get('simulator_calls', '?')} sims, "
            f"{time.time() - started:.0f}s  |  memory={len(store.records)}"
        )

    summary = summarize(records, config, store, manifest=manifest)
    (out_dir / f"summary_{config.method}.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print_summary(summary)
    return summary


def summarize(
    records: Sequence[dict],
    config: StreamConfig,
    store: MemoryStore,
    *,
    manifest: Optional[dict] = None,
) -> dict:
    total = len(records)
    successes = sum(1 for r in records if r.get("success"))

    rounds = []
    per_round = max(1, config.seeds_per_round)
    for start in range(0, total, per_round):
        chunk = records[start : start + per_round]
        if not chunk:
            continue
        chunk_successes = sum(1 for r in chunk if r.get("success"))
        rounds.append(
            {
                "round": start // per_round + 1,
                "seeds": [r.get("seed") for r in chunk],
                "successes": chunk_successes,
                "total": len(chunk),
                "accuracy": chunk_successes / len(chunk),
            }
        )

    cumulative = []
    running = 0
    for i, record in enumerate(records, 1):
        running += 1 if record.get("success") else 0
        cumulative.append(round(running / i, 4))

    def mean(key: str) -> float:
        vals = [r.get(key) for r in records if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "method": config.method,
        "level": config.level,
        "seeds": list(config.seeds),
        "total_seeds": total,
        "successes": successes,
        "success_rate": successes / total if total else 0.0,
        "avg_iterations": mean("iterations"),
        "avg_simulator_calls": mean("simulator_calls"),
        "avg_added_context_tokens": mean("added_context_tokens"),
        "total_simulator_calls": sum(
            r.get("simulator_calls") or 0 for r in records
        ),
        "rounds": rounds,
        "running_success_rate": cumulative,
        "memory_stats": store.stats(),
        "manifest": manifest or {},
        "per_seed": [
            {
                "seed": r.get("seed"),
                "success": bool(r.get("success")),
                "iterations": r.get("iterations"),
                "simulator_calls": r.get("simulator_calls"),
                "added_context_tokens": r.get("added_context_tokens"),
                "retrieved_memory_ids": r.get("retrieved_memory_ids", []),
                "memory_frozen": r.get("memory_frozen", False),
            }
            for r in records
        ],
    }


def print_summary(summary: dict) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {summary['method']} on {summary['level']}")
    print(f"{'=' * 70}")
    for entry in summary["rounds"]:
        print(
            f"    Round {entry['round']:>2}: {entry['successes']}/{entry['total']} "
            f"= {entry['accuracy']:.0%}  seeds={entry['seeds']}"
        )
    print(
        f"\n  Overall: {summary['successes']}/{summary['total_seeds']} "
        f"= {summary['success_rate']:.1%}"
    )
    print(f"  Avg iterations:            {summary['avg_iterations']:.1f}")
    print(f"  Avg simulator calls:       {summary['avg_simulator_calls']:.1f}")
    print(f"  Avg added context tokens:  {summary['avg_added_context_tokens']:.0f}")
    print(f"  Memory: {summary['memory_stats']}")
