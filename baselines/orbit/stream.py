"""The two passes: build memory on the train pool, then evaluate it frozen.

This is `baselines/runner/stream.py` for ORBIT, and the protocol is deliberately
*not* that module's.  On InterPhyre both methods run one cumulative stream over
the 50-seed eval grid, so memory grows across the very instances being scored
and the reported number is a learning curve.  The ORBIT arms already in
`MAZE_RESULTS.md` and `MASTERMIND_RESULTS.md` are not that shape: HeXA trains
over a train pool for some rounds, freezes the bank, and evaluates on a disjoint
held-out pool (`hexa_orbit.loop.frozen_eval`), and the no-bank baseline plays
that same held-out pool.  A memory baseline scored on a growing stream would not
be comparable to either.

So:

  **build**  train-pool instances, in order, serially.  Per instance: retrieve →
             play → update utilities → write back.  Memory grows.
  **eval**   held-out instances, memory read-only.  Retrieval still happens per
             instance -- that is what these methods *are* -- but nothing is
             written, so the memory every eval instance sees is identical and
             the arm is paired with the others by `instance_id`.

**The build pass is serial by default, and that is a correctness requirement
rather than caution.**  Instance *n*'s retrieval must see what instances 1..n-1
wrote; run two at once and which memories exist when a given instance retrieves
depends on thread timing, so the run is not reproducible and the Q values are
not comparable to another run of the same configuration.  The eval pass has no
such constraint -- memory is frozen -- and inherits `play_batch`'s concurrency,
including the maze-generation lock that makes it safe.

`build_batch_size > 1` relaxes seriality into *rounds*, and does it in the one
way that stays reproducible: every instance in a batch retrieves against the
memory as it stood at the **start** of the batch, the batch plays concurrently,
and the write-backs are then applied in **input order**.  So the result depends
on the batch boundaries, which are fixed by the flag, and not on thread timing.

This is a real deviation from both papers, which update after every episode, and
it is recorded as one.  It is also exactly what HeXA does -- a round plays 20
instances against one frozen bank and evolves afterwards -- so a batched memory
arm is if anything closer to the arm it is being compared with.

**Leave it at 1 unless the actor is slow.**  Measured from the gpt-oss-120b
no-bank Maze arm (256 instances, 375 s at concurrency 16), an instance costs
~0.4 min serial, so a 200-instance build is well under two hours and there is
nothing to buy.  It matters for a slow actor: Qwen3-8B with thinking enabled
runs ~3.5 min/instance, where the same build is twelve hours.

Resume is keyed on the **store**, not on the transcript file.  The store is the
memory state; a transcript is a record of how it got there.  Saving the store
before appending the transcript line means a crash in between costs one
transcript line and leaves the memory correct, where the reverse ordering would
replay an instance and write its memory twice.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from hexa_orbit import loop as orbit_loop
from hexa_orbit import report as report_mod
from hexa_orbit import trajectory as trajectory_mod
from hexa_orbit.actor import InstanceRun, MazeActor

from baselines.memory.blocks import count_tokens
from baselines.orbit.intent import seed_key
from baselines.orbit.rollout import play_and_record, to_result


@dataclass
class OrbitStreamConfig:
    task: str
    method: str                     # "episodic_rag" | "memrl"
    output_dir: Path
    eval_concurrency: int = 8
    verbose: bool = False
    #: Reuse of an existing store across runs is a footgun, not a feature: two
    #: methods writing one store is what `baselines/memory/store.py` supports on
    #: InterPhyre, but here it would mix MemRL summaries into an Episodic RAG
    #: arm. The runner gives each method its own output dir; this is the check.
    strict_store: bool = True


@dataclass
class Retrieval:
    """What retrieval produced for one instance, kept for the run record."""

    memory_ids: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    block: str = ""
    tokens: Optional[int] = 0


def retrieve_for(
    memory, embedder, intents, orbit_task, spec, block_formatter
) -> tuple[Any, Retrieval]:
    """Retrieve for one instance and render its block.

    The instance's own past memories are excluded by `seed_key`, which matters
    only if the same instance is ever played twice; the pools are disjoint, so
    in a normal run this excludes nothing and is here so that it cannot start
    mattering silently.
    """
    intent = intents.get(orbit_task, spec)
    embedding = embedder.embed_one(intent.text)
    retrieved = memory.retrieve(
        embedding,
        level=orbit_task.name,
        exclude_seeds=[seed_key(spec.instance_id)],
    )
    block = block_formatter(retrieved.records)
    return (
        (intent, embedding, retrieved),
        Retrieval(
            memory_ids=[r.memory_id for r in retrieved.records],
            diagnostics=dict(retrieved.diagnostics),
            block=block,
            tokens=count_tokens(block),
        ),
    )


# ── pass 1: build ─────────────────────────────────────────────────────────

def stored_instance_ids(store) -> set[str]:
    """Instances already written into memory. The resume key."""
    return {
        str(r.meta.get("instance_id"))
        for r in store.records
        if r.meta.get("instance_id")
    }


def build_memory(
    orbit_task,
    specs: Sequence[Any],
    *,
    memory,
    store,
    embedder,
    intents,
    block_formatter: Callable,
    actor_config,
    out_dir: Path,
    method: str,
    client_factory: Optional[Callable[[Any], Any]] = None,
    batch_size: int = 1,
    verbose: bool = False,
) -> dict:
    """Grow memory over the training pool. See the module docstring on batching."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    episodes_path = out_dir / "episodes.jsonl"

    done = stored_instance_ids(store)
    if done:
        print(f"  Resuming: {len(done)} instances already in memory")

    runs: list[InstanceRun] = []
    errored: list[InstanceRun] = []
    started = time.monotonic()

    todo = [(i, s) for i, s in enumerate(specs) if s.instance_id not in done]
    batch_size = max(1, int(batch_size))

    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]

        # Retrieve for the whole batch first, against the memory as it stands
        # now. At batch_size 1 this is the strict stream; above it, every
        # instance in the batch sees the same memory, which is what makes the
        # result independent of thread timing.
        prepared = []
        for index, spec in batch:
            (intent, embedding, retrieved), record = retrieve_for(
                memory, embedder, intents, orbit_task, spec, block_formatter
            )
            prepared.append((index, spec, intent, embedding, retrieved, record))
            print(
                f"\n[build {index + 1}/{len(specs)}] {spec.instance_id}  "
                f"memory={len(store.records)}  retrieved={len(record.memory_ids)} "
                f"({record.tokens} tok)"
            )
            if verbose and record.diagnostics:
                print(f"    retrieval: {record.diagnostics}")

        def play(item):
            index, spec, _intent, _embedding, _retrieved, record = item
            return play_and_record(
                orbit_task, spec, record.block, actor_config,
                leaked=None, client_factory=client_factory,
                retrieval_diagnostics=record.diagnostics,
            )

        if len(prepared) > 1:
            with ThreadPoolExecutor(max_workers=len(prepared)) as pool:
                played_batch = list(pool.map(play, prepared))
        else:
            played_batch = [play(prepared[0])] if prepared else []

        # Write back in input order, never completion order, so the store is a
        # deterministic function of the pool and the batch size.
        for (index, spec, intent, embedding, retrieved, record), (run, result) in zip(
            prepared, played_batch
        ):
            if result is None:
                print(f"    {spec.instance_id} ERROR: {run.error}")
                errored.append(run)
                _append(runs_path, run.to_dict())
                continue

            result["method"] = method
            result["stream_index"] = index
            result["retrieved_memory_ids"] = record.memory_ids
            result["memory_size_before"] = len(store.records)

            update_info = memory.update_utilities(retrieved, result)
            written = memory.add_episode(
                result=result,
                level=orbit_task.name,
                seed=seed_key(spec.instance_id),
                intent=intent.text,
                embedding=embedding,
                step_index=index,
                source="build",
            )
            if written is not None:
                # The store's `seed` is a truncated hash; the real id is what
                # every report, every runs.jsonl row and every paired test keys
                # on, so it travels with the record rather than being
                # recoverable only by re-hashing every instance in the pool.
                written.meta["instance_id"] = spec.instance_id
            # Store first, transcript second -- see the module docstring on resume.
            store.save()

            result["utility_updates"] = update_info
            result["memory_size_after"] = len(store.records)
            runs.append(run)
            _append(runs_path, run.to_dict())
            _append(episodes_path, {
                "task": orbit_task.name,
                "instance_id": spec.instance_id,
                "stream_index": index,
                "success": bool(result.get("success")),
                "success_count": result.get("success_count"),
                "episode_successes": result.get("episode_successes"),
                "reward": result.get("reward"),
                "steps": result.get("iterations"),
                "added_context_tokens": result.get("added_context_tokens"),
                "retrieved_memory_ids": record.memory_ids,
                "retrieval_diagnostics": record.diagnostics,
                "utility_updates": update_info,
                "memory_size_after": len(store.records),
                "elapsed_time": result.get("elapsed_time"),
            })

            print(
                f"    {spec.instance_id} -> R={result.get('success_count')}/3 "
                f"reward={float(result.get('reward', 0)):+.2f}  "
                f"memory={len(store.records)}"
            )

    played = len(runs)
    summary = {
        "n_specs": len(specs),
        "batch_size": batch_size,
        "n_played_this_pass": played,
        "n_errored": len(errored),
        "wall_clock_s": round(time.monotonic() - started, 1),
        "memory_stats": store.stats(),
        "report": report_mod.aggregate(runs, errored=errored) if runs else {},
    }
    # The single most informative diagnostic a memory run has, and the one the
    # InterPhyre port learned the hard way: if the store did not grow roughly
    # one record per instance, `add_similarity_threshold` is collapsing it and
    # every instance is retrieving the same two memories.
    if played:
        grew = len(store.records)
        print(
            f"\n  Memory: {grew} records from {len(stored_instance_ids(store))} "
            f"instances ({store.stats()['num_success']} success / "
            f"{store.stats()['num_failure']} failure)"
        )
        if grew < 0.5 * len(stored_instance_ids(store)):
            print(
                "  [!] The store holds far fewer records than instances played: "
                "near-duplicate\n      intents are being merged in place "
                "(MemRL's add_similarity_threshold). Retrieval\n      is drawing "
                "from a bank that is not growing -- check the calibration spread."
            )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _append(path: Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        handle.flush()


# ── pass 2: frozen eval ───────────────────────────────────────────────────

def _memory_fingerprint(store) -> tuple:
    """Everything about the store an eval pass must not change.

    Ids and Q values, not just the record count: MemRL's `update_utilities`
    rewrites Q in place without adding a record, so a count-only check would
    pass while the memory shifted underneath the pool.
    """
    return tuple(
        (r.memory_id, round(r.q, 9), r.q_visits, len(r.content))
        for r in store.records
    )


def frozen_eval(
    orbit_task,
    specs: Sequence[Any],
    *,
    memory,
    store,
    embedder,
    intents,
    block_formatter: Callable,
    actor_config,
    out_dir: Path,
    train_specs: Sequence[Any] = (),
    concurrency: int = 8,
    client_factory: Optional[Callable[[Any], Any]] = None,
) -> dict:
    """Play the held-out pool with memory read-only.

    Retrieval is per instance, so `play_batch`'s single `guidelines` argument is
    unused and the block is built inside the actor factory instead -- the
    factory is handed the live instance, and `instance.spec` is the retrieval
    key's source.  Everything else about the batch (input-order results, crash
    -safe partial file, per-instance error isolation, the maze generation lock)
    is `play_batch`'s and is not reimplemented here.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Warm every intent before the batch starts: building one resets an
    # environment, and on Maze that touches the global RNG. Doing it lazily
    # inside a worker would race with another instance's maze generation.
    intents.warm(orbit_task, specs)

    leaked_ids = orbit_task.leaked_ids(specs, train_specs)
    if leaked_ids:
        print(
            f"  {len(leaked_ids)} of {len(specs)} eval instances share content "
            f"with the train pool; reported separately as 'leaked'"
        )

    before = _memory_fingerprint(store)
    records: dict[str, Retrieval] = {}
    lock = threading.Lock()

    def actor_factory(_guidelines, instance=None):
        spec = instance.spec
        _, record = retrieve_for(
            memory, embedder, intents, orbit_task, spec, block_formatter
        )
        with lock:
            records[spec.instance_id] = record
        return MazeActor(
            actor_config,
            guidelines=record.block or None,
            client=client_factory(instance) if client_factory else None,
        )

    completed, errored = orbit_loop.play_batch(
        orbit_task, list(specs), actor_factory, None,
        leaked_ids=leaked_ids,
        concurrency=concurrency,
        partial_path=out_dir / "runs.partial.jsonl",
        on_instance=lambda i, r: print(
            f"[eval {i}/{len(specs)}] {r.instance_id} "
            f"R={r.success_count}/3 eps={r.episode_successes}"
            f"{' LEAKED' if r.leaked else ''}"
            + (f" ERROR: {r.error}" if r.error else "")
        ),
    )
    trajectory_mod.write_runs(out_dir / "runs.jsonl", completed + errored)

    after = _memory_fingerprint(store)
    if after != before:
        # "Frozen" is the whole basis for calling this arm paired: every eval
        # instance must see the same memory. A write here would mean retrieval
        # order silently determined the result, so fail rather than report it.
        raise AssertionError(
            "memory changed during the frozen eval — it is not read-only"
        )

    per_instance = []
    for run in completed:
        record = records.get(run.instance_id, Retrieval())
        result = to_result(orbit_task, run, memory_block=record.block)
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

    injected = [p["added_context_tokens"] or 0 for p in per_instance]
    summary = {
        "arm": f"frozen_memory_{orbit_task.name}",
        "report": report_mod.aggregate(completed, errored=errored),
        "n_eval": len(specs),
        "memory_stats": store.stats(),
        "retrieval": {
            "n_with_memory": sum(1 for p in per_instance if p["retrieved_memory_ids"]),
            "n_empty_block": sum(1 for p in per_instance if not p["retrieved_memory_ids"]),
            "mean_added_context_tokens": (
                sum(injected) / len(injected) if injected else 0.0
            ),
            "max_added_context_tokens": max(injected) if injected else 0,
        },
        "per_instance": per_instance,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    empty = summary["retrieval"]["n_empty_block"]
    if empty:
        print(
            f"\n  [!] {empty} of {len(per_instance)} eval instances got an EMPTY "
            f"memory block.\n      Those instances are the no-bank baseline, not "
            f"this method. On MemRL this is\n      Phase A rejecting the store: "
            f"check --delta against the calibration."
        )
    return summary
