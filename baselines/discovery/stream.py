"""Build-then-freeze driver for a DiscoverPhysics memory arm.

Two passes, in this order, mirroring `baselines/orbit/stream.py` and HeXA's own
DiscoverPhysics protocol:

  offline   the world's bank-free episodes are loaded straight into memory,
            the analogue of HeXA's "Round 0: distil from OFFLINE" step
  build     training episodes in rounds, memory growing between rounds
  eval      held-out episodes, memory frozen and read-only

**The build pass is round-batched, not serial.**  Every seed in a round
retrieves against the memory as it stood at the *round's start* and all of them
are written back afterwards, in seed order.  That is not a convenience: it is
precisely HeXA's round structure, where three episodes play against one frozen
bank before it evolves.  A serial build would give the memory arm a finer update
granularity than the arm it is being compared with.  The result depends only on
the round boundaries, which are fixed, so the run stays reproducible.

`--skip-build` evaluates an existing store; `--skip-eval` stops after building,
and a later run with the same output directory picks the store up.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from baselines.discovery import rollout
from baselines.discovery.intent import intent_for
from baselines.memory.store import MemoryStore


@dataclass
class ArmConfig:
    """Everything that defines one arm, all of it pinned by HeXA's protocol."""

    name: str                       # "oscillator" | "circle" | "hubble"
    eval_world: str
    eval_seeds: list[int]
    build: list[tuple[str, list[int]]]   # [(world, seeds), ...] one entry per round
    offline_dirs: dict[str, Path] = field(default_factory=dict)
    # The store's retrieval scope, and -- because `trajectory_fmt.instance_label`
    # prints it inside every stored memory -- a string the actor reads. It must
    # therefore be answer-free: "discovery" for the cross-world arm, and the
    # anonymous `intent.world_label` for a within-world one, never the world's
    # own name. A smoke run caught this: with `scope_level="oscillator"` each
    # retrieved memory opened "Level: oscillator", handing the actor the word
    # for the time-dependence it is supposed to discover.
    scope_level: str = "discovery"
    within_world: bool = False
    method: str = "episodic_rag"
    #: How retrieved records are rendered. Set by the runner from the method,
    #: because the two blocks differ in wording and in what they show per record
    #: (MemRL's header carries the utility Q; Episodic RAG has none).
    block_formatter: Callable[[Sequence], str] = None  # type: ignore[assignment]
    output_dir: Path = Path(".")
    cli_model: str = "sonnet"
    judge_model: str = "claude-opus-4-6"
    noise_std: float = 0.075
    max_tokens: int = 8192
    max_rounds: Optional[int] = None
    redact: bool = True


def episode_kwargs(cfg: ArmConfig) -> dict:
    kw = dict(
        model=cfg.cli_model,
        judge_model=cfg.judge_model,
        noise_std=cfg.noise_std,
        max_tokens=cfg.max_tokens,
        redact=cfg.redact,
    )
    if cfg.max_rounds is not None:
        kw["max_rounds"] = cfg.max_rounds
    return kw


# ── offline init ────────────────────────────────────────────────────────
def load_offline(directory: Path) -> list[dict]:
    """HeXA-format `trajectory_seed*_skillrl.json` episodes, seed-ordered."""
    paths = sorted(
        Path(directory).glob("trajectory_seed*_skillrl.json"),
        key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0),
    )
    out = []
    for path in paths:
        try:
            out.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):
            print(f"  [offline] skipping unreadable {path.name}")
    return out


# ── retrieval ───────────────────────────────────────────────────────────
def retrieve_block(
    memory, store: MemoryStore, embedder, cfg: ArmConfig, world: str, seed: int
) -> tuple[str, dict, object]:
    """The block for one episode, plus what retrieval actually did.

    `exclude_seeds` keeps an episode from retrieving itself. It is applied only
    on a within-world arm, where a seed identifies an episode. On the
    cross-world arm the memory holds no episode of the evaluated world at all,
    while seeds 100-102 are reused across the nine training worlds -- so
    excluding by bare seed number there would drop legitimate memories from
    other worlds for no reason.
    """
    query = embedder.embed_one(intent_for(world))
    retrieved = memory.retrieve(
        query,
        level=cfg.scope_level,
        exclude_seeds=({seed} if cfg.within_world else ()),
    )
    block = cfg.block_formatter(retrieved.records)
    diagnostics = dict(retrieved.diagnostics)
    diagnostics["records_in_scope"] = len(store.for_level(cfg.scope_level))
    diagnostics["retrieved_ids"] = [r.memory_id for r in retrieved.records]
    diagnostics["retrieved_worlds"] = [
        str(r.meta.get("world") or r.level) for r in retrieved.records
    ]
    diagnostics["n_empty_block"] = int(not block)
    return block, diagnostics, retrieved


def _ledger_path(cfg: "ArmConfig") -> Path:
    return Path(cfg.output_dir) / "written_back.json"


def _load_ledger(cfg: "ArmConfig") -> set[str]:
    path = _ledger_path(cfg)
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text()))
    except (json.JSONDecodeError, OSError):
        return set()


def _save_ledger(cfg: "ArmConfig", ledger: set[str]) -> None:
    _ledger_path(cfg).write_text(json.dumps(sorted(ledger), indent=2))


def write_back(memory, store: MemoryStore, embedder, cfg: ArmConfig, world: str,
               seed: int, result: dict, step_index: int,
               source: str = "online", ledger: Optional[set] = None) -> bool:
    """Store one episode. Returns False if it was already written back.

    Both passes resume from disk: an episode JSON that exists is reused rather
    than re-run, but the store it was written into is reloaded too, so an
    unguarded write-back replays every episode on the second launch. For
    Episodic RAG that duplicates records -- and since the within-world
    retrieval key is an exact tie, ordering alone decides which record wins, so
    a duplicate changes the answer.

    For MemRL it is worse and not fixable by inspecting the store. A
    near-duplicate intent is folded into an existing record *in place*,
    overwriting that record's `seed`, so after a within-world build the store
    holds two records claiming two seeds while fifteen episodes went into them.
    Membership simply is not recoverable from the store. Hence an explicit
    ledger of `(source, world, seed)` keys on disk: it is the only
    representation that survives a merge, and without it a resumed MemRL arm
    would re-merge every episode and inflate `q_visits`.
    """
    key = f"{source}:{world}:{seed}"
    if ledger is not None and key in ledger:
        return False
    record = memory.add_episode(
        result=result,
        level=cfg.scope_level,
        seed=seed,
        intent=intent_for(world),
        embedding=embedder.embed_one(intent_for(world)),
        step_index=step_index,
        source=source,
    )
    if record is not None:
        record.meta["world"] = world
    if ledger is not None:
        ledger.add(key)
    return True


# ── passes ──────────────────────────────────────────────────────────────
def run_offline_pass(memory, store, embedder, cfg: ArmConfig) -> int:
    """Seed the memory with bank-free episodes; returns how many were added."""
    ledger = _load_ledger(cfg)
    n = 0
    for label, directory in cfg.offline_dirs.items():
        episodes = load_offline(directory)
        for result in episodes:
            result.setdefault("simulator_calls", int(result.get("iterations") or 0))
            result.setdefault("verifier_feedback", rollout.verifier_feedback(result))
            # The trajectory's own `world` wins. The cross-world bootstrap is a
            # POOLED directory holding episodes from five different worlds under
            # one path (`_pooled_global_term/pooled_input`), so the dict key is
            # the pool's name and not any episode's world.
            world = str(result.get("world") or label)
            write_back(memory, store, embedder, cfg, world,
                       int(result.get("seed", 0)), result, n,
                       source="offline", ledger=ledger)
            n += 1
        worlds = sorted({str(r.get("world") or label) for r in episodes})
        print(f"  [offline] {label}: {len(episodes)} episodes "
              f"from {len(worlds)} world(s) -> {directory}")
    store.save()
    _save_ledger(cfg, ledger)
    return n


def run_build_pass(memory, store, embedder, cfg: ArmConfig, start_index: int) -> list[dict]:
    """Training rounds, memory frozen within a round and grown between them."""
    ledger = _load_ledger(cfg)
    results: list[dict] = []
    index = start_index
    for round_no, (world, seeds) in enumerate(cfg.build, 1):
        n_scope = len(store.for_level(cfg.scope_level))
        print(f"\n=== Build round {round_no}/{len(cfg.build)}: {world} seeds "
              f"{seeds} ({n_scope} memories in scope) ===")
        pending = []
        for seed in seeds:
            block, diagnostics, retrieved = retrieve_block(
                memory, store, embedder, cfg, world, seed
            )
            out = cfg.output_dir / f"build_round_{round_no:02d}" / world / \
                f"trajectory_seed{seed}_skillrl.json"
            result = rollout.play_and_record(
                world, seed, block, out_path=out,
                retrieval_diagnostics=diagnostics, **episode_kwargs(cfg)
            )
            if result is None:
                continue
            print(f"  seed {seed}: {'PASS' if result.get('success') else 'FAIL'}  "
                  f"expl={result.get('expl_score')}  "
                  f"reward={float(result.get('reward') or 0):+.2f}  "
                  f"+{result.get('added_context_tokens')} tok  "
                  f"[{diagnostics.get('mode') or diagnostics.get('phase_a')}]")
            # MemRL's Monte-Carlo utility update, applied to every memory that
            # was injected this episode. Episodic RAG has no utility signal and
            # its `update_utilities` is a documented no-op, so both methods run
            # the same loop. Applied immediately rather than with the batched
            # write-back: Q belongs to a memory slot that already exists, so
            # deferring it would mean the next round ranks on stale utilities.
            #
            # Skipped for an episode already in the ledger. A resumed run
            # replays finished episodes from disk, and Q is cumulative -- so
            # updating again would add a second visit for a single episode and
            # `q_visits` would grow on every relaunch, which is exactly what a
            # three-pass stub run showed (16 -> 30 -> 44).
            if f"online:{world}:{seed}" not in ledger:
                upd = memory.update_utilities(retrieved, result)
                if upd.get("updates"):
                    result["utility_updates"] = upd
            pending.append((seed, result))
            results.append(result)
        # Write back in seed order, after the whole round -- HeXA's granularity.
        for seed, result in pending:
            write_back(memory, store, embedder, cfg, world, seed, result, index,
                       ledger=ledger)
            index += 1
        store.save()
        _save_ledger(cfg, ledger)
    return results


def run_eval_pass(memory, store, embedder, cfg: ArmConfig) -> list[dict]:
    """Held-out episodes against the frozen memory. Nothing is written back."""
    print(f"\n=== Eval: {cfg.eval_world} seeds {cfg.eval_seeds} "
          f"({len(store.for_level(cfg.scope_level))} memories, FROZEN) ===")
    results = []
    for seed in cfg.eval_seeds:
        # No `update_utilities` here: the memory is frozen, so Q stays at the
        # value the build pass learned. Letting eval episodes move Q would make
        # a held-out seed's result depend on the seeds evaluated before it.
        block, diagnostics, _retrieved = retrieve_block(
            memory, store, embedder, cfg, cfg.eval_world, seed
        )
        out = cfg.output_dir / "eval_frozen" / cfg.eval_world / \
            f"trajectory_seed{seed}_skillrl.json"
        result = rollout.play_and_record(
            cfg.eval_world, seed, block, out_path=out,
            retrieval_diagnostics=diagnostics, **episode_kwargs(cfg)
        )
        if result is None:
            continue
        print(f"  seed {seed}: {'PASS' if result.get('success') else 'FAIL'}  "
              f"expl={result.get('expl_score')}  "
              f"reward={float(result.get('reward') or 0):+.2f}  "
              f"+{result.get('added_context_tokens')} tok  "
              f"[{diagnostics.get('mode') or diagnostics.get('phase_a')}]")
        results.append(result)
    return results


def summarise(results: Sequence[dict]) -> dict:
    """Means over completed episodes. Errored episodes never reach here."""
    if not results:
        return {"n": 0}
    n = len(results)
    expl = [float(r.get("expl_score") or 0.0) for r in results]
    reward = [float(r.get("reward") or 0.0) for r in results]
    added = [int(r.get("added_context_tokens") or 0) for r in results]
    return {
        "n": n,
        "pass_rate": sum(1 for r in results if r.get("success")) / n,
        "n_pass": sum(1 for r in results if r.get("success")),
        "mean_expl_score": round(sum(expl) / n, 4),
        "mean_reward": round(sum(reward) / n, 4),
        "mean_added_tokens": round(sum(added) / n, 1),
        "n_empty_block": sum(
            1 for r in results
            if (r.get("retrieval_diagnostics") or {}).get("n_empty_block")
        ),
    }
