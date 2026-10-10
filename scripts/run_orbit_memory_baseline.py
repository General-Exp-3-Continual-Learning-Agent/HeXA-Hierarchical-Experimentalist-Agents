#!/usr/bin/env python
"""Episodic RAG / MemRL on ORBIT Maze and Mastermind.

The memory-baseline counterpart to `run_orbit_maze_hexa.py`: same tasks, same
pools, same actor, same frozen-eval protocol, so the arm is paired per
`instance_id` with the no-bank baseline and with Design B.

    # 0. calibrate MemRL's similarity constants (cheap, no rollouts, no server)
    python -m baselines.orbit.calibrate --task mastermind

    # 1. server up:  sbatch scripts/serve_gpt_oss_vllm.sh

    # 2. run
    python scripts/run_orbit_memory_baseline.py --task mastermind --method memrl \
        --train-size 80 --eval-size 40 --out results/mastermind_memrl

    # plumbing check, no server, oracle plays and the summarizer is mocked:
    python scripts/run_orbit_memory_baseline.py --task maze --method episodic_rag \
        --train-size 3 --eval-size 2 --stub --out /tmp/smoke

Two passes, in this order (see `baselines/orbit/stream.py`):

  build   train-pool instances, serially, memory growing
  eval    held-out instances, memory frozen

`--skip-build` evaluates an existing store; `--skip-eval` stops after building,
and a later run with the same `--out` picks the store up and evaluates it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hexa_core.tokens import TOKENIZER_NOTE  # noqa: E402
from hexa_orbit import report  # noqa: E402
from hexa_orbit.actor import ActorConfig  # noqa: E402

from baselines.memory.memrl import MemRL, MemRLConfig  # noqa: E402
from baselines.memory.rag import EpisodicRAG, RAGConfig  # noqa: E402
from baselines.memory.store import MemoryStore  # noqa: E402
from baselines.memory.summarize import mock_summarizer  # noqa: E402
from baselines.orbit import blocks, stream, summarize  # noqa: E402
from baselines.orbit.embedding import (  # noqa: E402
    WORKING_DEFAULT,
    build_embedder,
    embedder_note,
)
from baselines.orbit.calibrate import calibration_path  # noqa: E402
from baselines.orbit.intent import (  # noqa: E402
    MODES as INTENT_MODES,
    IntentCache,
    cache_path,
    default_mode,
)
from baselines.orbit.tasks import TASKS, build_task  # noqa: E402

METHODS = ("episodic_rag", "memrl")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=Path("baselines/data/cache"))

    # ── pools ──
    p.add_argument("--train-size", type=int, default=None,
                   help="instances memory is built from. Default: 80 for "
                        "Mastermind (the whole non-eval half of the 120-code "
                        "space), 200 for Maze")
    p.add_argument("--eval-size", type=int, default=None,
                   help="held-out instances. Default: each task's full eval "
                        "pool -- 40 Mastermind, 256 Maze")

    # ── actor (defaults are ORBIT's, shared with every other arm) ──
    p.add_argument("--model", default=ActorConfig.model)
    p.add_argument("--base-url", default=None)
    p.add_argument("--temperature", type=float, default=ActorConfig.temperature)
    p.add_argument("--top-p", type=float, default=ActorConfig.top_p)
    p.add_argument("--max-response-tokens", type=int, default=None,
                   help="Default 2048 on Maze (ORBIT's cap) and 4096 on "
                        "Mastermind, where reasoning runs long enough that 2048 "
                        "truncates 6-15%% of turns and a truncated turn emits no "
                        "guess")
    p.add_argument("--reasoning-effort", default=ActorConfig.reasoning_effort,
                   help="pass '' for models with no reasoning channel")
    p.add_argument("--transcript-mode", choices=["final", "raw"], default="final")
    p.add_argument("--trajectory-token-budget", type=int, default=None)
    p.add_argument("--truncation-scoring", choices=["exclude", "orbit"],
                   default="exclude")
    p.add_argument("--eval-concurrency", type=int, default=8,
                   help="instances in flight during the frozen pass, where "
                        "memory is read-only and concurrency cannot change the "
                        "result")
    p.add_argument("--build-batch-size", type=int, default=1,
                   help="instances played together during the BUILD pass. 1 is "
                        "the strict stream both papers specify. Above 1 they "
                        "retrieve against the memory as of the batch start and "
                        "write back in input order, so the run stays "
                        "reproducible -- HeXA's round structure. Leave it at 1 "
                        "on gpt-oss-120b (~0.4 min/instance, so a 200-instance "
                        "build is <2h); raise it for a slow actor such as "
                        "Qwen3-8B with thinking (~3.5 min/instance). Recorded "
                        "as a deviation")
    p.add_argument("--seed", type=int, default=None,
                   help="actor sampling seed. Vary it across otherwise "
                        "identical eval passes to get the replicate spread the "
                        "Mastermind arms are reported with (n=40 is small "
                        "enough that one pass is eval noise)")
    p.add_argument("--eval-dir", default="eval_frozen",
                   help="subdirectory the frozen eval writes to. Give each "
                        "replicate its own (eval_frozen_run0, ...) so they "
                        "share one built memory instead of rebuilding it")

    # ── retrieval ──
    p.add_argument("--embedding-model", default=WORKING_DEFAULT,
                   help="Text embedder for the retrieval key, or 'hashing' for "
                        "a dependency-free character n-gram key. There is no "
                        "'numeric' mode here: the ORBIT equivalent of "
                        "InterPhyre's geometry is the hidden maze or code. The "
                        "CORE paper's jina-v2 is not the default because it does "
                        "not load under transformers >= 5 -- see "
                        "baselines/orbit/embedding.py")
    p.add_argument("--intent-mode", default=None, choices=INTENT_MODES,
                   help="must match the calibration run. Default is per task: "
                        "salient for maze (3x the similarity spread of the full "
                        "observation), observation for mastermind (whose salient "
                        "key is empty). See intent.py:DEFAULT_MODE_BY_TASK")

    # ── Episodic RAG ──
    p.add_argument("--k-success", type=int, default=1)
    p.add_argument("--k-failure", type=int, default=1)
    p.add_argument("--max-traj-steps", type=int, default=12,
                   help="steps kept PER EPISODE of a stored instance (the "
                        "grouping is why: see format_trajectory). 12 exceeds "
                        "both tasks' per-episode cap, so nothing is dropped by "
                        "default and stored transcripts are verbatim, as the "
                        "CORE paper specifies")
    p.add_argument("--max-obs-chars", type=int, default=600)

    # ── MemRL ──
    p.add_argument("--k1", type=int, default=5, help="Phase-A recall size")
    p.add_argument("--k2", type=int, default=3, help="Phase-B select size")
    p.add_argument("--delta", type=float, default=None)
    p.add_argument("--sim-norm-mean", type=float, default=None)
    p.add_argument("--sim-norm-std", type=float, default=None)
    p.add_argument("--lambda-q", type=float, default=0.5,
                   help="Q weight (1 - weight_sim). 0 is the Q-blind ablation")
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--q-init", type=float, default=0.0)
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--q-min-threshold", type=float, default=-10.0)
    p.add_argument("--add-similarity-threshold", type=float, default=None,
                   help="Merge-on-write threshold. Defaults to 0.99, or to "
                        "1.01 (never merge) when the calibration reports a "
                        "degenerate key -- otherwise every write collapses onto "
                        "one record per outcome class")
    p.add_argument("--novelty-threshold", type=float, default=0.85)
    p.add_argument("--reward-merge-gain", type=float, default=0.1)
    p.add_argument("--binary-reward", action="store_true",
                   help="score Q updates +-1 on solved-at-all, MemRL's own "
                        "config. Default is ORBIT's graded instance reward, "
                        "which is what every other arm on these tasks is scored "
                        "with and what distinguishes an episode-1 solve from an "
                        "episode-3 one")
    p.add_argument("--summarizer-max-tokens", type=int,
                   default=summarize.DEFAULT_SUMMARIZER_MAX_TOKENS,
                   help="generation cap per memory-building call. 4096 because "
                        "a thinking model spends its budget in the reasoning "
                        "channel first: at 1024, Qwen3-8B never reached its "
                        "answer channel on 44 of 80 Mastermind memories and the "
                        "store filled with raw chain of thought")
    p.add_argument("--random-seed", type=int, default=42)

    # ── flow ──
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--stub", action="store_true",
                   help="oracle actor and mocked summarizer; no server needed")
    p.add_argument("--verbose", action="store_true")
    return p


#: Per-task defaults that differ from ORBIT's shared ones, with the reason.
TASK_DEFAULTS = {
    "maze": {"train_size": 200, "eval_size": 256, "max_response_tokens": 2048},
    # 80 is the whole non-eval half of the 120-code space; asking for more raises.
    "mastermind": {"train_size": 80, "eval_size": 40, "max_response_tokens": 4096},
}


def resolve_calibration(args) -> tuple[float, float, float, bool, str]:
    """(delta, sim_norm_mean, sim_norm_std, degenerate, provenance).

    Refuses to fall back to MemRL's ALFWorld constants.  They were measured on a
    different similarity distribution, and on a degenerate key the paper's delta
    rejects the entire store — MemRL would then inject nothing and the run would
    report the no-bank arm under this method's name.
    """
    explicit = (
        args.delta is not None
        and args.sim_norm_mean is not None
        and args.sim_norm_std is not None
    )
    path = calibration_path(
        args.cache_dir, args.task, args.intent_mode, args.embedding_model
    )
    if not path.exists():
        if explicit:
            return args.delta, args.sim_norm_mean, args.sim_norm_std, False, "cli"
        raise SystemExit(
            f"No calibration for {args.task} at {path}, and --delta / "
            f"--sim-norm-mean / --sim-norm-std were not all given.\n"
            f"Run:  python -m baselines.orbit.calibrate --task {args.task} "
            f"--intent-mode {args.intent_mode} "
            f"--embedding-model {args.embedding_model}"
        )

    data = json.loads(path.read_text())
    if data.get("embedding_model") != args.embedding_model:
        raise SystemExit(
            f"Calibration in {path} used {data.get('embedding_model')}, this run "
            f"uses {args.embedding_model}. Re-run baselines.orbit.calibrate."
        )
    if data.get("intent_mode") != args.intent_mode:
        raise SystemExit(
            f"Calibration in {path} used intent mode {data.get('intent_mode')}, "
            f"this run uses {args.intent_mode}. Re-run baselines.orbit.calibrate."
        )
    rec = data["recommended"]
    return (
        args.delta if args.delta is not None else rec["delta"],
        args.sim_norm_mean if args.sim_norm_mean is not None else rec["sim_norm_mean"],
        args.sim_norm_std if args.sim_norm_std is not None else rec["sim_norm_std"],
        bool(data.get("degenerate")),
        "cli" if explicit else str(path),
    )


def deviations(args, *, degenerate: bool, merge_threshold: float) -> list[str]:
    """How this run differs from the published configurations. Always logged."""
    out = [
        "the memory unit is a three-episode ORBIT instance, not a single "
        "episode: within-instance memory is ORBIT's own cumulative chat, which "
        "every arm keeps, so what these methods add is cross-instance memory",
        f"retrieval is keyed on the episode-1 opening observation "
        f"({args.intent_mode} mode). Both papers key on a task description; the "
        f"ORBIT analogue would be the hidden maze or code, which the agent never "
        f"sees and which would hand the baseline the answer",
        embedder_note(args.embedding_model),
        f"actor is {args.model}; the CORE paper's baselines used gpt-oss-120b",
    ]
    if degenerate:
        out.append(
            "RETRIEVAL IS DEGENERATE on this task: every instance embeds to the "
            "same key, so Episodic RAG retrieves an arbitrary past success and "
            "failure, and MemRL's Phase A admits the whole store and selects on "
            "Q alone. delta was lowered below the observed floor so Phase A "
            "admits at all; the paper's top-20% quantile would reject everything"
        )
    if args.method == "memrl":
        if merge_threshold > 1.0:
            out.append(
                f"MemRL add_similarity_threshold set to {merge_threshold} (never "
                f"merge). Its configured 0.90-0.99 would fold every write into "
                f"one record per outcome class on a key this dense"
            )
        if not args.binary_reward:
            out.append(
                "Q updates use ORBIT's graded instance reward "
                "(hexa_orbit.reward.instance_reward) rather than MemRL's +-1, so "
                "utilities are on the same scale as every other arm on this task"
            )
        out.append(
            "memory-building and reflection calls go to the actor model at "
            "MemRL's own 0.0 / 0.3 temperatures"
        )
    if args.build_batch_size > 1:
        out.append(
            f"memory was updated every {args.build_batch_size} instances rather "
            f"than after every one: instances in a batch retrieved against the "
            f"memory as of the batch start and wrote back in input order. Both "
            f"papers update per episode; HeXA's rounds do exactly this"
        )
    if args.stub:
        out.append("SMOKE TEST ONLY: oracle actor and mocked summarizer")
    return out


def stub_client_factory(orbit_task):
    """A scripted client per instance, so the whole path runs without a server.

    Not a baseline and not a bound anyone should quote -- on Maze it plays the
    known-map optimum, which no real agent can do on episode 1. It exists to
    prove the plumbing end to end, and specifically to exercise the *success*
    path: MemRL's script branch and the SUCCESS record format are only reached
    when something actually solves an instance.

    Mastermind has no oracle -- the code is hidden, and letting the stub read it
    would make the smoke test exercise a leak rather than the plumbing. It
    cycles distinct legal guesses instead, which drives three episodes to their
    turn cap and exercises the failure path.
    """
    from hexa_core.llm.base import Generation
    from hexa_orbit import oracle

    GUESSES = ["1 2 3", "4 5 6", "1 4 2", "3 6 5", "2 1 4",
               "5 3 6", "6 2 1", "4 3 5", "2 5 1"]

    class Stub:
        def __init__(self, instance):
            self.instance = instance
            self.queue: list[str] = []
            self.index = 0

        def generate_structured(self, *, messages, **_):
            if orbit_task.name == "mastermind":
                guess = GUESSES[self.index % len(GUESSES)]
                self.index += 1
                return Generation.from_parts(
                    "", oracle.boxed(guess), finish_reason="stop"
                )
            if not self.queue:
                self.queue = oracle.instance_optimal_actions(self.instance)
            return Generation.from_parts(
                "", oracle.boxed(self.queue.pop(0)), finish_reason="stop"
            )

    return Stub


def main() -> int:
    args = build_parser().parse_args()
    defaults = TASK_DEFAULTS[args.task]
    if args.intent_mode is None:
        args.intent_mode = default_mode(args.task)
    if args.train_size is None:
        args.train_size = defaults["train_size"]
    if args.eval_size is None:
        args.eval_size = defaults["eval_size"]
    if args.max_response_tokens is None:
        args.max_response_tokens = defaults["max_response_tokens"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    orbit_task = build_task(args.task)
    train_specs = orbit_task.train_specs(args.train_size)
    eval_specs = orbit_task.eval_specs(args.eval_size)
    orbit_task.assert_disjoint(train_specs, eval_specs)

    actor_config = ActorConfig(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        top_p=args.top_p,
        max_response_tokens=args.max_response_tokens,
        # '' from the CLI means "no reasoning channel"; it must become None, or
        # VLLMClient sends reasoning_effort="".
        reasoning_effort=args.reasoning_effort or None,
        transcript_mode=args.transcript_mode,
        trajectory_token_budget=args.trajectory_token_budget,
        truncation_scoring=args.truncation_scoring,
        seed=args.seed,
    )

    store = MemoryStore(out_dir / "memory_store.json")
    embedder = build_embedder(args.embedding_model, cache_dir=args.cache_dir)
    intents = IntentCache(
        cache_path(args.cache_dir, args.task, args.intent_mode), args.intent_mode
    )

    degenerate = False
    summary_quality = None
    if args.method == "episodic_rag":
        memory = EpisodicRAG(store, RAGConfig(
            k_success=args.k_success,
            k_failure=args.k_failure,
            max_obs_chars=args.max_obs_chars,
            max_steps=args.max_traj_steps or None,
        ))
        block_formatter = blocks.format_rag_block
        method_config = memory.cfg.__dict__
        merge_threshold = float("nan")
    else:
        delta, sim_mean, sim_std, degenerate, provenance = resolve_calibration(args)
        print(f"  Calibration ({provenance}): delta={delta:.4f} "
              f"sim_norm_mean={sim_mean:.4f} sim_norm_std={sim_std:.4f}"
              f"{'  [DEGENERATE KEY]' if degenerate else ''}")
        merge_threshold = (
            args.add_similarity_threshold
            if args.add_similarity_threshold is not None
            else (1.01 if degenerate else 0.99)
        )
        model_fn = summarize.make_vllm_model_fn(actor_config)
        summary_quality = summarize.SummaryQuality()
        summarizer = (
            mock_summarizer if args.stub
            else summarize.make_summarizer(
                model_fn, args.task,
                max_new_tokens=args.summarizer_max_tokens,
                verbose=args.verbose,
                quality=summary_quality,
            )
        )
        memory = MemRL(store, MemRLConfig(
            k1=args.k1, k2=args.k2, delta=delta,
            sim_norm_mean=sim_mean, sim_norm_std=sim_std,
            weight_q=args.lambda_q, weight_sim=1.0 - args.lambda_q,
            epsilon=args.epsilon, q_min_threshold=args.q_min_threshold,
            alpha=args.alpha, gamma=args.gamma, q_init=args.q_init,
            use_shaped_reward=not args.binary_reward,
            add_similarity_threshold=merge_threshold,
            novelty_threshold=args.novelty_threshold,
            reward_merge_gain=args.reward_merge_gain,
            max_traj_steps=args.max_traj_steps or None,
            max_obs_chars=args.max_obs_chars,
            seed=args.random_seed,
        ), summarizer=summarizer)
        block_formatter = blocks.format_memrl_block
        method_config = memory.cfg.__dict__

    client_factory = stub_client_factory(orbit_task) if args.stub else None

    manifest = {
        "method": args.method,
        "task": args.task,
        "spec_source": (
            "CORE paper section 4.1 (retrieve entire past rollouts)"
            if args.method == "episodic_rag"
            else "MemTensor/MemRL official release, configs/rl_alf_config.yaml"
        ),
        "protocol": (
            "build memory on the train pool, then evaluate frozen on the "
            "held-out pool — hexa_orbit.loop.frozen_eval's shape, so this arm "
            "is paired per instance_id with the no-bank and Design B arms"
        ),
        "actor": actor_config.to_dict(),
        "embedding_model": args.embedding_model,
        "intent_mode": args.intent_mode,
        "degenerate_retrieval": degenerate,
        "config": dict(method_config),
        "n_train": len(train_specs),
        "n_eval": len(eval_specs),
        "build_batch_size": args.build_batch_size,
        "eval_dir": args.eval_dir,
        "stub": args.stub,
        "deviations": deviations(
            args, degenerate=degenerate, merge_threshold=merge_threshold
        ),
        "token_note": TOKENIZER_NOTE,
        **orbit_task.manifest(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    for note in manifest["deviations"]:
        print(f"  [deviation] {note}")

    started = time.monotonic()

    if not args.skip_build:
        print(f"\n{'=' * 70}\n  BUILD — {len(train_specs)} train instances, "
              f"{'serial' if args.build_batch_size == 1 else f'batches of {args.build_batch_size}'}"
              f"\n{'=' * 70}")
        stream.build_memory(
            orbit_task, train_specs,
            memory=memory, store=store, embedder=embedder, intents=intents,
            block_formatter=block_formatter, actor_config=actor_config,
            out_dir=out_dir / "build", method=args.method,
            client_factory=client_factory, batch_size=args.build_batch_size,
            verbose=args.verbose,
        )

    if args.skip_eval:
        print(f"\nbuilt only; wrote {out_dir}")
        return 0

    if not store.records:
        print("\nmemory is empty; nothing to evaluate")
        return 1

    print(f"\n{'=' * 70}\n  FROZEN EVAL — {len(eval_specs)} held-out instances, "
          f"concurrency {args.eval_concurrency}\n{'=' * 70}")
    summary = stream.frozen_eval(
        orbit_task, eval_specs,
        memory=memory, store=store, embedder=embedder, intents=intents,
        block_formatter=block_formatter, actor_config=actor_config,
        out_dir=out_dir / args.eval_dir, train_specs=train_specs,
        concurrency=args.eval_concurrency, client_factory=client_factory,
    )
    summary["manifest"] = manifest
    summary["wall_clock_s"] = round(time.monotonic() - started, 1)
    if summary_quality is not None and summary_quality.total:
        # On the record, not just in the log: a reader of summary_*.json must be
        # able to see whether the memories this arm injected were real summaries.
        summary["summarizer_quality"] = summary_quality.to_dict()
    (out_dir / f"summary_{args.method}_{args.eval_dir}.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    print()
    print(report.format_summary(
        summary["report"],
        title=f"{args.task} — {args.method}, frozen memory "
              f"({'ORACLE STUB' if args.stub else args.model})",
    ))
    if summary.get("summarizer_quality"):
        print(f"summarizer: {summary['summarizer_quality']}")
    print(f"\nmemory: {summary['memory_stats']}")
    print(f"retrieval: {summary['retrieval']}")
    print(f"\nwall clock {summary['wall_clock_s']:.0f}s -> {out_dir}")
    print("\nnext: diff against the no-bank arm on clean episode-1, paired by "
          "instance_id. Run run_orbit_maze_baseline.py / "
          "run_orbit_mastermind_baseline.py on the same --eval-size.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
