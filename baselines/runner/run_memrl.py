"""MemRL on InterPhyre.

    # 1. calibrate the similarity constants for this level (cheap, no rollouts)
    python -m baselines.calibrate --level pass_the_parcel --seeds 1-105

    # 2. run, passing the printed constants
    python -m baselines.runner.run_memrl \
        --level pass_the_parcel \
        --seeds 6-55 \
        --initial-traj-dir Initial_trajectories/pass_the_parcel \
        --output-dir baselines/data/memrl_ptp \
        --delta 0.71 --sim-norm-mean 0.66 --sim-norm-std 0.05

Omitting the three constants makes the runner read
`<cache-dir>/calibration_<level>.json` automatically; it refuses to fall back to
the ALFWorld defaults silently.

`--lambda-q 0` reduces MemRL to Q-blind similarity retrieval, which is the
runtime-RL ablation (MemRL vs MemRL-lambda0) at no extra implementation cost.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines.calibrate import parse_seed_spec
from baselines.memory.blocks import format_memrl_block
from baselines.memory.embedder import DEFAULT_MODEL, Embedder
from baselines.memory.intent import (
    DEFAULT_MODE as DEFAULT_INTENT_MODE,
    MODES as INTENT_MODES,
    IntentCache,
    cache_path,
)
from baselines.memory.memrl import MemRL, MemRLConfig
from baselines.memory.store import MemoryStore
from baselines.memory.summarize import (
    REFLECTION_TEMPERATURE,
    SCRIPT_TEMPERATURE,
    make_model_fn_summarizer,
    make_summarizer,
    mock_summarizer,
)
from baselines.runner.stream import (
    StreamConfig,
    load_offline_trajectories,
    run_stream,
)


#: Default retrieval key: the embedder MemRL's own release specifies
#: (`configs/rl_alf_config.yaml` -> embedding.model). Needs OPENAI_API_KEY, which
#: `baselines/bash_scripts/_baseline_env.sh` sources from outside the repo.
#: Episodic RAG defaults to jina instead, since that is what the CORE paper used
#: for *that* baseline — so each method runs on the embedder its own source names.
#: CAVEAT, measured on 105 seeds/level with jina: text embeddings of InterPhyre
#: intents rank instances near-randomly (Spearman rho +0.11 to +0.45 vs the
#: geometry) because intents in a level differ only in their numbers, and the
#: resulting delta made Phase A reject ~2/3 of episodes. Re-check the calibration
#: output for this embedder before trusting a run; `--embedding-model numeric`
#: keys on the standardized geometry and is better conditioned.
DEFAULT_KEY = "openai:text-embedding-3-large"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MemRL baseline on InterPhyre (official ALFWorld configuration)"
    )
    p.add_argument("--level", required=True)
    p.add_argument("--seeds", default="6-55")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--initial-traj-dir", default=None)
    p.add_argument(
        "--model", default="claude",
        help="Agent model: claude, mock, openai:<name>, or an HF path/name "
             "(e.g. Qwen2.5-7B-Instruct)",
    )
    p.add_argument(
        "--summarizer", default="auto",
        choices=("auto", "claude", "agent-model", "mock"),
        help="Backend for memory-building/reflection calls. auto = claude when "
             "--model is claude, otherwise reuse the agent model (which honours "
             "MemRL's 0.0/0.3 temperatures)",
    )
    p.add_argument(
        "--teacher-model", default="claude-sonnet-4-6",
        help="Model for the claude summarizer backend",
    )
    p.add_argument(
        "--summarizer-max-tokens", type=int, default=1024,
        help="Generation cap for the agent-model summarizer backend",
    )
    p.add_argument(
        "--embedding-model", default=DEFAULT_KEY,
        help="Retrieval key: default openai:text-embedding-3-large (MemRL's own "
             "config), a local sentence-transformers model such as "
             "jinaai/jina-embeddings-v2-base-en, or 'numeric' for the "
             "standardized geometry",
    )
    p.add_argument("--cache-dir", default="baselines/data/cache")
    p.add_argument(
        "--intent-mode", default=DEFAULT_INTENT_MODE, choices=INTENT_MODES,
        help="Retrieval key construction; must match the calibration run",
    )
    p.add_argument("--seeds-per-round", type=int, default=3)
    p.add_argument("--max-iterations", type=int, default=25)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-new-tokens", type=int, default=800)

    # ── MemRL hyperparameters (defaults = configs/rl_alf_config.yaml) ──
    p.add_argument("--k1", type=int, default=5, help="Phase-A recall size")
    p.add_argument("--k2", type=int, default=3, help="Phase-B select size")
    p.add_argument("--delta", type=float, default=None, help="Phase-A similarity threshold")
    p.add_argument("--sim-norm-mean", type=float, default=None)
    p.add_argument("--sim-norm-std", type=float, default=None)
    p.add_argument("--lambda-q", type=float, default=0.5, help="Q weight (1 - weight_sim)")
    p.add_argument("--alpha", type=float, default=0.3, help="Utility learning rate")
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--q-init", type=float, default=0.0)
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--q-min-threshold", type=float, default=-10.0)
    p.add_argument("--success-reward", type=float, default=1.0)
    p.add_argument("--failure-reward", type=float, default=-1.0)
    p.add_argument(
        "--use-shaped-reward", action="store_true",
        help="Use HeXA's iteration-aware reward in [-1,1] instead of +-1",
    )
    p.add_argument(
        "--add-similarity-threshold", type=float, default=0.99,
        help="Merge-on-write threshold. 0.99 is their LifelongBench value; "
             "their ALFWorld config uses 0.90, which collapses the bank on "
             "InterPhyre (99-100%% of scene pairs exceed it)",
    )
    p.add_argument("--novelty-threshold", type=float, default=0.85)
    p.add_argument("--reward-merge-gain", type=float, default=0.1)
    p.add_argument("--max-traj-steps", type=int, default=12)
    p.add_argument("--max-obs-chars", type=int, default=600)
    p.add_argument("--random-seed", type=int, default=42)

    p.add_argument("--freeze-after", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    return p


def resolve_calibration(args) -> tuple[float, float, float, str]:
    """Return (delta, sim_norm_mean, sim_norm_std, provenance)."""
    if (
        args.delta is not None
        and args.sim_norm_mean is not None
        and args.sim_norm_std is not None
    ):
        return args.delta, args.sim_norm_mean, args.sim_norm_std, "cli"

    from baselines.memory.features import calibration_path

    path = calibration_path(
        Path(args.cache_dir), args.level, args.intent_mode, args.embedding_model
    )
    if not path.exists():
        raise SystemExit(
            f"No calibration for {args.level} at {path}, and --delta / "
            f"--sim-norm-mean / --sim-norm-std were not all given.\n"
            f"Run:  python -m baselines.calibrate --level {args.level} "
            f"--seeds 1-105 --intent-mode {args.intent_mode} "
            f"--embedding-model {args.embedding_model}"
        )
    data = json.loads(path.read_text())
    # The constants are only meaningful for the exact key construction they were
    # measured on: a different embedder or intent mode gives a different
    # similarity distribution, so silently reusing them would be wrong.
    if data.get("embedding_model") != args.embedding_model:
        raise SystemExit(
            f"Calibration in {path} was computed with "
            f"{data.get('embedding_model')}, but this run uses "
            f"{args.embedding_model}. Re-run baselines.calibrate."
        )
    if data.get("intent_mode", "full") != args.intent_mode:
        raise SystemExit(
            f"Calibration in {path} was computed with intent mode "
            f"{data.get('intent_mode')}, but this run uses {args.intent_mode}. "
            f"Re-run baselines.calibrate."
        )
    rec = data["recommended"]
    return (
        args.delta if args.delta is not None else rec["delta"],
        args.sim_norm_mean if args.sim_norm_mean is not None else rec["sim_norm_mean"],
        args.sim_norm_std if args.sim_norm_std is not None else rec["sim_norm_std"],
        str(path),
    )


def deviations(args, summarizer_backend: str) -> list[str]:
    """Record how this run differs from the published configurations."""
    out = key_deviations(args)
    if summarizer_backend == "claude":
        out.append(
            "memory-building / reflection calls go through `claude -p`, which "
            "exposes no temperature flag, so MemRL's 0.0 / 0.3 split is not "
            "reproducible on this backend"
        )
    if summarizer_backend == "mock":
        out.append("SMOKE TEST ONLY: summarizer is mocked, no LLM memory-building")
    out.append(
        f"actor is {args.model}; the CORE paper's baselines used gpt-oss-120b"
    )
    return out


def key_deviations(args) -> list[str]:
    """State the retrieval-key choice, since it is the one substantive
    departure from how both papers key their memories."""
    if args.embedding_model == "numeric":
        return [
            "retrieval is keyed on the standardized scene geometry, not on an "
            "embedded task description as in the papers: measured on 105 seeds "
            "per level, text embeddings of InterPhyre intents rank instances "
            "near-randomly (Spearman rho +0.11 to +0.40 vs the geometry, ~10% "
            "top-1 overlap), because intents within a level differ only in "
            "their numbers. See baselines/memory/features.py",
        ]
    return [
        f"retrieval key is the text embedder {args.embedding_model}; note that "
        f"on InterPhyre this ranks instances close to randomly (see the "
        f"rank_agreement figures in the level's calibration JSON)",
    ]


def build_embedder(args):
    """Text embedder, or the numeric geometry key.

    `numeric` needs the scaler fitted during calibration, so it fails loudly
    rather than silently re-fitting on whatever seeds this run happens to see —
    a scaler fitted on a different pool would make Q values incomparable across
    runs.
    """
    from baselines.memory.features import FeatureScaler, NumericEmbedder, scaler_path

    if args.embedding_model != NumericEmbedder.name:
        return Embedder(args.embedding_model, cache_dir=Path(args.cache_dir))

    if args.intent_mode != "geometry":
        raise SystemExit(
            "--embedding-model numeric requires --intent-mode geometry."
        )
    path = scaler_path(Path(args.cache_dir), args.level, args.intent_mode)
    if not path.exists():
        raise SystemExit(
            f"No feature scaler at {path}.\nRun:  python -m baselines.calibrate "
            f"--level {args.level} --seeds 1-105 --intent-mode geometry "
            f"--embedding-model numeric"
        )
    scaler = FeatureScaler.load(path)
    print(f"  Numeric key: {scaler.dim} standardized geometry features")
    return NumericEmbedder(scaler)


def main() -> None:
    args = build_parser().parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    delta, sim_mean, sim_std, provenance = resolve_calibration(args)
    print(
        f"  Calibration ({provenance}): delta={delta:.4f} "
        f"sim_norm_mean={sim_mean:.4f} sim_norm_std={sim_std:.4f}"
    )

    seeds = parse_seed_spec(args.seeds)
    store = MemoryStore(out_dir / "memory_store.json")
    embedder = build_embedder(args)
    intents = IntentCache(
        cache_path(cache_dir, args.level, args.intent_mode), args.intent_mode
    )

    cfg = MemRLConfig(
        k1=args.k1,
        k2=args.k2,
        delta=delta,
        sim_norm_mean=sim_mean,
        sim_norm_std=sim_std,
        weight_q=args.lambda_q,
        weight_sim=1.0 - args.lambda_q,
        epsilon=args.epsilon,
        q_min_threshold=args.q_min_threshold,
        alpha=args.alpha,
        gamma=args.gamma,
        q_init=args.q_init,
        success_reward=args.success_reward,
        failure_reward=args.failure_reward,
        use_shaped_reward=args.use_shaped_reward,
        add_similarity_threshold=args.add_similarity_threshold,
        novelty_threshold=args.novelty_threshold,
        reward_merge_gain=args.reward_merge_gain,
        max_traj_steps=args.max_traj_steps,
        max_obs_chars=args.max_obs_chars,
        seed=args.random_seed,
    )

    from skillrl.runner.run_skill_agent import _load_model

    model_fn = _load_model(args.model)
    is_oss = "gpt-oss" in args.model.lower()

    backend = args.summarizer
    if backend == "auto":
        backend = "claude" if args.model == "claude" else "agent-model"

    if backend == "mock":
        summarizer = mock_summarizer
        summarizer_info = {"backend": "mock"}
    elif backend == "claude":
        summarizer = make_summarizer(model=args.teacher_model, verbose=args.verbose)
        summarizer_info = {"backend": "claude", "model": args.teacher_model}
    else:
        summarizer = make_model_fn_summarizer(
            model_fn,
            max_new_tokens=args.summarizer_max_tokens,
            verbose=args.verbose,
        )
        summarizer_info = {
            "backend": "agent-model",
            "model": args.model,
            "script_temperature": SCRIPT_TEMPERATURE,
            "reflection_temperature": REFLECTION_TEMPERATURE,
            "max_new_tokens": args.summarizer_max_tokens,
        }
    print(f"  Summarizer: {summarizer_info}")

    memory = MemRL(store, cfg, summarizer=summarizer)

    manifest = {
        "method": "memrl",
        "spec_source": (
            "MemTensor/MemRL official release, configs/rl_alf_config.yaml "
            "(ALFWorld: the closest analogue to InterPhyre)"
        ),
        "agent_model": args.model,
        "summarizer": summarizer_info,
        "embedding_model": args.embedding_model,
        "intent_mode": args.intent_mode,
        "calibration_source": provenance,
        "config": cfg.__dict__,
        "seeds": args.seeds,
        "initial_traj_dir": args.initial_traj_dir,
        "max_iterations": args.max_iterations,
        "deviations": deviations(args, backend),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    config = StreamConfig(
        level=args.level,
        seeds=seeds,
        output_dir=out_dir,
        method="memrl",
        seeds_per_round=args.seeds_per_round,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        is_oss=is_oss,
        verbose=args.verbose,
        freeze_after=args.freeze_after,
    )

    run_stream(
        config=config,
        memory=memory,
        store=store,
        embedder=embedder,
        intents=intents,
        model_fn=model_fn,
        block_formatter=format_memrl_block,
        offline_trajectories=load_offline_trajectories(
            Path(args.initial_traj_dir) if args.initial_traj_dir else None
        ),
        manifest=manifest,
    )


if __name__ == "__main__":
    main()
