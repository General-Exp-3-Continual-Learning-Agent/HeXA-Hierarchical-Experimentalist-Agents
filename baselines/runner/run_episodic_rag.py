"""Episodic RAG on InterPhyre.

    python -m baselines.runner.run_episodic_rag \
        --level pass_the_parcel \
        --seeds 6-55 \
        --initial-traj-dir Initial_trajectories/pass_the_parcel \
        --output-dir baselines/data/episodic_rag_ptp

Memory holds every past rollout verbatim, indexed by intent embedding; each
episode gets the nearest past success and nearest past failure in context with
their verifier feedback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines.calibrate import parse_seed_spec
from baselines.memory.blocks import format_rag_block
from baselines.memory.embedder import DEFAULT_MODEL, Embedder
from baselines.memory.intent import (
    DEFAULT_MODE as DEFAULT_INTENT_MODE,
    MODES as INTENT_MODES,
    IntentCache,
    cache_path,
)
from baselines.memory.rag import EpisodicRAG, RAGConfig
from baselines.memory.store import MemoryStore
from baselines.runner.stream import (
    StreamConfig,
    load_offline_trajectories,
    run_stream,
)


#: Default retrieval key. `jinaai/jina-embeddings-v2-base-en` is the CORE paper's
#: choice for Episodic RAG (MemRL's own paper used text-embedding-3-large).
#: CAVEAT, measured on 105 seeds/level: text embeddings of InterPhyre intents
#: rank instances near-randomly (Spearman rho +0.11 to +0.40 vs the geometry,
#: ~10% top-1 overlap), because intents in a level differ only in their numbers.
#: `--embedding-model numeric` keys on the standardized geometry instead and
#: gives a well-conditioned retrieval space; see docs/MEMORY_BASELINES.md.
DEFAULT_KEY = "jinaai/jina-embeddings-v2-base-en"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Episodic RAG baseline on InterPhyre")
    p.add_argument("--level", required=True)
    p.add_argument("--seeds", default="6-55", help='Seed spec (default: "6-55")')
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--initial-traj-dir", default=None,
        help="Offline init: HeXA-format trajectory_seed*.json directory",
    )
    p.add_argument(
        "--model", default="claude",
        help="Agent model: claude, mock, openai:<name>, or an HF path/name "
             "(e.g. Qwen2.5-7B-Instruct)",
    )
    p.add_argument(
        "--embedding-model", default=DEFAULT_KEY,
        help="Retrieval key: a text embedder (default jina, the CORE paper's "
             "choice) or 'numeric' for the standardized geometry",
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
    p.add_argument("--k-success", type=int, default=1)
    p.add_argument("--k-failure", type=int, default=1)
    p.add_argument("--max-obs-chars", type=int, default=600)
    p.add_argument(
        "--max-traj-steps", type=int, default=None,
        help="Truncate stored rollouts to their last N steps (default: keep all)",
    )
    p.add_argument(
        "--freeze-after", type=int, default=None,
        help="Stream index at which memory becomes read-only (transfer eval)",
    )
    p.add_argument("--verbose", action="store_true")
    return p


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

    seeds = parse_seed_spec(args.seeds)
    store = MemoryStore(out_dir / "memory_store.json")
    embedder = build_embedder(args)
    intents = IntentCache(
        cache_path(cache_dir, args.level, args.intent_mode), args.intent_mode
    )

    memory = EpisodicRAG(
        store,
        RAGConfig(
            k_success=args.k_success,
            k_failure=args.k_failure,
            max_obs_chars=args.max_obs_chars,
            max_steps=args.max_traj_steps,
        ),
    )

    from skillrl.runner.run_skill_agent import _load_model

    model_fn = _load_model(args.model)
    is_oss = "gpt-oss" in args.model.lower()

    manifest = {
        "method": "episodic_rag",
        "spec_source": "CORE paper section 4.1 (retrieve entire past rollouts)",
        "agent_model": args.model,
        "embedding_model": args.embedding_model,
        "intent_mode": args.intent_mode,
        "k_success": args.k_success,
        "k_failure": args.k_failure,
        "max_traj_steps": args.max_traj_steps,
        "max_obs_chars": args.max_obs_chars,
        "seeds": args.seeds,
        "initial_traj_dir": args.initial_traj_dir,
        "max_iterations": args.max_iterations,
        "deviations": [
            f"actor is {args.model}; the CORE paper's baselines used gpt-oss-120b",
        ]
        + key_deviations(args),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    config = StreamConfig(
        level=args.level,
        seeds=seeds,
        output_dir=out_dir,
        method="episodic_rag",
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
        block_formatter=format_rag_block,
        offline_trajectories=load_offline_trajectories(
            Path(args.initial_traj_dir) if args.initial_traj_dir else None
        ),
        manifest=manifest,
    )


if __name__ == "__main__":
    main()
