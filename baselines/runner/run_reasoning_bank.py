"""ReasoningBank on InterPhyre.

    python -m baselines.runner.run_reasoning_bank \
        --level down_to_earth \
        --seeds 6-106 --freeze-after 51 \
        --initial-traj-dir Initial_trajectories/DTE_Qwen7B \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output-dir baselines/data/results/reasoning_bank_dte_qwen7b

Each episode is distilled into <=3 generalizable reasoning items (success ->
strategies, failure -> lessons), appended to a flat, append-only bank; each new
episode retrieves the top-k most similar past experiences and reads their items.
The success/failure label is the InterPhyre environment reward (not the paper's
LLM-as-judge), so the only difference from HeXA is the memory mechanism.

Protocol note — to reproduce HeXA's frozen held-out eval, run the build and eval
as one stream and freeze memory at the boundary: HeXA trains on seeds 6-56 (51
seeds) then evaluates a frozen bank on 57-106, so `--seeds 6-106 --freeze-after
51`. Per-seed `memory_frozen` flags in the summary mark the held-out tail (57-106).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines.calibrate import parse_seed_spec
from baselines.memory.embedder import Embedder
from baselines.memory.intent import (
    DEFAULT_MODE as DEFAULT_INTENT_MODE,
    MODES as INTENT_MODES,
    IntentCache,
    cache_path,
)
from baselines.memory.store import MemoryStore
from baselines.reasoning_bank.blocks import format_reasoning_bank_block
from baselines.reasoning_bank.extract import make_cli_extractor, make_rb_extractor
from baselines.reasoning_bank.rb import ReasoningBank, ReasoningBankConfig
from baselines.runner.stream import (
    StreamConfig,
    load_offline_trajectories,
    run_stream,
)

#: Retrieval key. `numeric` (standardized scene geometry) is the default because,
#: as measured for the other InterPhyre memory baselines, text embeddings of
#: InterPhyre intents rank instances near-randomly (Spearman rho +0.11 to +0.40,
#: ~10% top-1 overlap) — a text key would make retrieval a straw man. It needs the
#: scaler fitted by `baselines.calibrate` (see build_embedder). Pass a text model
#: name (e.g. jinaai/jina-embeddings-v2-base-en, the paper's dense-retriever
#: family) to run that weaker configuration deliberately.
DEFAULT_KEY = "numeric"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ReasoningBank baseline on InterPhyre")
    p.add_argument("--level", required=True)
    p.add_argument("--seeds", default="6-55", help='Seed spec (default: "6-55")')
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--initial-traj-dir", default=None,
        help="Offline init: HeXA-format trajectory_seed*.json directory (distilled "
             "into the bank before the stream, matching HeXA's Off2On head start)",
    )
    p.add_argument(
        "--model", default="claude",
        help="Agent model: claude, mock, openai:<name>, or an HF path/name "
             "(e.g. Qwen2.5-7B-Instruct)",
    )
    p.add_argument(
        "--extractor-model", default=None,
        help="Model for memory extraction. Default: the agent model itself "
             "(the paper's same-backbone extractor). For a Claude actor the "
             "extraction goes through the `claude -p` CLI.",
    )
    p.add_argument(
        "--extractor-temperature", type=float, default=1.0,
        help="Extraction temperature (paper: 1.0; ignored by the claude -p backend)",
    )
    p.add_argument(
        "--extractor-max-new-tokens", type=int, default=1024,
        help="Max new tokens for the extractor call (default: 1024)",
    )
    p.add_argument(
        "--embedding-model", default=DEFAULT_KEY,
        help="Retrieval key: 'numeric' (standardized geometry, default) or a text "
             "embedder model name",
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
    p.add_argument(
        "--k", type=int, default=1,
        help="Past experiences retrieved per episode (paper default: 1; each "
             "carries up to --max-items items). Raise to widen the injected "
             "budget, e.g. toward HeXA's 6-skill/4-mistake real-estate.",
    )
    p.add_argument(
        "--max-items", type=int, default=3,
        help="Items extracted per trajectory (paper: at most 3)",
    )
    p.add_argument("--max-traj-steps", type=int, default=12)
    p.add_argument("--max-obs-chars", type=int, default=600)
    p.add_argument(
        "--freeze-after", type=int, default=None,
        help="Stream index at which memory becomes read-only (held-out eval). For "
             "HeXA's 6-56 build / 57-106 eval, use --seeds 6-106 --freeze-after 51.",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def key_deviations(args) -> list[str]:
    """State the retrieval-key choice, the one substantive departure from the paper."""
    if args.embedding_model == "numeric":
        return [
            "retrieval is keyed on the standardized scene geometry, not on an "
            "embedded task description as in the paper: text embeddings of "
            "InterPhyre intents rank instances near-randomly (see "
            "baselines/memory/features.py and the level's calibration JSON)",
        ]
    return [
        f"retrieval key is the text embedder {args.embedding_model}; on InterPhyre "
        f"this ranks instances close to randomly (see the rank_agreement figures "
        f"in the level's calibration JSON)",
    ]


def build_embedder(args):
    """Text embedder, or the numeric geometry key (needs the calibrated scaler).

    Mirrors `baselines/runner/run_episodic_rag.build_embedder`: `numeric` fails
    loudly rather than silently re-fitting a scaler on whatever seeds this run
    happens to see, which would make retrieval incomparable across runs.
    """
    from baselines.memory.features import FeatureScaler, NumericEmbedder, scaler_path

    if args.embedding_model != NumericEmbedder.name:
        return Embedder(args.embedding_model, cache_dir=Path(args.cache_dir))

    if args.intent_mode != "geometry":
        raise SystemExit("--embedding-model numeric requires --intent-mode geometry.")
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


def build_extractor(args, model_fn):
    """Same-backbone extractor by default; `claude -p` for a Claude actor."""
    ext_model = args.extractor_model or args.model
    if ext_model == "claude":
        return make_cli_extractor(max_items=args.max_items, verbose=args.verbose)

    if args.extractor_model and args.extractor_model != args.model:
        from skillrl.runner.run_skill_agent import _load_model

        extractor_fn = _load_model(args.extractor_model)
    else:
        extractor_fn = model_fn  # reuse the one model already on the GPU
    return make_rb_extractor(
        extractor_fn,
        temperature=args.extractor_temperature,
        max_new_tokens=args.extractor_max_new_tokens,
        max_items=args.max_items,
        verbose=args.verbose,
    )


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

    from skillrl.runner.run_skill_agent import _load_model

    model_fn = _load_model(args.model)
    is_oss = "gpt-oss" in args.model.lower()

    memory = ReasoningBank(
        store,
        ReasoningBankConfig(
            k=args.k,
            max_items=args.max_items,
            max_traj_steps=args.max_traj_steps,
            max_obs_chars=args.max_obs_chars,
        ),
        extractor=build_extractor(args, model_fn),
    )

    manifest = {
        "method": "reasoning_bank",
        "spec_source": "ReasoningBank (Ouyang et al., ICLR 2026, arXiv:2509.25140)",
        "agent_model": args.model,
        "extractor_model": args.extractor_model or args.model,
        "extractor_temperature": args.extractor_temperature,
        "embedding_model": args.embedding_model,
        "intent_mode": args.intent_mode,
        "k": args.k,
        "max_items": args.max_items,
        "max_traj_steps": args.max_traj_steps,
        "seeds": args.seeds,
        "freeze_after": args.freeze_after,
        "initial_traj_dir": args.initial_traj_dir,
        "max_iterations": args.max_iterations,
        "deviations": [
            f"actor is {args.model}",
            "correctness signal is the InterPhyre environment reward, not the "
            "paper's LLM-as-judge: with a real reward available, using it keeps "
            "the supervision identical to HeXA's and isolates the memory mechanism",
            "memory is append-only (no cap/merge/forgetting), as in the paper",
        ]
        + key_deviations(args),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    config = StreamConfig(
        level=args.level,
        seeds=seeds,
        output_dir=out_dir,
        method="reasoning_bank",
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
        block_formatter=format_reasoning_bank_block,
        offline_trajectories=load_offline_trajectories(
            Path(args.initial_traj_dir) if args.initial_traj_dir else None
        ),
        manifest=manifest,
    )


if __name__ == "__main__":
    main()
