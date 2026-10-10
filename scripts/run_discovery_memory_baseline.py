#!/usr/bin/env python
"""Episodic RAG on DiscoverPhysics, under HeXA's own protocol.

The memory-baseline counterpart to `hexa_discovery/within_world_loop.py` and
`cross_world_loop.py`: same worlds, same seeds, same round structure, same
actor, same freeze-then-evaluate split, so each arm lands paired *per seed*
against the no-bank baselines already on disk and against HeXA's bank arm.

    # plumbing check -- no model calls, scripted episodes
    python scripts/run_discovery_memory_baseline.py --arm oscillator --stub \
        --out /tmp/smoke

    # the real thing
    python scripts/run_discovery_memory_baseline.py --arm oscillator \
        --out discovery-agents/hexa_discovery/data/runs/osc_episodic_rag

Then re-judge with the independent judge before reading anything, exactly as
the HeXA arms were:

    cd discovery-agents && python -m hexa_discovery.rejudge \
        --run-dir hexa_discovery/data/runs/osc_episodic_rag/eval_frozen \
        --judge-model claude-opus-4-6

`--skip-build` evaluates an existing store; `--skip-eval` stops after the build.
Both passes resume from disk, so an interrupted arm is restarted with the same
command.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines.discovery import stream  # noqa: E402
from baselines.discovery.blocks import format_memrl_block, format_rag_block  # noqa: E402
from baselines.discovery.calibrate import calibrate, report as report_calibration  # noqa: E402
from baselines.discovery.summarize import make_summarizer  # noqa: E402
from baselines.discovery.intent import world_label  # noqa: E402
from baselines.discovery.store import RecencyTieStore  # noqa: E402
from baselines.discovery.stream import ArmConfig  # noqa: E402
from baselines.memory.memrl import MemRL, MemRLConfig  # noqa: E402
from baselines.memory.rag import EpisodicRAG, RAGConfig  # noqa: E402
from baselines.orbit.embedding import (  # noqa: E402
    HASHING,
    PAPER_DEFAULT,
    build_embedder,
    embedder_note,
)

#: Retrieval key for the DiscoverPhysics arms.
#:
#: Neither the CORE paper's jina-v2 nor the MiniLM substitute the ORBIT arms use
#: is available here: `HeXA/.venv` has no torch, and pulling one in would be
#: several GB for a key that does nothing on two of the three arms. Within one
#: world every episode has an identical problem embedding whatever embeds it
#: (see baselines/discovery/intent.py), so the choice can only matter on the
#: cross-world hubble arm -- where it has to separate eight world descriptions
#: that differ in wording throughout, which signed character-n-gram hashing does
#: cleanly and deterministically. Recorded as a deviation on every run, and
#: `--embedding-model` still takes a neural key if one is installed later.
DEFAULT_KEY = HASHING

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "discovery-agents" / "hexa_discovery" / "data"
OFFLINE = DATA / "offline" / "sonnet"

#: HeXA's cross-world training order, from `data/runs/logo_global_term/manifest.json`.
#: `gravity` is absent on purpose: it appears only in the pooled bootstrap.
_LOGO_WORLDS = [
    "yukawa", "fractional", "coulomb_easy", "extra_dimensions",
    "circle", "three_species", "dark_matter", "oscillator",
]


#: The bootstrap directory each within-world HeXA run actually distilled from,
#: and how many episodes it holds.
#:
#: This is not cosmetic. `offline/sonnet/circle/discovery` carries fifteen
#: episodes (seeds 100-114) but `within_circle.log`'s round 0 reports five --
#: "1 successes, 4 failures", rewards +0.90 / +0.85 +0.85 +0.80 +0.75 -- which
#: is `_circle_clean` exactly. Pointing the memory arm at the fifteen-episode
#: directory would hand it three times HeXA's bootstrap evidence and the arms
#: would no longer be the same protocol. Oscillator's directory already holds
#: only the five.
_BOOTSTRAP = {
    "oscillator": ("oscillator", 5),
    "circle": ("_circle_clean", 5),
}


def within_world_arm(world: str, out: Path) -> ArmConfig:
    """5 build rounds x 3 seeds, then 15 held-out seeds.

    Mirrors `PREREG_frozen_eval_oscillator.md`: train on seeds 200-214 in rounds
    of three, freeze, evaluate on 300-314. The offline directory is the same one
    HeXA's round 0 distilled from -- that world's five bank-free episodes.
    """
    return ArmConfig(
        name=world,
        eval_world=world,
        eval_seeds=list(range(300, 315)),
        build=[(world, list(range(s, s + 3))) for s in range(200, 215, 3)],
        offline_dirs={world: _bootstrap_dir(world)},
        # The anonymous label, not `world`: this string is printed inside every
        # retrieved memory (see ArmConfig.scope_level). Retrieval is still
        # scoped to this world alone, which is what a within-world arm means.
        scope_level=world_label(world),
        within_world=True,
        output_dir=out,
    )


def hubble_arm(out: Path) -> ArmConfig:
    """Leave-one-group-out: 8 build worlds x seeds 100-102, evaluate on hubble.

    `global_term` (ether + hubble) is held out whole, so neither hubble's scene
    topology nor its "central law + global term" structure is in memory. The
    bootstrap is the pooled directory the HeXA run used, not a single world's:
    every world is unanimous at this actor, so no one of them supplies both a
    success and a failure for the contrastive step HeXA does at round 0.

    `scope_level` is the synthetic "discovery" level, so retrieval ranges over
    all eight training worlds -- which is the only arm where the problem
    embedding has anything to separate.
    """
    return ArmConfig(
        name="hubble",
        eval_world="hubble",
        eval_seeds=list(range(100, 115)),
        build=[(w, [100, 101, 102]) for w in _LOGO_WORLDS],
        offline_dirs={
            "_pooled_global_term":
                OFFLINE / "_pooled_global_term" / "pooled_input" / "discovery"
        },
        scope_level="discovery",
        within_world=False,
        output_dir=out,
    )


def _bootstrap_dir(world: str) -> Path:
    """The bootstrap directory, with its episode count asserted.

    A silent change of bootstrap size is exactly the kind of protocol drift
    that is invisible in the output and fatal to the comparison, so this fails
    loudly instead.
    """
    name, expected = _BOOTSTRAP[world]
    path = OFFLINE / name / "discovery"
    found = len(list(path.glob("trajectory_seed*_skillrl.json")))
    if found != expected:
        raise SystemExit(
            f"{path} holds {found} episodes; HeXA's round 0 for {world} used "
            f"{expected}. Fix the directory or _BOOTSTRAP before running."
        )
    return path


#: Worlds in the cross-world bootstrap (`_pooled_global_term`), for calibration.
#: From `data/runs/logo_global_term/manifest.json`'s `bootstrap.pooled_from`.
_BOOTSTRAP_WORLDS = ["gravity", "yukawa", "fractional", "three_species",
                     "dark_matter"] * 3 + ["gravity", "yukawa", "fractional"]


ARMS = {
    "oscillator": lambda out: within_world_arm("oscillator", out),
    "circle": lambda out: within_world_arm("circle", out),
    "hubble": hubble_arm,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", required=True, choices=sorted(ARMS))
    p.add_argument("--out", required=True)
    p.add_argument("--method", default="episodic_rag",
                   choices=["episodic_rag", "memrl"])
    p.add_argument("--cli-model", default="sonnet",
                   help="Actor AND judge: cli_client patches every claude call")
    p.add_argument("--embedding-model", default=DEFAULT_KEY)
    p.add_argument("--cache-dir", default="baselines/data/cache")
    p.add_argument("--k-success", type=int, default=1)
    p.add_argument("--k-failure", type=int, default=1)
    p.add_argument("--max-obs-chars", type=int, default=600)
    p.add_argument("--max-traj-steps", type=int, default=12,
                   help="Never binds here: a DiscoverPhysics episode is capped "
                        "at 10 rounds, so stored rollouts are already verbatim")
    p.add_argument("--noise-std", type=float, default=0.075)
    p.add_argument("--max-rounds", type=int, default=None,
                   help="Default None = the agent's own MAX_ROUNDS (10), which "
                        "is what every HeXA DiscoverPhysics arm used")
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--stub", action="store_true",
                   help="Scripted episodes, no model calls: plumbing only")
    return p


def deviations(args, cfg: ArmConfig) -> list[str]:
    """Everything this run does that the CORE paper's §4.1 does not."""
    out = [
        f"actor is the Claude CLI model '{args.cli_model}'; the CORE paper's "
        f"baselines used gpt-oss-120b",
        "the build pass is round-batched (3 episodes retrieve against one "
        "frozen memory, then all write back), matching HeXA's round structure "
        "rather than the paper's per-inference update",
        "ties in retrieval break towards the MOST RECENT memory; the paper "
        "states no tie-break because its problems differ, whereas every episode "
        "of one DiscoverPhysics world has an identical problem embedding "
        "(see baselines/discovery/store.py)",
        "stored rollouts carry the same numeric and law-form redaction HeXA's "
        "teacher sees (traj_adapter.redact_text), and worlds are labelled "
        "anonymously; the judge's critique is excluded from memory entirely, "
        "because the judge reads the ground-truth explanation",
        embedder_note(args.embedding_model),
    ]
    if args.embedding_model != PAPER_DEFAULT:
        out.append(
            f"the CORE paper's embedder is {PAPER_DEFAULT}; it does not load "
            f"under transformers >= 5 (see baselines/orbit/embedding.py)"
        )
    if cfg.within_world:
        out.append(
            "WITHIN-WORLD arm: every memory comes from the evaluated world, so "
            "this measures accumulation, not transfer -- the same claim boundary "
            "HeXA's own pre-registration draws for oscillator and circle"
        )
    return out


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = ARMS[args.arm](out)
    cfg.cli_model = args.cli_model
    # `cli_client.patch_llm_client` forces EVERY claude call to --cli-model, so
    # the judge is the actor model whatever is asked for here. Recording
    # anything else would put a model in the trajectory that was never invoked.
    # The reported numbers come from a separate `hexa_discovery.rejudge` pass
    # with claude-opus-4-6 -- exactly how the HeXA arms were scored.
    cfg.judge_model = args.cli_model
    cfg.noise_std = args.noise_std
    cfg.max_rounds = args.max_rounds

    store = RecencyTieStore(out / "memory_store.json")
    embedder = build_embedder(args.embedding_model, cache_dir=Path(args.cache_dir))

    calibration = None
    if args.method == "memrl":
        # The memory's eventual world composition, known from the protocol
        # before a rollout is spent. Calibrating on it rather than on a finished
        # store means delta and the z-score constants are fixed in advance
        # instead of being fitted to the run they will be used in.
        composition = ([cfg.eval_world] * 20 if cfg.within_world
                       else [w for w, seeds in cfg.build for _ in seeds]
                            + _BOOTSTRAP_WORLDS)
        calibration = calibrate(composition, embedder, eval_world=cfg.eval_world)
        print(f"\n  MemRL calibration ({args.method}):")
        report_calibration(calibration)
        rec = calibration["recommended"]
        memory = MemRL(
            store,
            MemRLConfig(
                delta=rec["delta"],
                sim_norm_mean=rec["sim_norm_mean"],
                sim_norm_std=rec["sim_norm_std"],
                max_obs_chars=args.max_obs_chars,
                max_traj_steps=args.max_traj_steps or None,
            ),
            summarizer=(None if args.stub
                        else make_summarizer(model=args.cli_model, verbose=True)),
        )
        cfg.block_formatter = format_memrl_block
    else:
        memory = EpisodicRAG(
            store,
            RAGConfig(
                k_success=args.k_success,
                k_failure=args.k_failure,
                max_obs_chars=args.max_obs_chars,
                max_steps=args.max_traj_steps or None,
            ),
        )
        cfg.block_formatter = format_rag_block
    cfg.method = args.method

    if args.stub:
        from baselines.discovery import stub
        stub.install()
    else:
        from baselines.discovery import rollout
        rollout.ensure_cli(args.cli_model)

    manifest = {
        "method": args.method,
        "spec_source": "CORE paper section 4.1 (retrieve entire past rollouts)",
        "arm": args.arm,
        "protocol_source": (
            "PREREG_frozen_eval_oscillator.md" if cfg.within_world
            else "data/runs/logo_global_term/manifest.json"
        ),
        "eval_world": cfg.eval_world,
        "eval_seeds": cfg.eval_seeds,
        "build_rounds": [{"world": w, "seeds": s} for w, s in cfg.build],
        "offline_dirs": {k: str(v) for k, v in cfg.offline_dirs.items()},
        "retrieval_scope": cfg.scope_level,
        "within_world": cfg.within_world,
        "actor_model": args.cli_model,
        "judge_model_at_runtime": cfg.judge_model,
        "reported_judge": "claude-opus-4-6 (separate hexa_discovery.rejudge pass)",
        "embedding_model": args.embedding_model,
        "k_success": args.k_success,
        "k_failure": args.k_failure,
        "noise_std": cfg.noise_std,
        "max_rounds": cfg.max_rounds or "agent default (10)",
        "paired_against": (
            f"the no-bank baseline on the same seeds; see "
            f"DOC_Results/DISCOVERY_RESULTS.md for the HeXA arm"
        ),
        "memrl_calibration": calibration,
        "deviations": deviations(args, cfg),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in
                      ("arm", "eval_world", "retrieval_scope", "within_world")},
                     indent=2))

    started = time.perf_counter()
    summary = {"arm": args.arm}

    if not args.skip_build:
        n_offline = stream.run_offline_pass(memory, store, embedder, cfg)
        build = stream.run_build_pass(memory, store, embedder, cfg, n_offline)
        summary["offline"] = n_offline
        summary["build"] = stream.summarise(build)
        print(f"\n  build: {json.dumps(summary['build'])}")
    if not args.skip_eval:
        evaluated = stream.run_eval_pass(memory, store, embedder, cfg)
        summary["eval_frozen"] = stream.summarise(evaluated)
        print(f"\n  eval : {json.dumps(summary['eval_frozen'])}")

    summary["memory"] = store.stats()
    summary["elapsed_minutes"] = round((time.perf_counter() - started) / 60, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out/'summary.json'}  ({summary['elapsed_minutes']} min)")


if __name__ == "__main__":
    main()
