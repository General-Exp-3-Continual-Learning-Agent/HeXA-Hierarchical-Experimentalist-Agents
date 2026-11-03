"""ACE (Agentic Context Engineering, Zhang et al., ICLR 2026) on InterPhyre.

    python -m baselines.runner.run_ace \
        --level down_to_earth \
        --seeds 6-56 \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output-dir baselines/data/results/ace_down_to_earth_qwen7b

One evolving playbook per level, injected whole into the actor's system prompt;
after every seed a Reflector diagnoses the rollout and a Curator appends delta
bullets.  Generator, Reflector and Curator are the same model.  See
`baselines/ace/loop.py` for the mapping to the reference implementation and
every deviation from it (also written to manifest.json).
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import socket
from pathlib import Path

from baselines.calibrate import parse_seed_spec
from baselines.runner.stream import load_offline_trajectories


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ACE baseline on InterPhyre")
    p.add_argument("--level", required=True)
    p.add_argument("--seeds", default="6-56",
                   help='Seed spec (default "6-56": HeXA\'s stream, 17 rounds of 3)')
    p.add_argument("--output-dir", required=True)
    p.add_argument("--initial-traj-dir", default=None,
                   help="Offline warmup: HeXA-format trajectory_seed*.json directory")
    p.add_argument("--model", default="claude",
                   help="Actor (and Reflector/Curator) model: claude, mock, "
                        "openai:<name>, or an HF path/name")
    p.add_argument("--claude-model", default="claude-sonnet-4-6",
                   help="Model for the Reflector/Curator when --model claude")
    p.add_argument("--seeds-per-round", type=int, default=3)
    p.add_argument("--max-iterations", type=int, default=25)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-new-tokens", type=int, default=800)
    p.add_argument("--resolve-rounds", type=int, default=0,
                   help="Reference max_num_rounds: reflect-and-re-solve rounds on a "
                        "failed seed before curation (0 = off; paper used 5)")
    p.add_argument("--curate-every", type=int, default=1,
                   help="Reference curator_frequency (1 = curate after every seed)")
    p.add_argument("--token-budget", type=int, default=80000,
                   help="Reference playbook_token_budget (shown to the Curator only)")
    p.add_argument("--role-max-tokens", type=int, default=4096,
                   help="Generation cap for Reflector/Curator calls (reference max_tokens)")
    p.add_argument("--max-obs-chars", type=int, default=600)
    p.add_argument("--max-traj-steps", type=int, default=None,
                   help="Truncate the Reflector's view to the last N steps (default: all)")
    p.add_argument("--dedup", action="store_true",
                   help="Grow-and-refine: ACE's BulletpointAnalyzer after every Curator update")
    p.add_argument("--dedup-threshold", type=float, default=0.90,
                   help="Reference bulletpoint_analyzer_threshold (paper's best: 0.9)")
    p.add_argument("--token-cap", type=int, default=None,
                   help="Deviation: the paper's pruning trigger. After each curation, drop "
                        "lowest-utility (helpful-harmful), then oldest, bullets until the "
                        "playbook is <= this many tokens (paper tested 10K-100K)")
    p.add_argument("--edit-ops", action="store_true",
                   help="Deviation: the Curator may also UPDATE/MERGE/DELETE bullets "
                        "(the paper's refine step; the reference code is ADD only)")
    p.add_argument("--init-from", default=None,
                   help="Start from the final playbook of a finished ACE run dir "
                        "(frozen held-out eval: combine with --freeze-after 0)")
    p.add_argument("--freeze-after", type=int, default=None,
                   help="Stream index at which the playbook becomes read-only")
    p.add_argument("--verbose", action="store_true")
    return p


def acquire_writer_lock(out_dir: Path) -> None:
    """One writer per output dir, enforced on the shared filesystem."""
    lock = out_dir / ".writer.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(
            f"FATAL: {lock} is held by {lock.read_text().strip() or '?'}. "
            f"Another process is writing this run; remove the lock only if it is dead."
        )
    os.write(fd, f"{socket.gethostname()}:{os.getpid()}:{os.environ.get('SLURM_JOB_ID', '-')}".encode())
    os.close(fd)
    atexit.register(lambda: lock.unlink(missing_ok=True))


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    acquire_writer_lock(out_dir)

    from baselines.ace.llm import ACE_TEMPERATURE, make_text_llm
    from baselines.ace.loop import ACE, ACEConfig, seed_state_from
    from baselines.runner.memory_runner import run_memory_augmented
    from react_agent.level_prompts import build_initial_user_message
    from skillrl.runner.run_skill_agent import _load_model

    seeds = parse_seed_spec(args.seeds)
    if args.init_from and args.initial_traj_dir:
        raise SystemExit("--init-from and --initial-traj-dir are mutually exclusive")
    # Before the model loads, so a bad --init-from fails in seconds, not minutes.
    init_info = seed_state_from(Path(args.init_from), out_dir) if args.init_from else {}
    model_fn = _load_model(args.model)
    is_oss = "gpt-oss" in args.model.lower()
    text_llm = make_text_llm(args.model, model_fn, claude_model=args.claude_model)

    config = ACEConfig(
        level=args.level, seeds=seeds, output_dir=out_dir,
        seeds_per_round=args.seeds_per_round, max_iterations=args.max_iterations,
        temperature=args.temperature, max_new_tokens=args.max_new_tokens,
        is_oss=is_oss, verbose=args.verbose, freeze_after=args.freeze_after,
        resolve_rounds=args.resolve_rounds, curate_every=args.curate_every,
        token_budget=args.token_budget, role_max_tokens=args.role_max_tokens,
        max_obs_chars=args.max_obs_chars, max_traj_steps=args.max_traj_steps,
        dedup=args.dedup, dedup_threshold=args.dedup_threshold,
        edit_ops=args.edit_ops, token_cap=args.token_cap,
    )

    task = (
        f"Level: {args.level}\n"
        + build_initial_user_message(args.level, is_oss=is_oss, max_iterations=args.max_iterations)
    )

    def solve(seed: int, block: str) -> dict:
        return run_memory_augmented(
            model_fn, args.level, seed, block,
            max_iterations=args.max_iterations, verbose=args.verbose,
            temperature=args.temperature, max_new_tokens=args.max_new_tokens,
            is_oss=is_oss,
        )

    role_model = args.claude_model if args.model == "claude" else args.model
    manifest = {
        "method": "ace",
        "spec_source": "Zhang et al., Agentic Context Engineering (ICLR 2026); "
                       "github.com/ace-agent/ace @ 82709de (online mode, no GT labels)",
        "agent_model": args.model,
        "reflector_model": role_model,
        "curator_model": role_model,
        "role_temperature": None if args.model == "claude" else ACE_TEMPERATURE,
        "seeds": args.seeds,
        "initial_traj_dir": args.initial_traj_dir,
        "max_iterations": args.max_iterations,
        "resolve_rounds": args.resolve_rounds,
        "curate_every": args.curate_every,
        "token_budget": args.token_budget,
        "role_max_tokens": args.role_max_tokens,
        "max_obs_chars": args.max_obs_chars,
        "max_traj_steps": args.max_traj_steps,
        "freeze_after": args.freeze_after,
        "dedup": args.dedup,
        "dedup_threshold": args.dedup_threshold if args.dedup else None,
        "edit_ops": args.edit_ops,
        "token_cap": args.token_cap,
        **init_info,
        "deviations": [
            "label-free: the Reflector sees the simulator verdict and observations; "
            "InterPhyre has no reference answer",
            ("the whole playbook is injected every seed (as in ACE); the Curator may ADD, "
             "UPDATE, MERGE and DELETE bullets (baselines/ace/edit_ops.py): the paper's refine "
             "step, which the reference code leaves as a TODO (its Curator is ADD only)")
            if args.edit_ops else
            ("the whole playbook is injected every seed (as in ACE); it is never pruned, "
             "because the reference Curator implements ADD only and dedup is off by default"),
            "the post-curate re-generation of the reference loop is skipped "
            "(training-accuracy logging only)",
            "the actor cites bullet IDs in its Thoughts; this is ACE's own Generator "
            "contract (bullet_ids), and is what lets the Reflector tag bullets",
            "the Reflector prompt's example bullet IDs are 'xmpl-0000N' instead of the "
            "reference's 'calc-00001'/'fin-00002': 'calc-00001' is a real bullet ID, and "
            "template-copying credited it on every seed",
        ]
        + (["reflect-and-re-solve is OFF (reference default max_num_rounds=3, paper 5): "
            "each re-solve is a full extra episode with its own simulator budget"]
           if args.resolve_rounds == 0 else
           [f"reflect-and-re-solve ON with {args.resolve_rounds} rounds; retries are "
            "not scored and their simulator calls are logged under "
            "ace.counters.training_simulator_calls"])
        + (["grow-and-refine ON: ACE's BulletpointAnalyzer (all-mpnet-base-v2, "
            f"threshold {args.dedup_threshold}, LLM merge at T=0.3) after every Curator update"]
           if args.dedup else
           ["grow-and-refine (dedup) OFF, the reference default"])
        + ([f"pruning trigger ON (baselines/ace/prune.py): after each curation and dedup, "
             f"bullets are dropped lowest utility (helpful-harmful) first, oldest first on ties, "
             f"until the playbook is <= {args.token_cap} tokens; the paper describes this "
             "trigger (Table 21) but the reference code has none"]
           if args.token_cap else [])
        + ([f"curate_every={args.curate_every}: reflections since the last curation "
            "are concatenated (the reference keeps only the latest)"]
           if args.curate_every > 1 else [])
        + ([f"actor is {args.model}; the paper used DeepSeek-V3.1 for all roles"])
        + (["claude backend: `claude -p` exposes no temperature, so the roles (and the "
            "actor) run at the CLI default, not ACE's 0.0; HExA's Claude runs share this"]
           if args.model == "claude" else []),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    ace = ACE(config, model_fn=model_fn, text_llm=text_llm,
              task_text=lambda: task, solve=solve)
    ace.run(
        offline_trajectories=load_offline_trajectories(
            Path(args.initial_traj_dir) if args.initial_traj_dir else None
        ),
        manifest=manifest,
    )


if __name__ == "__main__":
    main()
