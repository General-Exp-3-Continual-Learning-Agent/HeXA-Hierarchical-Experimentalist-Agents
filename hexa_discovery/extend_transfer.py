"""Add seeds to an existing transfer arm without disturbing the rest of a run.

Re-running `cross_world_loop` with a larger `--seeds-per-world` would also widen
every TRAINING world (8 worlds x 2 extra seeds = 16 episodes) whose results the
frozen bank no longer depends on — pure waste, since the bank files are reused.
This runs only the held-out episodes, against the run's existing
`skill_bank_final.json`, and writes them into the same transfer directories so
the arm simply gets deeper.

Motivation: the DiscoverPhysics paper reports 5 seeds per world with
bootstrapped standard errors. At 3 seeds our hubble baseline carried an SE of
+/-0.252 — one collapsed seed dominating — making a +0.361 difference only 1.4
sigma. More seeds go to the arm that needs them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hexa_discovery import _paths

_paths.bootstrap()

from hexa_discovery.config import DISCOVERY_LEVEL
from hexa_discovery.cross_world_loop import _install_seams
from hexa_discovery.discovery_runner import run_batch_discovery
from skillrl.core.skill_bank import SkillBank


def main() -> None:
    p = argparse.ArgumentParser(description="Extend a transfer arm with more seeds")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--world", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--bank", default="skill_bank_final.json")
    p.add_argument("--cli-model", default="sonnet")
    p.add_argument("--noise-std", type=float, default=0.075)
    p.add_argument("--arms", nargs="+", default=["with_bank", "baseline"])
    p.add_argument("--max-skills", type=int, default=None,
                   help="Skills injected at solve time. MUST match the arm's "
                        "existing episodes: run_episode's default was raised 8 -> 12 "
                        "mid-project, so an arm started under the old default and "
                        "extended under the new one mixes two configurations and "
                        "cannot be pooled. hubble seeds 100-104 used 8.")
    p.add_argument("--max-mistakes", type=int, default=None)
    args = p.parse_args()

    run = Path(args.run_dir)
    _install_seams()
    bank = SkillBank.load(run / args.bank)
    n = len(bank.level_skills.get(DISCOVERY_LEVEL, []))
    print(f"loaded {args.bank}: {n} skills\n")

    kwargs = dict(noise_std=args.noise_std, use_cli=True, cli_model=args.cli_model)
    if args.max_skills is not None:
        kwargs["max_skills"] = args.max_skills
    if args.max_mistakes is not None:
        kwargs["max_mistakes"] = args.max_mistakes
    for arm in args.arms:
        print(f"\n{'='*60}\n  {args.world} / {arm} / seeds {args.seeds}\n{'='*60}")
        run_batch_discovery(
            args.world, args.seeds,
            bank if arm == "with_bank" else None,
            run / "transfer" / args.world / arm,
            **kwargs,
        )


if __name__ == "__main__":
    main()
