"""Offline bootstrap: run the PLAIN DiscoveryAgent (no skill bank) on a world
for a set of seeds and write HeXA-schema trajectory JSONs.

These seed the cross-world bank in round 1 of the sequential loop (or form the
pooled set). Run with a mix of noise seeds so the round-1 contrast has both
successes and failures where possible.

Example:
    python -m hexa_discovery.run_offline \
        --world oscillator --seeds 0 1 2 3 4 5 \
        --output-dir hexa_discovery/data/offline/oscillator \
        --model claude-opus-4-6
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hexa_discovery import _paths

_paths.bootstrap()

from hexa_discovery.discovery_runner import run_batch_discovery


def main() -> None:
    p = argparse.ArgumentParser(description="Generate offline seed trajectories for a world")
    p.add_argument("--world", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--output-dir", required=True,
                   help="Trajectories are written under <output-dir>/discovery/")
    p.add_argument("--model", default="claude-opus-4-6")
    p.add_argument("--judge-model", default="claude-opus-4-6")
    p.add_argument("--noise-std", type=float, default=0.075)
    p.add_argument("--max-rounds", type=int, default=None)
    p.add_argument("--use-critic", action="store_true")
    p.add_argument("--cli-model", default=None,
                   help="Force this model for the Claude CLI (e.g. 'sonnet', "
                        "'opus'); overrides --model/--judge-model at the CLI.")
    p.add_argument("--no-cli", action="store_true",
                   help="Use scienceagent's API providers instead of the Claude "
                        "CLI (requires the relevant API key).")
    p.add_argument("--no-redact", action="store_true",
                   help="Disable numeric redaction of law/fit sources (NOT "
                        "recommended for the cross-world bank).")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    run_batch_discovery(
        world=args.world,
        seeds=args.seeds,
        bank=None,
        eval_dir=Path(args.output_dir),
        use_cli=not args.no_cli,
        cli_model=args.cli_model,
        model=args.model,
        judge_model=args.judge_model,
        noise_std=args.noise_std,
        max_rounds=args.max_rounds,
        use_critic=args.use_critic,
        redact=not args.no_redact,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
