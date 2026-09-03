#!/usr/bin/env python
"""Phase 3: synthesise a target bank from a source bank, with no target data.

Near-transfer pilot, per the plan's staging:

    K_maze        <- Minesweeper alone
    K_mastermind  <- Wordle alone

The synthesis itself is `skillrl.distillation.cross_level_synthesis`, generalised
out of InterPhyre in Phase 0a; this script only picks the right source bank, points
it at `hexa_orbit.cross_task_blocks`, and runs the inspection that must happen
before any eval compute is spent.

**No target trajectories are read at any point.** The only inputs are the source
bank and the factual target description; the target's own pools are never
constructed here.

    python scripts/run_design_c_phase3.py --target maze --run-index 0 --dry-run
    python scripts/run_design_c_phase3.py --target maze --run-index 0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hexa_orbit import cross_task_blocks  # noqa: E402

#: The near-transfer pairing the plan argues for: spatial hidden-state probing
#: for Maze, hypothesis-narrowing-under-feedback for Mastermind.
NEAR_SOURCE = {"maze": "minesweeper", "mastermind": "wordle"}


def latest_bank(run_dir: Path) -> Path:
    """The last round's bank in a Phase 2 run directory.

    Rounds are `round0 .. roundN`; the bank to transfer is the one the final round
    left, not `round0`'s seeding bank.
    """
    rounds = sorted(
        (p for p in run_dir.glob("round*") if (p / "bank.json").exists()),
        key=lambda p: int(p.name.removeprefix("round")),
    )
    if not rounds:
        raise SystemExit(f"no round*/bank.json under {run_dir}")
    return rounds[-1] / "bank.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=sorted(NEAR_SOURCE))
    ap.add_argument("--run-index", type=int, default=0, choices=(0, 1, 2),
                    help="which Phase 2 replicate's bank to transfer; each gets "
                         "its own synthesis, so Phase 4 has three of them too")
    ap.add_argument("--source", action="append", default=None,
                    help="repeatable. One source evolves the near/far transfer "
                         "arms; two or more exercise the design's actual "
                         "mechanism, K_target = pi_ev(K_1..K_S, description). "
                         f"Default: the near-transfer pairing {NEAR_SOURCE}.")
    ap.add_argument("--source-bank", action="append", default=None,
                    help="repeatable, as '<source>=<path>'. Overrides the default "
                         "of each source's last Phase 2 round.")
    ap.add_argument("--teacher-model", default="openai/gpt-oss-120b",
                    help="same model as the actor -- HExA is self-improvement")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="build and dump the synthesis prompt, call nothing")
    args = ap.parse_args()

    sources = args.source or [NEAR_SOURCE[args.target]]
    explicit = dict(pair.split("=", 1) for pair in (args.source_bank or []))
    #: Phase 2 writes each source's arm to a directory keyed by the config it ran
    #: under, so Wordle's 5-turn arm is `orbit_wordle5_hexa` (D-C6). Mapped here
    #: rather than at the call site, so a wrong path fails loudly.
    ARM_DIR = {"minesweeper": "orbit_minesweeper_hexa", "wordle": "orbit_wordle5_hexa"}
    banks = {
        source: Path(explicit[source]) if source in explicit
        else latest_bank(Path(f"results/{ARM_DIR[source]}/run{args.run_index}"))
        for source in sources
    }
    for source, bank in banks.items():
        if not bank.exists():
            raise SystemExit(f"{source} bank not found at {bank}")

    tag = "_".join(sorted(sources))
    out = args.out or Path(
        f"results/design_c/{args.target}_from_{tag}/run{args.run_index}")
    out.mkdir(parents=True, exist_ok=True)

    if args.target not in cross_task_blocks.TARGET_BLOCKS:
        raise SystemExit(f"no factual block for target {args.target!r}")

    cmd = [
        sys.executable, "-m", "skillrl.distillation.cross_level_synthesis",
        "--target", args.target,
        "--blocks-module", "hexa_orbit.cross_task_blocks",
        "--output-dir", str(out),
        "--teacher-model", args.teacher_model,
        "--prompt-variant", "text-game",
    ]
    for source, bank in banks.items():
        cmd += ["--source-bank", f"{source}={bank}"]
    if args.dry_run:
        cmd.append("--dry-run")

    print(f"[phase 3] {args.target} <- {'+'.join(sources)}")
    for source, bank in banks.items():
        print(f"           {source}: {bank}")
    rc = subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent).returncode
    if rc != 0 or args.dry_run:
        return rc

    (out / "provenance.json").write_text(json.dumps({
        "target": args.target, "source": sources[0] if len(sources) == 1 else sources,
        "sources": sources, "run_index": args.run_index,
        "source_banks": {k: str(v) for k, v in banks.items()},
        "teacher_model": args.teacher_model,
        "target_trajectories_used": 0,
    }, indent=2))

    print("\n[phase 3] inspection follows -- read it before spending eval compute")
    return subprocess.run(
        [sys.executable, "scripts/inspect_cross_task_bank.py", "--dir", str(out),
         "--target", args.target, *sum((["--source", x] for x in sources), [])],
        cwd=Path(__file__).resolve().parent.parent,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
