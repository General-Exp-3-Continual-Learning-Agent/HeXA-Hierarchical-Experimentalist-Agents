#!/usr/bin/env python
"""Phase 3 in bulk: every source set x target x replicate, then the verdicts.

Runs the 18 syntheses the pre-registered lattice needs (`DESIGN_C_STATUS.md` 6b)
and collects each one's inspection into a single table, then **stops**. It does
not launch Phase 4.

That is deliberate. The plan calls manual inspection the cheapest place to catch a
broken transfer, and with zero target trajectories there is nothing downstream that
would catch a plausible-sounding skill built on a premise the target does not have.
The mechanical gates (citations resolve, no source entity survives) are enforced
here; the inverted-premise flags are printed for a person to read. Chaining an eval
onto this would quietly convert "flagged for reading" into "passed".

    python scripts/run_design_c_phase3_all.py --dry-run    # prompts only, no teacher
    python scripts/run_design_c_phase3_all.py
    python scripts/run_design_c_phase3_all.py --target maze --runs 0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent

#: The source-set lattice. `near` is the structurally paired source, `far` the
#: other, `both` the design's actual mechanism.
NEAR = {"maze": "minesweeper", "mastermind": "wordle"}
FAR = {"maze": "wordle", "mastermind": "minesweeper"}


def source_sets(target: str) -> dict[str, list[str]]:
    return {
        "near": [NEAR[target]],
        "far": [FAR[target]],
        "both": sorted([NEAR[target], FAR[target]]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", action="append", default=None,
                    choices=("maze", "mastermind"))
    ap.add_argument("--arm", action="append", default=None,
                    choices=("near", "far", "both"))
    ap.add_argument("--runs", action="append", type=int, default=None)
    ap.add_argument("--teacher-model", default="openai/gpt-oss-120b")
    ap.add_argument("--dry-run", action="store_true",
                    help="build every synthesis prompt and stop; calls no teacher")
    args = ap.parse_args()

    targets = args.target or ["maze", "mastermind"]
    arms = args.arm or ["near", "far", "both"]
    runs = args.runs or [0, 1, 2]

    results = []
    for target in targets:
        sets = source_sets(target)
        for arm in arms:
            for run in runs:
                sources = sets[arm]
                cmd = [sys.executable, "scripts/run_design_c_phase3.py",
                       "--target", target, "--run-index", str(run),
                       "--teacher-model", args.teacher_model]
                for s in sources:
                    cmd += ["--source", s]
                if args.dry_run:
                    cmd.append("--dry-run")
                print(f"\n{'=' * 72}\n[{target} | {arm} | run{run}] {'+'.join(sources)}")
                proc = subprocess.run(cmd, cwd=ROOT)
                results.append({
                    "target": target, "arm": arm, "run": run,
                    "sources": sources, "returncode": proc.returncode,
                    "dir": f"results/design_c/{target}_from_{'_'.join(sorted(sources))}/run{run}",
                })

    print(f"\n\n{'=' * 72}\nPHASE 3 VERDICTS\n{'=' * 72}")
    print(f"{'target':<11} {'arm':<6} {'run':<4} {'sources':<24} verdict")
    blocked = 0
    for r in results:
        if args.dry_run:
            verdict = "prompt built" if r["returncode"] == 0 else "FAILED"
        elif r["returncode"] == 0:
            verdict = "mechanical checks pass -- needs a human read"
        else:
            verdict = "BLOCKED (see the run's output above)"
            blocked += 1
        print(f"{r['target']:<11} {r['arm']:<6} {r['run']:<4} "
              f"{'+'.join(r['sources']):<24} {verdict}")

    summary = ROOT / "results/design_c/phase3_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {summary}")

    if blocked:
        print(f"\n{blocked} bank(s) blocked. Do not queue Phase 4 for those.")
        return 1
    if not args.dry_run:
        print("\nNo bank is cleared for Phase 4 by this script alone -- read the "
              "inverted-premise flags above first. For maze <- minesweeper "
              "especially: a skill about acting only where safety is proven is "
              "correct in a game that ends on one wrong probe and ruinous in one "
              "with no losing move.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
