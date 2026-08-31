#!/usr/bin/env python
"""Phase 4 -- zero-shot eval of every synthesised target bank.

Six transfer arms plus two raw-dump ablations. The ablation injects the SOURCE
bank verbatim (no relevance filtering, no re-grounding); if it matches the
synthesised arm, Phase 3's contribution is unearned. Baselines already exist at
results/design_c/{target}_baseline/run{0,1,2} and are not re-run.

Primary metric per the 2026-08-31 pre-registration amendment:
  maze       -> episode 1
  mastermind -> episode 3 and `any` (ep1 is a declared structural non-mover)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (target, arm, sources, bank_level)  bank_level=None -> synthesised (keyed on target)
ARMS = [
    ("maze", "near", ["minesweeper"], None),
    ("maze", "far", ["wordle"], None),
    ("maze", "both", ["minesweeper", "wordle"], None),
    ("mastermind", "near", ["wordle"], None),
    ("mastermind", "far", ["minesweeper"], None),
    ("mastermind", "both", ["minesweeper", "wordle"], None),
]
# raw-dump ablations: the best-covered single source for each target, injected raw
ABLATIONS = [
    ("maze", "rawdump", ["minesweeper"], "minesweeper"),
    ("mastermind", "rawdump", ["wordle"], "wordle"),
]


def bank_path(target: str, sources: list[str]) -> Path:
    return (ROOT / "results/design_c" / f"{target}_from_{'_'.join(sources)}"
            / "run0" / f"skill_bank_xl_{target}.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-index", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    arms = ARMS + ([] if args.no_ablation else ABLATIONS)
    results = []
    for target, arm, sources, bank_level in arms:
        bank = bank_path(target, sources)
        if not bank.exists():
            print(f"MISSING BANK, skipping: {bank}", flush=True)
            results.append({"target": target, "arm": arm, "sources": sources,
                            "returncode": None, "note": "bank missing"})
            continue
        out = (ROOT / "results/design_c/phase4"
               / f"{target}_{arm}_{'_'.join(sources)}" / f"run{args.run_index}")
        cmd = [sys.executable, "-u", str(ROOT / "scripts/run_orbit_target_zeroshot.py"),
               "--target", target, "--bank", str(bank),
               "--run-index", str(args.run_index),
               "--concurrency", str(args.concurrency),
               "--model", args.model, "--out", str(out)]
        if bank_level:
            cmd += ["--bank-level", bank_level]
        if args.base_url:
            cmd += ["--base-url", args.base_url]

        print("\n" + "=" * 72, flush=True)
        print(f"[phase 4] {target} | {arm} | {'+'.join(sources)}"
              f"{' | RAW DUMP ABLATION' if bank_level else ''}", flush=True)
        print("  " + " ".join(cmd), flush=True)
        if args.dry_run:
            results.append({"target": target, "arm": arm, "sources": sources,
                            "returncode": 0, "dir": str(out.relative_to(ROOT))})
            continue
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        print(f"[phase 4] {target}/{arm} exit={rc}", flush=True)
        results.append({"target": target, "arm": arm, "sources": sources,
                        "bank_level": bank_level, "returncode": rc,
                        "dir": str(out.relative_to(ROOT))})

    summary = ROOT / "results/design_c/phase4_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {summary}", flush=True)

    bad = [r for r in results if r.get("returncode") not in (0,)]
    print(f"\n{len(results) - len(bad)}/{len(results)} arms completed.", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
