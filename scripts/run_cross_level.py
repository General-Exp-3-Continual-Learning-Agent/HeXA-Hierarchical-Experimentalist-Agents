#!/usr/bin/env python3
"""Cross-level meta-transfer: synthesise a target-level skill bank from
fully-evolved source banks, then evaluate on the target via the offline loop.

Step 1: Run ``skillrl.distillation.cross_level_synthesis`` to produce
``skill_bank_xl_<target>.json`` and an audit JSON at ``--synthesis-output-dir``.
The teacher reads only the source banks + factual target-level scene
descriptions — no target-level trajectories are used.

Step 2: Move the synthesised bank into ``<offline-output-dir>/skill_bank_offline.json``
and run ``skillrl.loops.offline_loop`` with the supplied evaluation seeds.
``offline_loop`` reuses an existing skill_bank_offline.json verbatim, so no
re-distillation occurs.

Usage:
    python scripts/run_cross_level.py \\
        --target catapult \\
        --source-bank down_to_earth=results/skill_bank_dte.json \\
        --source-bank two_body_problem=results/skill_bank_tbp.json \\
        --source-bank pass_the_parcel=results/skill_bank_ptp.json \\
        --synthesis-output-dir results/cross_level/catapult \\
        --offline-output-dir results/cross_level/catapult/offline_eval \\
        --eval-seeds 6 7 8 9 10 11 12 13 14 15
"""
import argparse, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env() -> dict:
    """Repo-root + interphyre/ on PYTHONPATH so subprocesses import correctly."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(ROOT), str(ROOT / "interphyre"), env.get("PYTHONPATH", "")) if p
    )
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--target", required=True, help="Target level name.")
    parser.add_argument(
        "--source-bank", action="append", required=True,
        help="Repeatable: '<source_level>=<path_to_evolved_bank.json>'.",
    )
    parser.add_argument(
        "--synthesis-output-dir", required=True, type=Path,
        help="Where the synthesised bank + audit JSON are written.",
    )
    parser.add_argument(
        "--offline-output-dir", required=True, type=Path,
        help="Output directory for the offline evaluation phase.",
    )
    parser.add_argument(
        "--eval-seeds", nargs="+", type=int, required=True,
        help="Seeds to evaluate on the target level.",
    )
    parser.add_argument(
        "--teacher-model", default="claude-sonnet-4-6",
        help="Teacher model used for synthesis (default: claude-sonnet-4-6).",
    )
    parser.add_argument(
        "--structural-hint", default=None,
        help="Optional one-paragraph hint relating source and target levels.",
    )
    parser.add_argument(
        "--skip-synthesis", action="store_true",
        help="Skip step 1 if a synthesised bank already exists at the expected path.",
    )
    args = parser.parse_args()

    syn_dir = args.synthesis_output_dir.resolve()
    syn_dir.mkdir(parents=True, exist_ok=True)
    synthesised = syn_dir / f"skill_bank_xl_{args.target}.json"

    if args.skip_synthesis and synthesised.exists():
        print(f"[1/2] Skipping synthesis — reusing {synthesised}")
    else:
        cmd = [
            sys.executable, "-m", "skillrl.distillation.cross_level_synthesis",
            "--target", args.target,
            "--output-dir", str(syn_dir),
            "--teacher-model", args.teacher_model,
        ]
        for sb in args.source_bank:
            cmd += ["--source-bank", sb]
        if args.structural_hint:
            cmd += ["--structural-hint", args.structural_hint]
        print(f"[1/2] Synthesising target bank → {synthesised}")
        rc = subprocess.run(cmd, cwd=ROOT, env=_env()).returncode
        if rc != 0:
            sys.exit(rc)
        if not synthesised.exists():
            sys.exit(f"Synthesis completed but {synthesised} is missing.")

    # Stage the synthesised bank where offline_loop expects it.
    off_dir = args.offline_output_dir.resolve()
    off_dir.mkdir(parents=True, exist_ok=True)
    staged = off_dir / "skill_bank_offline.json"
    if staged.resolve() != synthesised.resolve():
        shutil.copy2(synthesised, staged)
    print(f"[2/2] Running offline evaluation with {staged}")

    cmd = [
        sys.executable, "-m", "skillrl.loops.offline_loop",
        "--level", args.target,
        # offline_loop requires this flag; the bank already exists so distillation is skipped.
        "--initial-traj-dir", str(ROOT / "Initial_trajectories" / args.target),
        "--output-dir", str(off_dir),
        "--seeds", *map(str, args.eval_seeds),
    ]
    sys.exit(subprocess.run(cmd, cwd=ROOT, env=_env()).returncode)


if __name__ == "__main__":
    main()
