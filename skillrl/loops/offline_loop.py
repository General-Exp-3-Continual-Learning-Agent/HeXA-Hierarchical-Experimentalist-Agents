"""Offline (static) skill bank variant.

Usage:
    python -m skillrl.loops.offline_loop --level catapult --initial-traj-dir Initial_trajectories/catapult/ --seeds 6 --output-dir skillrl/data/Hexa_offline_catapult

Distills a skill bank ONCE from initial trajectories, then runs ALL seeds
with that same fixed bank. No evolution, no re-distillation between rounds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skillrl.core.config import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_NEW_TOKENS,
    TEACHER_MODEL,
)
from skillrl.core.skill_bank import SkillBank
from skillrl.distillation.distill import run_distillation
from skillrl.runner.run_skill_agent import run_batch, _load_model


def run_offline(
    level_name: str,
    initial_traj_dir: Path,
    output_dir: Path,
    seeds: list[int],
    batch_size: int = 5,
    model_name: str = "claude",
    teacher_model: str = TEACHER_MODEL,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    skip_general: bool = True,
    max_general_skills: int = 0,
    max_specific_skills: int = 6,
    max_mistakes: int = 4,
    verbose: bool = False,
):
    """Distill once, then run all seeds with the fixed skill bank.

    Parameters
    ----------
    level_name : Level to solve.
    initial_traj_dir : Directory with initial seed trajectories for distillation.
    output_dir : Root output directory.
    seeds : All seeds to evaluate.
    batch_size : Seeds per batch for progress tracking (default: 5).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if level_name == "catapult":
        from skillrl.distillation import teacher_prompts_catapult as _tpc
        _tpc.patch()
        print(
            "  [Prompts] Patched distill + evolving_distill with the catapult-specific "
            "factual-block variants (no strategy hints leaked to teacher)."
        )

    progress_path = output_dir / "progress_offline.json"
    skill_bank_path = output_dir / "skill_bank_offline.json"

    # ── Phase 1: Distill skill bank (once) ───────────────────────
    if skill_bank_path.exists():
        bank = SkillBank.load(skill_bank_path)
        print(f"\n  Reusing existing skill bank: {skill_bank_path}")
        print(f"  {bank}")
    else:
        print(f"\n{'='*70}")
        print(f"  OFFLINE VARIANT: Distilling skill bank from initial trajectories")
        print(f"{'='*70}")

        traj_dirs = {level_name: initial_traj_dir}
        bank = run_distillation(
            traj_dirs=traj_dirs,
            levels=[level_name],
            output_path=skill_bank_path,
            teacher_model=teacher_model,
            skip_general=skip_general,
        )
        print(f"\n  Skill bank saved: {skill_bank_path}")
        print(f"  {bank}")

    # ── Resume: skip already-completed seeds ─────────────────────
    eval_dir = output_dir / "results"
    traj_dir = eval_dir / level_name

    completed_seeds = set()
    if traj_dir.exists():
        for f in traj_dir.glob("trajectory_seed*_skillrl.json"):
            try:
                data = json.loads(f.read_text())
                if "seed" in data:
                    completed_seeds.add(data["seed"])
            except (json.JSONDecodeError, KeyError):
                pass

    remaining_seeds = [s for s in seeds if s not in completed_seeds]

    print(f"\n{'='*70}")
    print(f"  OFFLINE VARIANT: {level_name}")
    print(f"  Total seeds: {len(seeds)}, Already done: {len(completed_seeds)}, Remaining: {len(remaining_seeds)}")
    print(f"  Skill bank: {skill_bank_path} (FIXED -- same for all seeds)")
    print(f"{'='*70}")

    if not remaining_seeds:
        print(f"\n  All seeds already completed.")
    else:
        # Load model once
        model_fn = _load_model(model_name)
        is_oss = "gpt-oss" in model_name.lower() if isinstance(model_name, str) else False

        # Run in batches for progress tracking
        for i in range(0, len(remaining_seeds), batch_size):
            batch = remaining_seeds[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(remaining_seeds) + batch_size - 1) // batch_size

            print(f"\n  Batch {batch_num}/{total_batches}: seeds {batch}")

            run_batch(
                model_fn=model_fn,
                level_name=level_name,
                seeds=batch,
                skill_bank=bank,
                eval_dir=eval_dir,
                max_iterations=max_iterations,
                verbose=verbose,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                is_oss=is_oss,
                max_general_skills=max_general_skills,
                max_specific_skills=max_specific_skills,
                max_mistakes=max_mistakes,
            )

    # ── Compute final stats from all trajectory files ────────────
    all_results = []
    for s in seeds:
        traj_file = traj_dir / f"trajectory_seed{s}_skillrl.json"
        if traj_file.exists():
            try:
                all_results.append(json.loads(traj_file.read_text()))
            except json.JSONDecodeError:
                pass

    total = len(all_results)
    successes = sum(1 for r in all_results if r.get("success"))
    accuracy = successes / total if total else 0

    progress = {
        "variant": "offline",
        "level": level_name,
        "skill_bank": str(skill_bank_path),
        "seeds_requested": seeds,
        "seeds_completed": [r.get("seed") for r in all_results],
        "total": total,
        "successes": successes,
        "accuracy": accuracy,
        "avg_iterations": sum(r.get("iterations", 0) for r in all_results) / total if total else 0,
    }
    progress_path.write_text(json.dumps(progress, indent=2))

    print(f"\n{'='*70}")
    print(f"  OFFLINE VARIANT COMPLETE")
    print(f"{'='*70}")
    print(f"  {level_name}: {successes}/{total} = {accuracy:.0%}")
    print(f"  Skill bank (fixed): {skill_bank_path}")
    print(f"  Results: {eval_dir}")
    print(f"  Progress: {progress_path}")

    return progress


def main():
    parser = argparse.ArgumentParser(
        description="Offline (static) skill bank variant -- distill once, run all seeds"
    )
    parser.add_argument("--level", type=str, required=True,
                        help="Puzzle level to solve")
    parser.add_argument("--initial-traj-dir", type=str, required=True,
                        help="Directory with initial trajectories for distillation")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: skillrl/data/offline/{level})")
    parser.add_argument("--seeds", nargs="+", type=int, required=True,
                        help="Seeds to evaluate (e.g. --seeds 6 7 8 9 10)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Seeds per batch for progress tracking (default: 5)")
    parser.add_argument("--model", type=str, default="claude",
                        help="Agent model (default: claude)")
    parser.add_argument("--teacher-model", type=str, default=TEACHER_MODEL,
                        help="Teacher model for distillation")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--skip-general", action="store_true", default=True)
    parser.add_argument("--max-general-skills", type=int, default=0)
    parser.add_argument("--max-specific-skills", type=int, default=6)
    parser.add_argument("--max-mistakes", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(
            Path(__file__).resolve().parent / "data" / "offline" / args.level
        )

    run_offline(
        level_name=args.level,
        initial_traj_dir=Path(args.initial_traj_dir),
        output_dir=Path(args.output_dir),
        seeds=args.seeds,
        batch_size=args.batch_size,
        model_name=args.model,
        teacher_model=args.teacher_model,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        skip_general=args.skip_general,
        max_general_skills=args.max_general_skills,
        max_specific_skills=args.max_specific_skills,
        max_mistakes=args.max_mistakes,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
