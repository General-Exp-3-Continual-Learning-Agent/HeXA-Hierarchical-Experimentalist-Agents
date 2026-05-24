"""Automated iterative skill refinement loop for a single level.

Usage:
    python -m skillrl.iterative_loop \
        --level pass_the_parcel \
        --initial-traj-dir results/pass_the_parcel \
        --num-rounds 3 \
        --seeds-per-round 5 \
        --start-seed 6

Each round:
  1. Distill skill bank from all accumulated trajectories starting from initial trajectories
     (latest round's trajectories)
  2. Run skill-augmented agent on the next batch of seeds
  3. Collect results, carry forward successes + new trajectories
  4. Repeat
"""

from __future__ import annotations

import argparse
import json
import shutil
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
from skillrl.distillation.distill import run_distillation
from skillrl.core.skill_bank import SkillBank
from skillrl.runner.run_skill_agent import run_batch, _load_model


def collect_trajectories(traj_dir: Path) -> tuple[list[dict], list[dict]]:
    """Load all trajectories from a directory, split into successes/failures."""
    successes, failures = [], []
    if not traj_dir.exists():
        return successes, failures
    for f in sorted(traj_dir.glob("trajectory_seed*.json")):
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("success"):
            successes.append(data)
        else:
            failures.append(data)
    return successes, failures


def copy_trajectories(src_dir: Path, dst_dir: Path):
    """Copy all trajectory JSON files from src to dst."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in src_dir.glob("trajectory_seed*.json"):
        shutil.copy2(f, dst_dir / f.name)


def select_best_successes(
    all_successes: list[dict], max_count: int = 2
) -> list[dict]:
    """Pick the best (fewest iterations) successes to carry forward."""
    sorted_s = sorted(all_successes, key=lambda t: t.get("iterations", 999))
    return sorted_s[:max_count]


def run_iterative_loop(
    level_name: str,
    initial_traj_dir: Path,
    output_dir: Path,
    num_rounds: int = 3,
    seeds_per_round: int = 5,
    start_seed: int = 6,
    max_carry_successes: int = 2,
    model_name: str = "claude",
    teacher_model: str = TEACHER_MODEL,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    skip_general: bool = True,
    max_general_skills: int = 2,
    max_specific_skills: int = 5,
    max_mistakes: int = 3,
    verbose: bool = False,
    skip_distill: bool = False,
):
    """Run the full iterative skill refinement loop.

    Parameters
    ----------
    level_name : Level to train on.
    initial_traj_dir : Directory with initial seed trajectories.
    output_dir : Root output directory for all rounds.
    num_rounds : Number of distill-then-run rounds.
    seeds_per_round : How many new seeds to run each round.
    start_seed : First seed number for round 1.
    max_carry_successes : Max previous successes to include in distillation.
    model_name : Model for the agent (default: claude).
    teacher_model : Model for the teacher/distiller.
    skip_general : Skip cross-level generalization (recommended for single-level).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume detection ─────────────────────────────────────────
    progress_path = output_dir / "progress.json"
    resume_from_round = 0
    round_stats = []

    if progress_path.exists():
        try:
            round_stats = json.loads(progress_path.read_text())
            if round_stats:
                last = round_stats[-1]
                resume_from_round = last["round"]
                # Advance start_seed past all seeds already run
                start_seed = max(s for s in last["seeds"]) + 1
                print(f"\n  RESUMING from round {resume_from_round + 1}")
                print(f"  Previous rounds: {len(round_stats)}, next start_seed: {start_seed}")
        except (json.JSONDecodeError, KeyError):
            round_stats = []

    # Load initial trajectories
    print(f"\n{'='*70}")
    print(f"  ITERATIVE SKILL REFINEMENT: {level_name}")
    print(f"  Rounds: {resume_from_round + 1}-{num_rounds}, Seeds/round: {seeds_per_round}, Start seed: {start_seed}")
    print(f"{'='*70}")

    init_successes, init_failures = collect_trajectories(initial_traj_dir)
    print(f"\nInitial trajectories: {len(init_successes)} successes, {len(init_failures)} failures")

    # Track accumulated trajectories across rounds
    all_trajectories_dir = output_dir / "accumulated_trajectories" / level_name
    all_trajectories_dir.mkdir(parents=True, exist_ok=True)

    # Copy initial trajectories to accumulated dir (safe to re-copy on resume)
    copy_trajectories(initial_traj_dir, all_trajectories_dir)

    # On resume, count all accumulated successes so far
    acc_succ, _ = collect_trajectories(all_trajectories_dir)
    all_successes = list(acc_succ) if acc_succ else list(init_successes)

    # Load model once
    model_fn = _load_model(model_name)
    is_oss = "gpt-oss" in model_name.lower() if isinstance(model_name, str) else False

    current_seed = start_seed

    for round_num in range(resume_from_round + 1, num_rounds + 1):
        round_start = time.perf_counter()
        print(f"\n{'#'*70}")
        print(f"  ROUND {round_num}/{num_rounds}")
        print(f"{'#'*70}")

        # ── Phase 1: Distill skill bank ──────────────────────────────
        skill_bank_path = output_dir / f"skill_bank_{round_num}.json"

        # Build distillation input:
        #   Round 1: all initial trajectories
        #   Round 2+: latest round trajectories + best N previous successes
        distill_traj_dir = output_dir / f"distill_input_{round_num}" / level_name
        if distill_traj_dir.exists():
            shutil.rmtree(distill_traj_dir)
        distill_traj_dir.mkdir(parents=True, exist_ok=True)

        if round_num == 1:
            # Round 1: use all initial trajectories
            copy_trajectories(initial_traj_dir, distill_traj_dir)
            print(f"\n  Phase 1: Distilling from initial trajectories")
        else:
            # Round 2+: latest round trajectories + best previous successes
            prev_round_dir = output_dir / f"round_{round_num - 1}" / level_name
            copy_trajectories(prev_round_dir, distill_traj_dir)

            # Add best N previous successes (by fewest iterations)
            best_prev = select_best_successes(all_successes, max_carry_successes)
            for traj in best_prev:
                seed = traj.get("seed", -1)
                traj_path = distill_traj_dir / f"trajectory_seed{seed}_skillrl.json"
                if not traj_path.exists():  # don't overwrite if already in latest round
                    traj_path.write_text(json.dumps(traj, indent=2, default=str))

            print(f"\n  Phase 1: Distilling from round {round_num - 1} trajectories + {len(best_prev)} best previous successes")

        if skip_distill and skill_bank_path.exists():
            bank = SkillBank.load(skill_bank_path)
            print(f"\n  Phase 1: SKIPPED — loading existing skill bank: {skill_bank_path}")
            print(f"  {bank}")
        else:
            succ, fail = collect_trajectories(distill_traj_dir)
            print(f"    Distillation input: {len(succ)} successes, {len(fail)} failures")

            traj_dirs = {level_name: distill_traj_dir}
            bank = run_distillation(
                traj_dirs=traj_dirs,
                levels=[level_name],
                output_path=skill_bank_path,
                teacher_model=teacher_model,
                skip_general=skip_general,
            )

            print(f"\n  Skill bank saved: {skill_bank_path}")
            print(f"  {bank}")

        # ── Phase 2: Run skill-augmented agent ───────────────────────
        seeds = list(range(current_seed, current_seed + seeds_per_round))
        round_eval_dir = output_dir / f"round_{round_num}"

        print(f"\n  Phase 2: Running agent on seeds {seeds}")

        results = run_batch(
            model_fn=model_fn,
            level_name=level_name,
            seeds=seeds,
            skill_bank=bank,
            eval_dir=round_eval_dir,
            max_iterations=max_iterations,
            verbose=verbose,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            is_oss=is_oss,
            max_general_skills=max_general_skills,
            max_specific_skills=max_specific_skills,
            max_mistakes=max_mistakes,
        )

        # ── Phase 3: Collect results ─────────────────────────────────
        round_successes = [r for r in results if r.get("success")]
        round_failures = [r for r in results if not r.get("success")]
        round_accuracy = len(round_successes) / len(results) if results else 0

        print(f"\n  Round {round_num} results: {len(round_successes)}/{len(results)} = {round_accuracy:.0%}")

        # Copy new trajectories to accumulated dir
        round_traj_dir = round_eval_dir / level_name
        copy_trajectories(round_traj_dir, all_trajectories_dir)

        # Update accumulated successes
        all_successes.extend(round_successes)

        # If we have too many trajectories, keep only the best successes
        # plus ALL failures (failures are valuable for learning)
        if max_carry_successes and len(all_successes) > max_carry_successes + seeds_per_round:
            # Rebuild accumulated dir with best successes + all failures + latest round
            best = select_best_successes(all_successes, max_carry_successes)
            # Keep all files but this ensures distillation sees the right mix
            print(f"    Carrying forward {len(best)} best successes + all failures")

        round_elapsed = time.perf_counter() - round_start

        stats = {
            "round": round_num,
            "seeds": seeds,
            "successes": len(round_successes),
            "failures": len(round_failures),
            "accuracy": round_accuracy,
            "skill_bank": str(skill_bank_path),
            "elapsed_seconds": round_elapsed,
            "accumulated_successes": len(all_successes),
        }
        round_stats.append(stats)

        # Save progress after each round
        progress_path.write_text(json.dumps(round_stats, indent=2))

        current_seed += seeds_per_round

        print(f"\n  Round {round_num} complete in {round_elapsed:.0f}s")
        print(f"  Accumulated: {len(all_successes)} total successes across all rounds")

    # ── Final summary ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ITERATIVE LOOP COMPLETE")
    print(f"{'='*70}")
    print(f"\n  Round-by-round results:")
    for s in round_stats:
        print(f"    Round {s['round']}: {s['successes']}/{s['successes']+s['failures']} "
              f"= {s['accuracy']:.0%} ({s['elapsed_seconds']:.0f}s)")

    total_successes = sum(s["successes"] for s in round_stats)
    total_runs = sum(s["successes"] + s["failures"] for s in round_stats)
    print(f"\n  Overall: {total_successes}/{total_runs} = {total_successes/total_runs:.0%}" if total_runs else "")
    print(f"  Skill banks saved in: {output_dir}")
    print(f"  Progress log: {output_dir / 'progress.json'}")

    return round_stats


def main():
    parser = argparse.ArgumentParser(
        description="Iterative skill refinement loop for a single level"
    )
    parser.add_argument("--level", type=str, required=True,
                        help="Puzzle level to train on")
    parser.add_argument("--initial-traj-dir", type=str, required=True,
                        help="Directory with initial seed trajectories")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: skillrl/data/iterative/{level})")
    parser.add_argument("--num-rounds", type=int, default=3,
                        help="Number of distill-then-run rounds (default: 3)")
    parser.add_argument("--seeds-per-round", type=int, default=3,
                        help="New seeds to run each round (default: 5)")
    parser.add_argument("--start-seed", type=int, default=6,
                        help="First seed number (default: 6)")
    parser.add_argument("--max-carry-successes", type=int, default=0,
                        help="Max previous successes to carry forward (default: 2)")
    parser.add_argument("--model", type=str, default="claude",
                        help="Agent model (default: claude)")
    parser.add_argument("--teacher-model", type=str, default=TEACHER_MODEL,
                        help="Teacher model for distillation")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--skip-general", action="store_true", default=True,
                        help="Skip cross-level generalization (default: True for single level)")
    parser.add_argument("--skip-distill", action="store_true", default=False,
                        help="Skip distillation and load existing skill bank from disk")
    parser.add_argument("--max-general-skills", type=int, default=0)
    parser.add_argument("--max-specific-skills", type=int, default=6)
    parser.add_argument("--max-mistakes", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(
            Path(__file__).resolve().parent / "data" / "iterative" / args.level
        )

    run_iterative_loop(
        level_name=args.level,
        initial_traj_dir=Path(args.initial_traj_dir),
        output_dir=Path(args.output_dir),
        num_rounds=args.num_rounds,
        seeds_per_round=args.seeds_per_round,
        start_seed=args.start_seed,
        max_carry_successes=args.max_carry_successes,
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
        skip_distill=args.skip_distill,
    )


if __name__ == "__main__":
    main()
