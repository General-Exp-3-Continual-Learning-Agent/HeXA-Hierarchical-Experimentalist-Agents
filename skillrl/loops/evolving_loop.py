"""Iterative skill refinement loop using evolving skill banks (v2 variant).

Usage:
    python -m skillrl.evolving_loop \
        --level pass_the_parcel \
        --initial-traj-dir results/pass_the_parcel \
        --num-rounds 3 \
        --seeds-per-round 5 \
        --start-seed 6

Round 1: Distill skill bank from initial trajectories (using standard distill.py)
Round 2+: Evolve skill bank using previous bank + new trajectories

This is the v2 approach where the skill bank is incrementally refined with a fixed
maximum capacity, rather than being re-distilled from scratch each round.
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
    MAX_SKILLS_PER_LEVEL,
)
from skillrl.core.skill_bank import SkillBank
from skillrl.distillation.distill import load_trajectories, run_distillation
from skillrl.distillation.evolving_distill import evolve_skill_bank
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


def get_already_run_seeds(eval_dir: Path, level_name: str) -> set[int]:
    """Return set of seed numbers that have already been run for this level."""
    level_dir = eval_dir / level_name
    if not level_dir.exists():
        return set()

    already_run = set()
    for f in level_dir.glob("trajectory_seed*_skillrl.json"):
        # Extract seed number from "trajectory_seed{N}_skillrl.json"
        try:
            seed_str = f.stem.split("trajectory_seed")[1].split("_skillrl")[0]
            already_run.add(int(seed_str))
        except (IndexError, ValueError):
            continue
    return already_run


def generate_new_seeds(start_seed: int, num_seeds: int, already_run: set[int]) -> list[int]:
    """Generate num_seeds seed numbers, skipping any in already_run."""
    seeds = []
    current = start_seed
    while len(seeds) < num_seeds:
        if current not in already_run:
            seeds.append(current)
        current += 1
    return seeds


def run_evolving_loop(
    level_name: str,
    initial_traj_dir: Path,
    output_dir: Path,
    num_rounds: int = 3,
    seeds_per_round: int = 5,
    start_seed: int = 6,
    max_skills: int = MAX_SKILLS_PER_LEVEL,
    max_mistakes: int = 5,
    model_name: str = "claude",
    teacher_model: str = TEACHER_MODEL,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    skip_general: bool = True,
    max_general_skills: int = 2,
    max_specific_skills: int = 6,
    max_mistakes_agent: int = 4,
    verbose: bool = False,
):
    """Run the evolving skill refinement loop.

    This variant evolves the skill bank each round rather than re-distilling it.

    Parameters
    ----------
    level_name : Level to train on.
    initial_traj_dir : Directory with initial seed trajectories.
    output_dir : Root output directory for all rounds.
    num_rounds : Number of rounds (evolution starting from round 2).
    seeds_per_round : How many new seeds to run each round.
    start_seed : First seed number for round 1.
    max_skills : Maximum total skills per level (default: 10).
    max_mistakes : Maximum total mistakes per level (default: 5).
    model_name : Model for the agent (default: claude).
    teacher_model : Model for the teacher/distiller.
    skip_general : Skip cross-level generalization (recommended for single-level).
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

    # ── Resume detection ─────────────────────────────────────────
    progress_path = output_dir / "progress_evolving.json"
    resume_from_round = 0
    round_stats = []

    if progress_path.exists():
        try:
            round_stats = json.loads(progress_path.read_text())
            if round_stats:
                last = round_stats[-1]
                resume_from_round = last["round"]
                # Advance start_seed past all seeds already run in completed rounds.
                # Mid-round gaps (e.g. seed 20 missing while 19 and 21 ran) are
                # handled by generate_new_seeds, which scans forward and skips
                # anything already in `already_run` — so DO NOT bump start_seed
                # past max(mid_round_seeds), or gaps will be jumped over.
                start_seed = max(s for s in last["seeds"]) + 1

                print(f"\n  RESUMING from round {resume_from_round + 1}")
                print(f"  Previous rounds: {len(round_stats)}, next start_seed: {start_seed}")
        except (json.JSONDecodeError, KeyError):
            round_stats = []

    # Load initial trajectories
    print(f"\n{'='*70}")
    print(f"  EVOLVING SKILL LOOP: {level_name}")
    print(f"  Rounds: {resume_from_round + 1}-{num_rounds}, Seeds/round: {seeds_per_round}, Start seed: {start_seed}")
    print(f"  Max skills per level: {max_skills}, Max mistakes per level: {max_mistakes}")
    print(f"{'='*70}")

    init_successes, init_failures = collect_trajectories(initial_traj_dir)
    print(f"\nInitial trajectories: {len(init_successes)} successes, {len(init_failures)} failures")

    # Load model once
    model_fn = _load_model(model_name)
    is_oss = "gpt-oss" in model_name.lower() if isinstance(model_name, str) else False

    current_seed = start_seed

    for round_num in range(resume_from_round + 1, num_rounds + 1):
        round_start = time.perf_counter()
        print(f"\n{'#'*70}")
        print(f"  ROUND {round_num}/{num_rounds}")
        print(f"{'#'*70}")

        # ── Phase 1: Create or evolve skill bank ──────────────────────────────
        skill_bank_path = output_dir / f"skill_bank_evolving_{round_num}.json"

        if round_num == 1:
            # Round 1: Seed the bank using initial trajectories + original distill method
            # (or load existing if already distilled)
            if skill_bank_path.exists():
                print(f"\n  Phase 1: Skill bank already exists, skipping distillation")
                bank = SkillBank.load(skill_bank_path)
                print(f"\n  Loaded existing skill bank: {skill_bank_path}")
            else:
                print(f"\n  Phase 1: Seeding skill bank from initial trajectories")

                traj_dirs = {level_name: initial_traj_dir}
                bank = run_distillation(
                    traj_dirs=traj_dirs,
                    levels=[level_name],
                    output_path=skill_bank_path,
                    teacher_model=teacher_model,
                    skip_general=skip_general,
                )
                print(f"\n  Skill bank seeded: {skill_bank_path}")

        else:
            # Round 2+: Evolve the previous bank using new trajectories
            # (or load existing if already evolved)
            if skill_bank_path.exists():
                print(f"\n  Phase 1: Skill bank already exists, skipping evolution")
                bank = SkillBank.load(skill_bank_path)
                print(f"\n  Loaded existing skill bank: {skill_bank_path}")
            else:
                print(f"\n  Phase 1: Evolving skill bank from round {round_num - 1}")

                # Load previous bank
                prev_skill_bank_path = output_dir / f"skill_bank_evolving_{round_num - 1}.json"
                if not prev_skill_bank_path.exists():
                    print(f"  [Error] Previous skill bank not found: {prev_skill_bank_path}")
                    raise FileNotFoundError(f"Cannot resume: {prev_skill_bank_path} not found")

                prev_bank = SkillBank.load(prev_skill_bank_path)
                print(f"  Loaded previous bank: {prev_bank}")

                # Collect new trajectories from previous round
                prev_round_dir = output_dir / f"round_{round_num - 1}" / level_name
                if not prev_round_dir.exists():
                    print(f"  [Error] Previous round trajectory directory not found: {prev_round_dir}")
                    raise FileNotFoundError(f"Cannot resume: {prev_round_dir} not found")

                # Evolve the bank
                bank = evolve_skill_bank(
                    level_name=level_name,
                    prev_bank=prev_bank,
                    new_trajs_dir=prev_round_dir,
                    output_path=skill_bank_path,
                    max_skills=max_skills,
                    max_mistakes=max_mistakes,
                    teacher_model=teacher_model,
                )
                print(f"\n  Evolved skill bank saved: {skill_bank_path}")

        # ── Phase 2: Run skill-augmented agent ───────────────────────────────
        round_eval_dir = output_dir / f"round_{round_num}"

        # Detect which seeds have already been run for this round
        already_run = get_already_run_seeds(round_eval_dir, level_name)
        if already_run:
            print(f"\n  Phase 2: Detected already-run seeds: {sorted(already_run)}")

        # Determine which seeds to use for this round
        if len(already_run) >= seeds_per_round:
            # The intended batch for this round is already complete — load from disk, skip run_batch
            seeds = sorted(list(already_run))[:seeds_per_round]
            print(f"  Phase 2: Round {round_num} batch {seeds} already complete. Loading existing results (skipping agent re-run)...")
            results = []
            for seed in seeds:
                traj_path = round_eval_dir / level_name / f"trajectory_seed{seed}_skillrl.json"
                if traj_path.exists():
                    try:
                        results.append(json.loads(traj_path.read_text()))
                    except json.JSONDecodeError:
                        print(f"  [Warning] Could not parse {traj_path}, skipping")
        else:
            # Generate new seeds to fill the batch, skipping any already done; only run the missing ones
            missing_count = seeds_per_round - len(already_run)
            new_seeds = generate_new_seeds(current_seed, missing_count, already_run)
            if already_run:
                print(f"  Phase 2: Round partially complete, running missing seeds: {new_seeds}")
            print(f"\n  Phase 2: Running agent on seeds {new_seeds}")

            fresh_results = run_batch(
                model_fn=model_fn,
                level_name=level_name,
                seeds=new_seeds,
                skill_bank=bank,
                eval_dir=round_eval_dir,
                max_iterations=max_iterations,
                verbose=verbose,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                is_oss=is_oss,
                max_general_skills=max_general_skills,
                max_specific_skills=max_specific_skills,
                max_mistakes=max_mistakes_agent,
            )

            # Combine with already-run seeds (loaded from disk) so round_stats reflects the full batch
            seeds = sorted(list(already_run)) + new_seeds
            results = []
            for seed in sorted(already_run):
                traj_path = round_eval_dir / level_name / f"trajectory_seed{seed}_skillrl.json"
                if traj_path.exists():
                    try:
                        results.append(json.loads(traj_path.read_text()))
                    except json.JSONDecodeError:
                        print(f"  [Warning] Could not parse {traj_path}, skipping")
            results.extend(fresh_results)

        # ── Phase 3: Collect results ─────────────────────────────────────────
        round_successes = [r for r in results if r.get("success")]
        round_failures = [r for r in results if not r.get("success")]
        round_accuracy = len(round_successes) / len(results) if results else 0

        print(f"\n  Round {round_num} results: {len(round_successes)}/{len(results)} = {round_accuracy:.0%}")

        round_elapsed = time.perf_counter() - round_start

        stats = {
            "round": round_num,
            "seeds": seeds,
            "successes": len(round_successes),
            "failures": len(round_failures),
            "accuracy": round_accuracy,
            "skill_bank": str(skill_bank_path),
            "elapsed_seconds": round_elapsed,
        }
        round_stats.append(stats)

        # Save progress after each round
        progress_path.write_text(json.dumps(round_stats, indent=2))

        # Update current_seed to one past the highest seed run
        current_seed = max(seeds) + 1 if seeds else current_seed + seeds_per_round

        print(f"\n  Round {round_num} complete in {round_elapsed:.0f}s")

    # ── Final summary ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  EVOLVING LOOP COMPLETE")
    print(f"{'='*70}")
    print(f"\n  Round-by-round results:")
    for s in round_stats:
        print(f"    Round {s['round']}: {s['successes']}/{s['successes']+s['failures']} "
              f"= {s['accuracy']:.0%} ({s['elapsed_seconds']:.0f}s)")

    total_successes = sum(s["successes"] for s in round_stats)
    total_runs = sum(s["successes"] + s["failures"] for s in round_stats)
    print(f"\n  Overall: {total_successes}/{total_runs} = {total_successes/total_runs:.0%}" if total_runs else "")
    print(f"  Skill banks saved in: {output_dir}")
    print(f"  Progress log: {progress_path}")

    return round_stats


def main():
    parser = argparse.ArgumentParser(
        description="Evolving skill refinement loop for a single level (v2 variant)"
    )
    parser.add_argument("--level", type=str, required=True,
                        help="Puzzle level to train on")
    parser.add_argument("--initial-traj-dir", type=str, required=True,
                        help="Directory with initial seed trajectories")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: skillrl/data/evolving/{level})")
    parser.add_argument("--num-rounds", type=int, default=3,
                        help="Number of rounds (default: 3)")
    parser.add_argument("--seeds-per-round", type=int, default=5,
                        help="New seeds to run each round (default: 5)")
    parser.add_argument("--start-seed", type=int, default=6,
                        help="First seed number (default: 6)")
    parser.add_argument("--max-skills", type=int, default=MAX_SKILLS_PER_LEVEL,
                        help=f"Max skills per level (default: {MAX_SKILLS_PER_LEVEL})")
    parser.add_argument("--max-mistakes", type=int, default=5,
                        help="Max mistakes per level (default: 5)")
    parser.add_argument("--model", type=str, default="claude",
                        help="Agent model (default: claude)")
    parser.add_argument("--teacher-model", type=str, default=TEACHER_MODEL,
                        help="Teacher model for evolution")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--skip-general", action="store_true", default=True,
                        help="Skip cross-level generalization (default: True for single level)")
    parser.add_argument("--max-general-skills", type=int, default=0)
    parser.add_argument("--max-specific-skills", type=int, default=6)
    parser.add_argument("--max-mistakes-agent", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(
            Path(__file__).resolve().parent / "data" / "evolving" / args.level
        )

    run_evolving_loop(
        level_name=args.level,
        initial_traj_dir=Path(args.initial_traj_dir),
        output_dir=Path(args.output_dir),
        num_rounds=args.num_rounds,
        seeds_per_round=args.seeds_per_round,
        start_seed=args.start_seed,
        max_skills=args.max_skills,
        max_mistakes=args.max_mistakes,
        model_name=args.model,
        teacher_model=args.teacher_model,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        skip_general=args.skip_general,
        max_general_skills=args.max_general_skills,
        max_specific_skills=args.max_specific_skills,
        max_mistakes_agent=args.max_mistakes_agent,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
