"""Online-start evolving skill refinement loop.

Usage:
    python -m skillrl.online_evolving_loop \
        --level pass_the_parcel \
        --num-rounds 5 \
        --seeds-per-round 3 \
        --start-seed 6

Round 1: No skill bank -- agent runs fully online (baseline).
Round 2: Distill initial skill bank from round 1 trajectories (standard distillation).
Round 3+: Evolve skill bank using previous bank + new trajectories (same as evolving v2).
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
    MAX_SKILLS_PER_LEVEL,
)
from skillrl.core.skill_bank import SkillBank
from skillrl.distillation.distill import run_distillation
from skillrl.distillation.evolving_distill import evolve_skill_bank
from skillrl.loops.evolving_loop import collect_trajectories
from skillrl.runner.run_skill_agent import run_batch, _load_model


def run_online_evolving_loop(
    level_name: str,
    output_dir: Path,
    num_rounds: int = 5,
    seeds_per_round: int = 3,
    start_seed: int = 6,
    max_skills: int = MAX_SKILLS_PER_LEVEL,
    max_mistakes: int = 5,
    model_name: str = "claude",
    teacher_model: str = TEACHER_MODEL,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    skip_general: bool = True,
    max_general_skills: int = 0,
    max_specific_skills: int = 6,
    max_mistakes_agent: int = 4,
    verbose: bool = False,
):
    """Run the online-start evolving skill refinement loop.

    Round 1: No skill bank (online baseline).
    Round 2: Distill from round 1 trajectories.
    Round 3+: Evolve previous bank with new trajectories.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume detection ─────────────────────────────────────────
    progress_path = output_dir / "progress_online_evolving.json"
    resume_from_round = 0
    round_stats = []

    if progress_path.exists():
        try:
            round_stats = json.loads(progress_path.read_text())
            if round_stats:
                last = round_stats[-1]
                resume_from_round = last["round"]
                start_seed = max(s for s in last["seeds"]) + 1
                print(f"\n  RESUMING from round {resume_from_round + 1}")
                print(f"  Previous rounds: {len(round_stats)}, next start_seed: {start_seed}")
        except (json.JSONDecodeError, KeyError):
            round_stats = []

    print(f"\n{'='*70}")
    print(f"  ONLINE-EVOLVING LOOP: {level_name}")
    print(f"  Rounds: {resume_from_round + 1}-{num_rounds}, Seeds/round: {seeds_per_round}, Start seed: {start_seed}")
    print(f"  Round 1 = online (no skills), Round 2 = distill, Round 3+ = evolve")
    print(f"  Max skills/level: {max_skills}, Max mistakes/level: {max_mistakes}")
    print(f"{'='*70}")

    # Load model once
    model_fn = _load_model(model_name)
    is_oss = "gpt-oss" in model_name.lower() if isinstance(model_name, str) else False

    current_seed = start_seed

    for round_num in range(resume_from_round + 1, num_rounds + 1):
        round_start = time.perf_counter()
        print(f"\n{'#'*70}")
        print(f"  ROUND {round_num}/{num_rounds}")
        print(f"{'#'*70}")

        # ── Phase 1: Get or create/evolve skill bank ─────────────────
        skill_bank_path = output_dir / f"skill_bank_online_evolving_{round_num}.json"

        if round_num == 1:
            # Round 1: NO skill bank -- fully online
            print(f"\n  Phase 1: ONLINE round (no skill bank)")
            bank = SkillBank()
            skill_bank_path = None

        elif round_num == 2:
            # Round 2: Distill from round 1 trajectories (standard distillation)
            if skill_bank_path.exists():
                bank = SkillBank.load(skill_bank_path)
                print(f"\n  Phase 1: Reusing existing skill bank: {skill_bank_path}")
                print(f"  {bank}")
            else:
                round1_traj_dir = output_dir / "round_1" / level_name
                print(f"\n  Phase 1: Distilling initial bank from round 1 trajectories")
                succ, fail = collect_trajectories(round1_traj_dir)
                print(f"    Input: {len(succ)} successes, {len(fail)} failures")

                traj_dirs = {level_name: round1_traj_dir}
                bank = run_distillation(
                    traj_dirs=traj_dirs,
                    levels=[level_name],
                    output_path=skill_bank_path,
                    teacher_model=teacher_model,
                    skip_general=skip_general,
                )
                print(f"\n  Skill bank saved: {skill_bank_path}")
                print(f"  {bank}")

        else:
            # Round 3+: Evolve previous bank with new trajectories
            if skill_bank_path.exists():
                bank = SkillBank.load(skill_bank_path)
                print(f"\n  Phase 1: Reusing existing skill bank: {skill_bank_path}")
                print(f"  {bank}")
            else:
                prev_bank_path = output_dir / f"skill_bank_online_evolving_{round_num - 1}.json"
                if not prev_bank_path.exists():
                    raise FileNotFoundError(f"Previous skill bank not found: {prev_bank_path}")

                prev_bank = SkillBank.load(prev_bank_path)
                prev_round_dir = output_dir / f"round_{round_num - 1}" / level_name
                if not prev_round_dir.exists():
                    raise FileNotFoundError(f"Previous round trajectories not found: {prev_round_dir}")

                print(f"\n  Phase 1: Evolving skill bank from round {round_num - 1}")
                print(f"  Previous bank: {prev_bank}")

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
                print(f"  {bank}")

        # ── Phase 2: Run agent (skip already-completed seeds) ────────
        seeds = list(range(current_seed, current_seed + seeds_per_round))
        round_eval_dir = output_dir / f"round_{round_num}"
        round_traj_dir = round_eval_dir / level_name

        # Detect seeds that already have trajectory files
        completed_seeds = set()
        if round_traj_dir.exists():
            for f in round_traj_dir.glob("trajectory_seed*_skillrl.json"):
                try:
                    data = json.loads(f.read_text())
                    if "seed" in data:
                        completed_seeds.add(data["seed"])
                except (json.JSONDecodeError, KeyError):
                    pass

        remaining_seeds = [s for s in seeds if s not in completed_seeds]

        if completed_seeds & set(seeds):
            print(f"\n  Phase 2: Skipping already-completed seeds: {sorted(completed_seeds & set(seeds))}")

        if remaining_seeds:
            print(f"  Running agent on seeds {remaining_seeds}"
                  + (" (no skills)" if round_num == 1 else ""))

            new_results = run_batch(
                model_fn=model_fn,
                level_name=level_name,
                seeds=remaining_seeds,
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
        else:
            print(f"\n  Phase 2: All seeds already completed, skipping.")

        # Reload all seed results for full round stats
        results = []
        for s in seeds:
            traj_file = round_traj_dir / f"trajectory_seed{s}_skillrl.json"
            if traj_file.exists():
                try:
                    results.append(json.loads(traj_file.read_text()))
                except json.JSONDecodeError:
                    pass

        # ── Phase 3: Collect results ─────────────────────────────────
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
            "skill_bank": str(skill_bank_path) if skill_bank_path else None,
            "elapsed_seconds": round_elapsed,
            "online": round_num == 1,
        }
        round_stats.append(stats)

        progress_path.write_text(json.dumps(round_stats, indent=2))

        current_seed += seeds_per_round

        print(f"\n  Round {round_num} complete in {round_elapsed:.0f}s")

    # ── Final summary ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ONLINE-EVOLVING LOOP COMPLETE")
    print(f"{'='*70}")
    print(f"\n  Round-by-round results:")
    for s in round_stats:
        tag = " [ONLINE]" if s.get("online") else ""
        print(f"    Round {s['round']}: {s['successes']}/{s['successes']+s['failures']} "
              f"= {s['accuracy']:.0%} ({s['elapsed_seconds']:.0f}s){tag}")

    total_successes = sum(s["successes"] for s in round_stats)
    total_runs = sum(s["successes"] + s["failures"] for s in round_stats)
    print(f"\n  Overall: {total_successes}/{total_runs} = {total_successes/total_runs:.0%}" if total_runs else "")
    print(f"  Output: {output_dir}")
    print(f"  Progress log: {progress_path}")

    return round_stats


def main():
    parser = argparse.ArgumentParser(
        description="Online-start evolving skill refinement loop"
    )
    parser.add_argument("--level", type=str, required=True,
                        help="Puzzle level to train on")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: skillrl/data/online_evolving/{level})")
    parser.add_argument("--num-rounds", type=int, default=5,
                        help="Number of rounds (default: 5)")
    parser.add_argument("--seeds-per-round", type=int, default=3,
                        help="New seeds to run each round (default: 3)")
    parser.add_argument("--start-seed", type=int, default=6,
                        help="First seed number (default: 6)")
    parser.add_argument("--max-skills", type=int, default=MAX_SKILLS_PER_LEVEL,
                        help=f"Max skills per level (default: {MAX_SKILLS_PER_LEVEL})")
    parser.add_argument("--max-mistakes", type=int, default=5,
                        help="Max mistakes per level (default: 5)")
    parser.add_argument("--model", type=str, default="claude",
                        help="Agent model (default: claude)")
    parser.add_argument("--teacher-model", type=str, default=TEACHER_MODEL,
                        help="Teacher model for distillation/evolution")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--skip-general", action="store_true", default=True,
                        help="Skip cross-level generalization (default: True)")
    parser.add_argument("--max-general-skills", type=int, default=0)
    parser.add_argument("--max-specific-skills", type=int, default=6)
    parser.add_argument("--max-mistakes-agent", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(
            Path(__file__).resolve().parent / "data" / "online_evolving" / args.level
        )

    run_online_evolving_loop(
        level_name=args.level,
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
