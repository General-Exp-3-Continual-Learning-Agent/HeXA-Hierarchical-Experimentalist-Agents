"""Offline-to-online evolving loop for the *contrastive-only* SkillRL variant.

Mirrors evolving_loop.py but uses the contrastive-only distillation/evolution
modules under the hood. There are NO trajectory rewards and NO per-skill
confidence values — skills are formed purely by contrasting successful and
failed trajectories.

Usage:
    python -m skillrl.loops.contrastive_only_loop \\
        --level pass_the_parcel \\
        --initial-traj-dir results/pass_the_parcel \\
        --num-rounds 3 \\
        --seeds-per-round 5 \\
        --start-seed 6

Round 1: Distill skill bank from initial (offline) trajectories using
         contrastive_only_distill.run_distillation_contrastive_only.
Round 2+: Evolve the skill bank from the previous round + new trajectories
          using contrastive_only_evolve.evolve_skill_bank_contrastive_only.
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
from skillrl.distillation.contrastive_only_distill import (
    run_distillation_contrastive_only,
)
from skillrl.distillation.contrastive_only_evolve import (
    evolve_skill_bank_contrastive_only,
)
from skillrl.loops.evolving_loop import (
    collect_trajectories,
    generate_new_seeds,
    get_already_run_seeds,
)
from skillrl.runner.run_skill_agent import run_batch, _load_model


def run_contrastive_only_loop(
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
    max_general_skills: int = 0,
    max_specific_skills: int = 6,
    max_mistakes_agent: int = 4,
    verbose: bool = False,
):
    """Run the offline-to-online contrastive-only refinement loop."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if level_name == "catapult":
        from skillrl.distillation import teacher_prompts_contrastive_only as _tpco
        _tpco.patch()
        print(
            "[Prompts] Patched contrastive-only distill + evolve with the "
            "catapult-specific factual-block variants (no strategy hints leaked)."
        )

    progress_path = output_dir / "progress_contrastive_only.json"
    resume_from_round = 0
    round_stats: list[dict] = []

    if progress_path.exists():
        try:
            round_stats = json.loads(progress_path.read_text())
            if round_stats:
                last = round_stats[-1]
                resume_from_round = last["round"]
                # Advance start_seed past all seeds in completed rounds.
                # Mid-round gaps (e.g. seed 20 missing while 19 and 21 ran)
                # are filled by generate_new_seeds — which scans forward and
                # skips anything in `already_run` — so do NOT bump start_seed
                # past max(mid_round_seeds), or gaps will be jumped over.
                start_seed = max(s for s in last["seeds"]) + 1

                print(f"\n  RESUMING from round {resume_from_round + 1}")
                print(
                    f"  Previous rounds: {len(round_stats)}, "
                    f"next start_seed: {start_seed}"
                )
        except (json.JSONDecodeError, KeyError):
            round_stats = []

    print(f"\n{'=' * 70}")
    print(f"  CONTRASTIVE-ONLY LOOP: {level_name}")
    print(
        f"  Rounds: {resume_from_round + 1}-{num_rounds}, "
        f"Seeds/round: {seeds_per_round}, Start seed: {start_seed}"
    )
    print(f"  No rewards, no confidence — pure contrastive distillation")
    print(f"  Max skills/level: {max_skills}, Max mistakes/level: {max_mistakes}")
    print(f"{'=' * 70}")

    init_successes, init_failures = collect_trajectories(initial_traj_dir)
    print(
        f"\nInitial trajectories: "
        f"{len(init_successes)} successes, {len(init_failures)} failures"
    )

    model_fn = _load_model(model_name)
    is_oss = (
        "gpt-oss" in model_name.lower()
        if isinstance(model_name, str) else False
    )

    current_seed = start_seed

    for round_num in range(resume_from_round + 1, num_rounds + 1):
        round_start = time.perf_counter()
        print(f"\n{'#' * 70}")
        print(f"  ROUND {round_num}/{num_rounds}")
        print(f"{'#' * 70}")

        skill_bank_path = output_dir / f"skill_bank_contrastive_only_{round_num}.json"

        # ── Phase 1: build / evolve skill bank ──────────────────────────
        if round_num == 1:
            if skill_bank_path.exists():
                print(f"\n  Phase 1: Skill bank already exists, skipping distillation")
                bank = SkillBank.load(skill_bank_path)
            else:
                print(f"\n  Phase 1: Seeding skill bank from initial trajectories")
                traj_dirs = {level_name: initial_traj_dir}
                bank = run_distillation_contrastive_only(
                    traj_dirs=traj_dirs,
                    levels=[level_name],
                    output_path=skill_bank_path,
                    teacher_model=teacher_model,
                )
                print(f"\n  Skill bank seeded: {skill_bank_path}")
        else:
            if skill_bank_path.exists():
                print(f"\n  Phase 1: Skill bank already exists, skipping evolution")
                bank = SkillBank.load(skill_bank_path)
            else:
                prev_path = (
                    output_dir / f"skill_bank_contrastive_only_{round_num - 1}.json"
                )
                if not prev_path.exists():
                    raise FileNotFoundError(
                        f"Cannot resume: previous bank not found: {prev_path}"
                    )
                prev_bank = SkillBank.load(prev_path)
                print(f"\n  Phase 1: Evolving skill bank from round {round_num - 1}")
                print(f"  Previous bank: {prev_bank}")

                prev_round_dir = output_dir / f"round_{round_num - 1}" / level_name
                if not prev_round_dir.exists():
                    raise FileNotFoundError(
                        f"Cannot resume: previous round trajectories not found: "
                        f"{prev_round_dir}"
                    )

                bank = evolve_skill_bank_contrastive_only(
                    level_name=level_name,
                    prev_bank=prev_bank,
                    new_trajs_dir=prev_round_dir,
                    output_path=skill_bank_path,
                    max_skills=max_skills,
                    max_mistakes=max_mistakes,
                    teacher_model=teacher_model,
                )

        # ── Phase 2: run skill-augmented agent ──────────────────────────
        round_eval_dir = output_dir / f"round_{round_num}"
        already_run = get_already_run_seeds(round_eval_dir, level_name)
        if already_run:
            print(f"\n  Phase 2: Detected already-run seeds: {sorted(already_run)}")

        if len(already_run) >= seeds_per_round:
            seeds = sorted(list(already_run))[:seeds_per_round]
            print(
                f"  Phase 2: Round {round_num} batch {seeds} already complete. "
                f"Loading existing results."
            )
            results = []
            for seed in seeds:
                traj_path = round_eval_dir / level_name / f"trajectory_seed{seed}_skillrl.json"
                if traj_path.exists():
                    try:
                        results.append(json.loads(traj_path.read_text()))
                    except json.JSONDecodeError:
                        print(f"  [Warning] Could not parse {traj_path}, skipping")
        else:
            missing_count = seeds_per_round - len(already_run)
            new_seeds = generate_new_seeds(current_seed, missing_count, already_run)
            if already_run:
                print(
                    f"  Phase 2: Round partially complete, "
                    f"running missing seeds: {new_seeds}"
                )
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

        # ── Phase 3: collect results ────────────────────────────────────
        round_successes = [r for r in results if r.get("success")]
        round_failures = [r for r in results if not r.get("success")]
        round_accuracy = (
            len(round_successes) / len(results) if results else 0.0
        )
        round_elapsed = time.perf_counter() - round_start

        print(
            f"\n  Round {round_num} results: "
            f"{len(round_successes)}/{len(results)} = {round_accuracy:.0%}"
        )

        round_stats.append({
            "round": round_num,
            "seeds": seeds,
            "successes": len(round_successes),
            "failures": len(round_failures),
            "accuracy": round_accuracy,
            "skill_bank": str(skill_bank_path),
            "elapsed_seconds": round_elapsed,
        })
        progress_path.write_text(json.dumps(round_stats, indent=2))

        current_seed = max(seeds) + 1 if seeds else current_seed + seeds_per_round
        print(f"\n  Round {round_num} complete in {round_elapsed:.0f}s")

    # ── Final summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  CONTRASTIVE-ONLY LOOP COMPLETE")
    print(f"{'=' * 70}")
    print(f"\n  Round-by-round results:")
    for s in round_stats:
        print(
            f"    Round {s['round']}: "
            f"{s['successes']}/{s['successes'] + s['failures']} "
            f"= {s['accuracy']:.0%} ({s['elapsed_seconds']:.0f}s)"
        )

    total_successes = sum(s["successes"] for s in round_stats)
    total_runs = sum(s["successes"] + s["failures"] for s in round_stats)
    if total_runs:
        print(
            f"\n  Overall: {total_successes}/{total_runs} "
            f"= {total_successes / total_runs:.0%}"
        )
    print(f"  Skill banks saved in: {output_dir}")
    print(f"  Progress log: {progress_path}")

    return round_stats


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Offline-to-online contrastive-only skill refinement loop "
            "(no rewards, no confidence)"
        )
    )
    parser.add_argument("--level", type=str, required=True,
                        help="Puzzle level to train on")
    parser.add_argument("--initial-traj-dir", type=str, required=True,
                        help="Directory with initial seed trajectories")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory "
                             "(default: skillrl/data/contrastive_only/{level})")
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--seeds-per-round", type=int, default=5)
    parser.add_argument("--start-seed", type=int, default=6)
    parser.add_argument("--max-skills", type=int, default=MAX_SKILLS_PER_LEVEL)
    parser.add_argument("--max-mistakes", type=int, default=5)
    parser.add_argument("--model", type=str, default="claude")
    parser.add_argument("--teacher-model", type=str, default=TEACHER_MODEL)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-general-skills", type=int, default=0)
    parser.add_argument("--max-specific-skills", type=int, default=5)
    parser.add_argument("--max-mistakes-agent", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(
            Path(__file__).resolve().parent
            / "data" / "contrastive_only" / args.level
        )

    run_contrastive_only_loop(
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
        max_general_skills=args.max_general_skills,
        max_specific_skills=args.max_specific_skills,
        max_mistakes_agent=args.max_mistakes_agent,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
