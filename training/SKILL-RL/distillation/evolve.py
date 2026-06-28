"""Phase 3: Recursive skill evolution — evaluate, find weak levels, generate new skills."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from skillrl.core.config import (
    ALL_LEVELS,
    ACCURACY_THRESHOLD,
    MAX_EVOLUTION_ROUNDS,
    SKILL_BANK_PATH,
    EVAL_RESULTS_DIR,
    EVOLUTION_LOG_DIR,
    TEACHER_MODEL,
)
from skillrl.core.config import LEVEL_DESCRIPTIONS
from skillrl.core.skill_bank import Skill, Mistake, SkillBank
from skillrl.runner.augmented_runner import run_skill_augmented
from skillrl.distillation.distill import call_teacher, _parse_json_object
from skillrl.distillation.teacher_prompts import EVOLUTION_PROMPT, format_trajectories_block


# ── Evaluation ─────────────────────────────────────────────────────────

def evaluate_batch(
    model_fn,
    skill_bank: SkillBank,
    levels: list[str],
    seeds: list[int],
    **kwargs,
) -> tuple[dict[str, float], dict[str, list[dict]]]:
    """Run skill-augmented agent on levels × seeds.

    Returns
    -------
    accuracies : {level_name: success_rate}
    all_results : {level_name: [result_dicts]}
    """
    accuracies = {}
    all_results = {}

    for level in levels:
        results = []
        for seed in seeds:
            result = run_skill_augmented(
                model_fn=model_fn,
                level_name=level,
                seed=seed,
                skill_bank=skill_bank,
                **kwargs,
            )
            results.append(result)
            status = "OK" if result.get("success") else "FAIL"
            print(f"  [{level}][seed={seed}] {status} ({result.get('iterations', '?')} iters)")

        successes = sum(1 for r in results if r.get("success"))
        acc = successes / len(seeds) if seeds else 0.0
        accuracies[level] = acc
        all_results[level] = results
        print(f"  {level}: {successes}/{len(seeds)} = {acc:.1%}")

    return accuracies, all_results


def identify_weak_levels(
    accuracies: dict[str, float],
    threshold: float = ACCURACY_THRESHOLD,
) -> list[str]:
    """Return levels below the accuracy threshold."""
    return [lvl for lvl, acc in accuracies.items() if acc < threshold]


def collect_failure_traces(results: list[dict]) -> list[dict]:
    """Filter results to get only failure trajectories."""
    return [r for r in results if not r.get("success")]


# ── Skill evolution ────────────────────────────────────────────────────

def evolve_skills_for_level(
    skill_bank: SkillBank,
    level_name: str,
    failure_traces: list[dict],
    accuracy: float,
    teacher_model: str = TEACHER_MODEL,
    generation: int = 1,
) -> tuple[list[Skill], list[Mistake]]:
    """Call teacher to generate new skills + mistakes targeting observed failure modes.

    Returns (new_skills, new_mistakes).
    """
    if not failure_traces:
        return [], []

    # Format failure trajectories
    traj_block = format_trajectories_block(failure_traces, max_trajs=5)

    # Format existing skills
    existing_skills = skill_bank.level_skills.get(level_name, [])
    existing_skills_block = "\n".join(
        f"- [{s.title}] {s.principle}" for s in existing_skills
    ) if existing_skills else "(no existing skills)"

    # Format existing mistakes
    existing_mistakes = skill_bank.level_mistakes.get(level_name, [])
    existing_mistakes_block = "\n".join(
        f"- {m.description} → {m.how_to_avoid}" for m in existing_mistakes
    ) if existing_mistakes else "(no existing mistakes)"

    level_desc = LEVEL_DESCRIPTIONS.get(level_name, "")

    prompt = EVOLUTION_PROMPT.format(
        level_name=level_name,
        level_description=level_desc,
        accuracy=accuracy,
        failure_trajectories=traj_block,
        existing_skills=existing_skills_block,
        existing_mistakes=existing_mistakes_block,
    )

    print(f"  Evolving skills for {level_name} (accuracy={accuracy:.0%})...")
    response = call_teacher(prompt, model=teacher_model)
    parsed = _parse_json_object(response)

    # Parse new skills
    new_skills = []
    base_sk = len(existing_skills)
    for i, raw in enumerate(parsed.get("skills", [])):
        new_skills.append(Skill(
            skill_id=f"{level_name[:3]}_evo{generation}_{base_sk + i:03d}",
            title=raw.get("title", ""),
            principle=raw.get("principle", ""),
            when_to_apply=raw.get("when_to_apply", ""),
            example=raw.get("example", ""),
            source_level=level_name,
            source_type="contrastive",
            source_seeds=[t.get("seed", -1) for t in failure_traces[:5]],
            generation=generation,
            confidence=0.5,
        ))

    # Parse new mistakes
    new_mistakes = []
    base_mk = len(existing_mistakes)
    for i, raw in enumerate(parsed.get("mistakes", [])):
        new_mistakes.append(Mistake(
            mistake_id=f"{level_name[:3]}_evo{generation}_err_{base_mk + i:03d}",
            description=raw.get("description", ""),
            why_it_happens=raw.get("why_it_happens", ""),
            how_to_avoid=raw.get("how_to_avoid", ""),
            source_level=level_name,
            source_seeds=[t.get("seed", -1) for t in failure_traces[:5]],
            generation=generation,
        ))

    return new_skills, new_mistakes


# ── Full evolution loop ────────────────────────────────────────────────

def run_evolution_loop(
    model_fn,
    skill_bank_path: Optional[Path] = None,
    levels: Optional[list[str]] = None,
    seeds: Optional[list[int]] = None,
    max_rounds: int = MAX_EVOLUTION_ROUNDS,
    threshold: float = ACCURACY_THRESHOLD,
    teacher_model: str = TEACHER_MODEL,
    log_dir: Optional[Path] = None,
    **agent_kwargs,
) -> SkillBank:
    """Full Phase 3: evaluate → find weak levels → evolve skills → repeat.

    Parameters
    ----------
    model_fn : Callable for the student agent.
    skill_bank_path : Path to load/save the skill bank.
    levels : Which levels to evaluate.
    seeds : Seeds to use for evaluation.
    max_rounds : Maximum evolution rounds.
    threshold : Accuracy threshold below which evolution triggers.
    teacher_model : Model for skill generation.
    log_dir : Directory for evolution logs.
    **agent_kwargs : Passed to run_skill_augmented.
    """
    if skill_bank_path is None:
        skill_bank_path = SKILL_BANK_PATH
    if levels is None:
        levels = ALL_LEVELS
    if seeds is None:
        seeds = list(range(5))
    if log_dir is None:
        log_dir = EVOLUTION_LOG_DIR

    skill_bank_path = Path(skill_bank_path)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    skill_bank = SkillBank.load(skill_bank_path)
    print(f"Loaded {skill_bank}")

    for round_num in range(1, max_rounds + 1):
        print(f"\n{'='*60}")
        print(f"  EVOLUTION ROUND {round_num}/{max_rounds}")
        print(f"{'='*60}")

        # 1. Evaluate all levels
        accuracies, all_results = evaluate_batch(
            model_fn, skill_bank, levels, seeds, **agent_kwargs
        )

        # Log round results
        round_log = {
            "round": round_num,
            "accuracies": accuracies,
            "skill_count": skill_bank.skill_count(),
        }

        # 2. Identify weak levels
        weak = identify_weak_levels(accuracies, threshold)
        round_log["weak_levels"] = weak
        print(f"\n  Weak levels (below {threshold:.0%}): {weak if weak else 'NONE — converged!'}")

        if not weak:
            round_log["status"] = "converged"
            log_path = log_dir / f"round_{round_num}.json"
            log_path.write_text(json.dumps(round_log, indent=2, default=str))
            print("  All levels above threshold. Evolution complete.")
            break

        # 3. Evolve skills + mistakes for weak levels
        total_new_skills = 0
        total_new_mistakes = 0
        for level in weak:
            failures = collect_failure_traces(all_results[level])
            new_skills, new_mistakes = evolve_skills_for_level(
                skill_bank, level, failures, accuracies[level],
                teacher_model=teacher_model,
                generation=round_num,
            )
            for skill in new_skills:
                skill_bank.add_skill(skill)
            for mistake in new_mistakes:
                skill_bank.add_mistake(mistake)
            total_new_skills += len(new_skills)
            total_new_mistakes += len(new_mistakes)
            print(f"  +{len(new_skills)} skills, +{len(new_mistakes)} mistakes for {level}")

        round_log["new_skills_added"] = total_new_skills
        round_log["new_mistakes_added"] = total_new_mistakes
        round_log["status"] = "evolved"

        # Save updated bank and log
        skill_bank.save(skill_bank_path)
        log_path = log_dir / f"round_{round_num}.json"
        log_path.write_text(json.dumps(round_log, indent=2, default=str))

        print(f"\n  Round {round_num}: +{total_new_skills} skills, +{total_new_mistakes} mistakes → {skill_bank}")

    return skill_bank


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    parser = argparse.ArgumentParser(description="Phase 3: Recursive skill evolution")
    parser.add_argument("--skill-bank", type=str, default=str(SKILL_BANK_PATH))
    parser.add_argument("--levels", nargs="+", default=ALL_LEVELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--rounds", type=int, default=MAX_EVOLUTION_ROUNDS)
    parser.add_argument("--threshold", type=float, default=ACCURACY_THRESHOLD)
    parser.add_argument("--model", type=str, default="claude",
                        help="Student agent model")
    parser.add_argument("--teacher-model", type=str, default=TEACHER_MODEL)
    parser.add_argument("--max-iterations", type=int, default=25)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Load student model
    from skillrl.runner.run_skill_agent import _load_model
    model_fn = _load_model(args.model)

    run_evolution_loop(
        model_fn=model_fn,
        skill_bank_path=Path(args.skill_bank),
        levels=args.levels,
        seeds=args.seeds,
        max_rounds=args.rounds,
        threshold=args.threshold,
        teacher_model=args.teacher_model,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
