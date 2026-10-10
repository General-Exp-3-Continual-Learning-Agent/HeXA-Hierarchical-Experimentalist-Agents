"""Phase 2 CLI: Run the skill-augmented ReAct agent on one or more seeds."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skillrl.core.config import (
    ALL_LEVELS,
    SKILL_BANK_PATH,
    EVAL_RESULTS_DIR,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_NEW_TOKENS,
    TEACHER_MODEL,
)
from skillrl.core.skill_bank import SkillBank
from skillrl.runner.augmented_runner import run_skill_augmented


def _load_model(model_name: str):
    """Load a model function by name, reusing existing loaders."""
    if model_name == "claude":
        from react_agent.run_react_claude import load_claude_cli_model
        return load_claude_cli_model()
    elif model_name == "mock":
        from react_agent.react_agent import MockModel
        return MockModel()
    elif model_name.startswith("openai:"):
        from react_agent.react_agent import load_openai_compatible_model
        actual_name = model_name.split(":", 1)[1]
        return load_openai_compatible_model(actual_name)
    else:
        from react_agent.react_agent import load_qwen_model
        return load_qwen_model(model_name)


def run_batch(
    model_fn,
    level_name: str,
    seeds: list[int],
    skill_bank: SkillBank,
    eval_dir: Path,
    **kwargs,
) -> list[dict]:
    """Run skill-augmented agent across multiple seeds and save results."""
    results = []
    level_dir = eval_dir / level_name
    level_dir.mkdir(parents=True, exist_ok=True)
    # Separate filenames so a within-seed run never resumes from (or is
    # mistaken for) static-injection results in the same eval dir.
    suffix = "skillrl_ws" if kwargs.get("within_seed") else "skillrl"

    for seed in seeds:
        traj_path = level_dir / f"trajectory_seed{seed}_{suffix}.json"

        # Skip seeds that already have a completed trajectory file
        if traj_path.exists():
            try:
                result = json.loads(traj_path.read_text())
                results.append(result)
                print(f"\n  Seed {seed}: SKIPPING (trajectory exists — "
                      f"{'SUCCESS' if result.get('success') else 'FAILURE'})")
                continue
            except (json.JSONDecodeError, OSError):
                pass  # corrupted file — re-run

        print(f"\n{'#'*60}")
        print(f"  Level: {level_name}  Seed: {seed}")
        print(f"{'#'*60}")

        result = run_skill_augmented(
            model_fn=model_fn,
            level_name=level_name,
            seed=seed,
            skill_bank=skill_bank,
            **kwargs,
        )
        results.append(result)

        traj_path.parent.mkdir(parents=True, exist_ok=True)
        traj_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"  → {'SUCCESS' if result.get('success') else 'FAILURE'} "
              f"in {result.get('iterations', '?')} iterations "
              f"({result.get('elapsed_time', 0):.1f}s)")

        # Free CUDA cache between seeds — prevents fragmentation buildup that
        # caused OOM mid-eval (~seed 67, iteration 12) on gpt-oss-120b runs.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # Save summary
    successes = sum(1 for r in results if r.get("success"))
    summary = {
        "level": level_name,
        "model": kwargs.get("model_name", "unknown"),
        "seeds": seeds,
        "total_seeds": len(seeds),
        "successes": successes,
        "success_rate": successes / len(seeds) if seeds else 0,
        "avg_iterations": sum(r.get("iterations", 0) for r in results) / len(results) if results else 0,
        "results": [
            {
                "seed": r.get("seed"),
                "success": r.get("success"),
                "action": r.get("action"),
                "iterations": r.get("iterations"),
                "elapsed_time": r.get("elapsed_time"),
                "skills_used": r.get("skill_titles", []),
                "within_seed_evolver_calls": r.get("within_seed_evolver_calls"),
            }
            for r in results
        ],
    }
    if kwargs.get("within_seed"):
        summary["inference_mode"] = "within_seed_free" if kwargs.get("ws_free") else "within_seed"
        summary["refine_on"] = kwargs.get("refine_on")
        summary["teacher_model"] = kwargs.get("teacher_model")
    summary_path = level_dir / f"summary_{suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print(f"  {level_name}: {successes}/{len(seeds)} = {summary['success_rate']:.1%}")
    print(f"  Results saved to {level_dir}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run skill-augmented ReAct agent")
    parser.add_argument("--level", type=str, required=True, choices=ALL_LEVELS,
                        help="Puzzle level to solve")
    parser.add_argument("--seeds", nargs="+", type=int, required=True,
                        help="Seeds to run")
    parser.add_argument("--skill-bank", type=str, default=str(SKILL_BANK_PATH),
                        help="Path to skill_bank.json")
    parser.add_argument("--model", type=str, default="claude",
                        help="Model: claude, mock, openai:<name>, or HF model name")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--eval-dir", type=str, default=str(EVAL_RESULTS_DIR))
    parser.add_argument("--ablation-level", type=int, default=None, choices=range(7),
                        help="Ablation level 0-6 for base prompt (6 = no hints, skills only)")
    parser.add_argument("--max-general-skills", type=int, default=0,
                        help="Max general skills to inject (default: 0)")
    parser.add_argument("--max-specific-skills", type=int, default=6,
                        help="Max level-specific skills to inject (default: 6)")
    parser.add_argument("--max-mistakes", type=int, default=4,
                        help="Max mistakes to inject (default: 4)")
    parser.add_argument("--xl-framing", action="store_true",
                        help="Label level skills as cross-level transferred "
                             "(use when --skill-bank is a cross-level synthesised bank).")
    parser.add_argument("--within-seed", action="store_true",
                        help="Within-seed experimentation: refine a seed-local copy of the "
                             "bank after each turn (default: static injection).")
    parser.add_argument("--teacher-model", type=str, default=TEACHER_MODEL,
                        help="Evolver for --within-seed refinement")
    parser.add_argument("--refine-on", choices=["all", "sim", "event"], default="all",
                        help="When to call the within-seed evolver: every tool turn (all), "
                             "after simulation turns (sim), or when code detects a new valid "
                             "simulation result -- new best, new green behaviour, 2nd failure "
                             "in a region, or every 3rd new simulation (event)")
    parser.add_argument("--ws-announce", action="store_true",
                        help="With --ws-free: tell the actor (system prompt) that skills are "
                             "updated during the episode, tag seed skills '(this episode)', and "
                             "append each evolver update to the tool result")
    parser.add_argument("--ws-require-numbers", action="store_true",
                        help="With --ws-free: during an invalid streak, re-ask the evolver once "
                             "if no skill quotes a range from the computed valid-placement lines")
    parser.add_argument("--ws-regions", action="store_true",
                        help="With --ws-free: the evolver may rule out placement regions "
                             "(>=2 cited failed simulations inside, <=2.0 wide); the tools "
                             "refuse simulations inside them")
    parser.add_argument("--ws-max-new-skills", type=int, default=2,
                        help="Seed-specific skills allowed on top of the bank during --within-seed")
    parser.add_argument("--ws-max-new-mistakes", type=int, default=1,
                        help="Seed-specific mistakes allowed on top of the bank during --within-seed")
    parser.add_argument("--no-evolver", action="store_true",
                        help="With --within-seed: notes-only ablation (episode notes "
                             "recomputed each turn, evolver never called, bank stays K*)")
    parser.add_argument("--ws-free", action="store_true",
                        help="With --within-seed: unguarded evolver -- may rewrite, re-weight "
                             "or delete any entry (incl. K*) and add seed-specific skills/mistakes up to "
                             "--ws-max-new-skills/--ws-max-new-mistakes; the actor sees the refined "
                             "bank each turn under the same --max-specific-skills/--max-mistakes caps")
    parser.add_argument("--prediction-cap", type=int, default=None,
                        help="Max predict_first_contact calls between simulations "
                             "(default: unlimited)")
    parser.add_argument("--explore", action="store_true",
                        help="Enforce test-time exploration: radius-band coverage over "
                             "the first simulations + stall breaker (ExploreToolkit)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Load skill bank
    bank_path = Path(args.skill_bank)
    if not bank_path.exists():
        print(f"[Error] Skill bank not found at {bank_path}. Run distillation first.")
        sys.exit(1)
    skill_bank = SkillBank.load(bank_path)
    print(f"Loaded {skill_bank}")

    # Load model
    model_fn = _load_model(args.model)
    is_oss = "gpt-oss" in args.model.lower() if isinstance(args.model, str) else False

    run_batch(
        model_fn=model_fn,
        level_name=args.level,
        seeds=args.seeds,
        skill_bank=skill_bank,
        eval_dir=Path(args.eval_dir),
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        is_oss=is_oss,
        ablation_level=args.ablation_level,
        max_general_skills=args.max_general_skills,
        max_specific_skills=args.max_specific_skills,
        max_mistakes=args.max_mistakes,
        xl_framing=args.xl_framing,
        model_label=args.model,
        within_seed=args.within_seed,
        teacher_model=args.teacher_model,
        refine_on=args.refine_on,
        ws_max_new_skills=args.ws_max_new_skills,
        ws_max_new_mistakes=args.ws_max_new_mistakes,
        ws_evolver=not args.no_evolver,
        ws_free=args.ws_free,
        ws_regions=args.ws_regions,
        ws_announce=args.ws_announce,
        ws_require_numbers=args.ws_require_numbers,
        prediction_cap=args.prediction_cap,
        explore=args.explore,
    )


if __name__ == "__main__":
    main()
