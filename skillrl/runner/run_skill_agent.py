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

    for seed in seeds:
        traj_path = level_dir / f"trajectory_seed{seed}_skillrl.json"

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
            }
            for r in results
        ],
    }
    summary_path = level_dir / "summary_skillrl.json"
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
    )


if __name__ == "__main__":
    main()
