"""Phase 1: Teacher model reads trajectories and distills them into skills.

Three passes per level (matching the SkillRL paper):
1. Contrastive skill extraction — successes + failures shown together
2. Common mistakes — dedicated pass on failures
3. Cross-level generalization — general physics principles
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from skillrl.core.config import (
    ALL_LEVELS,
    EXISTING_RESULTS,
    LEVEL_DESCRIPTIONS,
    SKILL_BANK_PATH,
    TEACHER_MODEL,
)
from skillrl.core.skill_bank import Skill, Mistake, SkillBank
from skillrl.distillation.teacher_prompts import (
    CONTRASTIVE_DISTILLATION_PROMPT,
    COMMON_MISTAKES_PROMPT,
    CROSS_LEVEL_GENERALIZATION_PROMPT,
    compute_trajectory_reward,
    format_trajectories_block,
)


# ── Teacher model interface ────────────────────────────────────────────

_HF_TEACHER_CACHE: dict[str, object] = {}


def _get_hf_teacher(model: str, max_new_tokens: int = 2048):
    """Lazy-load and cache a HuggingFace teacher (e.g. Qwen) so it is only
    loaded once per process."""
    if model not in _HF_TEACHER_CACHE:
        from react_agent.react_agent import load_qwen_model
        _HF_TEACHER_CACHE[model] = load_qwen_model(
            model, temperature=0.3, max_new_tokens=max_new_tokens,
        )
    return _HF_TEACHER_CACHE[model]


def call_teacher(prompt: str, model: str = TEACHER_MODEL) -> str:
    """Call the teacher model and return the text response.

    Dispatches based on model name:
      - names starting with "claude-"       → Claude CLI subprocess
      - names starting with "vllm:<url>"    → OpenAI-compatible vLLM server
      - anything else                       → local HuggingFace model (Qwen by default)
    """
    if model.startswith("vllm:"):
        endpoint = model[len("vllm:"):]  # e.g. "http://gpu014:8000"
        import requests
        try:
            resp = requests.post(
                f"{endpoint}/v1/chat/completions",
                json={
                    "model": "Qwen/Qwen2.5-7B-Instruct",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 2048,
                },
                timeout=300,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[Teacher Error] vLLM at {endpoint} — {type(e).__name__}: {e}")
            return ""

    if model.startswith("claude-"):
        try:
            result = subprocess.run(
                [
                    "claude", "-p", prompt,
                    "--model", model,
                    "--max-turns", "1",
                    "--no-session-persistence",
                    "--output-format", "json",
                ],
                capture_output=True,
                text=True,
                env=os.environ.copy(),
                timeout=300,
            )
            if result.returncode != 0:
                print(f"[Teacher Error] Exit code {result.returncode}")
                print(f"  stderr: {result.stderr[:500]}")
                print(f"  stdout: {result.stdout[:500]}")
                return ""
            output = json.loads(result.stdout)
            return output.get("result", "").strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[Teacher Error] {type(e).__name__}: {e}")
            return ""

    try:
        generate = _get_hf_teacher(model)
        messages = [{"role": "user", "content": prompt}]
        return generate(messages).strip()
    except Exception as e:
        print(f"[Teacher Error] HF model '{model}' — {type(e).__name__}: {e}")
        return ""


def _parse_json_array(text: str) -> list[dict]:
    """Extract a JSON array from the teacher's response."""
    # Try code block first
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Fall back to first array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    print(f"[Warning] Could not parse JSON array (first 200 chars): {text[:200]}")
    return []


def _parse_json_object(text: str) -> dict:
    """Extract a JSON object from the teacher's response (for evolution)."""
    # Try code block first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Fall back to first object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


# ── Trajectory loading ─────────────────────────────────────────────────

def load_trajectories(
    traj_dir: Path, level_name: Optional[str] = None
) -> tuple[list[dict], list[dict]]:
    """Load trajectory JSON files, split into successes and failures.

    Returns (success_trajectories, failure_trajectories).
    """
    successes, failures = [], []
    if not traj_dir.exists():
        print(f"[Warning] Trajectory directory not found: {traj_dir}")
        return successes, failures

    for f in sorted(traj_dir.glob("trajectory_seed*.json")):
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            print(f"[Warning] Bad JSON in {f}")
            continue
        if data.get("success"):
            successes.append(data)
        else:
            failures.append(data)

    return successes, failures


# ── Pass 1: Contrastive skill extraction ───────────────────────────────

def _compute_initial_confidence(
    source_seeds: list[int],
    all_trajs: list[dict],
    fallback_confidence: float = 0.5,
) -> float:
    """Compute initial skill confidence from the rewards of its specific source trajectories.

    Maps average reward from [-1, 1] to confidence in [0.1, 1.0].
    Falls back to level-wide average if no matching seeds are found.
    """
    if not source_seeds or not all_trajs:
        return fallback_confidence

    # Build seed → trajectory lookup
    seed_to_traj: dict[int, dict] = {}
    for t in all_trajs:
        seed = t.get("seed", -1)
        # Keep the first occurrence (trajectories are already sorted by reward)
        if seed not in seed_to_traj:
            seed_to_traj[seed] = t

    # Collect rewards for the specific seeds this skill was derived from
    matched_rewards = [
        compute_trajectory_reward(seed_to_traj[s])
        for s in source_seeds
        if s in seed_to_traj
    ]

    if not matched_rewards:
        # No matching seeds found — fall back to level-wide average
        all_rewards = [compute_trajectory_reward(t) for t in all_trajs]
        avg_reward = sum(all_rewards) / len(all_rewards)
    else:
        avg_reward = sum(matched_rewards) / len(matched_rewards)

    return max(0.1, min(1.0, (avg_reward + 1.0) / 2.0))


def distill_contrastive(
    level_name: str,
    successes: list[dict],
    failures: list[dict],
    teacher_model: str = TEACHER_MODEL,
    max_success_trajs: int = 5,
    max_failure_trajs: int = 5,
) -> list[Skill]:
    """Send successes + failures together to teacher for contrastive distillation."""
    if not successes and not failures:
        return []

    success_block = format_trajectories_block(successes, max_trajs=max_success_trajs) if successes else "(no successes)"
    failure_block = format_trajectories_block(failures, max_trajs=max_failure_trajs) if failures else "(no failures)"
    level_desc = LEVEL_DESCRIPTIONS.get(level_name, "No description available.")

    prompt = CONTRASTIVE_DISTILLATION_PROMPT.format(
        level_name=level_name,
        level_description=level_desc,
        n_successes=len(successes),
        n_failures=len(failures),
        success_block=success_block,
        failure_block=failure_block,
    )

    # Sort and select the same way format_trajectories_block does, then log
    sorted_succ = sorted(successes, key=lambda t: compute_trajectory_reward(t), reverse=True)[:max_success_trajs]
    sorted_fail = sorted(failures, key=lambda t: compute_trajectory_reward(t), reverse=True)[:max_failure_trajs]

    s_rewards = [f"{compute_trajectory_reward(t):+.2f}" for t in sorted_succ]
    f_rewards = [f"{compute_trajectory_reward(t):+.2f}" for t in sorted_fail]

    print(f"  Contrastive distillation: {len(sorted_succ)}/{len(successes)} successes + {len(sorted_fail)}/{len(failures)} failures sent to teacher")
    if s_rewards:
        print(f"    Success rewards (sent): {s_rewards}")
    if f_rewards:
        print(f"    Failure rewards (sent): {f_rewards}")
    response = call_teacher(prompt, model=teacher_model)
    raw_skills = _parse_json_array(response)

    all_trajs = successes + failures

    skills = []
    for i, raw in enumerate(raw_skills):
        prefix = level_name[:3]
        # Use teacher-reported source_seeds; fall back to first 3+3 if not provided
        skill_seeds = raw.get("source_seeds", [])
        if not skill_seeds:
            skill_seeds = (
                [t.get("seed", -1) for t in successes[:3]]
                + [t.get("seed", -1) for t in failures[:3]]
            )

        confidence = _compute_initial_confidence(skill_seeds, all_trajs)

        skills.append(Skill(
            skill_id=f"{prefix}_sk_{i:03d}",
            title=raw.get("title", ""),
            principle=raw.get("principle", ""),
            when_to_apply=raw.get("when_to_apply", ""),
            example=raw.get("example", ""),
            source_level=level_name,
            source_type="contrastive",
            source_seeds=skill_seeds,
            generation=0,
            confidence=confidence,
        ))
        print(f"    Skill '{raw.get('title', '')}': seeds={skill_seeds} → confidence={confidence:.3f}")
    return skills


# ── Pass 2: Common mistakes + partial skills extraction ──────────────

def distill_mistakes(
    level_name: str,
    failures: list[dict],
    teacher_model: str = TEACHER_MODEL,
    max_failure_trajs: int = 8,
) -> tuple[list[Mistake], list[Skill]]:
    """Dedicated pass on failures to extract mistakes and partial skills.

    Returns (mistakes, partial_skills).
    """
    if not failures:
        return [], []

    failure_block = format_trajectories_block(failures, max_trajs=max_failure_trajs)
    level_desc = LEVEL_DESCRIPTIONS.get(level_name, "No description available.")

    prompt = COMMON_MISTAKES_PROMPT.format(
        level_name=level_name,
        level_description=level_desc,
        failure_block=failure_block,
    )

    print(f"  Extracting mistakes + partial skills from {len(failures)} failures...")
    response = call_teacher(prompt, model=teacher_model)
    parsed = _parse_json_object(response)

    prefix = level_name[:3]

    # Parse mistakes
    raw_mistakes = parsed.get("mistakes", [])
    mistakes = []
    for i, raw in enumerate(raw_mistakes):
        mistakes.append(Mistake(
            mistake_id=f"{prefix}_err_{i:03d}",
            description=raw.get("description", ""),
            why_it_happens=raw.get("why_it_happens", ""),
            how_to_avoid=raw.get("how_to_avoid", ""),
            source_level=level_name,
            source_seeds=[t.get("seed", -1) for t in failures[:5]],
            generation=0,
        ))

    # Parse partial skills from failures (good steps within bad trajectories)
    raw_partial = parsed.get("partial_skills", [])
    partial_skills = []
    for i, raw in enumerate(raw_partial):
        skill_seeds = raw.get("source_seeds", [t.get("seed", -1) for t in failures[:3]])
        # Partial skills from failures get a confidence penalty (capped at 0.5)
        conf = min(0.5, _compute_initial_confidence(skill_seeds, failures, fallback_confidence=0.3))
        partial_skills.append(Skill(
            skill_id=f"{prefix}_fsk_{i:03d}",
            title=raw.get("title", ""),
            principle=raw.get("principle", ""),
            when_to_apply=raw.get("when_to_apply", ""),
            example=raw.get("example", ""),
            source_level=level_name,
            source_type="failure",
            source_seeds=skill_seeds,
            generation=0,
            confidence=conf,
        ))
        print(f"    Partial skill '{raw.get('title', '')}': seeds={skill_seeds} -> confidence={conf:.3f}")

    return mistakes, partial_skills


# ── Pass 3: Cross-level generalization ─────────────────────────────────

def distill_general_skills(
    all_level_skills: dict[str, list[Skill]], teacher_model: str = TEACHER_MODEL
) -> list[Skill]:
    """Extract general physics principles from all level-specific skills."""
    sections = []
    for lvl, skills in all_level_skills.items():
        lines = [f"\n### {lvl}"]
        desc = LEVEL_DESCRIPTIONS.get(lvl, "")
        if desc:
            lines.append(f"Level description: {desc}")
        for s in skills:
            entry = f"- [{s.title}] {s.principle} (Apply when: {s.when_to_apply})"
            if s.example:
                entry += f" Example: {s.example}"
            lines.append(entry)
        sections.append("\n".join(lines))

    prompt = CROSS_LEVEL_GENERALIZATION_PROMPT.format(
        all_level_skills="\n".join(sections)
    )

    print("  Distilling general cross-level skills...")
    response = call_teacher(prompt, model=teacher_model)
    raw_skills = _parse_json_array(response)

    skills = []
    for i, raw in enumerate(raw_skills):
        skills.append(Skill(
            skill_id=f"general_{i:03d}",
            title=raw.get("title", ""),
            principle=raw.get("principle", ""),
            when_to_apply=raw.get("when_to_apply", ""),
            example=raw.get("example", ""),
            source_level="general",
            source_type="contrastive",
            generation=0,
        ))
    return skills


# ── Full pipeline ──────────────────────────────────────────────────────

def run_distillation(
    traj_dirs: Optional[dict[str, Path]] = None,
    output_path: Optional[Path] = None,
    levels: Optional[list[str]] = None,
    teacher_model: str = TEACHER_MODEL,
    skip_general: bool = False,
    max_success_trajs: int = 5,
    max_failure_trajs: int = 5,
) -> SkillBank:
    """Full Phase 1 pipeline.

    For each level:
      1. Contrastive distillation (successes + failures together → skills)
      2. Mistakes extraction (failures → mistake patterns)
    Then optionally:
      3. Cross-level generalization (all level skills → general skills)
    """
    if traj_dirs is None:
        traj_dirs = EXISTING_RESULTS
    if output_path is None:
        output_path = SKILL_BANK_PATH
    if levels is None:
        levels = ALL_LEVELS

    bank = SkillBank()
    all_level_skills: dict[str, list[Skill]] = {}

    for level_name in levels:
        traj_dir = traj_dirs.get(level_name)
        if traj_dir is None:
            print(f"  [Skip] No trajectory directory for {level_name}")
            continue

        successes, failures = load_trajectories(Path(traj_dir))
        print(f"\n{'='*50}")
        print(f"  {level_name}: {len(successes)} successes, {len(failures)} failures")

        # Pass 1: Contrastive skills
        skills = distill_contrastive(
            level_name, successes, failures, teacher_model,
            max_success_trajs=max_success_trajs,
            max_failure_trajs=max_failure_trajs,
        )
        for skill in skills:
            bank.add_skill(skill)
        all_level_skills[level_name] = skills

        # Pass 2: Common mistakes + partial skills from failures
        mistakes, partial_skills = distill_mistakes(
            level_name, failures, teacher_model,
            max_failure_trajs=max_failure_trajs,
        )
        for mistake in mistakes:
            bank.add_mistake(mistake)
        for skill in partial_skills:
            bank.add_skill(skill)
        all_level_skills[level_name].extend(partial_skills)

        print(f"  → {len(skills)} contrastive + {len(partial_skills)} failure-derived skills + {len(mistakes)} mistakes")

    # Pass 3: Cross-level generalization
    if all_level_skills and not skip_general:
        general = distill_general_skills(all_level_skills, teacher_model)
        for skill in general:
            bank.add_skill(skill)
        print(f"\n  → {len(general)} general cross-level skills")

    bank.save(output_path)
    print(f"\nSkill bank saved to {output_path}")
    print(f"  {bank}")
    return bank


# ── CLI entry point ────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1: Skill distillation from trajectories")
    parser.add_argument("--levels", nargs="+", default=ALL_LEVELS,
                        help="Levels to process (default: all)")
    parser.add_argument("--traj-dir", type=str, default=None,
                        help="Path to trajectory directory (overrides default for the given level)")
    parser.add_argument("--output", type=str, default=str(SKILL_BANK_PATH),
                        help="Output path for skill bank JSON")
    parser.add_argument("--teacher-model", type=str, default=TEACHER_MODEL,
                        help="Teacher model to use")
    parser.add_argument("--no-general", action="store_true",
                        help="Skip cross-level generalization (useful for single-level runs)")
    parser.add_argument("--max-success-trajs", type=int, default=5,
                        help="Max success trajectories to show the teacher (default: 5)")
    parser.add_argument("--max-failure-trajs", type=int, default=5,
                        help="Max failure trajectories to show the teacher (default: 5)")
    args = parser.parse_args()

    # If --traj-dir is given, override the default trajectory dirs for the specified levels
    traj_dirs = None
    if args.traj_dir:
        traj_dirs = {lvl: Path(args.traj_dir) for lvl in args.levels}

    run_distillation(
        traj_dirs=traj_dirs,
        levels=args.levels,
        output_path=Path(args.output),
        teacher_model=args.teacher_model,
        skip_general=args.no_general,
        max_success_trajs=args.max_success_trajs,
        max_failure_trajs=args.max_failure_trajs,
    )


if __name__ == "__main__":
    main()
