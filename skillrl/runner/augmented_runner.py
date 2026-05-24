"""Phase 2: Wrap ReactAgent with skill-injected prompts.

Uses the same monkey-patching pattern as experiments/experiment_runner.py
(lines 126-180) to inject skills into the system prompt without modifying
any existing files.

Supports ablation levels (0-6) from experiments/prompt_builder.py to control
how much built-in hint content the base prompt contains.  The key use-case is
ablation_level=6 ("no hints") so that the agent relies entirely on learned
skills rather than hand-written guidance.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

# Import the *module* so we can monkey-patch its top-level names
import react_agent.react_agent as agent_module
from react_agent.react_agent import ReactAgent
from react_agent.tools import InterphyreToolkit
from react_agent.level_prompts import build_system_prompt, build_initial_user_message

# Ablation support — imported lazily to avoid hard dependency when not used
try:
    from experiments.prompt_builder import (
        build_system_prompt_ablation,
        build_initial_user_message_ablation,
    )
    from experiments.feedback_reducer import AblationToolkitWrapper
    _HAS_ABLATION = True
except ImportError:
    _HAS_ABLATION = False

from skillrl.core.skill_bank import Skill, SkillBank
from skillrl.core.retriever import retrieve_skills


DIAGNOSTIC_ADDENDUM = """
IMPORTANT — After each FAILED simulation, before trying a new placement, you MUST:
1. Review the LEARNED PHYSICS SKILLS listed in the system prompt above.
2. Identify which skill applies to the failure you just observed.
3. Explicitly state which skill you are applying and why in your next Thought.
"""


def format_worked_examples(trajectories: list[dict], max_examples: int = 1) -> str:
    """Format successful trajectories as worked examples for cold-start.

    Used when the skill bank is sparse to give the agent an in-context
    demonstration of effective reasoning.
    """
    if not trajectories:
        return ""

    lines = ["\n=== WORKED EXAMPLE ==="]
    for traj in trajectories[:max_examples]:
        lines.append(f"\n[Seed {traj.get('seed', '?')} — SUCCESS in {traj.get('iterations', '?')} steps]")
        steps = traj.get("trajectory", [])
        # Show only the last few decisive steps
        for step in steps[-3:]:
            thought = step.get("thought", "")
            action = step.get("action", "")
            obs = step.get("observation", "")
            if len(obs) > 300:
                obs = obs[:300] + "..."
            lines.append(f"  Thought: {thought}")
            lines.append(f"  Action: {action}")
            lines.append(f"  Observation: {obs}")
            lines.append("")
        final = traj.get("action")
        if final:
            lines.append(f"  Final: x={final[0]}, y={final[1]}, r={final[2]}")
    lines.append("=== END WORKED EXAMPLE ===\n")
    return "\n".join(lines)


def run_skill_augmented(
    model_fn,
    level_name: str,
    seed: int,
    skill_bank: SkillBank,
    max_iterations: int = 25,
    verbose: bool = False,
    temperature: float = 0.3,
    max_new_tokens: int = 800,
    is_oss: bool = False,
    max_general_skills: int = 0,
    max_specific_skills: int = 6,
    max_mistakes: Optional[int] = 4,
    worked_examples: Optional[list[dict]] = None,
    ablation_level: Optional[int] = None,
    xl_framing: bool = False,
) -> dict:
    """Run ReactAgent with skill-augmented prompts.

    Parameters
    ----------
    model_fn : Callable compatible with ReactAgent (messages, temp, max_tokens) -> str
    level_name : Puzzle level name.
    seed : Random seed for level generation.
    skill_bank : The SkillBank to retrieve skills from.
    max_iterations : Max ReAct loop iterations.
    verbose : Print debug info.
    temperature : Sampling temperature.
    max_new_tokens : Max generation tokens.
    is_oss : Whether using an OSS model (affects prompt format).
    max_general_skills : Max general skills to inject.
    max_specific_skills : Max level-specific skills to inject.
    worked_examples : Optional list of successful trajectory dicts for cold-start.
    ablation_level : If set (0-6), use the ablation prompt as the base instead of
                     the full hint prompt.  Level 6 = no hints at all, so the agent
                     relies entirely on learned skills.  Requires the experiments/
                     package to be importable.

    Returns
    -------
    dict with keys: success, action, iterations, trajectory, final_observation,
                    seed, skills_used, elapsed_time
    """
    # 1. Get base prompts — either ablated or full
    if ablation_level is not None:
        if not _HAS_ABLATION:
            raise ImportError(
                "ablation_level requires experiments.prompt_builder and "
                "experiments.feedback_reducer — make sure the experiments/ "
                "package is on PYTHONPATH."
            )
        original_system = build_system_prompt_ablation(level_name, ablation_level)
        original_initial = build_initial_user_message_ablation(level_name, ablation_level, is_oss=is_oss)
    else:
        original_system = build_system_prompt(level_name, is_oss=is_oss)
        original_initial = build_initial_user_message(level_name, is_oss=is_oss)

    # 2. Retrieve relevant skills
    skills = retrieve_skills(
        skill_bank, level_name,
        max_general=max_general_skills,
        max_specific=max_specific_skills,
    )

    # 3. Retrieve mistakes
    mistakes = skill_bank.get_mistakes_for_level(level_name)
    if max_mistakes is not None:
        mistakes = mistakes[:max_mistakes]

    # 4. Format skills + mistakes block
    skills_block = skill_bank.format_skills_as_prompt(
        level_name, skills=skills, mistakes=mistakes,
    )

    # 4b. Honest provenance label for cross-level transferred banks.
    # Single-line header swap — no extra paragraph, so prompt size is unchanged.
    if xl_framing and skills_block:
        original_header = f"## {level_name.replace('_', ' ').title()}-Specific Skills"
        skills_block = skills_block.replace(
            original_header,
            "## Cross-Level Skills (transferred from other puzzles — verify with simulation)",
        )

    # 5. Augment system prompt
    augmented_system = original_system
    if skills_block:
        augmented_system += "\n\n" + skills_block

    # Optional worked examples for cold-start
    if worked_examples:
        augmented_system += "\n" + format_worked_examples(worked_examples)

    # 6. Augment initial user message with diagnostic reminder
    augmented_initial = original_initial + "\n" + DIAGNOSTIC_ADDENDUM

    if verbose:
        print(f"\n[SkillRL] Injecting {len(skills)} skills + {len(mistakes)} mistakes for {level_name} seed={seed}")
        for s in skills:
            print(f"  - [skill] [{s.source_level}] {s.title}")
        for m in mistakes:
            print(f"  - [mistake] {m.description}")

    # 6. Monkey-patch (same pattern as experiment_runner.py lines 132-135)
    def patched_system(level: str, **kwargs) -> str:
        return augmented_system

    def patched_initial(level: str, **kwargs) -> str:
        return augmented_initial

    orig_build_system = agent_module.build_system_prompt
    orig_build_initial = agent_module.build_initial_user_message

    agent_module.build_system_prompt = patched_system
    agent_module.build_initial_user_message = patched_initial

    try:
        toolkit = InterphyreToolkit(level_name=level_name, seed=seed, is_oss=is_oss)

        # Wrap toolkit for ablation levels 5/6 (reduced feedback + blocked tools)
        if ablation_level is not None and ablation_level in (5, 6):
            toolkit = AblationToolkitWrapper(toolkit, ablation_level, level_name=level_name)

        agent = ReactAgent(
            model_fn=model_fn,
            toolkit=toolkit,
            level_name=level_name,
            max_iterations=max_iterations,
            verbose=verbose,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            is_oss=is_oss,
        )

        start = time.perf_counter()
        result = agent.solve()
        elapsed = time.perf_counter() - start

        toolkit.close()

        # Enrich result with SkillRL metadata
        result["seed"] = seed
        result["skills_used"] = [s.skill_id for s in skills]
        result["skill_titles"] = [s.title for s in skills]
        result["mistakes_injected"] = [m.mistake_id for m in mistakes]
        result["elapsed_time"] = elapsed
        if ablation_level is not None:
            result["ablation_level"] = ablation_level
        return result

    finally:
        # Restore originals
        agent_module.build_system_prompt = orig_build_system
        agent_module.build_initial_user_message = orig_build_initial
