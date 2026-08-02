"""Run ReactAgent with a retrieved-memory block injected into the system prompt.

This is the Episodic RAG / MemRL analogue of
`skillrl/runner/augmented_runner.py:run_skill_augmented` and uses the same
monkey-patching seam, so the only difference between methods is the text of the
injected block:

    HeXA          LEARNED PHYSICS SKILLS + MISTAKES TO AVOID + DIAGNOSTIC_ADDENDUM
    Episodic RAG  Retrieved Past Rollouts (verbatim)
    MemRL         Archived Memories (summarized, Q-ranked)
    ReAct         nothing

The base prompt, toolkit, iteration cap and actor model are untouched.
"""

from __future__ import annotations

import time
from typing import Optional

import react_agent.react_agent as agent_module
from react_agent.react_agent import ReactAgent
from react_agent.level_prompts import build_initial_user_message, build_system_prompt
from react_agent.tools import InterphyreToolkit

from baselines.memory.blocks import count_tokens
from baselines.memory.trajectory_fmt import count_simulator_calls


def run_memory_augmented(
    model_fn,
    level_name: str,
    seed: int,
    memory_block: str,
    *,
    max_iterations: int = 25,
    verbose: bool = False,
    temperature: float = 0.3,
    max_new_tokens: int = 800,
    is_oss: bool = False,
    retrieval_diagnostics: Optional[dict] = None,
) -> dict:
    """Solve one (level, seed) with `memory_block` appended to the system prompt."""
    original_system = build_system_prompt(level_name, is_oss=is_oss)
    original_initial = build_initial_user_message(level_name, is_oss=is_oss)

    augmented_system = original_system
    if memory_block:
        augmented_system = original_system + "\n\n" + memory_block

    def patched_system(level: str, **kwargs) -> str:
        return augmented_system

    def patched_initial(level: str, **kwargs) -> str:
        return original_initial

    orig_build_system = agent_module.build_system_prompt
    orig_build_initial = agent_module.build_initial_user_message
    agent_module.build_system_prompt = patched_system
    agent_module.build_initial_user_message = patched_initial

    try:
        toolkit = InterphyreToolkit(level_name=level_name, seed=seed, is_oss=is_oss)
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

        result["seed"] = seed
        result["level"] = level_name
        result["elapsed_time"] = elapsed
        result["simulator_calls"] = count_simulator_calls(result)
        result["added_context_tokens"] = count_tokens(memory_block)
        result["added_context_chars"] = len(memory_block)
        result["retrieval_diagnostics"] = retrieval_diagnostics or {}
        return result

    finally:
        agent_module.build_system_prompt = orig_build_system
        agent_module.build_initial_user_message = orig_build_initial
