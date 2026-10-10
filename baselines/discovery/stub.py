"""A scripted DiscoverPhysics episode, for checking the plumbing without a model.

Replaces `discovery_runner.run_episode` only. Everything above it is the real
code path -- retrieval, block rendering, the round-batched write-back, store
persistence, the summary -- so a stub run exercises exactly the logic that would
otherwise only be tested by a several-hour arm.

Outcomes are deterministic in the seed, and deliberately mixed: without both a
success and a failure in memory, Episodic RAG never leaves its "similarity only"
fallback and the retrieval branch that matters is never taken.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DISCOVERY = Path(__file__).resolve().parents[2] / "discovery-agents"
if str(_DISCOVERY) not in sys.path:
    sys.path.insert(0, str(_DISCOVERY))


def fake_episode(world, seed, bank, *, prompt_block=None, **kwargs):
    passed = (seed % 3) == 0
    expl = 0.9 if passed else 0.4
    return {
        "seed": seed,
        "world": world,
        "iterations": 4 + (seed % 3),
        "max_rounds": 10,
        "success": passed,
        "trajectory": [
            {"thought": f"probe {world} at radius <n> (stub, seed {seed})",
             "action": "run_experiment: {\"measurement_times\": [<n>]}",
             "observation": "positions: [[<n>, <n>]]"},
            {"thought": "submit", "action": "submit_final_law: def law(...)",
             "observation": f"submitted final law; passed={passed}"},
        ],
        "action": None,
        "norm_mse": 0.01 if passed else 0.9,
        "norm_mse_own": None,
        "expl_score": expl,
        "mean_pos_error": 0.05,
        "reward": 0.9 if passed else -0.2,
        "expl_reasoning": "STUB CRITIQUE — must never appear in a memory block",
        "final_law_raw": "def law(): ...",
        "explanation": "stub explanation",
        "redacted": True,
        "elapsed_time": 0.0,
        "model": "stub",
        "noise_std": kwargs.get("noise_std", 0.075),
        "skills_injected": bool(prompt_block),
        "skill_ids_injected": [],
        "skill_titles_injected": [],
        "mistake_ids_injected": [],
    }


def install() -> None:
    import hexa_discovery.discovery_runner as runner

    runner.run_episode = fake_episode
    print("  [stub] run_episode replaced; no model calls will be made")
