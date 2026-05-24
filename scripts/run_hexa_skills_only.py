#!/usr/bin/env python3
"""HeXA skills-only ablation: success trajectories only, no mistakes in the bank.

Same Init × Update shape as HeXA (offline → online evolving, batch 3), with two
ablations applied:

1. **Input filter**: failed trajectories are dropped before any teacher call —
   the teacher only ever sees successful trajectories.
2. **Output schema**: the bank contains only level-specific skills. There are
   no Mistake entries for the target level.

When ``--level catapult`` is passed, the loop auto-patches the catapult-specific
factual prompts (no strategy hints leaked) — same discipline as
``run_hexa_catapult.py``. All other levels use the generic ``LEVEL_DESCRIPTIONS``
from ``skillrl/core/config.py``.

Works with Claude (default) and Qwen actors / teachers via ``--model`` /
``--teacher-model``.

Example:

    python scripts/run_hexa_skills_only.py \\
        --level pass_the_parcel \\
        --initial-traj-dir Initial_trajectories/Pass_the_parcel \\
        --num-rounds 3 --seeds-per-round 3 --start-seed 6

    python scripts/run_hexa_skills_only.py \\
        --level catapult \\
        --initial-traj-dir Initial_trajectories/catapult \\
        --num-rounds 3 --seeds-per-round 3 --start-seed 6
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _launch import run_module

run_module("skillrl.loops.skills_only_evolving_loop")
