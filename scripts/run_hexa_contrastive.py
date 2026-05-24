#!/usr/bin/env python3
"""HeXA-Contrastive (reward ablation): no rewards, no per-skill confidence.

Drop-in offline-to-online evolving variant where the teacher forms skills
purely by CONTRASTING successful and failed trajectories — no per-trajectory
reward, no per-skill confidence calibration. Used in the paper as the reward
ablation, run on Qwen-7B over ``down_to_earth`` and ``two_body_problem``.

When ``--level catapult`` is passed, the loop auto-patches the catapult-specific
factual prompts (no strategy hints leaked) — same discipline as
``run_hexa_catapult.py`` and ``run_hexa_skills_only.py``. All other levels use
the generic ``LEVEL_DESCRIPTIONS`` from ``skillrl/core/config.py``.

Forwards every flag to ``skillrl.loops.contrastive_only_loop``. Help:

    python scripts/run_hexa_contrastive.py --help

Examples:

    python scripts/run_hexa_contrastive.py \\
        --level two_body_problem \\
        --initial-traj-dir Initial_trajectories/two_body_problem \\
        --num-rounds 3 --seeds-per-round 5 \\
        --model qwen7b

    python scripts/run_hexa_contrastive.py \\
        --level catapult \\
        --initial-traj-dir Initial_trajectories/catapult \\
        --num-rounds 3 --seeds-per-round 3 --start-seed 6
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _launch import run_module

run_module("skillrl.loops.contrastive_only_loop")
