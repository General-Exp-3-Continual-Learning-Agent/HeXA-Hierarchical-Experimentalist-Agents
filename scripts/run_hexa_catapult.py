#!/usr/bin/env python3
"""HeXA on `catapult` with the catapult-specific FACTUAL teacher prompts.

The standard HeXA loop (`run_hexa.py`) uses generic, single-paragraph level
descriptions from `skillrl/core/config.py`. For the catapult level the paper
uses a longer FACTUAL scene block (no strategy hints leaked to the teacher)
to test whether the teacher can derive strategies from trajectories alone.

This wrapper invokes ``skillrl.distillation.teacher_prompts_catapult`` —
that module monkey-patches the catapult prompts into the generic
distill/evolve pipeline before calling ``run_evolving_loop`` with
``level_name="catapult"``.

Same Init × Update shape as HeXA: offline → online evolving, batch 3.

Example:

    python scripts/run_hexa_catapult.py \\
        --initial-traj-dir Initial_trajectories/catapult \\
        --num-rounds 3 --seeds-per-round 3 --start-seed 6 \\
        --model claude --teacher-model claude-sonnet-4-6
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _launch import run_module

run_module("skillrl.distillation.teacher_prompts_catapult")
