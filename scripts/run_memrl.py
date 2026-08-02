#!/usr/bin/env python3
"""MemRL baseline (value-aware episodic memory).

Intent-Experience-Utility triplets with two-phase retrieval (similarity recall,
then value-aware selection on a z-scored similarity+Q composite) and Monte-Carlo
utility updates from the environment's verdict. Ported from the official release
(MemTensor/MemRL) using its ALFWorld configuration.

Run ``scripts/calibrate_intents.py`` for the level first — MemRL's Phase-A
threshold and similarity normalization constants are dataset-dependent and must
be measured on InterPhyre intents, not copied from ALFWorld.

Forwards every flag to ``baselines.runner.run_memrl``. Help:

    python scripts/run_memrl.py --help

Example:

    python scripts/calibrate_intents.py --level pass_the_parcel --seeds 1-105
    python scripts/run_memrl.py \\
        --level pass_the_parcel \\
        --seeds 6-55 \\
        --initial-traj-dir Initial_trajectories/pass_the_parcel \\
        --output-dir baselines/data/memrl_ptp \\
        --seeds-per-round 3 --verbose

The runtime-RL ablation (Q-blind retrieval) is ``--lambda-q 0``.
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _launch import run_module

run_module("baselines.runner.run_memrl")
