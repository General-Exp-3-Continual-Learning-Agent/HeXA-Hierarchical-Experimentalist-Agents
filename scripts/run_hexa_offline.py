#!/usr/bin/env python3
"""HeXA-Offline: distill once, run all seeds with a fixed bank.

Pure-offline configuration: the teacher distills a single skill bank from the
initial trajectories and the agent then runs all evaluation seeds against
that frozen bank — no rounds, no evolution.

Forwards every flag to ``skillrl.loops.offline_loop``. Help:

    python scripts/run_hexa_offline.py --help

Example:

    python scripts/run_hexa_offline.py \\
        --level catapult \\
        --initial-traj-dir Initial_trajectories/catapult \\
        --output-dir Hexa_catapult_offline/
        --seeds 6 
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _launch import run_module

run_module("skillrl.loops.offline_loop")
