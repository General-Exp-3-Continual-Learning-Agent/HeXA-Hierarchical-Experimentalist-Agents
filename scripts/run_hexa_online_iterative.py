#!/usr/bin/env python3
"""HeXA-Online-Iterative: pure online start, then re-distill each round.

Round 1 runs the agent with NO skill bank (online baseline). Rounds 2+
RE-DISTILL from scratch on the latest round's trajectories + best
carry-over successes (no bank context provided). Combines online seeding
with the Iterative update rule (contrast with ``run_hexa_online_evolving``).

Forwards every flag to ``skillrl.loops.online_iterative_loop``. Help:

    python scripts/run_hexa_online_iterative.py --help

Example:

    python scripts/run_hexa_online_iterative.py \\
        --level pass_the_parcel \\
        --initial-traj-dir Initial_trajectories/pass_the_parcel \\
        --num-rounds 3 --seeds-per-round 5
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _launch import run_module

run_module("skillrl.loops.online_iterative_loop")
