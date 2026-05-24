#!/usr/bin/env python3
"""HeXA: offline-to-online evolving skill refinement (headline method).

Round 1 distills a skill bank from initial offline trajectories. Rounds 2+
EVOLVE the bank by feeding the previous bank + new round trajectories back
to the teacher, which curates redundancy and adds new skills under a hard
budget (max 10 skills/level by default).

This is the configuration reported as "HeXA" in the paper:
- Initialisation: Offline-to-Online (Off2On)
- Update rule:    Evolving
- Seeds/round:    3
- Rounds:         3

Forwards every flag to ``skillrl.loops.evolving_loop``. See the full flag set:

    python scripts/run_hexa.py --help

Example:

    python scripts/run_hexa.py \\
        --level pass_the_parcel \\
        --initial-traj-dir Initial_trajectories/pass_the_parcel \\
        --num-rounds 3 --seeds-per-round 3 --start-seed 6
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _launch import run_module

run_module("skillrl.loops.evolving_loop")
