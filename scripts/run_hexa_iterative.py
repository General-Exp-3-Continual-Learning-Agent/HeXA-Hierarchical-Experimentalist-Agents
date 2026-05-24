#!/usr/bin/env python3
"""HeXA-Iterative: offline-to-online with re-distillation each round.

Round 1 distills a skill bank from initial offline trajectories. Rounds 2+
RE-DISTILL from scratch on the latest round's trajectories plus the best N
carry-over successes (no bank-context provided to the teacher). Skills can
accumulate without redundancy removal — contrast with the Evolving update
rule used by ``run_hexa.py``.

Forwards every flag to ``skillrl.loops.iterative_loop``. Help:

    python scripts/run_hexa_iterative.py --help

Example:

    python scripts/run_hexa_iterative.py \\
        --level pass_the_parcel \\
        --initial-traj-dir Initial_trajectories/pass_the_parcel \\
        --num-rounds 3 --seeds-per-round 3
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _launch import run_module

run_module("skillrl.loops.iterative_loop")
