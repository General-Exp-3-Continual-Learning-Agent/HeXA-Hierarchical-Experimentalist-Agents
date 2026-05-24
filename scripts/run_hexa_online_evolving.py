#!/usr/bin/env python3
"""HeXA-Online-Evolving: pure online start, then evolve the bank.

Round 1 runs the agent with NO skill bank (online baseline collection).
Round 2 distills an initial bank from those trajectories. Round 3+ EVOLVES
the bank using the previous bank + new round trajectories (same evolution
rule used by HeXA, but seeded online instead of from offline trajectories).

Forwards every flag to ``skillrl.loops.online_evolving_loop``. Help:

    python scripts/run_hexa_online_evolving.py --help

Example:

    python scripts/run_hexa_online_evolving.py \\
        --level pass_the_parcel \\
        --num-rounds 5 --seeds-per-round 3 --start-seed 6
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _launch import run_module

run_module("skillrl.loops.online_evolving_loop")
