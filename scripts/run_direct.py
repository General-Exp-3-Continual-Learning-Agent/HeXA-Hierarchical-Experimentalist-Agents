#!/usr/bin/env python3
"""Direct-answer baseline (Claude, 2 iterations).

Pure zero-shot baseline: the agent is allowed only ``--max-iterations 2``
(one peek at the scene, one action commit). Used in the paper to bound
the difficulty of each level when the agent gets essentially no chance to
reason or course-correct.

Forwards every flag to ``react_agent.run_react_claude_direct``. Help:

    python scripts/run_direct.py --help

Example:

    python scripts/run_direct.py \\
        --level pass_the_parcel \\
        --seeds 0 1 2 3 4 5 6 7 8 9 \\
        --max-iterations 2 \\
        --eval-dir results/direct_pass_the_parcel
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _launch import run_module

run_module("react_agent.run_react_claude_direct")
