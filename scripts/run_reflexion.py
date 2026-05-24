#!/usr/bin/env python3
"""Reflexion baseline (Claude, K=2 trials, mid-seed resume).

Per seed, run K trials of ReAct. After each failed trial, prompt Claude to
write a verbal self-reflection grounded in the failed trajectory; that
reflection is injected into the next trial's system prompt under a
``## Past Lessons`` block. Stops early on success. Both the actor and the
reflection step use the same ``claude -p`` CLI subprocess — no SDK, no API
key. Per-trial trajectories and per-seed summaries are saved to disk so
runs can be resumed mid-seed with ``--resume``.

Forwards every flag to ``react_agent.run_react_claude_reflexion``. Help:

    python scripts/run_reflexion.py --help

Example:

    python scripts/run_reflexion.py \\
        --level catapult \\
        --seeds 0 1 2 3 4 5 6 7 8 9 \\
        --k-trials 2 --max-iterations 25 \\
        --eval-dir results/reflexion_catapult --verbose

Resume an interrupted run (skips completed seeds AND completed trials):

    python scripts/run_reflexion.py \\
        --level catapult --seeds 0 1 2 3 4 5 6 7 8 9 \\
        --k-trials 2 --eval-dir results/reflexion_catapult --resume
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _launch import run_module

run_module("react_agent.run_react_claude_reflexion")
