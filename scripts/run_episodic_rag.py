#!/usr/bin/env python3
"""Episodic RAG baseline (retrieval of entire past rollouts).

Memory holds every past rollout verbatim, indexed by the intent embedding of its
(level, seed). Each episode is given the nearest past *successful* rollout and
the nearest past *unsuccessful* rollout, with the verifier's feedback attached —
the definition used in the CORE paper. Memory grows across the seed stream; no
summarization, no utility scores.

Forwards every flag to ``baselines.runner.run_episodic_rag``. Help:

    python scripts/run_episodic_rag.py --help

Example (50-seed grid, matching the ReAct/Reflexion/Direct protocol):

    python scripts/run_episodic_rag.py \\
        --level pass_the_parcel \\
        --seeds 6-55 \\
        --initial-traj-dir Initial_trajectories/pass_the_parcel \\
        --output-dir baselines/data/episodic_rag_ptp \\
        --seeds-per-round 3 --verbose
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _launch import run_module

run_module("baselines.runner.run_episodic_rag")
