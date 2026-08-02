#!/usr/bin/env python3
"""Calibrate MemRL's similarity constants on a level's intents.

Embeds the (level, seed) intent texts, reports the pairwise cosine-similarity
distribution, and prints the three dataset-dependent MemRL constants: the
Phase-A threshold ``delta`` (the paper's top-20% quantile of that distribution)
and the ``sim_norm_mean`` / ``sim_norm_std`` used for similarity z-normalization.

Costs no simulator rollouts and no LLM calls. Run it before ``run_memrl.py``.

Forwards every flag to ``baselines.calibrate``. Help:

    python scripts/calibrate_intents.py --help

Example:

    python scripts/calibrate_intents.py --level pass_the_parcel --seeds 1-105
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _launch import run_module

run_module("baselines.calibrate")
