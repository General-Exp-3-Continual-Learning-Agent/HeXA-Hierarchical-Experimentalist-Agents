#!/usr/bin/env python3
"""GEPA prompt optimization for the pass_the_parcel level (Claude actor + reflection).

Uses the GEPA reflective prompt optimizer to rewrite the pass_the_parcel system
prompt and initial user message against real Claude rollouts. The actor and
reflection LM are both Claude Sonnet 4.6 via the Claude Code CLI.

Seed design — identical to catapult for a clean cross-level comparison:
  TEST  : seeds  6–55  (50 seeds; same benchmark as all other methods)
  TRAIN : seeds 57–71 (15 seeds)
  VAL   : seeds 72–76  (5 seeds)

Budget: 50 metric calls (binary reward). Each call = one full ReAct episode
(up to 25 Claude CLI turns).

Quick wiring check:
    python scripts/run_gepa_ptp.py \\
        --train-seeds 100 101 --val-seeds 102 \\
        --max-metric-calls 6 --workers 2 --test-seeds 60 61

Full optimization run:
    python scripts/run_gepa_ptp.py --output-dir gepa_opt/runs/ptp_binary_50

Held-out test (after optimization completes):
    python scripts/run_gepa_ptp.py --test-only --output-dir gepa_opt/runs/ptp_binary_50
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _launch import run_module

run_module("gepa_opt.run_gepa_ptp")
