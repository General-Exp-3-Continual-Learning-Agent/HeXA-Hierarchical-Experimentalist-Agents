#!/usr/bin/env python3
"""GEPA prompt optimization for the catapult level (Claude actor + reflection).

Uses the GEPA reflective prompt optimizer to rewrite the catapult system prompt
and initial user message against real Claude rollouts. The actor solving the
puzzle and the reflection LM that rewrites the prompt are both Claude Sonnet 4.6
via the Claude Code CLI (no API key needed).

Forwards every flag to ``gepa_opt.run_gepa_catapult``. See the full flag set:

    python scripts/run_gepa_catapult.py --help

Defaults: 6 train seeds / 4 val seeds / 60 metric calls, 4 parallel rollouts.

Cost note: each metric call is a full ReAct episode (up to 25 Claude CLI turns,
several minutes). Start small, e.g.:

    python scripts/run_gepa_catapult.py \\
        --train-seeds 6 7 --val-seeds 8 \\
        --max-metric-calls 6 --reflection-minibatch-size 2 --workers 2
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _launch import run_module

run_module("gepa_opt.run_gepa_catapult")
