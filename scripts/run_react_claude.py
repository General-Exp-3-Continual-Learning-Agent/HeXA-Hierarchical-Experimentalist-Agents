#!/usr/bin/env python3
"""ReAct baseline (Claude via the Claude Code CLI).

Standard ReAct loop where the actor is Claude Sonnet, invoked by spawning
``claude -p`` as a subprocess (no Anthropic SDK and no API key — the same
auth that ``claude --version`` uses). ``--max-iterations`` defaults to 25.

Forwards every flag to ``react_agent.run_react_claude``. Help:

    python scripts/run_react_claude.py --help

Example:

    python scripts/run_react_claude.py \\
        --level pass_the_parcel \\
        --seeds 0 1 2 3 4 5 6 7 8 9 \\
        --eval-dir results/react_claude_pass_the_parcel
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _launch import run_module

run_module("react_agent.run_react_claude")
