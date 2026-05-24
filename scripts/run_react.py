#!/usr/bin/env python3
"""ReAct baseline (Qwen / vLLM / mock): single-shot ReAct on a level.

The standard ReAct agent with no skill bank, no reflection, no curriculum.
Use this for the open-source / Qwen baseline numbers in the paper.

Forwards every flag to ``react_agent.run_react``. Help:

    python scripts/run_react.py --help

Example (mock dry-run):

    python scripts/run_react.py --model mock --level catapult --seed 42 --verbose

Example (Qwen via local vLLM):

    python scripts/run_react.py \\
        --model "Qwen2.5-7B-Instruct" \\
        --vllm-url http://localhost:8000/v1 \\
        --level catapult --seed 42 --verbose
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _launch import run_module

run_module("react_agent.run_react")
