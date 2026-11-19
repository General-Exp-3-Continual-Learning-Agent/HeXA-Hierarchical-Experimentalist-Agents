#!/bin/bash
# Frozen-bank evaluation of the three HeXA catapult runs (Claude Sonnet).
#
# Each run's bank was evolved over 17 rounds x 3 seeds = seeds 6-56.
# skill_bank_evolving_17.json is the final bank; it is FROZEN here (no further
# distillation or evolution) and evaluated on the next 50 held-out seeds 57-106.
#
# Injection config is left at run_skill_agent defaults, which are exactly what
# the evolving loop used during training: 0 general skills, 6 level-specific,
# 4 mistakes, 25 max iterations, temperature 0.3.
#
# Usage: ./eval_frozen_banks_catapult.sh <run_number 1|2|3>
# Resumable: run_batch skips any seed that already has a trajectory file.

set -u
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-python}"
RUN="${1:?usage: $0 <1|2|3>}"

BASE="$REPO/skillrl/RESULTS/catapult_claude/HeXA_catapult/HeXA_catapult_claude_run${RUN}"
BANK="$BASE/skill_bank_evolving_17.json"
OUT="$BASE/frozen_eval_seeds57-106"

cd "$REPO"
exec "$PY" -m skillrl.runner.run_skill_agent \
    --level catapult \
    --seeds $(seq 57 106) \
    --skill-bank "$BANK" \
    --model claude \
    --eval-dir "$OUT" \
    --verbose
