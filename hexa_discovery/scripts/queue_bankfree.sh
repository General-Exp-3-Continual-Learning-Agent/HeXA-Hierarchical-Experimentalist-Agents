#!/bin/bash
# Bank-free (bank=None) episodes for per-world distillation K_l.
# These double as the missing no-bank baselines for these worlds.
# Stops on a session limit rather than reporting success: run_offline exits
# non-zero, and completed episodes are skipped on resume.
cd "$(dirname "$0")/../.."   # HeXA root
PY="${PY:-.venv/bin/python}"
while pgrep -f "[e]xtend_transfer" >/dev/null; do sleep 30; done
for W in coulomb_easy extra_dimensions oscillator three_species dark_matter; do
  echo "=== $W ==="
  if ! "$PY" -u -m hexa_discovery.run_offline \
      --world $W --seeds 100 101 102 103 104 \
      --output-dir hexa_discovery/data/offline/sonnet/$W \
      --cli-model sonnet --noise-std 0.075; then
    echo "STOPPED on $W (likely session limit) $(date) — re-run this script to resume"
    exit 1
  fi
done
echo "ALL BANK-FREE EPISODES DONE $(date)"
