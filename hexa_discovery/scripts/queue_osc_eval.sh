#!/bin/bash
# Frozen-bank held-out eval for oscillator (see PREREG_frozen_eval_oscillator.md).
# Waits for training to finish, then runs both arms in parallel on seeds 300-314.
cd "$(dirname "$0")/../.."   # HeXA root
PY="${PY:-.venv/bin/python}"
while pgrep -f "[w]ithin_world_loop" >/dev/null; do sleep 30; done
B=hexa_discovery/data/runs/within_oscillator_v2/skill_bank_round05.json
[ -f "$B" ] || { echo "FATAL: round-05 bank missing; training did not reach round 5"; exit 1; }
echo "round-05 bank frozen; starting eval $(date)"
mkdir -p hexa_discovery/data/runs/osc_eval
cp "$B" hexa_discovery/data/runs/osc_eval/skill_bank_frozen.json
S="300 301 302 303 304 305 306 307 308 309 310 311 312 313 314"
"$PY" -u -m hexa_discovery.extend_transfer \
  --run-dir hexa_discovery/data/runs/osc_eval --bank skill_bank_frozen.json \
  --world oscillator --seeds $S --arms with_bank --cli-model sonnet \
  > hexa_discovery/data/runs/osc_eval/bank_arm.log 2>&1 &
"$PY" -u -m hexa_discovery.run_offline \
  --world oscillator --seeds $S \
  --output-dir hexa_discovery/data/runs/osc_eval/baseline \
  --cli-model sonnet --noise-std 0.075 \
  > hexa_discovery/data/runs/osc_eval/baseline_arm.log 2>&1 &
wait
echo "EVAL COMPLETE $(date)"
