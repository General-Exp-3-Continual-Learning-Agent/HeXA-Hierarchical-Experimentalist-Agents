#!/bin/bash
cd "$(dirname "$0")/../.."   # HeXA root
PY="${PY:-.venv/bin/python}"
B=hexa_discovery/data/runs/within_circle/skill_bank_round05.json
[ -f "$B" ] || { echo "FATAL: circle round-05 bank missing"; exit 1; }
mkdir -p hexa_discovery/data/runs/circle_eval
cp "$B" hexa_discovery/data/runs/circle_eval/skill_bank_frozen.json
S="300 301 302 303 304 305 306 307 308 309 310 311 312 313 314"
"$PY" -u -m hexa_discovery.extend_transfer \
  --run-dir hexa_discovery/data/runs/circle_eval --bank skill_bank_frozen.json \
  --world circle --seeds $S --arms with_bank --cli-model sonnet \
  > hexa_discovery/data/runs/circle_eval/bank_arm.log 2>&1 &
"$PY" -u -m hexa_discovery.run_offline \
  --world circle --seeds $S \
  --output-dir hexa_discovery/data/runs/circle_eval/baseline \
  --cli-model sonnet --noise-std 0.075 \
  > hexa_discovery/data/runs/circle_eval/baseline_arm.log 2>&1 &
wait
echo "CIRCLE EVAL COMPLETE $(date)"
