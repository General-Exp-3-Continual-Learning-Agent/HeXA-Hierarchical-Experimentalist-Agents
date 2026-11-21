#!/bin/bash
# Run only spatial_task task 9 (with-bank arm) as a separate job writing to its own
# output dir, so it never shares a directory with the main resume job. Merge afterwards.
set -euo pipefail
DRY=0; [[ ${1:-} == --dry-run ]] && DRY=1
SETUP=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export REPO_DIR=$(cd -- "$SETUP/../.." && pwd)
export PIGEY_REPO="$REPO_DIR/baselines/Pigey"
export PIGEY_ROOT=${PIGEY_ROOT:?set PIGEY_ROOT to the shadow root}
source "$SETUP/experiments/libero_spatial_task/protocol.env"
export PIGEY_SUITE="$SUITE" PIGEY_TASKS="9" PIGEY_EPISODES="$EVAL_STATES"
export PIGEY_ER_MODEL="$ER_MODEL" PIGEY_ER_GPU_UTIL="$ER_GPU_UTIL"
export PIGEY_POLICY_GPU_FRACTION=0.35
export PIGEY_OUT_DIR="$PIGEY_ROOT/results/eval/spatial_task_bank_t9"
export PIGEY_SKILL_BANK="$PIGEY_ROOT/results/hexa/spatial_task_train/bank_round_16.json"
export PIGEY_CLAUDE_MAX_TURNS=3
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$PIGEY_OUT_DIR"
cmd=(sbatch --parsable --account="${SLURM_ACCOUNT:-pi_sniekum_umass_edu}" --export=ALL
    --output="$PIGEY_ROOT/logs/%x-%j.out" --error="$PIGEY_ROOT/logs/%x-%j.err"
    --exclude="${EXCLUDE_NODES:-gpu023}" --job-name=spatial_task-t9-bank --time=05:00:00
    "$SETUP/slurm/run_trial.sbatch" --model "$MODEL" --claude-timeout 300 --claude-max-turns 3
    --perception local --verifier claude --verify-model claude-haiku-4-5 --sam service --phase0 skip)
if (( DRY )); then printf '%q ' "${cmd[@]}"; printf '\n'; else "${cmd[@]}"; fi
