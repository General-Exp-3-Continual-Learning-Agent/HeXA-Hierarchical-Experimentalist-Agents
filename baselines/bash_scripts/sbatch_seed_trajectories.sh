#!/bin/bash
#SBATCH --job-name=seed_trajs
#SBATCH --partition=superpod-a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH -t 8:00:00
#SBATCH --mem=64G
#SBATCH -A pi_sniekum_umass_edu
#SBATCH -o logs/seed_trajs_%j.out
#SBATCH -e logs/seed_trajs_%j.err
#
# Generate offline starter trajectories for a level with plain ReAct (no memory,
# no skill bank) — the same way `Initial_trajectories/catapult` was produced.
#
# Two uses:
#   1. A capability probe. If the actor scores 0/N here, both memory baselines
#      will floor at 0% on this level and there is no point running them —
#      Qwen2.5-7B did exactly that on catapult (0/50).
#   2. The Off2On starter set. Output lands in the trajectory_seed*.json layout
#      that `--initial-traj-dir` reads, so HeXA and both baselines can share it.
#
# Usage:
#   sbatch baselines/bash_scripts/sbatch_seed_trajectories.sh --level down_to_earth
#   sbatch baselines/bash_scripts/sbatch_seed_trajectories.sh \
#       --level pass_the_parcel --seeds "1 2 3 4 5" --model claude
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

LEVEL=""
SEEDS="1 2 3 4 5"
MAX_ITERATIONS=25
MODEL=""          # empty -> QWEN_PATH from _baseline_env.sh
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --level)           LEVEL="$2";           shift 2 ;;
        --seeds)           SEEDS="$2";           shift 2 ;;
        --max-iterations)  MAX_ITERATIONS="$2";  shift 2 ;;
        --model)           MODEL="$2";           shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2";      shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -n "$LEVEL" ]] || { echo "[ERROR] --level is required"; exit 1; }

# sbatch runs a *copy* of this script from the Slurm spool dir, so the repo
# cannot be located relative to $BASH_SOURCE. Anchor it explicitly.
REPO_DIR="${REPO_DIR:-/project/pi_sniekum_umass_edu/vgandhi/HExA_Baseline_Exp/HeXA-Hierarchical-Experimentalist-Agents}"
export REPO_DIR
source "$REPO_DIR/baselines/bash_scripts/_baseline_env.sh"

ACTOR="${MODEL:-$QWEN_PATH}"
# Write into the repo's Initial_trajectories/ so --initial-traj-dir finds it the
# same way it finds the shipped catapult set.
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/Initial_trajectories/$LEVEL}"
mkdir -p "$OUTPUT_DIR"

echo "================================================================"
echo "  Starter trajectories / capability probe"
echo "  Job      : ${SLURM_JOB_ID:-<interactive>}"
echo "  Level    : $LEVEL"
echo "  Seeds    : $SEEDS"
echo "  Actor    : $ACTOR"
echo "  Output   : $OUTPUT_DIR"
echo "================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

if [[ "$ACTOR" == "claude" ]]; then
    MODULE="react_agent.run_react_claude"
else
    MODULE="react_agent.run_react"
fi

$PYTHON -m "$MODULE" \
    --level "$LEVEL" \
    --model "$ACTOR" \
    --seeds $SEEDS \
    --max-iterations "$MAX_ITERATIONS" \
    --eval-dir "$OUTPUT_DIR" \
    --verbose

echo
echo "Trajectories written to $OUTPUT_DIR:"
ls -1 "$OUTPUT_DIR"/trajectory_seed*.json 2>/dev/null || echo "  (none — check the log above)"
