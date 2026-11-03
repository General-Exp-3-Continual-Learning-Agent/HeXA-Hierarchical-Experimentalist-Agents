#!/bin/bash
#SBATCH --job-name=rbank_qwen
#SBATCH --partition=superpod-a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH --mem=64G
# Same GPU story as sbatch_baseline_qwen.sh: superpod-a100 is uniformly A100-80GB.
# To spread onto the general partitions (override on the command line, since
# #SBATCH lines are parsed before this script runs):
#
#   # Qwen-7B actor+extractor: bf16 weights ~15GB + 25-iteration context -> 40GB card
#   sbatch -p gpu,gpu-preempt --constraint='sm_80&vram40' \
#       baselines/bash_scripts/sbatch_reasoning_bank_qwen.sh --level down_to_earth ...
#
# gpu-preempt is safe despite the 24h walltime: the stream resumes from existing
# per-seed trajectory JSONs, so a preemption costs one episode.
#SBATCH -A pi_sniekum_umass_edu
#SBATCH -o logs/rbank_qwen_%j.out
#SBATCH -e logs/rbank_qwen_%j.err
#
# ReasoningBank with a local Qwen2.5-7B-Instruct actor on one GPU. The SAME model
# serves as the memory extractor (the paper's same-backbone choice) at extraction
# temperature 1.0, so a run needs no second model in memory.
#
# Retrieval key defaults to `numeric` (standardized scene geometry). Run the
# calibration job for the level FIRST so the scaler exists:
#   sbatch baselines/bash_scripts/sbatch_calibrate.sh --level down_to_earth
#
# HeXA-matched build + frozen held-out eval (train 6-56, freeze, eval 57-106):
#   sbatch baselines/bash_scripts/sbatch_reasoning_bank_qwen.sh \
#       --level down_to_earth \
#       --seeds 6-106 --freeze-after 51 \
#       --initial-traj-dir Initial_trajectories/DTE_Qwen7B
#
#   sbatch baselines/bash_scripts/sbatch_reasoning_bank_qwen.sh \
#       --level two_body_problem \
#       --seeds 6-106 --freeze-after 51 \
#       --initial-traj-dir Initial_trajectories/TwoBody_7B
#
# The Qwen seed-trajectory dirs (DTE_Qwen7B / TwoBody_7B) live in the live
# research repo, not the public HeXA tree. Point --initial-traj-dir at an absolute
# path if $REPO_DIR does not contain them, or omit it for a cold start.
#
# Variant with HeXA-matched injection budget (more retrieved experiences):
#   ... --extra "--k 3" --tag k3
#
# Runs resume: re-submitting skips seeds whose trajectory JSON already exists.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

LEVEL=""
SEEDS="6-55"
FREEZE_AFTER=""
SEEDS_PER_ROUND=3
MAX_ITERATIONS=25
INITIAL_TRAJ_DIR=""
EMBEDDING_MODEL=""       # empty -> runner default (numeric)
INTENT_MODE="geometry"
ACTOR_LABEL=""
TAG=""
EXTRA=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --level)             LEVEL="$2";             shift 2 ;;
        --seeds)             SEEDS="$2";             shift 2 ;;
        --freeze-after)      FREEZE_AFTER="$2";      shift 2 ;;
        --seeds-per-round)   SEEDS_PER_ROUND="$2";   shift 2 ;;
        --max-iterations)    MAX_ITERATIONS="$2";    shift 2 ;;
        --initial-traj-dir)  INITIAL_TRAJ_DIR="$2";  shift 2 ;;
        --embedding-model)   EMBEDDING_MODEL="$2";   shift 2 ;;
        --intent-mode)       INTENT_MODE="$2";       shift 2 ;;
        --model)             QWEN_OVERRIDE="$2";     shift 2 ;;
        --actor-label)       ACTOR_LABEL="$2";       shift 2 ;;
        --tag)               TAG="$2";               shift 2 ;;
        --extra)             EXTRA="$2";             shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -n "$LEVEL" ]] || { echo "[ERROR] --level is required"; exit 1; }

# sbatch runs a *copy* of this script from the Slurm spool dir, so the repo cannot
# be located relative to $BASH_SOURCE. Anchor it explicitly.
REPO_DIR="${REPO_DIR:-/project/pi_sniekum_umass_edu/vgandhi/HExA_Baseline_Exp/HeXA-Hierarchical-Experimentalist-Agents}"
export REPO_DIR
source "$REPO_DIR/baselines/bash_scripts/_baseline_env.sh"
ACTOR="${QWEN_OVERRIDE:-$QWEN_PATH}"

if [[ -z "$ACTOR_LABEL" ]]; then
    case "$ACTOR" in
        *3B*|*3b*) ACTOR_LABEL="qwen3b" ;;
        *7B*|*7b*) ACTOR_LABEL="qwen7b" ;;
        *) ACTOR_LABEL="$(basename "$ACTOR" | tr -cd '[:alnum:]-' | cut -c1-20)" ;;
    esac
fi
RUN_NAME="reasoning_bank_${LEVEL}_${ACTOR_LABEL}${TAG:+_$TAG}"
OUTPUT_DIR="$BASELINE_ROOT/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"

echo "================================================================"
echo "  ReasoningBank baseline (local Qwen actor + same-backbone extractor)"
echo "  Job          : ${SLURM_JOB_ID:-<interactive>}"
echo "  Level        : $LEVEL"
echo "  Seeds        : $SEEDS  (rounds of $SEEDS_PER_ROUND)"
echo "  Freeze after : ${FREEZE_AFTER:-<none>}"
echo "  Actor        : $ACTOR"
echo "  Actor label  : $ACTOR_LABEL"
echo "  Embedder     : ${EMBEDDING_MODEL:-<runner default: numeric>}"
echo "  Intent mode  : $INTENT_MODE"
echo "  Offline init : ${INITIAL_TRAJ_DIR:-<none>}"
echo "  Output       : $OUTPUT_DIR"
echo "  Cache        : $CACHE_DIR"
echo "================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

args=(
    --level "$LEVEL"
    --seeds "$SEEDS"
    --seeds-per-round "$SEEDS_PER_ROUND"
    --max-iterations "$MAX_ITERATIONS"
    --output-dir "$OUTPUT_DIR"
    --cache-dir "$CACHE_DIR"
    --model "$ACTOR"
    --intent-mode "$INTENT_MODE"
    --verbose
)
[[ -n "$FREEZE_AFTER" ]] && args+=(--freeze-after "$FREEZE_AFTER")
[[ -n "$INITIAL_TRAJ_DIR" ]] && args+=(--initial-traj-dir "$INITIAL_TRAJ_DIR")
[[ -n "$EMBEDDING_MODEL" ]] && args+=(--embedding-model "$EMBEDDING_MODEL")

# shellcheck disable=SC2206
[[ -n "$EXTRA" ]] && args+=(${EXTRA})

$PYTHON -m baselines.runner.run_reasoning_bank "${args[@]}"

echo "Done. Results in $OUTPUT_DIR"
