#!/bin/bash
#SBATCH --job-name=baseline_calibrate
#SBATCH --partition=cpu
#SBATCH -c 8
#SBATCH -t 1:00:00
#SBATCH --mem 24G
#SBATCH -A pi_sniekum_umass_edu
#SBATCH -o logs/calibrate_%j.out
#SBATCH -e logs/calibrate_%j.err
#
# Step 1 of the memory baselines: calibrate MemRL's dataset-dependent
# similarity constants (delta, sim_norm_mean, sim_norm_std) on a level's
# intent embeddings.  Costs no simulator rollouts and no LLM calls, and runs on
# CPU: a 137M embedder over ~100 short texts does not need a GPU, and the `gpu`
# partition can hand out Maxwell cards that current torch cannot drive at all.
#
# Usage:
#   sbatch baselines/bash_scripts/sbatch_calibrate.sh --level pass_the_parcel
#   sbatch baselines/bash_scripts/sbatch_calibrate.sh --level catapult --seeds 1-105
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

LEVEL=""
SEEDS="1-105"
# Calibrate the key we actually run with (see docs/MEMORY_BASELINES.md).
# Pass a text embedder name to measure that alternative instead.
EMBEDDING_MODEL="jinaai/jina-embeddings-v2-base-en"
INTENT_MODE="geometry"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --level)            LEVEL="$2";            shift 2 ;;
        --seeds)            SEEDS="$2";            shift 2 ;;
        --embedding-model)  EMBEDDING_MODEL="$2";  shift 2 ;;
        --intent-mode)      INTENT_MODE="$2";      shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -n "$LEVEL" ]] || { echo "[ERROR] --level is required"; exit 1; }

# sbatch runs a *copy* of this script from the Slurm spool dir, so the repo
# cannot be located relative to $BASH_SOURCE. Anchor it explicitly.
REPO_DIR="${REPO_DIR:-/project/pi_sniekum_umass_edu/vgandhi/HExA_Baseline_Exp/HeXA-Hierarchical-Experimentalist-Agents}"
export REPO_DIR
source "$REPO_DIR/baselines/bash_scripts/_baseline_env.sh"

echo "================================================================"
echo "  MemRL similarity calibration"
echo "  Job       : ${SLURM_JOB_ID:-<interactive>}"
echo "  Level     : $LEVEL"
echo "  Seeds     : $SEEDS"
echo "  Embedder  : $EMBEDDING_MODEL"
echo "  Intent    : $INTENT_MODE"
echo "  Cache dir : $CACHE_DIR"
echo "================================================================"

$PYTHON -m baselines.calibrate \
    --level "$LEVEL" \
    --seeds "$SEEDS" \
    --embedding-model "$EMBEDDING_MODEL" \
    --intent-mode "$INTENT_MODE" \
    --cache-dir "$CACHE_DIR"

echo "Done. Calibration written under $CACHE_DIR"
