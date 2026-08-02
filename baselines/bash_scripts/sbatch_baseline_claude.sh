#!/bin/bash
#SBATCH --job-name=baseline_claude
#SBATCH --partition=cpu
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH --mem 32G
#SBATCH -A pi_sniekum_umass_edu
#SBATCH -o logs/baseline_claude_%j.out
#SBATCH -e logs/baseline_claude_%j.err
#
# Episodic RAG / MemRL with the Claude Sonnet actor (via the `claude -p` CLI),
# matching HeXA's headline configuration so the numbers are comparable with the
# published ReAct / Reflexion / Direct / HeXA results.
#
# CPU-only: the actor is an outbound `claude -p` call and the jina embedder runs
# fine on CPU, so this job never needs a GPU.
#
# Requires the `claude` CLI to be on PATH and authenticated on compute nodes.
#
# Run the calibration job for the level FIRST (MemRL only):
#   sbatch baselines/bash_scripts/sbatch_calibrate.sh --level pass_the_parcel
#
# Usage:
#   sbatch baselines/bash_scripts/sbatch_baseline_claude.sh \
#       --method episodic_rag --level pass_the_parcel \
#       --initial-traj-dir Initial_trajectories/pass_the_parcel
#
#   sbatch baselines/bash_scripts/sbatch_baseline_claude.sh \
#       --method memrl --level catapult \
#       --initial-traj-dir Initial_trajectories/catapult
#
# Runs resume: re-submitting skips seeds whose trajectory JSON already exists.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

METHOD=""
LEVEL=""
SEEDS="6-55"
SEEDS_PER_ROUND=3
MAX_ITERATIONS=25
INITIAL_TRAJ_DIR=""
# Retrieval key. `numeric` (standardized scene geometry) is the default for the
# reason measured in docs/MEMORY_BASELINES.md: text embeddings rank InterPhyre
# instances near-randomly (Spearman rho +0.11 to +0.40, ~10% top-1 overlap), so a
# text key would make both baselines straw men. Pass a model name here only to
# run that weaker configuration deliberately as a secondary comparison.
EMBEDDING_MODEL="jinaai/jina-embeddings-v2-base-en"
INTENT_MODE="geometry"
TEACHER_MODEL="claude-sonnet-4-6"
TAG=""
EXTRA=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)            METHOD="$2";            shift 2 ;;
        --level)             LEVEL="$2";             shift 2 ;;
        --seeds)             SEEDS="$2";             shift 2 ;;
        --seeds-per-round)   SEEDS_PER_ROUND="$2";   shift 2 ;;
        --max-iterations)    MAX_ITERATIONS="$2";    shift 2 ;;
        --initial-traj-dir)  INITIAL_TRAJ_DIR="$2";  shift 2 ;;
        --embedding-model)   EMBEDDING_MODEL="$2";   shift 2 ;;
        --intent-mode)       INTENT_MODE="$2";       shift 2 ;;
        --teacher-model)     TEACHER_MODEL="$2";     shift 2 ;;
        --tag)               TAG="$2";               shift 2 ;;
        --extra)             EXTRA="$2";             shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -n "$METHOD" && -n "$LEVEL" ]] || { echo "[ERROR] --method and --level are required"; exit 1; }
case "$METHOD" in
    episodic_rag|memrl) ;;
    *) echo "[ERROR] --method must be episodic_rag or memrl"; exit 1 ;;
esac

# sbatch runs a *copy* of this script from the Slurm spool dir, so the repo
# cannot be located relative to $BASH_SOURCE. Anchor it explicitly.
REPO_DIR="${REPO_DIR:-/project/pi_sniekum_umass_edu/vgandhi/HExA_Baseline_Exp/HeXA-Hierarchical-Experimentalist-Agents}"
export REPO_DIR
source "$REPO_DIR/baselines/bash_scripts/_baseline_env.sh"

command -v claude >/dev/null || {
    echo "[ERROR] 'claude' CLI not found on PATH for this node."
    exit 1
}

RUN_NAME="${METHOD}_${LEVEL}_claude${TAG:+_$TAG}"
OUTPUT_DIR="$BASELINE_ROOT/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"

echo "================================================================"
echo "  Memory baseline (Claude actor)"
echo "  Job          : ${SLURM_JOB_ID:-<interactive>}"
echo "  Method       : $METHOD"
echo "  Level        : $LEVEL"
echo "  Seeds        : $SEEDS  (rounds of $SEEDS_PER_ROUND)"
echo "  Actor        : claude ($(claude --version 2>/dev/null || echo '?'))"
echo "  Embedder     : $EMBEDDING_MODEL"
echo "  Intent mode  : $INTENT_MODE"
echo "  Offline init : ${INITIAL_TRAJ_DIR:-<none>}"
echo "  Output       : $OUTPUT_DIR"
echo "  Cache        : $CACHE_DIR"
echo "================================================================"

args=(
    --level "$LEVEL"
    --seeds "$SEEDS"
    --seeds-per-round "$SEEDS_PER_ROUND"
    --max-iterations "$MAX_ITERATIONS"
    --output-dir "$OUTPUT_DIR"
    --cache-dir "$CACHE_DIR"
    --model claude
    --embedding-model "$EMBEDDING_MODEL"
    --intent-mode "$INTENT_MODE"
    --verbose
)
[[ -n "$INITIAL_TRAJ_DIR" ]] && args+=(--initial-traj-dir "$INITIAL_TRAJ_DIR")
[[ "$METHOD" == "memrl" ]] && args+=(--summarizer claude --teacher-model "$TEACHER_MODEL")

# shellcheck disable=SC2206
[[ -n "$EXTRA" ]] && args+=(${EXTRA})

if [[ "$METHOD" == "memrl" ]]; then
    $PYTHON -m baselines.runner.run_memrl "${args[@]}"
else
    $PYTHON -m baselines.runner.run_episodic_rag "${args[@]}"
fi

echo "Done. Results in $OUTPUT_DIR"
