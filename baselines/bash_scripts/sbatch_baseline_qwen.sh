#!/bin/bash
#SBATCH --job-name=baseline_qwen
#SBATCH --partition=superpod-a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH --mem=64G
# superpod-a100 is uniformly A100-80GB, so no --constraint is needed and there is
# no risk of landing on the sm_52 Maxwell cards that sit in the general `gpu`
# partition (torch ships no kernels for those; a job there dies at the first
# tensor op). It is also far less contended: submitting into `gpu` with a 24h
# walltime sat at Reason=Priority behind ~210 queued jobs.
#SBATCH -A pi_sniekum_umass_edu
#SBATCH -o logs/baseline_qwen_%j.out
#SBATCH -e logs/baseline_qwen_%j.err
#
# Episodic RAG / MemRL with a local Qwen2.5-7B-Instruct actor on one GPU.
# The same model serves as MemRL's memory-building/reflection summarizer, which
# also means MemRL's configured temperatures (0.0 script / 0.3 reflection) are
# honoured — unlike the `claude -p` backend, which has no temperature flag.
#
# Run the calibration job for the level FIRST (MemRL only):
#   sbatch baselines/bash_scripts/sbatch_calibrate.sh --level pass_the_parcel
#
# Usage:
#   sbatch baselines/bash_scripts/sbatch_baseline_qwen.sh \
#       --method memrl --level pass_the_parcel \
#       --initial-traj-dir Initial_trajectories/pass_the_parcel
#
#   sbatch baselines/bash_scripts/sbatch_baseline_qwen.sh \
#       --method episodic_rag --level catapult \
#       --initial-traj-dir Initial_trajectories/catapult
#
#   # runtime-RL ablation: Q-blind retrieval
#   sbatch baselines/bash_scripts/sbatch_baseline_qwen.sh \
#       --method memrl --level catapult --extra "--lambda-q 0" --tag lambda0
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
# Empty -> let each runner apply its own default (MemRL: OpenAI
# text-embedding-3-large, its config's choice; Episodic RAG: jina, the CORE
# paper's choice for that baseline). Set this to force one key for both.
EMBEDDING_MODEL=""
INTENT_MODE="geometry"
ACTOR_LABEL=""
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
        --model)             QWEN_OVERRIDE="$2";     shift 2 ;;
        --actor-label)       ACTOR_LABEL="$2";       shift 2 ;;
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
ACTOR="${QWEN_OVERRIDE:-$QWEN_PATH}"

# Derive a short actor label from the model path when not given, so 7B and 3B
# runs never share an output directory (their stores are not interchangeable).
if [[ -z "$ACTOR_LABEL" ]]; then
    case "$ACTOR" in
        *3B*|*3b*) ACTOR_LABEL="qwen3b" ;;
        *7B*|*7b*) ACTOR_LABEL="qwen7b" ;;
        *gpt-oss-120b*) ACTOR_LABEL="gptoss120b" ;;
        *) ACTOR_LABEL="$(basename "$ACTOR" | tr -cd '[:alnum:]-' | cut -c1-20)" ;;
    esac
fi
RUN_NAME="${METHOD}_${LEVEL}_${ACTOR_LABEL}${TAG:+_$TAG}"
OUTPUT_DIR="$BASELINE_ROOT/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"

echo "================================================================"
echo "  Memory baseline (local Qwen actor)"
echo "  Job          : ${SLURM_JOB_ID:-<interactive>}"
echo "  Method       : $METHOD"
echo "  Level        : $LEVEL"
echo "  Seeds        : $SEEDS  (rounds of $SEEDS_PER_ROUND)"
echo "  Actor        : $ACTOR"
echo "  Actor label  : $ACTOR_LABEL"
echo "  Embedder     : ${EMBEDDING_MODEL:-<runner default>}"
echo "  Intent mode  : $INTENT_MODE"
echo "  Offline init : ${INITIAL_TRAJ_DIR:-<none>}"
echo "  Output       : $OUTPUT_DIR"
echo "  Cache        : $CACHE_DIR"
echo "  HF_HOME      : $HF_HOME"
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
[[ -n "$INITIAL_TRAJ_DIR" ]] && args+=(--initial-traj-dir "$INITIAL_TRAJ_DIR")
[[ -n "$EMBEDDING_MODEL" ]] && args+=(--embedding-model "$EMBEDDING_MODEL")
# MemRL summarizes with the same local model, at the paper's temperatures.
[[ "$METHOD" == "memrl" ]] && args+=(--summarizer agent-model)

# shellcheck disable=SC2206
[[ -n "$EXTRA" ]] && args+=(${EXTRA})

if [[ "$METHOD" == "memrl" ]]; then
    $PYTHON -m baselines.runner.run_memrl "${args[@]}"
else
    $PYTHON -m baselines.runner.run_episodic_rag "${args[@]}"
fi

echo "Done. Results in $OUTPUT_DIR"
