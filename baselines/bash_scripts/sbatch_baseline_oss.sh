#!/bin/bash
#SBATCH --job-name=baseline_oss
#SBATCH --partition=superpod-a100
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH -c 16
#SBATCH -t 1-00:00:00
#SBATCH --mem=480G
#SBATCH -A pi_sniekum_umass_edu
#SBATCH -o logs/baseline_oss_%j.out
#SBATCH -e logs/baseline_oss_%j.err
#
# Episodic RAG / MemRL with gpt-oss-120b on 4x A100 80GB, mirroring
# react_agent/bash_scripts/run_gpt_oss_4xa100.sh.
#
# 117B * 2 bytes ~ 234GB BF16; 4x A100 80GB = 320GB VRAM, hence 480G host RAM.
# The OSS path in load_qwen_model applies: OSS_FORMAT_ADDENDUM in the system
# prompt, get_oss_tools() native function-calling via apply_chat_template,
# <|call|> as an EOS token, and _parse_oss_native to map harmony channels back to
# ReAct form. Our runners detect it from the model path (`"gpt-oss" in model`).
#
# Usage:
#   sbatch baselines/bash_scripts/sbatch_baseline_oss.sh \
#       --method memrl --level catapult \
#       --initial-traj-dir Initial_trajectories/catapult
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

METHOD=""
LEVEL=""
SEEDS="6-55"
SEEDS_PER_ROUND=3
MAX_ITERATIONS=25
MAX_NEW_TOKENS=512      # the OSS path caps generation internally too
INITIAL_TRAJ_DIR=""
EMBEDDING_MODEL=""      # empty -> each runner's own default (MemRL openai, RAG jina)
ACTOR_LABEL="gptoss120b"
TAG=""
EXTRA=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)            METHOD="$2";            shift 2 ;;
        --level)             LEVEL="$2";             shift 2 ;;
        --seeds)             SEEDS="$2";             shift 2 ;;
        --seeds-per-round)   SEEDS_PER_ROUND="$2";   shift 2 ;;
        --max-iterations)    MAX_ITERATIONS="$2";    shift 2 ;;
        --max-new-tokens)    MAX_NEW_TOKENS="$2";    shift 2 ;;
        --initial-traj-dir)  INITIAL_TRAJ_DIR="$2";  shift 2 ;;
        --embedding-model)   EMBEDDING_MODEL="$2";   shift 2 ;;
        --model)             OSS_OVERRIDE="$2";      shift 2 ;;
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

REPO_DIR="${REPO_DIR:-/project/pi_sniekum_umass_edu/vgandhi/HExA_Baseline_Exp/HeXA-Hierarchical-Experimentalist-Agents}"
export REPO_DIR
source "$REPO_DIR/baselines/bash_scripts/_baseline_env.sh"

ACTOR="${OSS_OVERRIDE:-$GPT_OSS_PATH}"
[[ -r "$ACTOR/config.json" ]] || { echo "[ERROR] model not readable: $ACTOR"; exit 1; }

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# gpt-oss needs the harmony/tiktoken vocabs; fetching them at runtime fails on
# compute nodes without egress, so cache them in-repo once (same fix as
# react_agent/bash_scripts/run_gpt_oss_4xa100.sh).
mkdir -p "$REPO_DIR/tiktoken_encodings"
for enc in o200k_base cl100k_base; do
    if [ ! -f "$REPO_DIR/tiktoken_encodings/$enc.tiktoken" ]; then
        wget -q -O "$REPO_DIR/tiktoken_encodings/$enc.tiktoken" \
            "https://openaipublic.blob.core.windows.net/encodings/$enc.tiktoken" || true
    fi
done
export TIKTOKEN_ENCODINGS_BASE="$REPO_DIR/tiktoken_encodings"

RUN_NAME="${METHOD}_${LEVEL}_${ACTOR_LABEL}${TAG:+_$TAG}"
OUTPUT_DIR="$BASELINE_ROOT/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"

echo "================================================================"
echo "  Memory baseline — gpt-oss-120b on 4x A100"
echo "  Job          : ${SLURM_JOB_ID:-<interactive>}"
echo "  Method       : $METHOD"
echo "  Level        : $LEVEL"
echo "  Seeds        : $SEEDS  (rounds of $SEEDS_PER_ROUND)"
echo "  Actor        : $ACTOR"
echo "  Embedder     : ${EMBEDDING_MODEL:-<runner default>}"
echo "  Offline init : ${INITIAL_TRAJ_DIR:-<none>}"
echo "  Output       : $OUTPUT_DIR"
echo "================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

args=(
    --level "$LEVEL"
    --seeds "$SEEDS"
    --seeds-per-round "$SEEDS_PER_ROUND"
    --max-iterations "$MAX_ITERATIONS"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --output-dir "$OUTPUT_DIR"
    --cache-dir "$CACHE_DIR"
    --model "$ACTOR"
    --verbose
)
[[ -n "$INITIAL_TRAJ_DIR" ]] && args+=(--initial-traj-dir "$INITIAL_TRAJ_DIR")
[[ -n "$EMBEDDING_MODEL" ]] && args+=(--embedding-model "$EMBEDDING_MODEL")
# MemRL summarizes with the same local model, honouring its 0.0/0.3 temperatures.
[[ "$METHOD" == "memrl" ]] && args+=(--summarizer agent-model)

# shellcheck disable=SC2206
[[ -n "$EXTRA" ]] && args+=(${EXTRA})

if [[ "$METHOD" == "memrl" ]]; then
    $PYTHON -m baselines.runner.run_memrl "${args[@]}"
else
    $PYTHON -m baselines.runner.run_episodic_rag "${args[@]}"
fi

echo "Done. Results in $OUTPUT_DIR"
