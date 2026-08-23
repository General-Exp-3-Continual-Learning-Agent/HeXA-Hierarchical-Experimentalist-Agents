#!/bin/bash
# ─── Direct-answer baseline for GPT-OSS-120B × 8 levels (INTERACTIVE) ─────────
#
# Runs the true direct-answer baseline (scripted get_level_state peek + ONE
# forced finish() call, NO ReAct loop) for gpt-oss-120b on all 8 Interphyre
# levels, to produce the O-120B reference column.
#
# This is for an INTERACTIVE session with GPUs already allocated — NOT sbatch.
# gpt-oss-120b is too large for one GPU; device_map="auto" in the runner shards
# it across every visible GPU, so expose all 4 via CUDA_VISIBLE_DEVICES.
#
# Usage (from an interactive node with 4 GPUs):
#   bash run_direct_baseline_gpt_oss_120b.sh
#
# Override anything via env:
#   CUDA_VISIBLE_DEVICES=0,1,2,3  SEED_END=99  bash run_direct_baseline_gpt_oss_120b.sh

set -Eeuo pipefail

# ─── Configurable ────────────────────────────────────────────────────────────
GPT_OSS_120B_PATH="${GPT_OSS_120B_PATH:-/datasets/ai/gpt/hub/models--openai--gpt-oss-120b/snapshots/eabf0c518da7584a2e7dab4ab272709785a72126/}"
MODEL_TAG="${MODEL_TAG:-gpt_oss_120b}"

# Shard across all 4 interactive GPUs (device_map="auto" needs them visible).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-phyreagent}"
HF_HOME_DIR="${HF_HOME_DIR:-${SCRATCH:-$HOME}/hf_cache}"

SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-99}"        # 100 seeds to match the reference table
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-700}"
TEMPERATURE="${TEMPERATURE:-0.3}"
# ─────────────────────────────────────────────────────────────────────────────

ALL_LEVELS=(
    down_to_earth
    two_body_problem
    falling_into_place
    basket_case
    tipping_point
    cliffhanger
    pass_the_parcel
    catapult
)

# ─── Environment setup ───────────────────────────────────────────────────────
module load conda/latest 2>/dev/null || true
conda activate "$CONDA_ENV" 2>/dev/null || source activate "$CONDA_ENV" 2>/dev/null || true

export HF_HOME="$HF_HOME_DIR"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO_DIR/interphyre:$REPO_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"

echo "================================================================"
echo "  Direct Baseline — GPT-OSS-120B — All 8 Levels (interactive)"
echo "  Model    : $MODEL_TAG  ($GPT_OSS_120B_PATH)"
echo "  GPUs     : $CUDA_VISIBLE_DEVICES"
echo "  Seeds    : $SEED_START .. $SEED_END"
echo "  Repo     : $REPO_DIR"
echo "  Started  : $(date)"
echo "================================================================"

if [[ ! -e "$GPT_OSS_120B_PATH" ]]; then
    echo "[ERROR] Model path does not exist: $GPT_OSS_120B_PATH"
    exit 1
fi

# ─── Main loop ───────────────────────────────────────────────────────────────
for LEVEL in "${ALL_LEVELS[@]}"; do
    EVAL_DIR="eval_results_direct_${LEVEL}_${MODEL_TAG}"

    echo ""
    echo "  ── Level: $LEVEL  →  $EVAL_DIR"

    python -m react_agent.run_direct_answer_oss \
        --model        "$GPT_OSS_120B_PATH" \
        --model-tag    "$MODEL_TAG" \
        --level        "$LEVEL" \
        --variant      cot \
        --seeds        $(seq $SEED_START $SEED_END) \
        --temperature  "$TEMPERATURE" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --eval-dir     "$EVAL_DIR" \
        --resume \
        --verbose

    python react_agent/plot_eval.py --eval-dir "$EVAL_DIR" 2>/dev/null || true

    echo "  ── Done: $LEVEL"
done

echo ""
echo "================================================================"
echo "  ALL 8 LEVELS DONE for $MODEL_TAG at $(date)"
echo "================================================================"
