#!/bin/bash
# ─── Direct-answer baseline for all OSS models × 8 levels ────────────────────
#
# Mirrors the ReAct table columns: Q-7B, Q-14B, O-20B, all 100 seeds per level.
#
# Qwen 7B and 14B use run_direct_answer_oss.py (true direct-answer: scripted
# get_level_state peek + ONE model call, same prompts as Claude direct baseline).
# GPT-OSS-20B falls back to run_react.py --max-iterations 2.
#
# Usage (interactive / local):
#   MODEL=7b  bash run_direct_baseline_all_models.sh
#   MODEL=14b bash run_direct_baseline_all_models.sh
#   MODEL=20b bash run_direct_baseline_all_models.sh
#   MODEL=all bash run_direct_baseline_all_models.sh   # runs sequentially (needs shared GPU)
#
# Or set paths via env before calling:
#   QWEN7B_PATH=/path/to/Qwen2.5-7B-Instruct  MODEL=7b  bash ...
#
# SLURM: submit one job per model (different GPU requirements):
#   sbatch --gres=gpu:1 --constraint=vram32 --mem=20G  -t 48:00:00 --wrap="MODEL=7b  bash run_direct_baseline_all_models.sh"
#   sbatch --gres=gpu:1 --constraint=vram48 --mem=64G  -t 48:00:00 --wrap="MODEL=14b bash run_direct_baseline_all_models.sh"
#   sbatch --gres=gpu:a100:1              --mem=80G  -t 48:00:00 --wrap="MODEL=20b bash run_direct_baseline_all_models.sh"

set -Eeuo pipefail

# ─── Configurable ────────────────────────────────────────────────────────────
MODEL="${MODEL:-all}"                          # 7b | 14b | 20b | all

QWEN7B_PATH="${QWEN7B_PATH:-/datasets/ai/qwen2/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28/}"
QWEN14B_PATH="${QWEN14B_PATH:-/datasets/ai/qwen2/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8/}"
GPT_OSS_20B_PATH="${GPT_OSS_20B_PATH:-/datasets/ai/gpt/hub/models--openai--gpt-oss-20b/snapshots/2e8f8052ee2aeee907f76e08c08b9fdde8677ca8/}"

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-phyreagent}"
HF_HOME_DIR="${HF_HOME_DIR:-${SCRATCH:-$HOME}/hf_cache}"

SEED_START=0
SEED_END=99        # 100 seeds to match the ReAct table
MAX_NEW_TOKENS=700
TEMPERATURE=0.3
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

# Map MODEL tag → (path, short_name, max_new_tokens)
case "$MODEL" in
    7b)   MODELS=("$QWEN7B_PATH:qwen7b:$MAX_NEW_TOKENS") ;;
    14b)  MODELS=("$QWEN14B_PATH:qwen14b:$MAX_NEW_TOKENS") ;;
    20b)  MODELS=("$GPT_OSS_20B_PATH:gpt_oss_20b:$MAX_NEW_TOKENS") ;;
    all)  MODELS=(
            "$QWEN7B_PATH:qwen7b:$MAX_NEW_TOKENS"
            "$QWEN14B_PATH:qwen14b:$MAX_NEW_TOKENS"
            "$GPT_OSS_20B_PATH:gpt_oss_20b:$MAX_NEW_TOKENS"
          ) ;;
    *)    echo "[ERROR] Unknown MODEL=$MODEL. Use: 7b | 14b | 20b | all"; exit 1 ;;
esac

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
echo "  Direct Baseline — All 8 Levels"
echo "  Model tag : $MODEL"
echo "  Seeds     : $SEED_START .. $SEED_END"
echo "  Repo      : $REPO_DIR"
echo "  Started   : $(date)"
echo "================================================================"

# ─── Main loop ───────────────────────────────────────────────────────────────
for MODEL_SPEC in "${MODELS[@]}"; do
    MODEL_PATH="${MODEL_SPEC%%:*}"
    REST="${MODEL_SPEC#*:}"
    MODEL_TAG="${REST%%:*}"
    MODEL_TOKENS="${REST#*:}"

    echo ""
    echo "################################################################"
    echo "  Model : $MODEL_TAG  ($MODEL_PATH)"
    echo "################################################################"

    # Skip models whose path is still the placeholder.
    if [[ "$MODEL_PATH" == *"/path/to/"* ]]; then
        echo "  [SKIP] $MODEL_TAG — placeholder path not set. Export ${MODEL_TAG^^}_PATH or edit this script."
        continue
    fi
    if [[ ! -e "$MODEL_PATH" ]]; then
        echo "  [SKIP] $MODEL_TAG — path does not exist: $MODEL_PATH"
        continue
    fi

    for LEVEL in "${ALL_LEVELS[@]}"; do
        EVAL_DIR="eval_results_direct_${LEVEL}_${MODEL_TAG}"

        echo ""
        echo "  ── Level: $LEVEL  →  $EVAL_DIR"

        # All three models use the true direct-answer baseline:
        # scripted get_level_state peek + ONE model call with DIRECT_SYSTEM_TEMPLATE.
        python -m react_agent.run_direct_answer_oss \
            --model        "$MODEL_PATH" \
            --model-tag    "$MODEL_TAG" \
            --level        "$LEVEL" \
            --variant      cot \
            --seeds        $(seq $SEED_START $SEED_END) \
            --temperature  "$TEMPERATURE" \
            --max-new-tokens "$MODEL_TOKENS" \
            --eval-dir     "$EVAL_DIR" \
            --resume \
            --verbose

        # Optional: generate plots (soft-fail if plot script missing)
        python react_agent/plot_eval.py --eval-dir "$EVAL_DIR" 2>/dev/null || true

        echo "  ── Done: $LEVEL"
    done

    echo ""
    echo "  All 8 levels done for $MODEL_TAG at $(date)"
done

echo ""
echo "================================================================"
echo "  ALL DONE at $(date)"
echo "================================================================"
