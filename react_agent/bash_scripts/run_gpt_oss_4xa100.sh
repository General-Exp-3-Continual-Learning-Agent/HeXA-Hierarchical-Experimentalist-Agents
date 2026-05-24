#!/bin/bash
#SBATCH --job-name=gpt_oss_120b_catapult
#SBATCH --partition=superpod-a100
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH -c 16
#SBATCH -t 1-00:00:00
#SBATCH --mem=480G
#SBATCH -o logs/gpt_oss_120b_catapult_%j.out
#SBATCH -e logs/gpt_oss_120b_catapult_%j.err
#SBATCH -A pi_sniekum_umass_edu

# ─── gpt-oss-120b on 4x A100 80GB via HF direct (load_qwen_model OSS path) ─
# Mirrors the working 20b interactive command, just bigger model + 4 GPUs.
# Routing: run_react.py → load_qwen_model(is_oss=True) which applies:
#   - OSS_FORMAT_ADDENDUM into system prompt
#   - get_oss_tools(level) into apply_chat_template (native function-calling)
#   - <|call|> as EOS so the model stops after a tool call
#   - _parse_oss_native to convert harmony channels → ReAct format
# 117B * 2 bytes ≈ 234GB BF16; 4x A100 80GB = 320GB VRAM + overhead → 480G RAM.

GPT_OSS_HF_PATH="${GPT_OSS_HF_PATH:-/datasets/ai/gpt/hub/models--openai--gpt-oss-120b/snapshots/eabf0c518da7584a2e7dab4ab272709785a72126/}"
REPO_DIR="/home/udhanuka_umass_edu/RL Med/physics-reasoning-agents"
CONDA_ENV="/home/udhanuka_umass_edu/.conda/envs/phyreagent"
HF_HOME_DIR="${HF_HOME_DIR:-/scratch4/workspace/svaidyanatha_umass_edu-phyre/hf_cache}"

# ─── Configurable ───────────────────────────────────────────────────────────
LEVEL="${LEVEL:-falling_into_place}"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-100}"
MAX_ITER="${MAX_ITER:-25}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
EVAL_DIR="${EVAL_DIR:-eval_results_gpt_oss_120b_${LEVEL}}"
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "  GPT-OSS-120B on 4x A100 80GB (HF direct, OSS-aware path)"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURMD_NODENAME"
echo "  Model  : $GPT_OSS_HF_PATH"
echo "  GPUs   : $CUDA_VISIBLE_DEVICES"
echo "  Level  : $LEVEL"
echo "  Seeds  : $SEED_START to $SEED_END"
echo "================================================================"

module load conda/latest
conda activate "$CONDA_ENV"

export HF_HOME="$HF_HOME_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fix tiktoken/harmony vocab download issue (known gpt-oss bug)
mkdir -p "$REPO_DIR/tiktoken_encodings"
if [ ! -f "$REPO_DIR/tiktoken_encodings/o200k_base.tiktoken" ]; then
    wget -q -O "$REPO_DIR/tiktoken_encodings/o200k_base.tiktoken" \
        "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
    wget -q -O "$REPO_DIR/tiktoken_encodings/cl100k_base.tiktoken" \
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
fi
export TIKTOKEN_ENCODINGS_BASE="$REPO_DIR/tiktoken_encodings"

mkdir -p "$REPO_DIR/logs"
cd "$REPO_DIR"

echo "[$(date)] Starting ReAct eval: level=$LEVEL seeds=$SEED_START..$SEED_END"

python -m react_agent.run_react \
    --model "$GPT_OSS_HF_PATH" \
    --level "$LEVEL" \
    --seeds $(seq $SEED_START $SEED_END) \
    --max-iterations "$MAX_ITER" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --eval-dir "$EVAL_DIR" \
    --verbose

python react_agent/plot_eval.py --eval-dir "$EVAL_DIR" 2>/dev/null || true

echo "[$(date)] Done. Results in $EVAL_DIR/"
