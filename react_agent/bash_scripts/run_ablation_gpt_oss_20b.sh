#!/bin/bash
#SBATCH --job-name=gpt_oss_20b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1          # Single A100 80GB is enough for GPT-OSS-20B
#SBATCH -c 8
#SBATCH -t 48:00:00
#SBATCH --mem=80G
#SBATCH -o logs/gpt_oss_20b_%j.out
#SBATCH -e logs/gpt_oss_20b_%j.err
#SBATCH -A <your-slurm-account>

# ─── User-configurable ───────────────────────────────────────────────────────
# HuggingFace model path for GPT-OSS-20B
GPT_OSS_HF_PATH="${GPT_OSS_HF_PATH:-/path/to/models--openai--gpt-oss-20b/snapshots/<snapshot>/}"

REPO_DIR="${REPO_DIR:-/path/to/HeXA}"
CONDA_ENV="phyreagent"
HF_HOME_DIR="${SCRATCH:-$HOME}/hf_cache"
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "  GPT-OSS-20B Ablation Job (direct HuggingFace, no vLLM)"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURMD_NODENAME"
echo "  Model  : $GPT_OSS_HF_PATH"
echo "  GPUs   : $CUDA_VISIBLE_DEVICES"
echo "================================================================"

module load conda/latest
conda activate "$CONDA_ENV"

export HF_HOME="$HF_HOME_DIR"
# Tell experiment_runner which HF model to load
export VLLM_MODEL_NAME="$GPT_OSS_HF_PATH"

cd "$REPO_DIR"

RUN_ID="gpt_oss_20b_down_to_earth"
echo "[$(date)] Starting ablation run: $RUN_ID"

python -m experiments.run_ablation \
    --config experiments/configs/ablation_gpt_oss_20b.yaml \
    --verbose

echo "[$(date)] Ablation complete. Results in results_ablations/$RUN_ID/"