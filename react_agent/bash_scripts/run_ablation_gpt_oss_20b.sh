#!/bin/bash
#SBATCH --job-name=gpt_oss_20b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1          # Single A100 80GB is enough for GPT-OSS-20B
#SBATCH -c 8
#SBATCH -t 48:00:00
#SBATCH --mem=80G
#SBATCH -o /home/udhanuka_umass_edu/gpt_oss_logs/gpt_oss_20b_%j.out
#SBATCH -e /home/udhanuka_umass_edu/gpt_oss_logs/gpt_oss_20b_%j.err
#SBATCH -A pi_hongyu_umass_edu

# ─── User-configurable ───────────────────────────────────────────────────────
# HuggingFace model path for GPT-OSS-20B
GPT_OSS_HF_PATH="${GPT_OSS_HF_PATH:-/datasets/ai/gpt/hub/models--openai--gpt-oss-20b/snapshots/2e8f8052ee2aeee907f76e08c08b9fdde8677ca8/}"

REPO_DIR="/home/udhanuka_umass_edu/RL Med/physics-reasoning-agents"
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