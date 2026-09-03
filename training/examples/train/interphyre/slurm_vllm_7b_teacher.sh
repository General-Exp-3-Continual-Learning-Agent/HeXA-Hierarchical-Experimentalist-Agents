#!/bin/bash
#SBATCH --job-name=vllm_7b_teacher
#SBATCH --partition=superpod-a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH -t 12:00:00
#SBATCH --mem=48G
#SBATCH -o logs/vllm_teacher_%j.out
#SBATCH -e logs/vllm_teacher_%j.err
#SBATCH -A pi_sniekum_umass_edu

set -euo pipefail

PROJECT_DIR="/project/pi_sniekum_umass_edu/vgandhi"
VERL_TOOL_DIR="$PROJECT_DIR/HeXA-Hierarchical-Experimentalist-Agents/training"
export PYTHONPATH="$(dirname "$VERL_TOOL_DIR")/interphyre:${PYTHONPATH:-}"   # makes repo-level interphyre package importable
CONDA_ENV="$PROJECT_DIR/conda/envs/VerlToolInterphyre"

MODEL_PATH="/scratch4/workspace/svaidyanatha_umass_edu-phyre/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
PORT="${VLLM_PORT:-8000}"

module load conda/latest
module load cuda/12.8
export PATH="$CONDA_ENV/bin:${PATH:-}"
export HF_HOME="/scratch4/workspace/svaidyanatha_umass_edu-phyre/hf_cache"

cd "$VERL_TOOL_DIR"
mkdir -p logs

echo "================================================================"
echo "  Qwen 2.5 7B Instruct — vLLM teacher server"
echo "  Node     : $SLURMD_NODENAME"
echo "  Port     : $PORT"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Model    : $MODEL_PATH"
echo "================================================================"

# Write the endpoint to a file so the orchestrator can read it
echo "http://${SLURMD_NODENAME}:${PORT}" > "$VERL_TOOL_DIR/logs/vllm_teacher_endpoint.txt"
echo "Endpoint written to logs/vllm_teacher_endpoint.txt"

vllm serve "$MODEL_PATH" \
    --port "$PORT" \
    --gpu-memory-utilization 0.5 \
    --max-model-len 8192 \
    --served-model-name "Qwen/Qwen2.5-7B-Instruct"
