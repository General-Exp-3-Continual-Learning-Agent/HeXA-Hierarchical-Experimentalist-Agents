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

# Paths: every one can be overridden from the environment.
VERL_TOOL_DIR="${VERL_TOOL_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"   # this training/ dir; sbatch from here
CKPT_ROOT="${CKPT_ROOT:-$VERL_TOOL_DIR/checkpoints}"
export PYTHONPATH="$(dirname "$VERL_TOOL_DIR")/interphyre:${PYTHONPATH:-}"   # makes repo-level interphyre package importable
CONDA_ENV="${CONDA_ENV:?set CONDA_ENV to the VerlToolInterphyre conda env path}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${VLLM_PORT:-8000}"

module load conda/latest
module load cuda/12.8
export PATH="$CONDA_ENV/bin:${PATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

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
