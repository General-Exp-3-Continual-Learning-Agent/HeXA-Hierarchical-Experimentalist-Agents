#!/bin/bash
#SBATCH -G 1
#SBATCH --partition=gpu
#SBATCH -c 2               
#SBATCH -t 10:00:00
#SBATCH --mem 20G           
#SBATCH --constraint=vram32

# Training script for OpenVLA on Interphyre dataset

export HF_HOME="${HF_HOME:-/path/to/hf_cache}"

module load conda/latest
export QWEN_PATH="${QWEN_PATH:-/path/to/models--Qwen--Qwen2.5-7B-Instruct/snapshots/<snapshot>/}"


# Set working directory
cd "${REPO_DIR:-/path/to/HeXA}"
# Run training with reduced batch size for memory efficiency # batch_size 16 -> 4, accumulation_steps 1 -> 4

# Run training on successes only
EVAL_DIR="eval_results_tippingpoint_qwen2.5_7b"

python -m react_agent.run_react \
    --level tipping_point\
    --model "$QWEN_PATH" \
    --seeds $(seq 0 50) \
    --max-iterations 25 \
    --eval-dir $EVAL_DIR \
    --verbose

python react_agent/plot_eval.py --eval-dir $EVAL_DIR


#sbatch -A <your-slurm-account> run.sh