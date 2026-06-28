#!/bin/bash
#SBATCH --job-name=interphyre_eval
#SBATCH --partition=superpod-a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH -t 4:00:00
#SBATCH --mem=32G
#SBATCH -o logs/eval_%j.out
#SBATCH -e logs/eval_%j.err
#SBATCH -A pi_sniekum_umass_edu
#SBATCH --reservation=sniekum

set -euo pipefail

PROJECT_DIR="/project/pi_sniekum_umass_edu/vgandhi"
VERL_TOOL_DIR="$PROJECT_DIR/HeXA-Hierarchical-Experimentalist-Agents/training"
export PYTHONPATH="$(dirname "$VERL_TOOL_DIR")/interphyre:${PYTHONPATH:-}"   # makes repo-level interphyre package importable
CONDA_ENV="$PROJECT_DIR/conda/envs/VerlToolInterphyre"
PYTHON="$CONDA_ENV/bin/python"

CHECKPOINT="/scratch4/workspace/svaidyanatha_umass_edu-phyre/checkpoints/interphyre/interphyre-a100-1gpu-qwen_qwen2.5-3b-instruct-grpo-n4-b1-t1.0-lr1e-6-skilled-dte/run35/global_step_50/actor/huggingface"

module load conda/latest
module load cuda/12.8
export PATH="$CONDA_ENV/bin:${PATH:-}"

cd "$VERL_TOOL_DIR"
mkdir -p logs/eval

OUTPUT="$VERL_TOOL_DIR/logs/eval/dte_run35_step50_skilled.jsonl"
echo "Eval output: $OUTPUT"

echo "================================================================"
echo "  Interphyre Eval — run35 step 50 (epoch 1) [SKILLED], down_to_earth, seeds 51-100"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Checkpoint: $CHECKPOINT"
echo "================================================================"

$PYTHON examples/eval/interphyre_eval.py \
    --checkpoint "$CHECKPOINT" \
    --level down_to_earth \
    --seed_start 51 \
    --num_seeds 50 \
    --max_turns 25 \
    --skill_bank SKILL-RL/Skill_banks/skill_bank_evolving_DTE.json \
    --output "$OUTPUT"

echo "Done. Results in $OUTPUT"
