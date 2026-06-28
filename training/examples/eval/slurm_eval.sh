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
#
# General interphyre eval — replaces the per-run slurm_eval_*.sh scripts.
#
# Usage:
#   sbatch examples/eval/slurm_eval.sh \
#       --checkpoint <path/to/actor/huggingface> \
#       --level <down_to_earth|two_body_problem> \
#       [--skill_bank <path>]   (omit for non-skilled runs) \
#       [--seed_start 51] [--num_seeds 50] [--max_turns 25] \
#       [--output <path>]       (default: logs/eval/eval<N>.jsonl, auto-incremented)
#
# Examples:
#   # non-skilled down_to_earth
#   sbatch examples/eval/slurm_eval.sh \
#       --checkpoint /scratch4/workspace/svaidyanatha_umass_edu-phyre/checkpoints/interphyre/interphyre-a100-1gpu-qwen_qwen2.5-3b-instruct-grpo-n4-b1-t1.0-lr1e-6/run24/global_step_1000/actor/huggingface \
#       --level down_to_earth
#
#   # skilled two_body_problem
#   sbatch examples/eval/slurm_eval.sh \
#       --checkpoint /scratch4/workspace/svaidyanatha_umass_edu-phyre/checkpoints/interphyre/interphyre-a100-1gpu-qwen_qwen2.5-3b-instruct-grpo-n4-b1-t1.0-lr1e-6-skilled-tbp/run34/global_step_50/actor/huggingface \
#       --level two_body_problem \
#       --skill_bank SKILL-RL/Skill_banks/skill_bank_evolving_two_body.json
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────
# Default checkpoint (HF Hub id or local path) and level; override with flags.
CHECKPOINT="vgandhi13/Qwen2.5-3B-Interphyre-Evolving-TwoBodyProblem"
LEVEL="two_body_problem"
SKILL_BANK=""
SEED_START=51
NUM_SEEDS=50
MAX_TURNS=25
OUTPUT=""

# ─── Parse args ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)  CHECKPOINT="$2"; shift 2 ;;
        --level)       LEVEL="$2";      shift 2 ;;
        --skill_bank)  SKILL_BANK="$2"; shift 2 ;;
        --seed_start)  SEED_START="$2"; shift 2 ;;
        --num_seeds)   NUM_SEEDS="$2";  shift 2 ;;
        --max_turns)   MAX_TURNS="$2";  shift 2 ;;
        --output)      OUTPUT="$2";     shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$CHECKPOINT" || -z "$LEVEL" ]]; then
    echo "[ERROR] --checkpoint and --level are required."
    echo "        See header of this script for usage."
    exit 1
fi

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_DIR="/project/pi_sniekum_umass_edu/vgandhi"
VERL_TOOL_DIR="$PROJECT_DIR/HeXA-Hierarchical-Experimentalist-Agents/training"
export PYTHONPATH="$(dirname "$VERL_TOOL_DIR")/interphyre:${PYTHONPATH:-}"   # makes repo-level interphyre package importable
CONDA_ENV="$PROJECT_DIR/conda/envs/VerlToolInterphyre"
PYTHON="$CONDA_ENV/bin/python"

module load conda/latest
module load cuda/12.8
export PATH="$CONDA_ENV/bin:${PATH:-}"

cd "$VERL_TOOL_DIR"
mkdir -p logs/eval

# ─── Output file (auto-increment if not given) ───────────────────────────────
if [[ -z "$OUTPUT" ]]; then
    eval_num=1
    while [ -f "logs/eval/eval${eval_num}.jsonl" ]; do
        eval_num=$((eval_num + 1))
    done
    OUTPUT="$VERL_TOOL_DIR/logs/eval/eval${eval_num}.jsonl"
fi

echo "================================================================"
echo "  Interphyre Eval"
echo "  Job ID     : ${SLURM_JOB_ID:-<interactive>}"
echo "  Checkpoint : $CHECKPOINT"
echo "  Level      : $LEVEL"
echo "  Skill bank : ${SKILL_BANK:-<none>}"
echo "  Seeds      : $SEED_START..$((SEED_START + NUM_SEEDS - 1))"
echo "  Max turns  : $MAX_TURNS"
echo "  Output     : $OUTPUT"
echo "================================================================"

# ─── Run ─────────────────────────────────────────────────────────────────────
eval_args=(
    --checkpoint "$CHECKPOINT"
    --level "$LEVEL"
    --seed_start "$SEED_START"
    --num_seeds "$NUM_SEEDS"
    --max_turns "$MAX_TURNS"
    --output "$OUTPUT"
)
if [[ -n "$SKILL_BANK" ]]; then
    eval_args+=(--skill_bank "$SKILL_BANK")
fi

$PYTHON examples/eval/interphyre_eval.py "${eval_args[@]}"

echo "Done. Results in $OUTPUT"
