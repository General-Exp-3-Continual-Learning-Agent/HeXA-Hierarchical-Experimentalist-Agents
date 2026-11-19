#!/bin/bash
#SBATCH --job-name=rbank
#SBATCH --partition=gpu-preempt
#SBATCH --constraint=sm_80&vram40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 10:00:00
#SBATCH -A pi_sniekum_umass_edu
#SBATCH -o logs/rbank_%j.out
#SBATCH -e logs/rbank_%j.err
#
# Generic ReasoningBank baseline launcher (any InterPhyre level), Qwen2.5-7B actor
# + same-backbone extractor, HeXA-matched frozen protocol (build 6-56, eval 57-106).
# Runs in the repo checkout ($REPO) with the project Python env ($PY).
#
# Required env: LEVEL, INIT.  Optional: SEEDS, FREEZE, TAG, K, EMB, MODEL
#   LEVEL=two_body_problem INIT=Initial_trajectories/TwoBody_7B \
#     sbatch baselines/bash_scripts/sbatch_reasoning_bank_qwen7b.sh
#   LEVEL=down_to_earth INIT=Initial_trajectories/DTE_Qwen7B TAG=rep2 \
#     sbatch baselines/bash_scripts/sbatch_reasoning_bank_qwen7b.sh
set -euo pipefail

LEVEL="${LEVEL:?set LEVEL, e.g. down_to_earth or two_body_problem}"
INIT="${INIT:?set INIT, e.g. Initial_trajectories/DTE_Qwen7B}"
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"  # submit from the repo root
PY="${PY:-python}"
MODEL="${MODEL:-/datasets/ai/qwen2/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}"
SEEDS="${SEEDS:-6-106}"
FREEZE="${FREEZE:-51}"
K="${K:-1}"
EMB="${EMB:-numeric}"
TAG="${TAG:-}"

export PYTHONPATH="$REPO:$REPO/interphyre:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
cd "$REPO"
mkdir -p logs

OUT="${OUT:-baselines/data/results/reasoning_bank_${LEVEL}_qwen7b${TAG:+_$TAG}}"
echo "=== RB $LEVEL  seeds $SEEDS  freeze ${FREEZE:-none}  k $K  emb $EMB  -> $OUT ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

args=(
    --level "$LEVEL"
    --seeds "$SEEDS"
    --model "$MODEL"
    --initial-traj-dir "$INIT"
    --output-dir "$OUT"
    --cache-dir baselines/data/cache
    --embedding-model "$EMB"
    --intent-mode geometry
    --k "$K"
    --max-iterations 25
    --verbose
)
[[ -n "$FREEZE" ]] && args+=(--freeze-after "$FREEZE")
$PY -m baselines.runner.run_reasoning_bank "${args[@]}"
echo "Done. Results in $OUT"
