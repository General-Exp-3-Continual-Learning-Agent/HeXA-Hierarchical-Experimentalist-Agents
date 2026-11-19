#!/bin/bash
#SBATCH --job-name=rbank_dte
#SBATCH --partition=gpu-preempt
#SBATCH --constraint=sm_80&vram40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 10:00:00
#SBATCH -A pi_sniekum_umass_edu
#SBATCH -o logs/rbank_dte_%j.out
#SBATCH -e logs/rbank_dte_%j.err
#
# ReasoningBank baseline on down_to_earth with the Qwen2.5-7B actor + same-backbone
# extractor, matching HeXA's Qwen7B protocol (build seeds 6-56, freeze, eval 57-106).
# Runs in the repo checkout ($REPO) with the project Python env ($PY), which
# is what every Qwen/Claude job here uses. gpu-preempt is safe: the stream resumes
# from existing per-seed trajectory JSONs, so a preemption costs one episode.
#
# Overridable via env:  SEEDS, FREEZE, TAG, K, EMB, MODEL
#   sbatch baselines/bash_scripts/sbatch_reasoning_bank_dte_qwen7b.sh                 # full: 6-106, freeze 51
#   SEEDS=6-8 FREEZE= TAG=smoke sbatch -t 02:00:00 baselines/bash_scripts/sbatch_reasoning_bank_dte_qwen7b.sh
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"  # submit from the repo root
PY="${PY:-python}"
MODEL="${MODEL:-/datasets/ai/qwen2/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}"
INIT="${INIT:-Initial_trajectories/DTE_Qwen7B}"
SEEDS="${SEEDS:-6-106}"
FREEZE="${FREEZE:-51}"        # seed 57 is stream index 51 (6..56 = 51 build seeds)
K="${K:-1}"
EMB="${EMB:-numeric}"
TAG="${TAG:-}"

export PYTHONPATH="$REPO:$REPO/interphyre:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
cd "$REPO"
mkdir -p logs

OUT="baselines/data/results/reasoning_bank_down_to_earth_qwen7b${TAG:+_$TAG}"

echo "================================================================"
echo "  ReasoningBank on down_to_earth (Qwen2.5-7B actor + extractor)"
echo "  Job     : ${SLURM_JOB_ID:-<interactive>}"
echo "  Seeds   : $SEEDS   Freeze-after: ${FREEZE:-<none>}   k: $K   emb: $EMB"
echo "  Model   : $MODEL"
echo "  Init    : $INIT"
echo "  Output  : $OUT"
echo "================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

args=(
    --level down_to_earth
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
