#!/bin/bash
#SBATCH --job-name=ace_qwen
#SBATCH --partition=superpod-a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH --mem=64G
#SBATCH -A pi_sniekum_umass_edu
#SBATCH -o logs/ace_qwen_%j.out
#SBATCH -e logs/ace_qwen_%j.err
#
# ACE (Agentic Context Engineering) on InterPhyre with a local Qwen actor on one
# GPU.  The same weights serve as Generator, Reflector and Curator (ACE's
# same-model setting), so no second model and no API are needed.
#
# Protocol matches the existing DTE/TBP Episodic RAG / MemRL Qwen runs:
# seeds 9-58 in rounds of 3, offline warmup from Initial_trajectories/<level>
# (seeds 1-8), 25 iterations, actor temperature 0.3, same Qwen2.5 snapshot.
#
# Why one A100: Qwen2.5-7B bf16 (~15 GB) + a 25-turn context + the playbook.
# superpod-a100 is capped by MaxGRESPerAccount; if it pends behind labmates,
# spread to the general partitions (sm_80 floor because weights load as bf16):
#   sbatch -p gpu,gpu-preempt --constraint='sm_80&vram40' baselines/bash_scripts/sbatch_ace_qwen.sh ...
# Preemption is safe: the run resumes from the per-seed trajectory JSONs and
# ace_state.json, and never curates a seed twice.
#
# Usage (submit from the repo root so logs/ resolves):
#   sbatch baselines/bash_scripts/sbatch_ace_qwen.sh --level down_to_earth
#   sbatch baselines/bash_scripts/sbatch_ace_qwen.sh --level two_body_problem
#   sbatch baselines/bash_scripts/sbatch_ace_qwen.sh --level down_to_earth --size 3b
#   # smoke test: 2 seeds, short walltime
#   sbatch -t 01:00:00 baselines/bash_scripts/sbatch_ace_qwen.sh --level down_to_earth --seeds 9-10 --tag smoke
#   # ablation: reference reflect-and-re-solve loop (paper used 5 rounds)
#   sbatch baselines/bash_scripts/sbatch_ace_qwen.sh --level down_to_earth --extra "--resolve-rounds 3" --tag resolve3
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

LEVEL=""
SEEDS="9-58"
SIZE="7b"
TAG=""
EXTRA=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --level) LEVEL="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --size)  SIZE="$2";  shift 2 ;;
        --tag)   TAG="$2";   shift 2 ;;
        --extra) EXTRA="$2"; shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -n "$LEVEL" ]] || { echo "[ERROR] --level is required"; exit 1; }

# sbatch runs a copy of this script from the spool dir, so anchor paths explicitly.
REPO_DIR="${REPO_DIR:-/home/udhanuka_umass_edu/Scalar_Research/HeXA}"
CONDA_ENV="${CONDA_ENV:-/project/pi_sniekum_umass_edu/vgandhi/conda/envs/VerlToolInterphyre}"
# Offline trajectories for DTE/TBP exist only in the shared baseline checkout
# (read-only here); catapult also ships in this repo.
INIT_ROOT="${INIT_ROOT:-/project/pi_sniekum_umass_edu/vgandhi/HExA_Baseline_Exp/HeXA-Hierarchical-Experimentalist-Agents/Initial_trajectories}"
case "$SIZE" in
    7b) QWEN_PATH="${QWEN_PATH:-/datasets/ai/qwen2/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}" ;;
    3b) QWEN_PATH="${QWEN_PATH:-/datasets/ai/qwen2/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1}" ;;
    *) echo "[ERROR] --size must be 7b or 3b"; exit 1 ;;
esac
RUN_NAME="ace_${LEVEL}_qwen${SIZE}${TAG:+_$TAG}"
OUTPUT_DIR="${BASELINE_ROOT:-$REPO_DIR/baselines/data/results}/$RUN_NAME"

module load conda/latest 2>/dev/null || true
module load cuda/12.8 2>/dev/null || true
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$REPO_DIR:$REPO_DIR/interphyre:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# Weights load from an absolute path; never touch the network for them.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# cl100k_base for context-token accounting, pre-fetched on the login node
# (compute nodes may have no internet; count_tokens would silently return None).
export TIKTOKEN_CACHE_DIR="$REPO_DIR/baselines/data/cache/tiktoken"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

cd "$REPO_DIR"
mkdir -p logs "$OUTPUT_DIR"

echo "================================================================"
echo "  ACE baseline (local Qwen actor = Reflector = Curator)"
echo "  Job        : ${SLURM_JOB_ID:-<interactive>} on $(hostname)  $(date)"
echo "  Level      : $LEVEL   seeds $SEEDS"
echo "  Actor      : $QWEN_PATH"
echo "  Warmup     : $INIT_ROOT/$LEVEL"
echo "  Output     : $OUTPUT_DIR"
echo "  Extra      : ${EXTRA:-<none>}"
echo "================================================================"
module list 2>&1 || true
which python && python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
[[ -d "$QWEN_PATH" ]] || { echo "[ERROR] model dir not found: $QWEN_PATH"; exit 1; }
[[ -d "$INIT_ROOT/$LEVEL" ]] || { echo "[ERROR] no offline trajectories: $INIT_ROOT/$LEVEL"; exit 1; }
# Record exactly which actor/tool code this run used (the working tree may be dirty).
git -C "$REPO_DIR" rev-parse HEAD > "$OUTPUT_DIR/code_rev.txt" 2>/dev/null || true
git -C "$REPO_DIR" diff -- react_agent interphyre > "$OUTPUT_DIR/code_diff.patch" 2>/dev/null || true
# baselines/ is untracked, so hash the method code instead of diffing it.
sha1sum baselines/ace/*.py baselines/runner/run_ace.py baselines/runner/memory_runner.py \
    baselines/memory/trajectory_fmt.py > "$OUTPUT_DIR/code_sha1.txt"

# shellcheck disable=SC2206
python -m baselines.runner.run_ace \
    --level "$LEVEL" \
    --seeds "$SEEDS" \
    --seeds-per-round 3 \
    --max-iterations 25 \
    --initial-traj-dir "$INIT_ROOT/$LEVEL" \
    --model "$QWEN_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --verbose \
    ${EXTRA}

echo "Done $(date). Results in $OUTPUT_DIR"
