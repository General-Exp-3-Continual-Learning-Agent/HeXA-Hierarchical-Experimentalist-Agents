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
# Protocol: HeXA's seed stream, seeds 6-56 (17 rounds of 3), starting from an
# EMPTY playbook (no offline warmup), 25 iterations, actor temperature 0.3.
# `--init-dir <Initial_trajectories/level>` turns ACE's offline warmup on (the
# protocol the labmate's DTE/TBP RAG/MemRL runs used with seeds 9-58).
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
#   sbatch -t 01:00:00 baselines/bash_scripts/sbatch_ace_qwen.sh --level down_to_earth --seeds 6-7 --tag smoke
#   # frozen held-out eval of a finished run's final playbook (no Reflector/Curator calls)
#   sbatch --dependency=afterok:<jobid> baselines/bash_scripts/sbatch_ace_qwen.sh --level down_to_earth \
#       --seeds 57-106 --frozen-from baselines/data/results/ace_down_to_earth_qwen7b_s6-56
#   # paper configuration: 5 reflect-and-re-solve rounds + grow-and-refine dedup
#   sbatch -t 16:00:00 baselines/bash_scripts/sbatch_ace_qwen.sh --level down_to_earth \
#       --init-dir <Initial_trajectories/DTE_Qwen7B> --extra "--resolve-rounds 5 --dedup" --tag paperconfig
#   # vLLM backend (HExA's serving stack): starts `vllm serve` on the job's GPU and
#   # runs the actor and all ACE roles through it (prefix caching, continuous
#   # batching).  HF `generate` decodes batch-1 in a Python loop and re-encodes the
#   # whole 25-turn prompt every call: ~3.5 s/call, CPU-bound, GPU mostly idle.
#   sbatch baselines/bash_scripts/sbatch_ace_qwen.sh --level down_to_earth --size 3b --backend vllm \
#       --init-dir <...> --extra "--resolve-rounds 5 --dedup" --tag paperconfig_vllm
#   # ablation: reference reflect-and-re-solve loop (paper used 5 rounds)
#   sbatch baselines/bash_scripts/sbatch_ace_qwen.sh --level down_to_earth --extra "--resolve-rounds 3" --tag resolve3
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

LEVEL=""
SEEDS="6-56"
INIT_DIR=""
FROZEN_FROM=""
SIZE="7b"
BACKEND="hf"
TAG=""
EXTRA=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --level) LEVEL="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --size)  SIZE="$2";  shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --tag)   TAG="$2";   shift 2 ;;
        --init-dir) INIT_DIR="$2"; shift 2 ;;
        --frozen-from) FROZEN_FROM="$2"; shift 2 ;;
        --extra) EXTRA="$2"; shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -n "$LEVEL" ]] || { echo "[ERROR] --level is required"; exit 1; }

[[ "$BACKEND" == hf || "$BACKEND" == vllm ]] || { echo "[ERROR] --backend must be hf or vllm"; exit 1; }

# sbatch runs a copy of this script from the spool dir, so anchor paths explicitly.
REPO_DIR="${REPO_DIR:-/home/udhanuka_umass_edu/Scalar_Research/HeXA}"
CONDA_ENV="${CONDA_ENV:-/project/pi_sniekum_umass_edu/vgandhi/conda/envs/VerlToolInterphyre}"
case "$SIZE" in
    7b) QWEN_PATH="${QWEN_PATH:-/datasets/ai/qwen2/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}" ;;
    3b) QWEN_PATH="${QWEN_PATH:-/datasets/ai/qwen2/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1}" ;;
    *) echo "[ERROR] --size must be 7b or 3b"; exit 1 ;;
esac
# Seed range and warmup are in the name so runs on different protocols can never
# share (and resume into) one output directory.
RUN_NAME="ace_${LEVEL}_qwen${SIZE}_s${SEEDS//,/_}${INIT_DIR:+_warmup}${FROZEN_FROM:+_frozen}${TAG:+_$TAG}"
OUTPUT_DIR="${BASELINE_ROOT:-$REPO_DIR/baselines/data/results}/$RUN_NAME"

module load conda/latest 2>/dev/null || true
module load cuda/12.8 2>/dev/null || true
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$REPO_DIR:$REPO_DIR/interphyre:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# The submitting shell's profile points HF_HOME/HF_HUB_CACHE/TRANSFORMERS_CACHE
# at another lab's /project dir (not writable here) and enables hf_transfer
# (not installed); sbatch inherits all of it.  Clear it and use the repo cache,
# which holds all-mpnet-base-v2 for --dedup (fetched once on the login node).
unset HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE HF_HUB_ENABLE_HF_TRANSFER \
      HF_HUB_TEMP_DIR HF_HUB_LOCAL_DIR_AUTO_EXTRACT
export HF_HOME="$REPO_DIR/baselines/data/cache/hf"
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
echo "  Actor      : $QWEN_PATH  (backend: $BACKEND)"
echo "  Warmup     : ${INIT_DIR:-<none>}"
echo "  Frozen from: ${FROZEN_FROM:-<no: playbook learns online>}"
echo "  Output     : $OUTPUT_DIR"
echo "  Extra      : ${EXTRA:-<none>}"
echo "================================================================"
module list 2>&1 || true
which python && python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
[[ -d "$QWEN_PATH" ]] || { echo "[ERROR] model dir not found: $QWEN_PATH"; exit 1; }
[[ -z "$INIT_DIR" || -d "$INIT_DIR" ]] || { echo "[ERROR] no offline trajectories: $INIT_DIR"; exit 1; }
# Record exactly which actor/tool code this run used (the working tree may be dirty).
git -C "$REPO_DIR" rev-parse HEAD > "$OUTPUT_DIR/code_rev.txt" 2>/dev/null || true
git -C "$REPO_DIR" diff -- react_agent interphyre > "$OUTPUT_DIR/code_diff.patch" 2>/dev/null || true
# baselines/ is untracked, so hash the method code instead of diffing it.
sha1sum baselines/ace/*.py baselines/runner/run_ace.py baselines/runner/memory_runner.py \
    baselines/memory/trajectory_fmt.py > "$OUTPUT_DIR/code_sha1.txt"

MODEL_ARG="$QWEN_PATH"
if [[ "$BACKEND" == vllm ]]; then
    # Same serving stack and settings as HExA's builds
    # (physics-reasoning-agents/skillrl/run_tbp_builds.sh); a per-run seed keeps
    # replicates from being byte-identical (vLLM defaults to seed 0).
    VLLM_ENV="${VLLM_ENV:-/work/pi_sniekum_umass_edu/${USER}/.conda/envs/vllm-gptoss}"
    PORT="${PORT:-$((8100 + ${SLURM_JOB_ID:-0} % 800))}"
    SERVED="qwen2.5-${SIZE}"
    VLLM_SEED="${VLLM_SEED:-${SLURM_JOB_ID:-0}}"
    export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
    export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$REPO_DIR/.cache/flashinfer}"
    export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$REPO_DIR/.cache/torch_extensions}"
    mkdir -p "$FLASHINFER_WORKSPACE_BASE" "$TORCH_EXTENSIONS_DIR"
    "$VLLM_ENV/bin/vllm" serve "$QWEN_PATH" --served-model-name "$SERVED" --port "$PORT" \
        --max-model-len 32768 --gpu-memory-utilization 0.85 --seed "$VLLM_SEED" \
        --enable-prefix-caching > "$OUTPUT_DIR/vllm_${SLURM_JOB_ID:-local}.log" 2>&1 &
    VLLM_PID=$!
    trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT INT TERM
    dl=$(( $(date +%s) + 900 ))
    until curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; do
        kill -0 "$VLLM_PID" 2>/dev/null || { echo "ERROR: vLLM died"; tail -30 "$OUTPUT_DIR/vllm_${SLURM_JOB_ID:-local}.log"; exit 1; }
        [ "$(date +%s)" -lt "$dl" ] || { echo "ERROR: vLLM not ready after 15 min"; exit 1; }
        sleep 10
    done
    echo "[vllm] ready on port $PORT (seed $VLLM_SEED)"
    export OPENAI_BASE_URL="http://localhost:${PORT}/v1"
    MODEL_ARG="openai:$SERVED"
fi

INIT_ARGS=()
[[ -n "$INIT_DIR" ]] && INIT_ARGS=(--initial-traj-dir "$INIT_DIR")
if [[ -n "$FROZEN_FROM" ]]; then
    [[ "$FROZEN_FROM" = /* ]] || FROZEN_FROM="$REPO_DIR/$FROZEN_FROM"
    INIT_ARGS+=(--init-from "$FROZEN_FROM" --freeze-after 0)
fi

# shellcheck disable=SC2206
python -m baselines.runner.run_ace \
    --level "$LEVEL" \
    --seeds "$SEEDS" \
    --seeds-per-round 3 \
    --max-iterations 25 \
    "${INIT_ARGS[@]}" \
    --model "$MODEL_ARG" \
    --output-dir "$OUTPUT_DIR" \
    --verbose \
    ${EXTRA}

echo "Done $(date). Results in $OUTPUT_DIR"
