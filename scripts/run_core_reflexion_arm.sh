#!/bin/bash
#SBATCH --job-name=core_reflexion
#SBATCH --partition=cpu-preempt
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 24:00:00
#SBATCH -o logs/core_reflexion_%j.log

# ─── Reflexion (K trials/item) on CORE's four tasks, no build phase ──────────
#
# CPU only: this process only makes HTTP calls to an already-served vLLM
# endpoint, exactly like scripts/run_core_memory_baseline_arm.sh.
#
#   sbatch scripts/serve_gpt_oss_vllm.sh
#   VLLM_URL=http://gpu017:8000/v1 sbatch scripts/run_core_reflexion_arm.sh
#
# No build phase to skip-and-recover into (baselines/core/reflexion.py has no
# persistent memory), so unlike the memory-baseline arm there is nothing to
# `--skip-build` back into after a dead endpoint -- a task that dies mid-run
# must be rerun with --force from scratch. Each of the 4 tasks is otherwise
# independent (separate output dirs), run sequentially against the one server.

set -Eeuo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"

# Force off the ambient HF_HOME (shared PI cache this account can't read the
# token file of) -- see run_core_memory_baseline_arm.sh for the full story.
export HF_HOME="$HOME/.cache/hf_baselines"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_TEMP_DIR="$HF_HOME/tmp"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_HUB_TEMP_DIR"

# Unbuffered: see run_core_memory_baseline_arm.sh for why this matters --
# without it, `tail`/`grep` on this log understate progress and can look like
# a stall when the run is actually fine.
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-$HOME/.conda/envs/phyreagent/bin/python}"
TASKS="${TASKS:-matchstick mathgap zebralogic tower-of-hanoi}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
K_TRIALS="${K_TRIALS:-3}"
EVAL_RUNS="${EVAL_RUNS:-3}"
MAX_TOKENS="${MAX_TOKENS:-60000}"
CONCURRENCY="${CONCURRENCY:-64}"
OUT_ROOT="${OUT_ROOT:-results_core_baselines/reflexion}"

if [[ -z "${VLLM_URL:-}" ]]; then echo "ERROR: set VLLM_URL."; exit 1; fi
if ! curl -sf "${VLLM_URL}/models" >/dev/null 2>&1; then
    echo "ERROR: ${VLLM_URL} is not answering."; exit 1
fi

echo "[$(date)] reflexion (K=$K_TRIALS) on: ${TASKS}"
echo "           model=$MODEL endpoint=$VLLM_URL eval_runs=$EVAL_RUNS"

for TASK in $TASKS; do
    OUT="${OUT_ROOT}/${TASK}"
    echo
    echo "================ reflexion / ${TASK} -> ${OUT} ================"

    if ! curl -sf "${VLLM_URL}/models" >/dev/null 2>&1; then
        echo "ERROR: endpoint died before ${TASK}; stopping."
        exit 1
    fi
    "$PYTHON" scripts/run_core_reflexion_baseline.py \
        --task "$TASK" --k-trials "$K_TRIALS" --eval-runs "$EVAL_RUNS" \
        --model "$MODEL" --base-url "$VLLM_URL" \
        --max-tokens "$MAX_TOKENS" --concurrency "$CONCURRENCY" \
        --output-dir "$OUT"
done

echo
echo "[$(date)] reflexion: all tasks done -> ${OUT_ROOT}"
