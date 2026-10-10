#!/bin/bash
#SBATCH --job-name=core_mem_baseline
#SBATCH --partition=cpu-preempt
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 24:00:00
#SBATCH -o logs/core_mem_baseline_%j.log

# ─── Episodic RAG / MemRL on CORE's four tasks, one method across all tasks ──
#
# CPU only, exactly like scripts/run_orbit_reflexion_arm.sh: this process only
# makes HTTP calls to an already-served vLLM endpoint. Bring the GPU up first:
#
#   sbatch scripts/serve_gpt_oss_vllm.sh
#
# then:
#
#   METHOD=episodic_rag VLLM_URL=http://gpu017:8000/v1 \
#     sbatch scripts/run_core_memory_baseline_arm.sh
#
# Runs all four CORE tasks for one method, sequentially, against the same
# server. Each task's build (n=1) + eval (n=3) is independent of the others --
# separate memory stores, separate output dirs -- so this is four independent
# jobs run back to back rather than anything that shares state across tasks.

set -Eeuo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"

# The ambient HF_HOME on this cluster (inherited from the submitting shell's
# environment, which Slurm exports by default) points at a shared PI cache
# (/project/pi_hongyu_umass_edu/cache/hf) this account cannot read the token
# file of -- PermissionError on the embedder's first SentenceTransformer load.
# Force it to somewhere this account owns, unconditionally: `${VAR:-default}`
# is a no-op here since the ambient value is already non-empty.
export HF_HOME="$HOME/.cache/hf_baselines"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_TEMP_DIR="$HF_HOME/tmp"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_HUB_TEMP_DIR"

# Unbuffered: stdout is redirected to a file (`-o` above), not a tty, so
# Python block-buffers print() by default. Without this, `tail`/`grep` on the
# log file understate progress by anywhere up to one buffer's worth of visits --
# which is what made a genuinely-progressing zebralogic run look stalled for
# 3+ hours and led to a since-fixed restart-into-the-same-dir corruption (see
# run_core_memory_baseline.py's --force guard). build_log.jsonl was never
# affected (it's opened and closed per line), only the human-facing log was.
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-$HOME/.conda/envs/phyreagent/bin/python}"
METHOD="${METHOD:-episodic_rag}"
TASKS="${TASKS:-matchstick mathgap zebralogic tower-of-hanoi}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
N_TRAIN="${N_TRAIN:-60}"
BUDGET="${BUDGET:-900}"
EVAL_RUNS="${EVAL_RUNS:-3}"
MAX_TOKENS="${MAX_TOKENS:-60000}"
# 64: the server's own pool cap (serve_gpt_oss_vllm.sh), not a guess -- at 40,
# GPU KV-cache usage sat at ~10% with 0 requests ever waiting, so the server
# was nowhere near saturated. Measured on the live episodic_rag/matchstick run.
CONCURRENCY="${CONCURRENCY:-64}"
OUT_ROOT="${OUT_ROOT:-results_core_baselines/${METHOD}}"

if [[ -z "${VLLM_URL:-}" ]]; then echo "ERROR: set VLLM_URL."; exit 1; fi
if ! curl -sf "${VLLM_URL}/models" >/dev/null 2>&1; then
    echo "ERROR: ${VLLM_URL} is not answering."; exit 1
fi

echo "[$(date)] ${METHOD} on: ${TASKS}"
echo "           model=$MODEL endpoint=$VLLM_URL n_train=$N_TRAIN budget=$BUDGET eval_runs=$EVAL_RUNS"

for TASK in $TASKS; do
    OUT="${OUT_ROOT}/${TASK}"
    echo
    echo "================ ${METHOD} / ${TASK} -> ${OUT} ================"

    if ! curl -sf "${VLLM_URL}/models" >/dev/null 2>&1; then
        echo "ERROR: endpoint died before ${TASK}; stopping."
        exit 1
    fi
    "$PYTHON" scripts/run_core_memory_baseline.py \
        --task "$TASK" --method "$METHOD" \
        --n-train "$N_TRAIN" --budget "$BUDGET" --eval-runs "$EVAL_RUNS" \
        --model "$MODEL" --base-url "$VLLM_URL" \
        --max-tokens "$MAX_TOKENS" --concurrency "$CONCURRENCY" \
        --output-dir "$OUT"
done

echo
echo "[$(date)] ${METHOD}: all tasks done -> ${OUT_ROOT}"
