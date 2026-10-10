#!/bin/bash
#SBATCH --job-name=orbit_memory
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -o logs/orbit_memory_%j.log
# No -e: stderr merges into the same file, matching run_maze_arm.sh -- reading a
# run's story across two files interleaved by nothing is a waste of time.

# ─── One Episodic RAG / MemRL arm on ORBIT Maze or Mastermind ────────────────
#
# CPU only, like run_maze_arm.sh: the client waits on the vLLM server the whole
# time. The GPU belongs to the separate serve_gpt_oss_vllm.sh job it talks to.
#
#   TASK=mastermind METHOD=memrl \
#   VLLM_URL=http://gpu013:8000/v1 \
#   OUT=results/orbit_mastermind_memrl \
#     sbatch scripts/run_orbit_memory_arm.sh
#
# 12 hours rather than 6: this runs TWO passes, and the first is serial by
# construction (instance n's retrieval must see what 1..n-1 wrote -- see
# baselines/orbit/stream.py). At ~1-3 min/instance an 80-instance build pass is
# 2-4 hours before the eval pass starts.
#
# Calibrate first for MemRL; it refuses to start without it:
#   python -m baselines.orbit.calibrate --task mastermind

set -Eeuo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"

PYTHON="${PYTHON:-$HOME/.conda/envs/phyreagent/bin/python}"
TASK="${TASK:-mastermind}"
METHOD="${METHOD:-memrl}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
OUT="${OUT:-results/orbit_${TASK}_${METHOD}}"
CACHE_DIR="${CACHE_DIR:-baselines/data/cache}"
ARM_ARGS="${ARM_ARGS:---eval-concurrency 8}"

case "$METHOD" in
    episodic_rag|memrl) ;;
    *) echo "ERROR: METHOD must be episodic_rag or memrl"; exit 1 ;;
esac
case "$TASK" in
    maze|mastermind) ;;
    *) echo "ERROR: TASK must be maze or mastermind"; exit 1 ;;
esac

# ── HuggingFace caches ───────────────────────────────────────────────────────
# The login environment on this cluster exports HF_HOME, HF_HUB_CACHE,
# HF_DATASETS_CACHE, TRANSFORMERS_CACHE and PIP_CACHE_DIR into
# /project/pi_hongyu_umass_edu/cache, which this account cannot write. The
# embedder is the first thing that needs them, so an uncorrected run dies with
# `PermissionError: .../cache/hf/token` several minutes in rather than at
# startup. Pinned here for the same reason baselines/bash_scripts/_baseline_env.sh
# pins HF_HOME: an inherited value that points somewhere we do not own is a hard
# failure the moment anything needs downloading.
HF_ROOT="${HF_ROOT:-$HOME/.cache/huggingface}"
export HF_HOME="$HF_ROOT"
export HF_HUB_CACHE="$HF_ROOT/hub"
export HF_DATASETS_CACHE="$HF_ROOT/datasets"
export TRANSFORMERS_CACHE="$HF_ROOT/transformers"
export HF_HUB_TEMP_DIR="$HF_ROOT/tmp"
export TORCH_HOME="${TORCH_HOME_OVERRIDE:-$HOME/.cache/torch}"
mkdir -p "$HF_HUB_CACHE" "$HF_HUB_TEMP_DIR" "$TORCH_HOME" logs

# Slurm block-buffers stdout, which makes an hours-long run look frozen.
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# The endpoint must be passed in, not guessed: vllm_endpoint_latest.json can be
# stale from a previous job, and silently aiming an arm at the wrong model would
# be worse than failing.
if [[ -z "${VLLM_URL:-}" ]]; then
    echo "ERROR: set VLLM_URL to the serving job's base_url."
    echo "       Read it from logs/vllm_endpoint_<jobid>.json -- NOT from"
    echo "       vllm_endpoint_latest.json, which may belong to an older job."
    exit 1
fi
export HEXA_VLLM_URL="$VLLM_URL"

if ! curl -sf "${VLLM_URL}/models" >/dev/null 2>&1; then
    echo "ERROR: ${VLLM_URL} is not answering. Is the serving job still running?"
    exit 1
fi

# MemRL cannot start without calibrated constants and says so, but it says so
# after the pools are drawn. Checking here turns a 12-hour queue wait followed by
# an immediate exit into an immediate exit.
if [[ "$METHOD" == "memrl" ]]; then
    if ! compgen -G "$CACHE_DIR/orbit_calibration_${TASK}_*" >/dev/null; then
        echo "ERROR: no calibration for ${TASK} in ${CACHE_DIR}."
        echo "       Run:  $PYTHON -m baselines.orbit.calibrate --task ${TASK}"
        exit 1
    fi
fi

echo "[$(date)] endpoint ${VLLM_URL} is live"
echo "[$(date)] ${TASK} / ${METHOD} / ${MODEL} -> ${OUT}"

exec "$PYTHON" -u scripts/run_orbit_memory_baseline.py \
    --task "$TASK" --method "$METHOD" --model "$MODEL" \
    --out "$OUT" --cache-dir "$CACHE_DIR" $ARM_ARGS
