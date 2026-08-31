#!/bin/bash
#SBATCH --job-name=dc_phase4
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 12:00:00
#SBATCH -o logs/dc_phase4_%j.log
# stderr merged into stdout, same reasoning as run_maze_arm.sh.

# ─── Design C Phase 4: all eight zero-shot eval arms, sequentially ───────────
#
# Batch rather than login node for the same reasons as run_maze_arm.sh: eight
# 256-instance arms is a multi-hour job, and the login node's `python` has no
# numpy -- the deps live in the phyreagent conda env.
#
#   VLLM_URL=http://gpu017:8000/v1 sbatch scripts/run_design_c_phase4.sh

set -Eeuo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"

PYTHON="${PYTHON:-$HOME/.conda/envs/phyreagent/bin/python}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
CONCURRENCY="${CONCURRENCY:-24}"
RUN_INDEX="${RUN_INDEX:-0}"

if [[ -z "${VLLM_URL:-}" ]]; then
    echo "ERROR: set VLLM_URL to the gpt-oss serving job's base_url."
    exit 1
fi
export HEXA_VLLM_URL="$VLLM_URL"

if ! curl -sf "${VLLM_URL}/models" >/dev/null 2>&1; then
    echo "ERROR: ${VLLM_URL} is not answering. Is the serving job still running?"
    exit 1
fi
echo "[$(date)] endpoint ${VLLM_URL} is live"

exec "$PYTHON" -u scripts/run_design_c_phase4_all.py \
    --run-index "$RUN_INDEX" --concurrency "$CONCURRENCY" \
    --model "$MODEL" --base-url "$VLLM_URL"
