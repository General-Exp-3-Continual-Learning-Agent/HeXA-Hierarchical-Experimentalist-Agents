#!/bin/bash
#SBATCH --job-name=dc_p4_mm
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 4:00:00
#SBATCH -o logs/dc_p4_mm_%j.log

# ─── Design C Phase 4, Mastermind only, on the ORBIT-paper config ────────────
#
# Run after Phase 3's gates have been read by a person. The redo job stops at
# Phase 3 by design: `run_design_c_phase3_all.py` exits non-zero when any bank is
# blocked and `set -e` halts there, so eval compute is never spent on a bank
# nobody looked at.
#
# --no-ablation: mastermind_rawdump_wordle already ran on this config and is
# unaffected by the target-block fix, since a raw dump injects the SOURCE bank
# and never reads the target description.
#
#   VLLM_URL=http://gpu017:8000/v1 sbatch scripts/run_design_c_phase4_mastermind.sh

set -Eeuo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"

PYTHON="${PYTHON:-$HOME/.conda/envs/phyreagent/bin/python}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
CONCURRENCY="${CONCURRENCY:-24}"

if [[ -z "${VLLM_URL:-}" ]]; then echo "ERROR: set VLLM_URL."; exit 1; fi
export HEXA_VLLM_URL="$VLLM_URL"
if ! curl -sf "${VLLM_URL}/models" >/dev/null 2>&1; then
    echo "ERROR: ${VLLM_URL} is not answering."; exit 1
fi
echo "[$(date)] endpoint ${VLLM_URL} is live"

exec "$PYTHON" -u scripts/run_design_c_phase4_all.py \
    --run-index 0 --concurrency "$CONCURRENCY" --model "$MODEL" \
    --base-url "$VLLM_URL" --target mastermind --no-ablation
