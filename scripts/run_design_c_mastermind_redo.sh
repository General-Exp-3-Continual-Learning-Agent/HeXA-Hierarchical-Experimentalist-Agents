#!/bin/bash
#SBATCH --job-name=dc_mm_redo
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 6:00:00
#SBATCH -o logs/dc_mm_redo_%j.log

# ─── Re-run the Mastermind half of Phase 3 + Phase 4 ─────────────────────────
#
# The first pass synthesised against a MASTERMIND_BLOCK that described a
# 4-digit / 1-8 / 1680-code task; the eval runs 3-digit / 1-6 / 120 codes. The
# block is fixed and pinned by a test; this redoes synthesis and eval on the
# corrected description. See
# results/design_c/_CONFOUNDED_mastermind_4digit_block/WHY_CONFOUNDED.md
#
#   VLLM_URL=http://gpu017:8000/v1 sbatch scripts/run_design_c_mastermind_redo.sh

set -Eeuo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"

PYTHON="${PYTHON:-$HOME/.conda/envs/phyreagent/bin/python}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
CONCURRENCY="${CONCURRENCY:-24}"

if [[ -z "${VLLM_URL:-}" ]]; then echo "ERROR: set VLLM_URL."; exit 1; fi
export HEXA_VLLM_URL="$VLLM_URL"
# gpt-oss is served at 65536; the teacher's evolution/synthesis response needs
# room or the call is rejected and the previous bank is silently kept.
export HEXA_TEACHER_MAX_TOKENS="${HEXA_TEACHER_MAX_TOKENS:-32768}"
if ! curl -sf "${VLLM_URL}/models" >/dev/null 2>&1; then
    echo "ERROR: ${VLLM_URL} is not answering."; exit 1
fi
echo "[$(date)] endpoint ${VLLM_URL} is live"

echo "########## PHASE 3 (mastermind only) ##########"
"$PYTHON" -u scripts/run_design_c_phase3_all.py --runs 0 --target mastermind

echo "########## PHASE 4 (mastermind only) ##########"
"$PYTHON" -u scripts/run_design_c_phase4_all.py \
    --run-index 0 --concurrency "$CONCURRENCY" --model "$MODEL" \
    --base-url "$VLLM_URL" --target mastermind --no-ablation
echo "[$(date)] done"
