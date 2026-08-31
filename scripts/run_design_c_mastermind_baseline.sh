#!/bin/bash
#SBATCH --job-name=dc_mm_base
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 4:00:00
#SBATCH -o logs/dc_mm_base_%j.log

# ─── Matched no-bank Mastermind baseline for Phase 4 ─────────────────────────
#
# The existing Mastermind baselines do NOT pair with the Phase 4 arms:
# `results/orbit_mastermind_baseline_oss` (n=256) and
# `results/design_c/mastermind_baseline/run0` (partial, n=226) share ZERO
# instance ids with the arms. The arms eval on MastermindTask.eval_specs(), the
# 40-instance held-out pool; the older baselines were produced by a different
# script and a different draw. A paired test needs the same instances through
# the same code path, so this runs `run_orbit_target_zeroshot.py --bank none`.
#
# Three runs: --run-index only reseeds the actor, so the instances are identical
# and the spread across runs is the noise band.
#
#   VLLM_URL=http://gpu017:8000/v1 sbatch scripts/run_design_c_mastermind_baseline.sh

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

for R in 0 1 2; do
    echo "=== mastermind no-bank baseline run${R} ==="
    "$PYTHON" -u scripts/run_orbit_target_zeroshot.py \
        --target mastermind --bank none --run-index "$R" \
        --concurrency "$CONCURRENCY" --model "$MODEL" --base-url "$VLLM_URL" \
        --out "results/design_c/phase4/mastermind_nobank/run${R}"
done
echo "[$(date)] done"
