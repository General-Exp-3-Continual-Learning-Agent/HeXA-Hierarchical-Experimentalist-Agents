#!/bin/bash
#SBATCH --job-name=mm_b_reps
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH -o logs/mm_b_reps_%j.log

# ─── Matched replicates of the Design B bank ─────────────────────────────────
#
# Design C's three runs are eval-SEED replicates of a single synthesised bank:
# the bank is fixed at mastermind_from_wordle/run0 and only the actor seed
# varies. The like-for-like control is therefore three eval-seed replicates of
# the ONE Design B bank -- not three fresh evolutions, which would measure a
# different source of variance than the number it is being compared against.
#
#   VLLM_URL=http://gpu017:8000/v1 sbatch scripts/run_designb_mastermind_evalreps.sh

set -Eeuo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"
PYTHON="${PYTHON:-$HOME/.conda/envs/phyreagent/bin/python}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
BANK="${BANK:-results/orbit_mastermind_hexa_oss_v4/round7/bank.json}"

if [[ -z "${VLLM_URL:-}" ]]; then echo "ERROR: set VLLM_URL."; exit 1; fi
export HEXA_VLLM_URL="$VLLM_URL"
curl -sf "${VLLM_URL}/models" >/dev/null || { echo "ERROR: ${VLLM_URL} not answering."; exit 1; }
echo "[$(date)] endpoint live; bank ${BANK}"

for R in 0 1 2; do
    echo "=== Design B bank, eval-seed replicate run${R} ==="
    "$PYTHON" -u scripts/run_orbit_target_zeroshot.py \
        --target mastermind --bank "$BANK" --run-index "$R" \
        --concurrency 24 --model "$MODEL" --base-url "$VLLM_URL" \
        --out "results/design_c/phase4/mastermind_designb_bank/run${R}"
done
echo "[$(date)] done"
