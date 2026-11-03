#!/bin/bash
#SBATCH --job-name=pigey-perception-check
#SBATCH --partition=superpod-a100
#SBATCH --account=pi_sniekum_umass_edu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00

# Smoke test: serve the local ER model and SAM, and run Pigey's localization
# (detection + SAM-masked depth unprojection) on a few LIBERO frames. No pi0.5 checkpoint, Claude, or Gemini calls are needed.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this check with sbatch}"
export REPO_DIR="${REPO_DIR:-/project/pi_sniekum_umass_edu/vgandhi/HeXA-Hierarchical-Experimentalist-Agents}"
SETUP="$REPO_DIR/baselines/pigey_setup"
hostname
date
nvidia-smi --query-gpu=name,memory.total --format=csv
source "$SETUP/activate.sh"
which python && python --version
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export TMPDIR="$PIGEY_ROOT/tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
# Nothing else shares the GPU in this check.
PIGEY_ER_GPU_UTIL="${PIGEY_ER_GPU_UTIL:-0.85}" bash "$SETUP/servers/serve_perception.sh"
trap 'kill "$(cat "$TMPDIR/er-server.pid")" "$(cat "$TMPDIR/sam-server.pid")" 2> /dev/null || true' EXIT
source "$TMPDIR/perception.env"
python "$SETUP/runner/perception_check.py" --out-dir "$PIGEY_ROOT/results/perception_check/$SLURM_JOB_ID" "$@"
date
