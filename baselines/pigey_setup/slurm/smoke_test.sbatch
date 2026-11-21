#!/bin/bash
#SBATCH --job-name=pigey-smoke
#SBATCH --partition=cpu
#SBATCH --account=pi_sniekum_umass_edu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00

# Live CLI smoke test: one native-image + tool-selection call through Claude Code,
# plus the verifier's yes/no call. No simulator, no GPU, no API key -- just the
# Claude Code CLI and outbound HTTPS. See README "Live checks" / run_claude.py
# --cli-smoke-test.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this with sbatch}"
export REPO_DIR="${REPO_DIR:-/home/udhanuka_umass_edu/Scalar_Research/HeXA}"
SETUP="$REPO_DIR/baselines/pigey_setup"
hostname
date
export PATH="$HOME/.local/bin:$PATH"
source "$SETUP/activate.sh"
which python && python --version
which claude && claude --version
export PYTHONUNBUFFERED=1
bash "$SETUP/run_claude.sh" --cli-smoke-test
date
