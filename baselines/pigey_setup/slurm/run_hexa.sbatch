#!/bin/bash
#SBATCH --job-name=pigey-hexa
#SBATCH --partition=superpod-a100
#SBATCH --account=pi_sniekum_umass_edu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# HeXA's evolving skill-bank loop over Pigey on one A100 (hexa/loop.py). The
# servers start once; each round runs every task on PIGEY_HEXA_EPISODES_PER_ROUND
# init states from the PIGEY_HEXA_EPISODES pool with the current bank injected,
# then the teacher evolves the bank for the next round. Round 1's bank is
#   PIGEY_HEXA_INITIAL_BANK   a finished bank (copied in as bank_round_1.json), or
#   PIGEY_HEXA_INITIAL        finished no-bank trial directories to distill it from, or
#   neither                   an online start: round 1 runs without a bank.
# Other environment: PIGEY_SUITE, PIGEY_TASKS (default 0-9), PIGEY_HEXA_OUT (the run
# directory; an existing one is resumed), PIGEY_ER_MODEL, PIGEY_ER_GPU_UTIL. Extra
# arguments go to hexa/loop.py and on to run_claude.sh, e.g.
#   PIGEY_HEXA_INITIAL_BANK=experiments/libero_goal_task/initial_bank.json PIGEY_HEXA_EPISODES="$(seq -s' ' 10 24)" \
#       sbatch --export=ALL slurm/run_hexa.sbatch --model claude-sonnet-5-5 --rounds 15
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this loop with sbatch}"
export REPO_DIR="${REPO_DIR:-/project/pi_sniekum_umass_edu/vgandhi/HeXA-Hierarchical-Experimentalist-Agents}"
SETUP="$REPO_DIR/baselines/pigey_setup"
hostname
date
nvidia-smi --query-gpu=name,memory.total --format=csv
source "$SETUP/activate.sh"
export PATH="$HOME/.local/bin:$PATH"
which python && python --version
which claude
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export TMPDIR="$PIGEY_ROOT/tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
OUT_DIR="${PIGEY_HEXA_OUT:-$PIGEY_ROOT/results/hexa/$SLURM_JOB_ID}"
initial=()
if [[ -n "${PIGEY_HEXA_INITIAL_BANK:-}" ]]; then
    [[ -f "$PIGEY_HEXA_INITIAL_BANK" ]] || { echo "PIGEY_HEXA_INITIAL_BANK not found: $PIGEY_HEXA_INITIAL_BANK" >&2; exit 1; }
    initial+=(--initial-bank "$PIGEY_HEXA_INITIAL_BANK")
elif [[ -n "${PIGEY_HEXA_INITIAL:-}" ]]; then
    # shellcheck disable=SC2206
    initial+=(--initial-trials $PIGEY_HEXA_INITIAL)
fi
printf 'Run directory: %s\n' "$OUT_DIR"

source "$SETUP/servers/start_servers.sh"

# shellcheck disable=SC2086
python "$SETUP/hexa/loop.py" \
    --out-dir "$OUT_DIR" \
    --suite "${PIGEY_SUITE:-libero_goal_task}" \
    --tasks ${PIGEY_TASKS:-0-9} \
    --episodes ${PIGEY_HEXA_EPISODES:-10 11 12 13 14 15 16 17 18 19} \
    --episodes-per-round "${PIGEY_HEXA_EPISODES_PER_ROUND:-1}" \
    ${initial[@]+"${initial[@]}"} \
    --policy-host localhost --policy-port "$POLICY_PORT" \
    --pid-files "$TMPDIR/policy-server.pid" "$TMPDIR/er-server.pid" "$TMPDIR/sam-server.pid" \
    "$@"
ls -la "$OUT_DIR" || true
date
