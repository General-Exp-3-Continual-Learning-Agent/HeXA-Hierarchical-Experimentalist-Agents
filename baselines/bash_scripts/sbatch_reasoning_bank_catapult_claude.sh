#!/bin/bash
#SBATCH --job-name=rbank_cat_claude
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 24:00:00
#SBATCH -A pi_sniekum_umass_edu
#SBATCH -o logs/rbank_cat_claude_%j.out
#SBATCH -e logs/rbank_cat_claude_%j.err
#
# ReasoningBank on catapult with Claude Sonnet 4.6 as BOTH actor and extractor,
# all via the `claude -p` CLI -- the same actor loader/model as HeXA's and ACE's
# catapult Claude runs. No GPU: every model call is a CLI call; 4 CPU / 16G cover
# the simulator and the numeric (geometry) retrieval scaler.
#
# Protocol matches HeXA/ACE catapult Claude: seeds 6-56 online, 57-106 frozen
# held-out (run as one stream --seeds 6-106 --freeze-after 51), seeded with the 5
# Initial_trajectories/catapult trajectories, numeric k=1 retrieval, env reward.
#
# Resumable: the stream skips seeds whose trajectory JSON already exists, so a
# Claude session-limit exit (nonzero) just needs the SAME command resubmitted.
#
# Overridable via env: SEEDS, FREEZE, TAG, K, EMB
#   sbatch baselines/bash_scripts/sbatch_reasoning_bank_catapult_claude.sh
#   TAG=rep2 sbatch baselines/bash_scripts/sbatch_reasoning_bank_catapult_claude.sh
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"  # submit from the repo root
PY="${PY:-python}"
LEVEL="${LEVEL:-catapult}"
INIT="${INIT:-Initial_trajectories/catapult}"
SEEDS="${SEEDS:-6-106}"
FREEZE="${FREEZE:-51}"        # seed 57 is stream index 51 (6..56 = 51 build seeds)
K="${K:-1}"
EMB="${EMB:-numeric}"
TAG="${TAG:-}"

export PATH="$HOME/.local/bin:$PATH"          # claude CLI
export PYTHONPATH="$REPO:$REPO/interphyre:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CUDA_VISIBLE_DEVICES=""
cd "$REPO"
mkdir -p logs

OUT="${OUT:-baselines/data/results/reasoning_bank_${LEVEL}_claude${TAG:+_$TAG}}"

echo "================================================================"
echo "  ReasoningBank on catapult (Claude Sonnet 4.6 actor + extractor)"
echo "  Job    : ${SLURM_JOB_ID:-<interactive>} on $(hostname)  $(date)"
echo "  Seeds  : $SEEDS   Freeze-after: ${FREEZE:-<none>}   k: $K   emb: $EMB"
echo "  Init   : $INIT"
echo "  Output : $OUT"
echo "================================================================"
which claude && claude --version
"$PY" --version

args=(
    --level "$LEVEL"
    --seeds "$SEEDS"
    --model claude
    --initial-traj-dir "$INIT"
    --output-dir "$OUT"
    --cache-dir baselines/data/cache
    --embedding-model "$EMB"
    --intent-mode geometry
    --k "$K"
    --max-iterations 25
    --verbose
)
[[ -n "$FREEZE" ]] && args+=(--freeze-after "$FREEZE")

"$PY" -m baselines.runner.run_reasoning_bank "${args[@]}"
echo "Done $(date). Results in $OUT"
