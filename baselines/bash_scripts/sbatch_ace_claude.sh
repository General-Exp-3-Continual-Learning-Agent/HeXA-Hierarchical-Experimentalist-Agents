#!/bin/bash
#SBATCH --job-name=ace_claude
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH --mem=16G
#SBATCH -o logs/ace_claude_%j.out
#SBATCH -e logs/ace_claude_%j.err
#
# ACE (Agentic Context Engineering) on InterPhyre with Claude Sonnet 4.6 as
# Generator, Reflector and Curator (ACE's same-model setting), all via the
# `claude -p` CLI -- the same actor loader and model as HExA's catapult Claude
# runs (physics-reasoning-agents/skillrl/RESULTS/catapult_claude/HeXA_catapult).
#
# No GPU: every model call is a CLI call.  4 CPUs / 16G cover the simulator
# and the all-mpnet-base-v2 embeddings that --dedup computes on CPU.
#
# Protocol (matches HExA catapult Claude): seeds 6-56 online (17 rounds x 3),
# seeded with the 5 initial trajectories in Initial_trajectories/catapult,
# then the final playbook frozen and evaluated on 57-106.
#
# A Claude usage/session limit raises SessionLimitError and the job exits
# nonzero with ace_state.json intact; resubmitting the SAME command resumes
# (finished seeds are skipped, a lagging curation is redone from the cached
# trajectory).  The frozen eval is chained with afterok, so it never starts on
# a half-built playbook.
#
# Usage (submit from the repo root so logs/ resolves):
#   INIT=/home/udhanuka_umass_edu/Scalar_Research/physics-reasoning-agents/Initial_trajectories/catapult
#   # smoke test: 2 seeds, default config
#   sbatch -t 01:00:00 baselines/bash_scripts/sbatch_ace_claude.sh --level catapult --seeds 6-7 --tag smoke
#   # paper configuration, then frozen held-out eval
#   sbatch baselines/bash_scripts/sbatch_ace_claude.sh --level catapult --init-dir $INIT \
#       --extra "--resolve-rounds 5 --dedup" --tag paperconfig
#   sbatch --dependency=afterok:<jobid> baselines/bash_scripts/sbatch_ace_claude.sh --level catapult \
#       --seeds 57-106 --frozen-from baselines/data/results/ace_catapult_claude_s6-56_warmup_paperconfig --tag paperconfig
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

LEVEL=""
SEEDS="6-56"
INIT_DIR=""
FROZEN_FROM=""
TAG=""
EXTRA=""
CLAUDE_MODEL="claude-sonnet-4-6"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --level) LEVEL="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --tag)   TAG="$2";   shift 2 ;;
        --init-dir) INIT_DIR="$2"; shift 2 ;;
        --frozen-from) FROZEN_FROM="$2"; shift 2 ;;
        --extra) EXTRA="$2"; shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -n "$LEVEL" ]] || { echo "[ERROR] --level is required"; exit 1; }

REPO_DIR="${REPO_DIR:-/home/udhanuka_umass_edu/Scalar_Research/HeXA}"
CONDA_ENV="${CONDA_ENV:-/project/pi_sniekum_umass_edu/vgandhi/conda/envs/VerlToolInterphyre}"
RUN_NAME="ace_${LEVEL}_claude_s${SEEDS//,/_}${INIT_DIR:+_warmup}${FROZEN_FROM:+_frozen}${TAG:+_$TAG}"
OUTPUT_DIR="${BASELINE_ROOT:-$REPO_DIR/baselines/data_ace/results}/$RUN_NAME"

module load conda/latest 2>/dev/null || true
export PATH="$CONDA_ENV/bin:$HOME/.local/bin:$PATH"
export PYTHONPATH="$REPO_DIR:$REPO_DIR/interphyre:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
unset HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE HF_HUB_ENABLE_HF_TRANSFER \
      HF_HUB_TEMP_DIR HF_HUB_LOCAL_DIR_AUTO_EXTRACT
export HF_HOME="$REPO_DIR/baselines/data_ace/cache/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TIKTOKEN_CACHE_DIR="$REPO_DIR/baselines/data_ace/cache/tiktoken"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CUDA_VISIBLE_DEVICES=""

cd "$REPO_DIR"
mkdir -p logs "$OUTPUT_DIR"

echo "================================================================"
echo "  ACE baseline (Claude actor = Reflector = Curator, via claude -p)"
echo "  Job        : ${SLURM_JOB_ID:-<interactive>} on $(hostname)  $(date)"
echo "  Level      : $LEVEL   seeds $SEEDS"
echo "  Model      : $CLAUDE_MODEL"
echo "  Warmup     : ${INIT_DIR:-<none>}"
echo "  Frozen from: ${FROZEN_FROM:-<no: playbook learns online>}"
echo "  Output     : $OUTPUT_DIR"
echo "  Extra      : ${EXTRA:-<none>}"
echo "================================================================"
module list 2>&1 || true
which python && python --version
which claude && claude --version
[[ -z "$INIT_DIR" || -d "$INIT_DIR" ]] || { echo "[ERROR] no offline trajectories: $INIT_DIR"; exit 1; }
git -C "$REPO_DIR" rev-parse HEAD > "$OUTPUT_DIR/code_rev.txt" 2>/dev/null || true
git -C "$REPO_DIR" diff -- react_agent interphyre > "$OUTPUT_DIR/code_diff.patch" 2>/dev/null || true
sha1sum baselines/ace/*.py baselines/runner/run_ace.py baselines/runner/memory_runner.py \
    baselines/memory/trajectory_fmt.py react_agent/run_react_claude.py > "$OUTPUT_DIR/code_sha1.txt"

INIT_ARGS=()
[[ -n "$INIT_DIR" ]] && INIT_ARGS=(--initial-traj-dir "$INIT_DIR")
if [[ -n "$FROZEN_FROM" ]]; then
    [[ "$FROZEN_FROM" = /* ]] || FROZEN_FROM="$REPO_DIR/$FROZEN_FROM"
    INIT_ARGS+=(--init-from "$FROZEN_FROM" --freeze-after 0)
fi

# shellcheck disable=SC2206
python -m baselines.runner.run_ace \
    --level "$LEVEL" \
    --seeds "$SEEDS" \
    --seeds-per-round 3 \
    --max-iterations 25 \
    "${INIT_ARGS[@]}" \
    --model claude \
    --claude-model "$CLAUDE_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --verbose \
    ${EXTRA}

echo "Done $(date). Results in $OUTPUT_DIR"
