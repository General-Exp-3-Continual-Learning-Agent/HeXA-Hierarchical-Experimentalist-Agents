#!/bin/bash
#SBATCH --job-name=qwen14b_two_body
#SBATCH -G 1
#SBATCH --partition=gpu
#SBATCH -c 4
#SBATCH -t 10:00:00
#SBATCH --mem=64G
#SBATCH --constraint=vram48
#SBATCH -o logs/qwen14b_two_body_%j.out
#SBATCH -e logs/qwen14b_two_body_%j.err
#SBATCH -A pi_sniekum_umass_edu

# ─── Qwen2.5-14B-Instruct on 1x 48GB GPU (two_body_problem) ───────────────────
# Loads from a local snapshot to skip HF Hub revalidation.

# ─── Debug / fail-fast ────────────────────────────────────────────────────────
set -Eeuo pipefail

dbg() { echo "[DBG $(date +%H:%M:%S)] $*" >&2; }
err() { echo "[ERR $(date +%H:%M:%S)] $*" >&2; }
on_err() {
    local exit_code=$?
    err "Script failed at line $1 (exit=$exit_code). Last command: $BASH_COMMAND"
    nvidia-smi >&2 || true
    free -h >&2 || true
    exit "$exit_code"
}
trap 'on_err $LINENO' ERR
# ──────────────────────────────────────────────────────────────────────────────

REPO_DIR="/home/udhanuka_umass_edu/RL Med/physics-reasoning-agents"
CONDA_ENV="/home/udhanuka_umass_edu/.conda/envs/phyreagent"
HF_HOME_DIR="${HF_HOME_DIR:-/scratch4/workspace/svaidyanatha_umass_edu-phyre/hf_cache}"
QWEN_PATH="${QWEN_PATH:-/datasets/ai/qwen2/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8/}"

# ─── Configurable ─────────────────────────────────────────────────────────────
LEVEL="${LEVEL:-tipping_point}"
SEED_START="${SEED_START:-26}"
SEED_END="${SEED_END:-50}"
MAX_ITER="${MAX_ITER:-25}"
EVAL_DIR="${EVAL_DIR:-eval_results_${LEVEL}_qwen2.5_14b}"
# ──────────────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "  Qwen2.5-14B-Instruct on 1x A40 (vram48)"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Node    : $SLURMD_NODENAME"
echo "  Model   : $QWEN_PATH"
echo "  Level   : $LEVEL"
echo "  Seeds   : $SEED_START to $SEED_END"
echo "  EvalDir : $EVAL_DIR"
echo "================================================================"

# ─── Pre-flight checks ────────────────────────────────────────────────────────
dbg "Checking paths..."
[ -d "$REPO_DIR" ]   || { err "REPO_DIR not found: $REPO_DIR"; exit 1; }
[ -d "$CONDA_ENV" ]  || { err "CONDA_ENV not found: $CONDA_ENV"; exit 1; }
[ -d "$QWEN_PATH" ]  || { err "QWEN_PATH not found: $QWEN_PATH"; exit 1; }
dbg "Paths OK."

dbg "GPU state at start:"
nvidia-smi >&2 || true
# ──────────────────────────────────────────────────────────────────────────────

module load conda/latest
conda activate "$CONDA_ENV"

export HF_HOME="$HF_HOME_DIR"
# Skip HF Hub revalidation/telemetry — the snapshot is already on disk.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO_DIR/interphyre:$REPO_DIR:${PYTHONPATH:-}"

mkdir -p "$REPO_DIR/logs"
cd "$REPO_DIR"

dbg "Python/torch versions:"
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| GPUs', torch.cuda.device_count())" >&2 || true

echo "[$(date)] Starting eval: level=$LEVEL seeds=$SEED_START..$SEED_END"

python -m react_agent.run_react \
    --level "$LEVEL" \
    --model "$QWEN_PATH" \
    --seeds $(seq $SEED_START $SEED_END) \
    --max-iterations "$MAX_ITER" \
    --eval-dir "$EVAL_DIR" \
    --verbose

python react_agent/plot_eval.py --eval-dir "$EVAL_DIR" 2>/dev/null || true

dbg "GPU state at end:"
nvidia-smi >&2 || true

echo "[$(date)] Done. Results in $EVAL_DIR/"
