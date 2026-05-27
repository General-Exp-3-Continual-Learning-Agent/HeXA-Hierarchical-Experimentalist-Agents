#!/bin/bash
#SBATCH --job-name=skillrl_oss120b_catapult
#SBATCH --partition=superpod-a100
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH -c 16
#SBATCH -t 1-12:00:00
#SBATCH --mem=480G
#SBATCH -o logs/skillrl_oss120b_catapult_%j.out
#SBATCH -e logs/skillrl_oss120b_catapult_%j.err
#SBATCH -A <your-slurm-account>

# ─── SkillRL evolving loop: gpt-oss-120b agent + Claude Sonnet teacher ──────
# Agent  : gpt-oss-120b via HF direct (~234GB BF16)
# Teacher: claude-sonnet-4-6 via Claude CLI (no GPU memory)
# Total  : ~234GB < 480GB (4x A100 80GB + headroom) — fits safely.

# ─── Debug / fail-fast ──────────────────────────────────────────────────────
set -Eeuo pipefail

dbg() { echo "[DBG $(date +%H:%M:%S)] $*" >&2; }
err() { echo "[ERR $(date +%H:%M:%S)] $*" >&2; }
on_err() {
    local exit_code=$?
    err "Script failed at line $1 (exit=$exit_code). Last command: $BASH_COMMAND"
    err "--- nvidia-smi at failure ---"
    nvidia-smi >&2 || true
    err "--- free -h at failure ---"
    free -h >&2 || true
    exit "$exit_code"
}
trap 'on_err $LINENO' ERR
# ────────────────────────────────────────────────────────────────────────────

GPT_OSS_HF_PATH="${GPT_OSS_HF_PATH:-/path/to/models--openai--gpt-oss-120b/snapshots/<snapshot>/}"
REPO_DIR="${REPO_DIR:-/path/to/HeXA}"
CONDA_ENV="${CONDA_ENV:-/path/to/conda/envs/phyreagent}"
HF_HOME_DIR="${HF_HOME_DIR:-/path/to/hf_cache}"

# ─── Configurable ───────────────────────────────────────────────────────────
INITIAL_TRAJ_DIR="${INITIAL_TRAJ_DIR:-"${REPO_DIR}/Initial_trajectories/catapult"}"
NUM_ROUNDS="${NUM_ROUNDS:-17}"
SEEDS_PER_ROUND="${SEEDS_PER_ROUND:-3}"
START_SEED="${START_SEED:-6}"
MAX_ITER="${MAX_ITER:-25}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEACHER_MODEL="${TEACHER_MODEL:-claude-sonnet-4-6}"
MAX_SKILLS="${MAX_SKILLS:-10}"
MAX_MISTAKES="${MAX_MISTAKES:-5}"
MAX_SPECIFIC_SKILLS="${MAX_SPECIFIC_SKILLS:-6}"
MAX_MISTAKES_AGENT="${MAX_MISTAKES_AGENT:-4}"
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "  SkillRL Evolving — GPT-OSS-120B agent + Claude Sonnet teacher"
echo "  Job ID       : ${SLURM_JOB_ID:-N/A}"
echo "  Node         : ${SLURMD_NODENAME:-N/A}"
echo "  GPUs         : ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "  Agent        : $GPT_OSS_HF_PATH"
echo "  Teacher      : $TEACHER_MODEL"
echo "  Init trajs   : $INITIAL_TRAJ_DIR"
echo "  Rounds       : $NUM_ROUNDS  Seeds/round: $SEEDS_PER_ROUND  Start seed: $START_SEED"
echo "================================================================"

# ─── Environment snapshot (stderr) ──────────────────────────────────────────
dbg "Hostname        : $(hostname)"
dbg "Date            : $(date)"
dbg "User            : $(whoami)"
dbg "PWD             : $(pwd)"
dbg "Shell           : $SHELL ($BASH_VERSION)"
dbg "SLURM_JOB_ID    : ${SLURM_JOB_ID:-N/A}"
dbg "SLURM_NODELIST  : ${SLURM_NODELIST:-N/A}"
dbg "SLURM_MEM_PER_NODE: ${SLURM_MEM_PER_NODE:-N/A}"
dbg "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"

dbg "--- nvidia-smi (pre-load) ---"
nvidia-smi >&2 || err "nvidia-smi failed"
dbg "--- free -h ---"
free -h >&2 || true
dbg "--- df -h on /scratch4 ---"
df -h /scratch4 2>&2 || true

# ─── Path validations ───────────────────────────────────────────────────────
dbg "Validating paths..."
[ -d "$REPO_DIR" ]            || { err "REPO_DIR missing: $REPO_DIR"; exit 1; }
[ -d "$CONDA_ENV" ]           || { err "CONDA_ENV missing: $CONDA_ENV"; exit 1; }
[ -d "$INITIAL_TRAJ_DIR" ]    || { err "INITIAL_TRAJ_DIR missing: $INITIAL_TRAJ_DIR"; exit 1; }
[ -d "$GPT_OSS_HF_PATH" ]     || { err "GPT_OSS_HF_PATH missing: $GPT_OSS_HF_PATH"; exit 1; }
dbg "All required paths exist."

dbg "--- GPT-OSS snapshot listing ---"
ls -lh "$GPT_OSS_HF_PATH" >&2 || true
dbg "Snapshot total size: $(du -sh "$GPT_OSS_HF_PATH" 2>/dev/null | cut -f1 || echo 'unknown')"

# ─── Conda + env vars ───────────────────────────────────────────────────────
dbg "Loading conda module..."
module load conda/latest
dbg "Activating conda env: $CONDA_ENV"
conda activate "$CONDA_ENV"

dbg "Python          : $(which python) ($(python --version 2>&1))"
dbg "Pip prefix      : $(python -c 'import sys; print(sys.prefix)')"
dbg "Torch / CUDA    : $(python -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} ngpus={torch.cuda.device_count()}")' 2>&1 || echo 'torch import failed')"
dbg "Transformers    : $(python -c 'import transformers; print(transformers.__version__)' 2>&1 || echo 'transformers import failed')"

export HF_HOME="$HF_HOME_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Surface Python tracebacks immediately even when stdout is buffered
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

dbg "HF_HOME         : $HF_HOME"
dbg "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"

mkdir -p "$REPO_DIR/logs"
cd "$REPO_DIR"

export PYTHONPATH="$(pwd)/interphyre:$(pwd):${PYTHONPATH:-}"
dbg "PYTHONPATH      : $PYTHONPATH"

# Fix tiktoken/harmony vocab download issue (known gpt-oss bug)
mkdir -p "$REPO_DIR/tiktoken_encodings"
if [ ! -f "$REPO_DIR/tiktoken_encodings/o200k_base.tiktoken" ]; then
    dbg "Downloading tiktoken vocabs..."
    wget -q -O "$REPO_DIR/tiktoken_encodings/o200k_base.tiktoken" \
        "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
    wget -q -O "$REPO_DIR/tiktoken_encodings/cl100k_base.tiktoken" \
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
else
    dbg "tiktoken vocabs already cached at $REPO_DIR/tiktoken_encodings"
fi
export TIKTOKEN_ENCODINGS_BASE="$REPO_DIR/tiktoken_encodings"

# ─── Progress file diagnostics ──────────────────────────────────────────────
PROGRESS_FILE="$REPO_DIR/skillrl/data/evolving_oss/catapult/progress_evolving.json"
if [ -f "$PROGRESS_FILE" ]; then
    dbg "Progress file exists: $PROGRESS_FILE"
    dbg "  modified: $(stat -c %y "$PROGRESS_FILE" 2>/dev/null || echo 'unknown')"
    dbg "  size    : $(stat -c %s "$PROGRESS_FILE" 2>/dev/null || echo 'unknown') bytes"
    dbg "  rounds completed: $(python -c "import json; print(len(json.load(open('$PROGRESS_FILE'))))" 2>&1 || echo 'parse failed')"
else
    dbg "No prior progress file at $PROGRESS_FILE — starting fresh."
fi

echo "[$(date)] Starting SkillRL evolving loop..."

PYTHON_START_TS=$(date +%s)
dbg "Launching python module: skillrl.distillation.teacher_prompts_catapult"
dbg "Python start timestamp: $PYTHON_START_TS"

# Disable fail-fast for the python call so we can capture its exit code and
# still run post-run diagnostics (nvidia-smi, elapsed time) before exiting.
set +e
python -u -m skillrl.distillation.teacher_prompts_catapult \
    --initial-traj-dir "$INITIAL_TRAJ_DIR" \
    --num-rounds "$NUM_ROUNDS" \
    --seeds-per-round "$SEEDS_PER_ROUND" \
    --start-seed "$START_SEED" \
    --model "$GPT_OSS_HF_PATH" \
    --teacher-model "$TEACHER_MODEL" \
    --max-iterations "$MAX_ITER" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-skills "$MAX_SKILLS" \
    --max-mistakes "$MAX_MISTAKES" \
    --max-specific-skills "$MAX_SPECIFIC_SKILLS" \
    --max-mistakes-agent "$MAX_MISTAKES_AGENT" \
    --verbose
PYTHON_EXIT=$?
set -e

PYTHON_END_TS=$(date +%s)
ELAPSED=$((PYTHON_END_TS - PYTHON_START_TS))
dbg "Python exit code: $PYTHON_EXIT"
dbg "Python elapsed  : ${ELAPSED}s ($((ELAPSED/60))m)"
dbg "--- nvidia-smi (post-run) ---"
nvidia-smi >&2 || true

echo
echo "[$(date)] Done. (exit=$PYTHON_EXIT)"
exit "$PYTHON_EXIT"
