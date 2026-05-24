#!/bin/bash
#SBATCH --job-name=skilleval_oss120b_catapult
#SBATCH --partition=superpod-a100
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH -c 16
#SBATCH -t 1-00:00:00
#SBATCH --mem=500G
#SBATCH -o logs/skilleval_oss120b_catapult_%j.out
#SBATCH -e logs/skilleval_oss120b_catapult_%j.err
#SBATCH -A pi_sniekum_umass_edu

# ─── SkillRL final eval: gpt-oss-120b with a fixed skill bank ────────────────
# Runs run_skill_agent (augmented_runner) on a held-out seed range using
# a skill bank produced by the evolving loop.

# ─── Debug / fail-fast ──────────────────────────────────────────────────────
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
# ────────────────────────────────────────────────────────────────────────────

GPT_OSS_HF_PATH="${GPT_OSS_HF_PATH:-/datasets/ai/gpt/hub/models--openai--gpt-oss-120b/snapshots/eabf0c518da7584a2e7dab4ab272709785a72126/}"
REPO_DIR="/home/udhanuka_umass_edu/RL Med/physics-reasoning-agents"
CONDA_ENV="/home/udhanuka_umass_edu/.conda/envs/phyreagent"
HF_HOME_DIR="${HF_HOME_DIR:-/scratch4/workspace/svaidyanatha_umass_edu-phyre/hf_cache}"

# ─── Configurable ────────────────────────────────────────────────────────────
LEVEL="${LEVEL:-catapult}"
SEED_START="${SEED_START:-62}"
SEED_END="${SEED_END:-99}"
SKILL_BANK="${SKILL_BANK:-"$REPO_DIR/skillrl/data/evolving_oss/catapult/skill_bank_evolving_17.json"}"
EVAL_DIR="${EVAL_DIR:-"$REPO_DIR/skillrl/data/evolving_oss/catapult/final_eval"}"
MAX_ITER="${MAX_ITER:-25}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_SPECIFIC_SKILLS="${MAX_SPECIFIC_SKILLS:-6}"
MAX_MISTAKES="${MAX_MISTAKES:-4}"
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "  SkillRL final eval — gpt-oss-120b on 4x A100"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Node     : $SLURMD_NODENAME"
echo "  Model    : $GPT_OSS_HF_PATH"
echo "  GPUs     : $CUDA_VISIBLE_DEVICES"
echo "  Level    : $LEVEL"
echo "  Seeds    : $SEED_START to $SEED_END"
echo "  Skill bk : $SKILL_BANK"
echo "  Eval dir : $EVAL_DIR"
echo "================================================================"

# ─── Pre-flight checks ───────────────────────────────────────────────────────
dbg "Checking paths..."
[ -d "$REPO_DIR" ]      || { err "REPO_DIR not found: $REPO_DIR"; exit 1; }
[ -d "$CONDA_ENV" ]     || { err "CONDA_ENV not found: $CONDA_ENV"; exit 1; }
[ -f "$SKILL_BANK" ]    || { err "SKILL_BANK not found: $SKILL_BANK"; exit 1; }
[ -d "$GPT_OSS_HF_PATH" ] || { err "GPT_OSS_HF_PATH not found: $GPT_OSS_HF_PATH"; exit 1; }
dbg "All paths OK."

dbg "GPU state at start:"
nvidia-smi >&2 || true
free -h >&2 || true
# ─────────────────────────────────────────────────────────────────────────────

module load conda/latest
conda activate "$CONDA_ENV"

export HF_HOME="$HF_HOME_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$(pwd)/interphyre:$(pwd):${PYTHONPATH:-}"

# Fix tiktoken/harmony vocab download issue (known gpt-oss bug)
mkdir -p "$REPO_DIR/tiktoken_encodings"
if [ ! -f "$REPO_DIR/tiktoken_encodings/o200k_base.tiktoken" ]; then
    wget -q -O "$REPO_DIR/tiktoken_encodings/o200k_base.tiktoken" \
        "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
    wget -q -O "$REPO_DIR/tiktoken_encodings/cl100k_base.tiktoken" \
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
fi
export TIKTOKEN_ENCODINGS_BASE="$REPO_DIR/tiktoken_encodings"

mkdir -p "$REPO_DIR/logs"
mkdir -p "$EVAL_DIR"
cd "$REPO_DIR"

dbg "Python/torch versions:"
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| GPUs', torch.cuda.device_count())" >&2 || true

echo "[$(date)] Starting skill eval: level=$LEVEL seeds=$SEED_START..$SEED_END"

python -m skillrl.run_skill_agent \
    --level "$LEVEL" \
    --seeds $(seq $SEED_START $SEED_END) \
    --skill-bank "$SKILL_BANK" \
    --model "$GPT_OSS_HF_PATH" \
    --eval-dir "$EVAL_DIR" \
    --max-iterations "$MAX_ITER" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-specific-skills "$MAX_SPECIFIC_SKILLS" \
    --max-mistakes "$MAX_MISTAKES" \
    --verbose

dbg "GPU state at end:"
nvidia-smi >&2 || true

echo "[$(date)] Done. Results in $EVAL_DIR/"
