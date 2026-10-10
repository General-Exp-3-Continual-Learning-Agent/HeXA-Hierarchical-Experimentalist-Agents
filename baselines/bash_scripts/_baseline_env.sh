#!/bin/bash
# Shared environment for the memory-baseline sbatch scripts.
#
# Sourced, not executed.  Sets the conda python, PYTHONPATH (so the vendored
# interphyre package resolves), the HF cache, and the scratch output roots.
# Override any of these by exporting them before sbatch.

REPO_DIR="${REPO_DIR:-/project/pi_sniekum_umass_edu/vgandhi/HExA_Baseline_Exp/HeXA-Hierarchical-Experimentalist-Agents}"
CONDA_ENV="${CONDA_ENV:-/project/pi_sniekum_umass_edu/vgandhi/conda/envs/VerlToolInterphyre}"
PYTHON="$CONDA_ENV/bin/python"

# Model weights stay on scratch (tens of GB); run outputs live in the repo so
# results sit next to the code that produced them. Override either with
# WORK_DIR / BASELINE_ROOT.
WORK_DIR="${WORK_DIR:-/scratch4/workspace/vgandhi_umass_edu-grasp}"
# Pin HF_HOME rather than deferring to an inherited value: sbatch --export=ALL
# carries the submitting shell's environment, and a stale HF_HOME can point at a
# cache directory we do not own (harmless while nothing needs downloading, a hard
# failure the moment something does). Set HF_CACHE_DIR to override deliberately.
export HF_HOME="${HF_CACHE_DIR:-$WORK_DIR/hf_cache}"

# OpenAI credentials for --embedding-model openai:text-embedding-3-large (the
# embedder MemRL's own config specifies). Kept outside the repo: this tree has a
# remote, so a key in a tracked file would be pushed. Override with
# OPENAI_ENV_FILE, or just export OPENAI_API_KEY yourself.
OPENAI_ENV_FILE="${OPENAI_ENV_FILE:-$HOME/.config/interphyre/openai.env}"
if [[ -z "${OPENAI_API_KEY:-}" && -r "$OPENAI_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$OPENAI_ENV_FILE"
fi
BASELINE_ROOT="${BASELINE_ROOT:-$REPO_DIR/baselines/data/results}"
# Cache is shared across ALL runs (intents, embeddings, calibrations), so it is
# anchored independently of BASELINE_ROOT — otherwise pointing results elsewhere
# would orphan the calibrations and force a re-measure.
CACHE_DIR="${CACHE_DIR:-$REPO_DIR/baselines/data/cache}"

# Local Qwen actor.  Defaults to the bare hub name, which resolves through
# HF_HOME (downloading on first use).  An already-materialised snapshot lives at
#   /scratch4/workspace/svaidyanatha_umass_edu-phyre/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/<hash>
# export QWEN_PATH=<that snapshot dir> to skip the download entirely.
QWEN_PATH="${QWEN_PATH:-Qwen/Qwen2.5-7B-Instruct}"

# Other actors, materialised on this cluster (the public react_agent scripts have
# these paths scrubbed to /path/to/...<snapshot>/, so they are recorded here).
QWEN3_PATH="${QWEN3_PATH:-/scratch4/workspace/svaidyanatha_umass_edu-phyre/hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1}"
GPT_OSS_PATH="${GPT_OSS_PATH:-/datasets/ai/gpt/hub/models--openai--gpt-oss-120b/snapshots/eabf0c518da7584a2e7dab4ab272709785a72126}"

module load conda/latest 2>/dev/null || true
module load cuda/12.8 2>/dev/null || true

export PATH="$CONDA_ENV/bin:${PATH:-}"
export PYTHONPATH="$REPO_DIR:$REPO_DIR/interphyre:${PYTHONPATH:-}"
# Tokenizers' fork warnings are noise in batch logs.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# Slurm redirects stdout to a file, which makes Python block-buffer it — progress
# then looks frozen for minutes. Stream it instead.
export PYTHONUNBUFFERED=1

mkdir -p "$BASELINE_ROOT" "$CACHE_DIR" "$REPO_DIR/logs"
cd "$REPO_DIR"
