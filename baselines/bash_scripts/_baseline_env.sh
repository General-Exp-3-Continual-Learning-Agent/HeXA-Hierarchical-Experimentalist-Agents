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
BASELINE_ROOT="${BASELINE_ROOT:-$REPO_DIR/baselines/data}"
CACHE_DIR="${CACHE_DIR:-$BASELINE_ROOT/cache}"

# Local Qwen actor.  Defaults to the bare hub name, which resolves through
# HF_HOME (downloading on first use).  An already-materialised snapshot lives at
#   /scratch4/workspace/svaidyanatha_umass_edu-phyre/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/<hash>
# export QWEN_PATH=<that snapshot dir> to skip the download entirely.
QWEN_PATH="${QWEN_PATH:-Qwen/Qwen2.5-7B-Instruct}"

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
