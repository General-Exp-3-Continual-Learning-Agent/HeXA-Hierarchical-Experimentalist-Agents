#!/bin/bash
#SBATCH --job-name=pigey-models-setup
#SBATCH --partition=cpu
#SBATCH --account=pi_sniekum_umass_edu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# Install vLLM in its own venv and download every model Pigey needs locally:
# the ER perception VLM, SAM ViT-B, and the pi0.5 LIBERO checkpoint.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this installation with sbatch}"
export REPO_DIR="${REPO_DIR:-/project/pi_sniekum_umass_edu/vgandhi/HeXA-Hierarchical-Experimentalist-Agents}"
export PIGEY_ROOT="${PIGEY_ROOT:-/scratch/workspace/vgandhi_umass_edu-roboprobe-sim/pigey}"
export PIGEY_ER_MODEL="${PIGEY_ER_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
export UV_CACHE_DIR="$PIGEY_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$PIGEY_ROOT/python"
export XDG_CACHE_HOME="$PIGEY_ROOT/cache/xdg"
export HF_HOME="$PIGEY_ROOT/cache/huggingface"
export OPENPI_DATA_HOME="$PIGEY_ROOT/cache/openpi"
export TMPDIR="$PIGEY_ROOT/tmp/$SLURM_JOB_ID"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export UV_LINK_MODE=hardlink
export UV_CONCURRENT_DOWNLOADS=4
export UV_CONCURRENT_INSTALLS=4
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$PIGEY_ROOT/logs" "$HF_HOME" "$OPENPI_DATA_HOME"
printf 'Installing Pigey local models on %s, Slurm job %s\n' "$(hostname)" "$SLURM_JOB_ID"
date
UV="$PIGEY_ROOT/tools/bin/uv"
VENV="$PIGEY_ROOT/vllm/.venv"
if [[ ! -x "$VENV/bin/python" ]]; then
    "$UV" venv --python 3.12 --seed "$VENV"
fi
# vLLM 0.11.0 is the first release with Qwen3-VL support; Qwen3-VL needs
# transformers >= 4.57, and vLLM 0.11.0 predates transformers 5.
"$UV" pip install --python "$VENV/bin/python" 'vllm==0.11.0' 'transformers==4.57.1'
"$UV" pip check --python "$VENV/bin/python"
"$UV" pip freeze --python "$VENV/bin/python" > "$PIGEY_ROOT/requirements.perception.lock.txt"
"$VENV/bin/python" --version

# Hugging Face models: ER perception VLM and SAM (used by Pigey's spatial tools).
"$VENV/bin/python" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
root = Path(os.environ['PIGEY_ROOT'])
lines = []
for model in (os.environ['PIGEY_ER_MODEL'], 'facebook/sam-vit-base'):
    path = Path(snapshot_download(model, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja']))
    lines.append(f'{model} {path.name}')
    print(f'{model} downloaded to {path}', flush=True)
(root / 'hf-models.revision').write_text('\n'.join(lines) + '\n')
PY

# pi0.5 LIBERO policy checkpoint, via openpi's own downloader (public bucket).
"$PIGEY_ROOT/openpi/.venv/bin/python" - <<'PY'
from openpi.shared import download
path = download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero')
assert (path / 'params').is_dir(), path
print(f'pi05_libero checkpoint at {path}', flush=True)
PY
du -sh "$HF_HOME" "$OPENPI_DATA_HOME"
date
printf 'Local model setup complete under %s\n' "$PIGEY_ROOT"
