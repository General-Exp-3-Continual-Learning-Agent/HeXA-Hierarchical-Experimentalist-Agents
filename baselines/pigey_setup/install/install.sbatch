#!/bin/bash
#SBATCH --job-name=pigey-setup
#SBATCH --partition=cpu
#SBATCH --account=pi_sniekum_umass_edu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this installation with sbatch}"
export REPO_DIR="${REPO_DIR:-/home/udhanuka_umass_edu/Scalar_Research/HeXA}"
export PIGEY_ROOT="${PIGEY_ROOT:-/scratch/workspace/vgandhi_umass_edu-roboprobe-sim/pigey}"
export UV_CACHE_DIR="$PIGEY_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$PIGEY_ROOT/python"
export PIP_CACHE_DIR="$PIGEY_ROOT/cache/pip"
export XDG_CACHE_HOME="$PIGEY_ROOT/cache/xdg"
export HF_HOME="$PIGEY_ROOT/cache/huggingface"
export OPENPI_DATA_HOME="$PIGEY_ROOT/cache/openpi"
export NUMBA_CACHE_DIR="$PIGEY_ROOT/cache/numba"
export TMPDIR="$PIGEY_ROOT/tmp/$SLURM_JOB_ID"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export UV_CONCURRENT_BUILDS="$SLURM_CPUS_PER_TASK"
export UV_LINK_MODE=hardlink
export UV_CONCURRENT_DOWNLOADS=4
export UV_CONCURRENT_INSTALLS=4
export GIT_LFS_SKIP_SMUDGE=1
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$PIGEY_ROOT/logs" "$PIGEY_ROOT/config/libero" "$PIGEY_ROOT/results" "$PIGEY_ROOT/datasets"
printf 'Installing Pigey on %s, Slurm job %s\n' "$(hostname)" "$SLURM_JOB_ID"
if [[ ! -x "$PIGEY_ROOT/tools/bin/uv" ]]; then
    /modules/opt/linux-ubuntu24.04-x86_64/miniforge3/24.7.1/bin/python -m venv "$PIGEY_ROOT/tools"
    "$PIGEY_ROOT/tools/bin/python" -m pip install uv
fi
UV="$PIGEY_ROOT/tools/bin/uv"
if [[ ! -d "$PIGEY_ROOT/LIBERO-PRO/.git" ]]; then
    git clone --depth 1 https://github.com/Zxy-MLlab/LIBERO-PRO.git "$PIGEY_ROOT/LIBERO-PRO"
fi
if [[ ! -d "$PIGEY_ROOT/openpi/.git" ]]; then
    git clone --depth 1 https://github.com/Physical-Intelligence/openpi.git "$PIGEY_ROOT/openpi"
fi
git -C "$PIGEY_ROOT/openpi" submodule update --init --recursive --depth 1
if [[ ! -x "$PIGEY_ROOT/.venv/bin/python" ]]; then
    "$UV" venv --python 3.10 --seed "$PIGEY_ROOT/.venv"
fi

# Modern LiteLLM needs newer tokenizers than transformers 4.21 permits.
# Retain the other LIBERO-PRO pins; keep server dependencies in another venv.
"$PIGEY_ROOT/tools/bin/python" - <<'PY'
import os
from pathlib import Path
root = Path(os.environ['PIGEY_ROOT'])
lines = (root / 'LIBERO-PRO/requirements.txt').read_text().splitlines()
lines = [line.strip() for line in lines if not line.strip().startswith('transformers==')]
lines += ['transformers>=4.53,<5', 'mujoco==2.3.7', 'litellm', 'anthropic', 'openai',
          'imageio[ffmpeg]', 'imageio-ffmpeg', 'torch==1.11.0+cpu',
          'torchvision==0.12.0+cpu', 'torchaudio==0.11.0+cpu']
(root / 'requirements.sim.txt').write_text('\n'.join(lines) + '\n')
libero = root / 'LIBERO-PRO/libero/libero'
paths = {'benchmark_root': libero, 'bddl_files': libero / 'bddl_files',
         'init_states': libero / 'init_files', 'assets': libero / 'assets',
         'datasets': root / 'datasets'}
(root / 'config/libero/config.yaml').write_text(''.join(f'{key}: {value}\n' for key, value in paths.items()))
PY
"$UV" pip install --python "$PIGEY_ROOT/.venv/bin/python" \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
    -r "$PIGEY_ROOT/requirements.sim.txt" \
    -e "$PIGEY_ROOT/LIBERO-PRO" -e "$PIGEY_ROOT/openpi/packages/openpi-client"
"$UV" pip check --python "$PIGEY_ROOT/.venv/bin/python"
"$UV" pip freeze --python "$PIGEY_ROOT/.venv/bin/python" > "$PIGEY_ROOT/requirements.sim.lock.txt"
"$PIGEY_ROOT/.venv/bin/python" "$REPO_DIR/baselines/pigey_setup/install/prepare_tasks.py"

cd "$PIGEY_ROOT/openpi"
"$UV" sync --locked --no-dev --python 3.11
"$UV" pip check --python "$PIGEY_ROOT/openpi/.venv/bin/python"
"$UV" pip freeze --python "$PIGEY_ROOT/openpi/.venv/bin/python" > "$PIGEY_ROOT/requirements.policy.lock.txt"
git -C "$PIGEY_ROOT/LIBERO-PRO" rev-parse HEAD > "$PIGEY_ROOT/libero-pro.revision"
git -C "$PIGEY_ROOT/openpi" rev-parse HEAD > "$PIGEY_ROOT/openpi.revision"
git -C "$REPO_DIR/baselines/Pigey" rev-parse HEAD > "$PIGEY_ROOT/pigey.revision"

bash "$REPO_DIR/baselines/pigey_setup/install/validate.sbatch"
printf 'Pigey setup complete: %s\n' "$PIGEY_ROOT"
