#!/bin/bash
#SBATCH --job-name=pigey-er-download
#SBATCH --partition=cpu
#SBATCH --account=pi_sniekum_umass_edu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00

# Download candidate ER detector models into the scratch Hugging Face cache.
# Usage: sbatch ... download_er_models.sbatch MODEL [MODEL ...]
# Gated models use the existing Hugging Face login token; a model whose
# license has not been accepted fails without stopping the others.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this download with sbatch}"
(( $# )) || { echo "Name at least one model" >&2; exit 2; }
export PIGEY_ROOT="${PIGEY_ROOT:-/scratch/workspace/vgandhi_umass_edu-roboprobe-sim/pigey}"
export HF_HOME="$PIGEY_ROOT/cache/huggingface"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
hostname
date
VENV="$PIGEY_ROOT/vllm/.venv"
which "$VENV/bin/python" && "$VENV/bin/python" --version
"$VENV/bin/python" - "$@" <<'PY'
import os, sys
from pathlib import Path
from huggingface_hub import snapshot_download
root = Path(os.environ['PIGEY_ROOT'])
failed = []
for model in sys.argv[1:]:
    try:
        path = Path(snapshot_download(model, max_workers=4, allow_patterns=[
            '*.json', '*.safetensors', '*.txt', '*.model', '*.jinja', '*.py']))
    except Exception as exc:
        failed.append(model)
        print(f'{model} FAILED: {type(exc).__name__}: {str(exc)[:300]}', flush=True)
        continue
    with open(root / 'hf-models.revision', 'a') as f:
        f.write(f'{model} {path.name}\n')
    print(f'{model} downloaded to {path}', flush=True)
sys.exit(1 if failed else 0)
PY
du -sh "$HF_HOME"
date
