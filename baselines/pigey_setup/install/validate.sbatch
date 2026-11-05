#!/bin/bash
#SBATCH --job-name=pigey-validate
#SBATCH --partition=cpu
#SBATCH --account=pi_sniekum_umass_edu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00

set -euo pipefail
: "${SLURM_JOB_ID:?Run runtime validation inside a Slurm allocation}"
export REPO_DIR="${REPO_DIR:-/home/udhanuka_umass_edu/Scalar_Research/HeXA}"
source "$REPO_DIR/baselines/pigey_setup/activate.sh"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export TMPDIR="$PIGEY_ROOT/tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"

# LIBERO-PRO's setup.py discovers no packages because its outer libero folder
# lacks __init__.py. Preserve the namespace layout using a source-root .pth.
python - <<'PY'
import os
from pathlib import Path
import sysconfig
(Path(sysconfig.get_path('purelib')) / 'libero_pro_source.pth').write_text(
    os.environ['PIGEY_ROOT'] + '/LIBERO-PRO\n'
)
PY
python "$PIGEY_REPO/sim/agent_sim.py" --help > "$PIGEY_ROOT/logs/harness-help.txt" 2>&1
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python "$REPO_DIR/baselines/pigey_setup/install/smoke_check.py" \
    > "$PIGEY_ROOT/logs/simulator-check.txt" 2>&1
cd "$PIGEY_ROOT/openpi"
JAX_PLATFORMS=cpu "$PIGEY_ROOT/openpi/.venv/bin/python" scripts/serve_policy.py --help \
    > "$PIGEY_ROOT/logs/policy-help.txt" 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$PIGEY_ROOT/SETUP_COMPLETE"
printf 'Pigey runtime checks passed: %s\n' "$PIGEY_ROOT"
