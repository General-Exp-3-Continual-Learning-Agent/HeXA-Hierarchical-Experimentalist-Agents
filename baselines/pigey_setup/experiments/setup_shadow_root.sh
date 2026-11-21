#!/bin/bash
# Build a private "shadow" PIGEY_ROOT so resume_eval.sh can run unchanged without
# write access to Varun's logs/ and *_bank dirs. Environment, caches, models, banks
# and no-bank results are symlinked read-through to Varun's root; logs/, tmp/ and the
# with-bank result dirs are real directories in your own workspace.
# Login-node safe: only mkdir/ln. Trial copying is done by slurm/copy_bank_trials.sbatch.
set -euo pipefail
VARUN=${VARUN_ROOT:-/scratch/workspace/vgandhi_umass_edu-roboprobe-sim/pigey}
SHADOW=${SHADOW_ROOT:-/scratch3/workspace/udhanuka_umass_edu-pigey/resume_shadow_root}
mkdir -p "$SHADOW/logs" "$SHADOW/tmp" "$SHADOW/results/eval" "$SHADOW/results/hexa"
for name in .venv cache config datasets LIBERO-PRO openpi python tools vllm \
    SETUP_COMPLETE hf-models.revision libero-pro-assets.revision libero-pro.revision \
    openpi.revision pigey.revision; do
    [[ -e "$VARUN/$name" ]] && ln -sfn "$VARUN/$name" "$SHADOW/$name"
done
for run in goal_swap spatial_task; do
    ln -sfn "$VARUN/results/hexa/${run}_train" "$SHADOW/results/hexa/${run}_train"
    ln -sfn "$VARUN/results/eval/${run}_nobank" "$SHADOW/results/eval/${run}_nobank"
    mkdir -p "$SHADOW/results/eval/${run}_bank"
done
echo "shadow root ready: $SHADOW"
ls -la "$SHADOW" "$SHADOW/results/eval"
