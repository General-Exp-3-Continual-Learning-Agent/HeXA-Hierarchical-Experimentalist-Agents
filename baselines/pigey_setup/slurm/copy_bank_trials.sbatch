#!/bin/bash
#SBATCH --job-name=copy-bank-trials
#SBATCH --partition=cpu
#SBATCH --account=pi_sniekum_umass_edu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
# Seed the shadow root's with-bank dirs with Varun's finished trials (read-only on
# his side) so resume_eval.sh only reruns the unfinished episodes. Resumable: rsync.
set -euo pipefail
hostname; date
VARUN=${VARUN_ROOT:-/scratch/workspace/vgandhi_umass_edu-roboprobe-sim/pigey}
SHADOW=${SHADOW_ROOT:-/scratch3/workspace/udhanuka_umass_edu-pigey/resume_shadow_root}
for run in goal_swap spatial_task; do
    rsync -a "$VARUN/results/eval/${run}_bank/" "$SHADOW/results/eval/${run}_bank/"
    printf '%s: %s trial.json copied\n' "$run" \
        "$(find "$SHADOW/results/eval/${run}_bank" -name trial.json | wc -l)"
done
date
