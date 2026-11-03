#!/bin/bash
# Launch ACE PTP rep3, STRICTLY matched to run1 (ACE_PTP_claude_run1).
#
# run1 config (from its manifest, verified 2026-10-29):
#   warmup seeds 6-56, seeds-per-round 3, max-iter 25, claude-sonnet-4-6
#   --dedup (threshold 0.9)  --edit-ops   (grow-and-refine + edit ops ON)
#   initial-traj-dir = physics-reasoning-agents/Initial_trajectories/Pass_the_parcel
#   eval seeds 57-106, frozen-from the warmup (dedup/edit-ops are inert at freeze)
#
# DEVIATION from run1 (deliberate): --token-cap 10000 on the warmup, bounding the
# learned playbook to <=10k tokens (run1/rep2 had no cap: 4k and 18k respectively).
# The cap only acts during warmup curation; it is inert at frozen eval.
#
# Guards the two footguns:
#   1. BASELINE_ROOT -> the ACE_PTP_claude_reps/ subdir (NOT the default results/),
#      so rep3 lands beside run1/rep2 and resume-by-file works.
#   2. --dedup --edit-ops on the warmup (the actual run1 method config).
#
# Chains eval afterok:<warmup> so the frozen eval never starts on a half-built
# playbook. Resume-safe: re-submitting skips seeds whose trajectory file exists.
#
#   bash baselines/bash_scripts/launch_ptp_ace_rep3.sh
set -euo pipefail

REPO_DIR="/home/udhanuka_umass_edu/Scalar_Research/HeXA"
cd "$REPO_DIR"
export BASELINE_ROOT="$REPO_DIR/baselines/data_ace/results/ACE_PTP_claude_reps"
INIT="/home/udhanuka_umass_edu/Scalar_Research/physics-reasoning-agents/Initial_trajectories/Pass_the_parcel"
WARMUP_DIR="$BASELINE_ROOT/ace_pass_the_parcel_claude_s6-56_warmup_rep3"

[ -d "$INIT" ] || { echo "ERROR: missing init-traj dir $INIT"; exit 1; }

echo "=== rep3 warmup (seeds 6-56, --dedup --edit-ops, matched to run1) ==="
WJOB=$(sbatch --parsable --export=ALL,BASELINE_ROOT="$BASELINE_ROOT" \
    baselines/bash_scripts/sbatch_ace_claude.sh \
    --level pass_the_parcel --seeds 6-56 \
    --init-dir "$INIT" \
    --extra "--dedup --dedup-threshold 0.9 --edit-ops --token-cap 10000" \
    --tag rep3)
echo "  warmup job: $WJOB -> $WARMUP_DIR"

echo "=== rep3 frozen eval (seeds 57-106, afterok:$WJOB) ==="
EJOB=$(sbatch --parsable --dependency=afterok:"$WJOB" \
    --export=ALL,BASELINE_ROOT="$BASELINE_ROOT" \
    baselines/bash_scripts/sbatch_ace_claude.sh \
    --level pass_the_parcel --seeds 57-106 \
    --frozen-from "$WARMUP_DIR" \
    --tag rep3)
echo "  eval job: $EJOB (pending on warmup)"
echo
echo "Submitted rep3: warmup $WJOB, eval $EJOB."
