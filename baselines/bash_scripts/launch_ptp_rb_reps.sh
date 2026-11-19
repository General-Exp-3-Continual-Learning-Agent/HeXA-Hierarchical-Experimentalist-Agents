#!/bin/bash
# Launch ReasoningBank PTP-claude REPLICATES, exact config as the existing
# (untagged = "rep1") run: baselines/data/results/reasoning_bank_pass_the_parcel_claude
#
# rep1 config (from its manifest, verified 2026-10-29):
#   level pass_the_parcel, seeds 6-106, freeze-after 51 (build 6-56, frozen 57-106)
#   model claude (actor + extractor), extractor_temperature 1.0, actor temp 0.3
#   embedding-model numeric, intent-mode geometry
#   k 5, max-items 3, max-iterations 25, max-traj-steps 12, max-obs-chars 600
#   initial-traj-dir Initial_trajectories/Pass_the_parcel
#   rep1 held-out (57-106) = 43/50 = 86.0%
#
# Each rep is ONE stream job (RB builds + freezes in a single pass). Output dirs
# are TAG'd so rep1 is never overwritten:
#   reasoning_bank_pass_the_parcel_claude_rep2, _rep3
# Resume-safe: the stream skips seeds whose trajectory JSON already exists, so a
# Claude session-limit exit just needs the SAME command resubmitted.
#
#   bash baselines/bash_scripts/launch_ptp_rb_reps.sh           # rep2 + rep3
#   REPS="rep2" bash baselines/bash_scripts/launch_ptp_rb_reps.sh   # just rep2
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
WRAP="baselines/bash_scripts/sbatch_reasoning_bank_catapult_claude.sh"   # level-agnostic
REPS="${REPS:-rep2 rep3}"

for TAG in $REPS; do
  echo "=== RB PTP $TAG (k=5, exact rep1 config) ==="
  JOB=$(LEVEL=pass_the_parcel \
        INIT=Initial_trajectories/Pass_the_parcel \
        SEEDS=6-106 FREEZE=51 K=5 EMB=numeric TAG="$TAG" \
        sbatch --parsable "$WRAP")
  echo "  $TAG job: $JOB -> baselines/data/results/reasoning_bank_pass_the_parcel_claude_$TAG"
done
echo "Submitted RB PTP reps: $REPS"
