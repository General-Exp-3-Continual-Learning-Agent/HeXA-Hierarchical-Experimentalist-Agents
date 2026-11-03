#!/bin/bash
# Queue ACE paper-config (5 reflect-and-re-solve rounds + dedup) on Qwen2.5-3B,
# DTE and TBP, 3 independent builds each, each followed by a frozen held-out
# eval on 57-106 (afterok).  Mirrors the 7B ACE runs and HExA's 3B builds
# (physics-reasoning-agents/skillrl/run_hexa_builds_qwen3b.sh, job 64508950):
# train 6-56 in 17 rounds of 3, same initial trajectories -- DTE_Qwen3B (identical
# to DTE_Qwen7B) and TwoBody_7B (no 3B TBP set exists).
#
# Usage (from the repo root; submits 12 one-GPU jobs: 6 trainings + 6 frozen evals):
#   bash baselines/bash_scripts/queue_ace_qwen3b.sh                 # both levels, runs 1-3
#   LEVELS=two_body_problem RUNS="2 3" bash baselines/bash_scripts/queue_ace_qwen3b.sh
#   DRY=1 bash baselines/bash_scripts/queue_ace_qwen3b.sh           # print, submit nothing
set -euo pipefail

INIT_ROOT=/home/udhanuka_umass_edu/Scalar_Research/physics-reasoning-agents/Initial_trajectories
LEVELS="${LEVELS:-down_to_earth two_body_problem}"
RUNS="${RUNS:-1 2 3}"
PARTITION="${PARTITION:-superpod-a100}"
S=baselines/bash_scripts/sbatch_ace_qwen.sh
R=baselines/data/results

for LEVEL in $LEVELS; do
    case "$LEVEL" in
        down_to_earth)    INIT="$INIT_ROOT/DTE_Qwen3B"; WALL=16:00:00 ;;
        two_body_problem) INIT="$INIT_ROOT/TwoBody_7B"; WALL=24:00:00 ;;
        *) echo "unknown level $LEVEL"; exit 1 ;;
    esac
    for RUN in $RUNS; do
        TAG=paperconfig; [[ "$RUN" == 1 ]] || TAG="paperconfig_run$RUN"
        TRAIN="$R/ace_${LEVEL}_qwen3b_s6-56_warmup_$TAG"
        FROZEN="$R/ace_${LEVEL}_qwen3b_s57-106_frozen_$TAG"
        train_cmd=(sbatch --parsable -p "$PARTITION" -t "$WALL" --job-name="ace3b_${LEVEL:0:3}_r$RUN"
                   "$S" --level "$LEVEL" --size 3b --init-dir "$INIT"
                   --extra "--resolve-rounds 5 --dedup" --tag "$TAG")
        if [[ -n "${DRY:-}" ]]; then
            echo "${train_cmd[*]}"; echo "  -> $TRAIN, then frozen -> $FROZEN"; continue
        fi
        j=$("${train_cmd[@]}")
        f=$(sbatch --parsable -p "$PARTITION" -t 08:00:00 --dependency=afterok:"$j" \
                --job-name="ace3b_${LEVEL:0:3}_frz_r$RUN" \
                "$S" --level "$LEVEL" --size 3b --seeds 57-106 --frozen-from "$TRAIN" --tag "$TAG")
        echo "$LEVEL run $RUN: train $j -> $TRAIN ; frozen $f -> $FROZEN"
    done
done
