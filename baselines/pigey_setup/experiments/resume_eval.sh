#!/bin/bash
# Resume the two unfinished 2026-11-08 frozen eval arms; never seed or train.
set -euo pipefail
RUN=${1:-}
case "$RUN" in
    goal_swap) EVAL_TIME=01:00:00 ;;
    spatial_task) EVAL_TIME=14:00:00 ;;
    *) echo "usage: bash $0 {goal_swap|spatial_task} [--dry-run]" >&2; exit 2 ;;
esac
DRY=0
if [[ ${2:-} == --dry-run ]]; then DRY=1; elif [[ -n ${2:-} ]]; then exit 2; fi
(( $# <= 2 )) || exit 2
SETUP=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export REPO_DIR=$(cd -- "$SETUP/../.." && pwd)
export PIGEY_REPO="$REPO_DIR/baselines/Pigey"
export PIGEY_ROOT=${PIGEY_ROOT:-/scratch/workspace/vgandhi_umass_edu-roboprobe-sim/pigey}
source "$SETUP/experiments/libero_$RUN/protocol.env"
export PIGEY_SUITE="$SUITE" PIGEY_TASKS="$TASKS" PIGEY_EPISODES="$EVAL_STATES"
export PIGEY_ER_MODEL="$ER_MODEL" PIGEY_ER_GPU_UTIL="$ER_GPU_UTIL"
export PIGEY_POLICY_GPU_FRACTION=0.35
export PIGEY_OUT_DIR="$PIGEY_ROOT/results/eval/${RUN}_bank"
export PIGEY_SKILL_BANK="$PIGEY_ROOT/results/hexa/${RUN}_train/bank_round_16.json"
export PIGEY_CLAUDE_MAX_TURNS=3
export PATH="$HOME/.local/bin:$PATH"
LOGS="$PIGEY_ROOT/logs"
SBATCH=(sbatch --parsable --account="${SLURM_ACCOUNT:-pi_sniekum_umass_edu}" --export=ALL
    --output="$LOGS/%x-%j.out" --error="$LOGS/%x-%j.err")
EXCLUDE=${EXCLUDE_NODES:-gpu023}
[[ -z "$EXCLUDE" ]] || SBATCH+=(--exclude="$EXCLUDE")
if (( ! DRY )); then
    for path in "$PIGEY_OUT_DIR" "$LOGS" "$PIGEY_ROOT/tmp"; do
        [[ -w "$path" ]] || { echo "No write access: $path; see RESUME_TEAMMATE.md" >&2; exit 1; }
    done
    [[ -r "$PIGEY_SKILL_BANK" ]] || { echo "Missing bank: $PIGEY_SKILL_BANK" >&2; exit 1; }
    [[ -f "$PIGEY_REPO/sim/agent_sim.py" ]] || { echo "Run git submodule update --init baselines/Pigey" >&2; exit 1; }
    "${CLAUDE_BINARY:-claude}" auth status
fi
submit() {
    if (( DRY )); then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}
# run_trial skips every finished pair; backend errors are retried.
eval_cmd=("${SBATCH[@]}" --job-name="${RUN}-eval-bank-resume" --time="$EVAL_TIME"
    "$SETUP/slurm/run_trial.sbatch" --model "$MODEL" --claude-timeout 300 --claude-max-turns 3
    --perception local --verifier claude --verify-model claude-haiku-4-5 --sam service --phase0 skip)
if (( DRY )); then
    submit "${eval_cmd[@]}"
    eval_id=DRY_RUN_JOB_ID
else
    eval_id=$(submit "${eval_cmd[@]}")
    eval_id=${eval_id%%;*}
    echo "eval-bank: $eval_id"
fi
# Paths travel as exported variables, avoiding shell interpolation in --wrap.
export PIGEY_RESUME_SETUP="$SETUP" PIGEY_RESUME_RUN="$RUN"
compare_cmd='set -euo pipefail; source "$PIGEY_RESUME_SETUP/activate.sh"; python "$PIGEY_RESUME_SETUP/hexa/compare.py" --with-bank "$PIGEY_OUT_DIR" --no-bank "$PIGEY_ROOT/results/eval/${PIGEY_RESUME_RUN}_nobank" --suite "$PIGEY_SUITE" --json "$PIGEY_ROOT/results/eval/${PIGEY_RESUME_RUN}_comparison.json" | tee "$PIGEY_ROOT/results/eval/${PIGEY_RESUME_RUN}_comparison.txt"'
compare_shell=$(printf '%q' "$compare_cmd")
if (( DRY )); then
    submit "${SBATCH[@]}" --job-name="${RUN}-compare-resume" --partition=cpu --cpus-per-task=2 --mem=8G \
        --time=00:15:00 --dependency="afterany:$eval_id" --wrap="bash -c $compare_shell"
else
    compare_id=$(submit "${SBATCH[@]}" --job-name="${RUN}-compare-resume" --partition=cpu --cpus-per-task=2 --mem=8G \
        --time=00:15:00 --dependency="afterany:$eval_id" --wrap="bash -c $compare_shell")
    compare_id=${compare_id%%;*}
    echo "compare: $compare_id"
    printf '%s eval-bank=%s compare=%s submitted-by=%s\n' "$(date -Is)" "$eval_id" "$compare_id" "$USER" \
        >> "$PIGEY_ROOT/results/eval/${RUN}_jobs.txt"
fi
