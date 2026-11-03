#!/bin/bash
# Submit the libero_goal_task protocol (protocol.env) as four Slurm jobs:
#   NAME-train        slurm/run_hexa.sbatch: initial_bank.json, ROUNDS rounds over the training
#                     init states; writes bank_round_<ROUNDS+1>.json, the final bank
#   NAME-eval-bank    slurm/run_trial.sbatch on the eval init states with that bank (afterok: train)
#   NAME-eval-nobank  the same episodes with no bank; independent, so it starts right away
#   NAME-compare      a cpu job running hexa/compare.py once both arms have ended (afterany)
# Usage: bash submit.sh [--dry-run] [--run-name NAME]
# NAME defaults to goal_task. Results: $PIGEY_ROOT/results/hexa/NAME_train, results/eval/NAME_{bank,nobank},
# results/eval/NAME_comparison.{txt,json}; logs in $PIGEY_ROOT/logs/NAME-*-<job>.{out,err}.
# Re-running with the same NAME resumes: the loop reuses finished rounds and banks,
# and each eval arm skips finished episodes.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SETUP="$(cd -- "$HERE/../.." && pwd)"
export REPO_DIR="${REPO_DIR:-$(cd -- "$SETUP/../.." && pwd)}"
export PIGEY_ROOT="${PIGEY_ROOT:-/scratch/workspace/vgandhi_umass_edu-roboprobe-sim/pigey}"
ACCOUNT="${SLURM_ACCOUNT:-pi_sniekum_umass_edu}"
source "$HERE/protocol.env"

dry=0
RUN=goal_task
while (($#)); do
    case $1 in
        --dry-run) dry=1 ;;
        --run-name) RUN=$2; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done
[[ -f "$HERE/initial_bank.json" ]] || { echo "missing $HERE/initial_bank.json" >&2; exit 1; }
TRAIN_OUT="$PIGEY_ROOT/results/hexa/${RUN}_train"
FINAL_BANK="$TRAIN_OUT/bank_round_$((ROUNDS + 1)).json"
EVAL_BANK_OUT="$PIGEY_ROOT/results/eval/${RUN}_bank"
EVAL_NOBANK_OUT="$PIGEY_ROOT/results/eval/${RUN}_nobank"
LOGS="$PIGEY_ROOT/logs"
mkdir -p "$LOGS" "$PIGEY_ROOT/results/eval"

run() {  # print the command in --dry-run mode, else run it and echo the job id
    if ((dry)); then
        printf '%q ' "$@"; echo; echo "<job id>"
    else
        local id
        id=$("$@")
        echo "${id%%;*}"
    fi
}
SBATCH=(sbatch --parsable --account="$ACCOUNT" --output="$LOGS/%x-%j.out" --error="$LOGS/%x-%j.err" --export=ALL)

echo "Protocol: $SUITE tasks [$TASKS]; train init states [$TRAIN_STATES]; eval init states [$EVAL_STATES]"
echo "Initial bank: $HERE/initial_bank.json"

train_id=$(run env PIGEY_SUITE="$SUITE" PIGEY_TASKS="$TASKS" \
    PIGEY_HEXA_EPISODES="$TRAIN_STATES" PIGEY_HEXA_EPISODES_PER_ROUND=1 \
    PIGEY_HEXA_INITIAL_BANK="$HERE/initial_bank.json" PIGEY_HEXA_OUT="$TRAIN_OUT" \
    PIGEY_ER_MODEL="$ER_MODEL" PIGEY_ER_GPU_UTIL="$ER_GPU_UTIL" \
    "${SBATCH[@]}" --job-name="${RUN}-train" --time="$TRAIN_TIME" \
    "$SETUP/slurm/run_hexa.sbatch" --model "$MODEL" --rounds "$ROUNDS" | tail -1)
echo "train:        $train_id  -> $TRAIN_OUT (final bank $FINAL_BANK)"

bank_id=$(run env PIGEY_SUITE="$SUITE" PIGEY_TASKS="$TASKS" PIGEY_EPISODES="$EVAL_STATES" \
    PIGEY_SKILL_BANK="$FINAL_BANK" PIGEY_OUT_DIR="$EVAL_BANK_OUT" \
    PIGEY_ER_MODEL="$ER_MODEL" PIGEY_ER_GPU_UTIL="$ER_GPU_UTIL" \
    "${SBATCH[@]}" --job-name="${RUN}-eval-bank" --time="$EVAL_BANK_TIME" --dependency="afterok:$train_id" \
    "$SETUP/slurm/run_trial.sbatch" --model "$MODEL" | tail -1)
echo "eval-bank:    $bank_id  -> $EVAL_BANK_OUT (after train succeeds)"

nobank_id=$(run env PIGEY_SUITE="$SUITE" PIGEY_TASKS="$TASKS" PIGEY_EPISODES="$EVAL_STATES" \
    PIGEY_OUT_DIR="$EVAL_NOBANK_OUT" \
    PIGEY_ER_MODEL="$ER_MODEL" PIGEY_ER_GPU_UTIL="$ER_GPU_UTIL" \
    "${SBATCH[@]}" --job-name="${RUN}-eval-nobank" --time="$EVAL_NOBANK_TIME" \
    "$SETUP/slurm/run_trial.sbatch" --model "$MODEL" | tail -1)
echo "eval-nobank:  $nobank_id  -> $EVAL_NOBANK_OUT"

compare_cmd="source $SETUP/activate.sh && python $SETUP/hexa/compare.py --with-bank $EVAL_BANK_OUT --no-bank $EVAL_NOBANK_OUT --suite $SUITE --json $PIGEY_ROOT/results/eval/${RUN}_comparison.json | tee $PIGEY_ROOT/results/eval/${RUN}_comparison.txt"
compare_id=$(run "${SBATCH[@]}" --job-name="${RUN}-compare" --partition=cpu --cpus-per-task=2 --mem=8G --time=00:15:00 \
    --dependency="afterany:$bank_id:$nobank_id" --wrap "bash -c '$compare_cmd'" | tail -1)
echo "compare:      $compare_id  -> $PIGEY_ROOT/results/eval/${RUN}_comparison.txt (after both arms end)"

if ((!dry)); then
    printf '%s train=%s eval-bank=%s eval-nobank=%s compare=%s\n' "$(date -Is)" "$train_id" "$bank_id" "$nobank_id" "$compare_id" \
        >> "$PIGEY_ROOT/results/eval/${RUN}_jobs.txt"
fi
