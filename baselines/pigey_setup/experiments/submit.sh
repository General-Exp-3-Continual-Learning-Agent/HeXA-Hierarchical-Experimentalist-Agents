#!/bin/bash
# Submit one experiment directory's protocol (its protocol.env) as Slurm jobs:
#   NAME-seed         only when the directory has no initial_bank.json: slurm/run_trial.sbatch,
#                     no bank, on the seed init states; the training job distills round 1's
#                     bank from these trials
#   NAME-train        slurm/run_hexa.sbatch: initial_bank.json (or the seed trials), ROUNDS rounds
#                     over the training init states; writes bank_round_<ROUNDS+1>.json, the final bank
#   NAME-eval-bank    slurm/run_trial.sbatch on the eval init states with that bank (afterok: train)
#   NAME-eval-nobank  the same episodes with no bank; independent, so it starts right away
#   NAME-compare      a cpu job running hexa/compare.py once both arms have ended (afterany)
# --one-job replaces NAME-seed, NAME-train and NAME-eval-bank with NAME-hexa
# (slurm/run_hexa_arm.sbatch, HEXA_ARM_TIME, default 36 h): one GPU job that runs the seed
# episodes (if any), trains and then runs the with-bank eval on the same servers, so no
# phase queues again.
# Usage: bash experiments/submit.sh EXPERIMENT_DIR [--dry-run] [--one-job] [--run-name NAME]
# NAME defaults to RUN_NAME from protocol.env. Results: $PIGEY_ROOT/results/hexa/NAME_{seed,train},
# results/eval/NAME_{bank,nobank}, results/eval/NAME_comparison.{txt,json}; logs in
# $PIGEY_ROOT/logs/NAME-*-<job>.{out,err}. Re-running with the same NAME resumes: finished seed
# episodes, rounds, banks and eval episodes are skipped.
set -euo pipefail
SETUP="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
(($#)) || { echo "usage: bash $0 EXPERIMENT_DIR [--dry-run] [--one-job] [--run-name NAME]" >&2; exit 2; }
HERE="$(cd -- "$1" && pwd)"
shift
export REPO_DIR="${REPO_DIR:-$(cd -- "$SETUP/../.." && pwd)}"
export PIGEY_ROOT="${PIGEY_ROOT:-/scratch/workspace/vgandhi_umass_edu-roboprobe-sim/pigey}"
ACCOUNT="${SLURM_ACCOUNT:-pi_sniekum_umass_edu}"
source "$HERE/protocol.env"

dry=0
one_job=0
RUN="$RUN_NAME"
while (($#)); do
    case $1 in
        --dry-run) dry=1 ;;
        --one-job) one_job=1 ;;
        --run-name) RUN=$2; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done
SEED_OUT="$PIGEY_ROOT/results/hexa/${RUN}_seed"
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
[[ -n "${EXCLUDE_NODES:-}" ]] && SBATCH+=(--exclude="$EXCLUDE_NODES")

echo "Protocol: $SUITE tasks [$TASKS]; seed init states [$SEED_STATES]; train [$TRAIN_STATES]; eval [$EVAL_STATES]"

seed_id=""
if [[ -f "$HERE/initial_bank.json" ]]; then
    echo "Initial bank: $HERE/initial_bank.json"
    initial=(PIGEY_HEXA_INITIAL_BANK="$HERE/initial_bank.json")
elif ((one_job)); then
    echo "No initial_bank.json: the HeXA job runs the seed episodes into $SEED_OUT and distills from them"
    initial=(PIGEY_SEED_EPISODES="$SEED_STATES" PIGEY_SEED_OUT="$SEED_OUT")
else
    echo "No initial_bank.json: round 1's bank is distilled from the seed trials in $SEED_OUT"
    seed_id=$(run env PIGEY_SUITE="$SUITE" PIGEY_TASKS="$TASKS" PIGEY_EPISODES="$SEED_STATES" \
        PIGEY_OUT_DIR="$SEED_OUT" PIGEY_ER_MODEL="$ER_MODEL" PIGEY_ER_GPU_UTIL="$ER_GPU_UTIL" \
        "${SBATCH[@]}" --job-name="${RUN}-seed" --time="$SEED_TIME" \
        "$SETUP/slurm/run_trial.sbatch" --model "$MODEL" | tail -1)
    echo "seed:         $seed_id  -> $SEED_OUT"
    initial=(PIGEY_HEXA_INITIAL="$SEED_OUT")
fi

if ((one_job)); then
    train_id=$(run env PIGEY_SUITE="$SUITE" PIGEY_TASKS="$TASKS" PIGEY_MODEL="$MODEL" \
        PIGEY_HEXA_EPISODES="$TRAIN_STATES" PIGEY_HEXA_EPISODES_PER_ROUND=1 PIGEY_HEXA_ROUNDS="$ROUNDS" \
        "${initial[@]}" PIGEY_HEXA_OUT="$TRAIN_OUT" \
        PIGEY_EVAL_EPISODES="$EVAL_STATES" PIGEY_EVAL_OUT="$EVAL_BANK_OUT" \
        PIGEY_ER_MODEL="$ER_MODEL" PIGEY_ER_GPU_UTIL="$ER_GPU_UTIL" \
        "${SBATCH[@]}" --job-name="${RUN}-hexa" --time="${HEXA_ARM_TIME:-36:00:00}" \
        "$SETUP/slurm/run_hexa_arm.sbatch" | tail -1)
    bank_id=$train_id
    seed_step=""
    [[ -f "$HERE/initial_bank.json" ]] || seed_step="$SEED_OUT -> "
    echo "hexa:         $train_id  -> $seed_step$TRAIN_OUT -> $EVAL_BANK_OUT"
else
    train_id=$(run env PIGEY_SUITE="$SUITE" PIGEY_TASKS="$TASKS" \
        PIGEY_HEXA_EPISODES="$TRAIN_STATES" PIGEY_HEXA_EPISODES_PER_ROUND=1 \
        "${initial[@]}" PIGEY_HEXA_OUT="$TRAIN_OUT" \
        PIGEY_ER_MODEL="$ER_MODEL" PIGEY_ER_GPU_UTIL="$ER_GPU_UTIL" \
        "${SBATCH[@]}" --job-name="${RUN}-train" --time="$TRAIN_TIME" \
        ${seed_id:+--dependency="afterok:$seed_id"} \
        "$SETUP/slurm/run_hexa.sbatch" --model "$MODEL" --rounds "$ROUNDS" | tail -1)
    echo "train:        $train_id  -> $TRAIN_OUT (final bank $FINAL_BANK)${seed_id:+ (after seed succeeds)}"

    bank_id=$(run env PIGEY_SUITE="$SUITE" PIGEY_TASKS="$TASKS" PIGEY_EPISODES="$EVAL_STATES" \
        PIGEY_SKILL_BANK="$FINAL_BANK" PIGEY_OUT_DIR="$EVAL_BANK_OUT" \
        PIGEY_ER_MODEL="$ER_MODEL" PIGEY_ER_GPU_UTIL="$ER_GPU_UTIL" \
        "${SBATCH[@]}" --job-name="${RUN}-eval-bank" --time="$EVAL_BANK_TIME" --dependency="afterok:$train_id" \
        "$SETUP/slurm/run_trial.sbatch" --model "$MODEL" | tail -1)
    echo "eval-bank:    $bank_id  -> $EVAL_BANK_OUT (after train succeeds)"
fi

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
    printf '%s seed=%s train=%s eval-bank=%s eval-nobank=%s compare=%s\n' "$(date -Is)" "${seed_id:-none}" \
        "$train_id" "$bank_id" "$nobank_id" "$compare_id" >> "$PIGEY_ROOT/results/eval/${RUN}_jobs.txt"
fi
