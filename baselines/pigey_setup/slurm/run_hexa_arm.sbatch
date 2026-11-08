#!/bin/bash
#SBATCH --job-name=pigey-hexa-arm
#SBATCH --partition=superpod-a100
#SBATCH --account=pi_sniekum_umass_edu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00

# The whole HeXA arm in one job on one set of servers, so no phase queues again:
#   phase 0  (only with PIGEY_SEED_EPISODES) no-bank seed episodes into PIGEY_SEED_OUT;
#            round 1's bank is then distilled from them (run_trial.sbatch without a bank)
#   phase 1  hexa/loop.py training, as run_hexa.sbatch
#   phase 2  frozen eval with the final bank, as run_trial.sbatch with PIGEY_SKILL_BANK
# Environment:
#   PIGEY_MODEL          orchestrator and teacher model, used by every phase (required)
#   PIGEY_SUITE          LIBERO-PRO suite (default libero_goal_task)
#   PIGEY_TASKS          task ids, space separated (default 0-9)
#   PIGEY_SEED_EPISODES, PIGEY_SEED_OUT   seed init states and results directory (phase 0)
#   PIGEY_HEXA_INITIAL_BANK   a finished bank for round 1 instead (no phase 0)
#   PIGEY_HEXA_EPISODES, PIGEY_HEXA_EPISODES_PER_ROUND, PIGEY_HEXA_OUT   training, as run_hexa.sbatch
#   PIGEY_HEXA_ROUNDS    training rounds (default 15); the final bank is bank_round_<rounds+1>.json
#   PIGEY_EVAL_EPISODES, PIGEY_EVAL_OUT   eval init states and results directory (required)
#   PIGEY_ER_MODEL, PIGEY_ER_GPU_UTIL   detector and its GPU share
# Extra arguments go to hexa/loop.py only. Every phase resumes, so resubmitting the same
# environment continues where a failed or timed-out job stopped.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this job with sbatch}"
: "${PIGEY_MODEL:?set PIGEY_MODEL}"
: "${PIGEY_EVAL_EPISODES:?set PIGEY_EVAL_EPISODES}"
: "${PIGEY_EVAL_OUT:?set PIGEY_EVAL_OUT}"
export REPO_DIR="${REPO_DIR:-/project/pi_sniekum_umass_edu/vgandhi/HeXA-Hierarchical-Experimentalist-Agents}"
SETUP="$REPO_DIR/baselines/pigey_setup"
hostname
date
nvidia-smi --query-gpu=name,memory.total --format=csv
source "$SETUP/activate.sh"
export PATH="$HOME/.local/bin:$PATH"
which python && python --version
which claude
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export TMPDIR="$PIGEY_ROOT/tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
SUITE="${PIGEY_SUITE:-libero_goal_task}"
TASKS="${PIGEY_TASKS:-0 1 2 3 4 5 6 7 8 9}"
ROUNDS="${PIGEY_HEXA_ROUNDS:-15}"
TRAIN_OUT="${PIGEY_HEXA_OUT:-$PIGEY_ROOT/results/hexa/$SLURM_JOB_ID}"
FINAL_BANK="$TRAIN_OUT/bank_round_$((ROUNDS + 1)).json"
initial=()
if [[ -n "${PIGEY_HEXA_INITIAL_BANK:-}" ]]; then
    [[ -f "$PIGEY_HEXA_INITIAL_BANK" ]] || { echo "PIGEY_HEXA_INITIAL_BANK not found: $PIGEY_HEXA_INITIAL_BANK" >&2; exit 1; }
    initial+=(--initial-bank "$PIGEY_HEXA_INITIAL_BANK")
elif [[ -n "${PIGEY_SEED_EPISODES:-}" ]]; then
    : "${PIGEY_SEED_OUT:?set PIGEY_SEED_OUT with PIGEY_SEED_EPISODES}"
    initial+=(--initial-trials "$PIGEY_SEED_OUT")
fi
printf 'Seed: %s\nTraining: %s\nEval: %s\n' "${PIGEY_SEED_OUT:-none}" "$TRAIN_OUT" "$PIGEY_EVAL_OUT"

source "$SETUP/servers/start_servers.sh"

# run_episodes OUT_DIR EPISODES [BANK]: every (task, episode) not yet finished under OUT_DIR,
# as run_trial.sbatch does; returns non-zero if any episode failed.
run_episodes() {
    local out=$1 episodes=$2 bank=${3:-} task episode name code
    local -a failed=()
    local -A finished=()
    mkdir -p "$out"
    while read -r task episode; do
        [[ -n "$task" ]] && finished["$task $episode"]=1
    done < <(python "$SETUP/runner/completed_trials.py" "$out" --suite "$SUITE")
    printf '%d finished trial(s) under %s, skipped\n' "${#finished[@]}" "$out"
    for task in $TASKS; do
        for episode in $episodes; do
            [[ -n "${finished[$task $episode]:-}" ]] && continue
            for name in policy er sam; do
                kill -0 "$(cat "$TMPDIR/$name-server.pid")" 2> /dev/null \
                    || { echo "$name server is no longer running; stopping before task $task episode $episode" >&2; exit 1; }
            done
            printf '\n##### task %s episode %s (%s)\n' "$task" "$episode" "$(date)"
            code=0
            PIGEY_SKILL_BANK="$bank" bash "$SETUP/run_claude.sh" \
                --suite "$SUITE" --task "$task" --episode "$episode" \
                --policy-host localhost --policy-port "$POLICY_PORT" \
                --max-llm-steps 35 --max-steps-per-rollout 500 \
                --out-dir "$out" --model "$PIGEY_MODEL" || code=$?
            (( code == 0 )) || failed+=("task $task episode $episode (exit $code)")
        done
    done
    (( ${#failed[@]} == 0 )) || { printf 'Failed: %s\n' "${failed[@]}" >&2; return 1; }
}

if [[ -z "${PIGEY_HEXA_INITIAL_BANK:-}" && -n "${PIGEY_SEED_EPISODES:-}" ]]; then
    printf '\n===== Phase 0: seed episodes without a bank (%s)\n' "$(date)"
    # Like the separate seed job's afterok: a failed seed episode stops the arm.
    run_episodes "$PIGEY_SEED_OUT" "$PIGEY_SEED_EPISODES" || exit 1
fi

printf '\n===== Phase 1: HeXA training, %s rounds (%s)\n' "$ROUNDS" "$(date)"
# shellcheck disable=SC2086
python "$SETUP/hexa/loop.py" \
    --out-dir "$TRAIN_OUT" \
    --suite "$SUITE" \
    --tasks $TASKS \
    --episodes ${PIGEY_HEXA_EPISODES:-10 11 12 13 14 15 16 17 18 19} \
    --episodes-per-round "${PIGEY_HEXA_EPISODES_PER_ROUND:-1}" \
    ${initial[@]+"${initial[@]}"} \
    --policy-host localhost --policy-port "$POLICY_PORT" \
    --pid-files "$TMPDIR/policy-server.pid" "$TMPDIR/er-server.pid" "$TMPDIR/sam-server.pid" \
    --model "$PIGEY_MODEL" --rounds "$ROUNDS" "$@"
[[ -f "$FINAL_BANK" ]] || { echo "training ended without $FINAL_BANK" >&2; exit 1; }

printf '\n===== Phase 2: frozen eval with %s (%s)\n' "$FINAL_BANK" "$(date)"
run_episodes "$PIGEY_EVAL_OUT" "$PIGEY_EVAL_EPISODES" "$FINAL_BANK"
date
