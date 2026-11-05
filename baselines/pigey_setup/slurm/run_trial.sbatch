#!/bin/bash
#SBATCH --job-name=pigey-trial
#SBATCH --partition=superpod-a100
#SBATCH --account=pi_sniekum_umass_edu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Pigey episodes on one A100: the pi0.5 policy, local ER (Qwen3-VL) and SAM
# servers start once, then every (task, init state) pair in PIGEY_TASKS x
# PIGEY_EPISODES runs in turn through run_claude.sh. Environment:
#   PIGEY_TASKS       task ids, space separated (default: the array index, else 0)
#   PIGEY_EPISODES    init states, space separated (default 2, Pigey's README example)
#   PIGEY_SUITE       LIBERO-PRO suite (default libero_goal_task)
#   PIGEY_OUT_DIR     results directory (default $PIGEY_ROOT/results/claude-code/<job id>).
#                     An existing directory is resumed: finished pairs are skipped.
#   PIGEY_SKILL_BANK  a HeXA bank to inject into the system prompt (the with-bank
#                     arm of a frozen evaluation); unset for plain Pigey
#   PIGEY_ER_MODEL, PIGEY_ER_GPU_UTIL   detector and its GPU share (servers/serve_perception.sh)
# Extra arguments go to run_claude.sh, e.g. --model claude-sonnet-5-5:
#   PIGEY_TASKS="0 8" PIGEY_EPISODES="2" sbatch --export=ALL slurm/run_trial.sbatch --model claude-sonnet-5-5
# A failed episode is recorded and the remaining episodes still run; the job
# exits non-zero at the end if any failed. In a job array without PIGEY_TASKS,
# the array index is the task: sbatch --array=0-9%4 --export=ALL,PIGEY_EPISODES="0 1 2" ...
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this trial with sbatch}"
export REPO_DIR="${REPO_DIR:-/home/udhanuka_umass_edu/Scalar_Research/HeXA}"
SETUP="$REPO_DIR/baselines/pigey_setup"
hostname
date
nvidia-smi --query-gpu=name,memory.total --format=csv
source "$SETUP/activate.sh"
which python && python --version
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export TMPDIR="$PIGEY_ROOT/tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
SUITE="${PIGEY_SUITE:-libero_goal_task}"
OUT_DIR="${PIGEY_OUT_DIR:-$PIGEY_ROOT/results/claude-code/$SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

# Resume: skip (task, episode) pairs that already finished under OUT_DIR.
declare -A finished=()
while read -r task episode; do
    [[ -n "$task" ]] && finished["$task $episode"]=1
done < <(python "$SETUP/runner/completed_trials.py" "$OUT_DIR" --suite "$SUITE")
printf 'Results: %s (%d finished trial(s) found, skipped)\n' "$OUT_DIR" "${#finished[@]}"
if [[ -n "${PIGEY_SKILL_BANK:-}" ]]; then
    [[ -f "$PIGEY_SKILL_BANK" ]] || { echo "PIGEY_SKILL_BANK not found: $PIGEY_SKILL_BANK" >&2; exit 1; }
    printf 'Skill bank: %s\n' "$PIGEY_SKILL_BANK"
else
    echo "No skill bank: plain Pigey"
fi

source "$SETUP/servers/start_servers.sh"

failed=()
for task in ${PIGEY_TASKS:-${SLURM_ARRAY_TASK_ID:-0}}; do
    for episode in ${PIGEY_EPISODES:-2}; do
        if [[ -n "${finished[$task $episode]:-}" ]]; then
            printf 'task %s episode %s already finished; skipping\n' "$task" "$episode"
            continue
        fi
        for name in policy er sam; do
            kill -0 "$(cat "$TMPDIR/$name-server.pid")" 2> /dev/null \
                || { echo "$name server is no longer running; stopping before task $task episode $episode" >&2; exit 1; }
        done
        printf '\n##### task %s episode %s (%s)\n' "$task" "$episode" "$(date)"
        code=0
        bash "$SETUP/run_claude.sh" \
            --suite "$SUITE" --task "$task" --episode "$episode" \
            --policy-host localhost --policy-port "$POLICY_PORT" \
            --max-llm-steps 35 --max-steps-per-rollout 500 \
            --out-dir "$OUT_DIR" "$@" || code=$?
        (( code == 0 )) || failed+=("task $task episode $episode (exit $code)")
    done
done
ls -la "$OUT_DIR" || true
date
(( ${#failed[@]} == 0 )) || { printf 'Failed: %s\n' "${failed[@]}" >&2; exit 1; }
