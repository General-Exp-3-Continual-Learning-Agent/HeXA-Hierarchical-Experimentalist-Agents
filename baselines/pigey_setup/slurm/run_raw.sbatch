#!/bin/bash
#SBATCH --job-name=pigey-raw
#SBATCH --partition=superpod-a100
#SBATCH --account=pi_sniekum_umass_edu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

# The paper's "pi0.5 alone" baseline: Pigey's agent_sim.py --mode raw, no LLM and
# no perception. pi0.5 gets the task string once and runs closed-loop until the
# simulator reports success or Pigey's step limit for the suite (400 for
# libero_goal_task) runs out. Only the policy server starts. Environment:
#   PIGEY_TASKS       task ids, space separated (default 0)
#   PIGEY_EPISODES    init states, space separated (default 2)
#   PIGEY_SUITE       LIBERO-PRO suite (default libero_goal_task)
#   PIGEY_OUT_DIR     results directory (default $PIGEY_ROOT/results/raw/<job id>);
#                     an existing one is resumed. Trials land in OUT_DIR/raw/<suite>_tNN_eNN/<time>/
#   PIGEY_TASKS="$(seq -s ' ' 0 9)" PIGEY_EPISODES="$(seq -s ' ' 25 49)" sbatch --export=ALL slurm/run_raw.sbatch
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this trial with sbatch}"
export REPO_DIR="${REPO_DIR:-/project/pi_sniekum_umass_edu/vgandhi/HeXA-Hierarchical-Experimentalist-Agents}"
SETUP="$REPO_DIR/baselines/pigey_setup"
hostname
date
nvidia-smi --query-gpu=name,memory.total --format=csv
source "$SETUP/activate.sh"
which python && python --version
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export TMPDIR="$PIGEY_ROOT/tmp/$SLURM_JOB_ID"
export HARNESS_PERCEPTION=none   # agent_sim's state_text would otherwise call Gemini
mkdir -p "$TMPDIR"
SUITE="${PIGEY_SUITE:-libero_goal_task}"
OUT_DIR="${PIGEY_OUT_DIR:-$PIGEY_ROOT/results/raw/$SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

declare -A finished=()
while read -r task episode; do
    [[ -n "$task" ]] && finished["$task $episode"]=1
done < <(python "$SETUP/runner/completed_trials.py" "$OUT_DIR" --suite "$SUITE")
printf 'Results: %s (%d finished trial(s) found, skipped)\n' "$OUT_DIR" "${#finished[@]}"

PIGEY_POLICY_ONLY=1 source "$SETUP/servers/start_servers.sh"

failed=()
for task in ${PIGEY_TASKS:-0}; do
    for episode in ${PIGEY_EPISODES:-2}; do
        [[ -n "${finished[$task $episode]:-}" ]] && continue
        kill -0 "$(cat "$TMPDIR/policy-server.pid")" 2> /dev/null \
            || { echo "policy server is no longer running; stopping before task $task episode $episode" >&2; exit 1; }
        printf '\n##### task %s episode %s (%s)\n' "$task" "$episode" "$(date)"
        code=0
        python "$PIGEY_REPO/sim/agent_sim.py" --mode raw \
            --suite "$SUITE" --task "$task" --episode "$episode" \
            --policy-host localhost --policy-port "$POLICY_PORT" \
            --out-dir "$OUT_DIR" || code=$?
        (( code == 0 )) || failed+=("task $task episode $episode (exit $code)")
    done
done
date
(( ${#failed[@]} == 0 )) || { printf 'Failed: %s\n' "${failed[@]}" >&2; exit 1; }
