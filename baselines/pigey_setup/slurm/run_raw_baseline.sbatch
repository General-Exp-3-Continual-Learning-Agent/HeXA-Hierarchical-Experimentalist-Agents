#!/bin/bash
#SBATCH --job-name=pigey-raw-baseline
#SBATCH --partition=superpod-a100
#SBATCH --account=pi_sniekum_umass_edu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Raw pi0.5-LIBERO baseline across all six LIBERO-PRO perturbation suites in
# one GPU allocation: no agent, no LLM, no ER/SAM perception servers.
# agent_sim.py --mode raw just feeds each task's literal instruction straight
# to pi0.5 and runs it closed-loop until done or SUITE_MAX_STEPS (400 env
# steps; none of these six suite names are in agent_sim's SUITE_MAX_STEPS
# dict, so they all fall back to the same default). Only the pi0.5 policy
# server is needed; it is started once and reused across all six suites.
# Every suite uses the same (task, episode) eval set as the HeXA bank/no-bank
# arms: PIGEY_TASKS x PIGEY_EPISODES.
#   PIGEY_SUITES     suites, space separated (default: all six LIBERO-PRO suites)
#   PIGEY_TASKS      task ids, space separated (default: all ten tasks)
#   PIGEY_EPISODES   init states, space separated (default: eval states 25-49)
#   PIGEY_OUT_BASE   results root; each suite gets its own $PIGEY_OUT_BASE/<suite> dir
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
OUT_BASE="${PIGEY_OUT_BASE:-$PIGEY_ROOT/results/eval/raw}"
mkdir -p "$OUT_BASE"

# Only the pi0.5 policy server; raw mode never calls Perceive/Grasp/VerifyCandidate.
CHECKPOINT="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero"
cleanup() { [[ -f "$TMPDIR/policy-server.pid" ]] && kill "$(cat "$TMPDIR/policy-server.pid")" 2> /dev/null || true; }
trap cleanup EXIT
POLICY_PORT="${PIGEY_POLICY_PORT:-$(python "$SETUP/runner/free_ports.py" 1)}"
(
    cd "$PIGEY_ROOT/openpi" || exit 1
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${PIGEY_POLICY_GPU_FRACTION:-0.5}"
    exec "$PIGEY_ROOT/openpi/.venv/bin/python" scripts/serve_policy.py --port "$POLICY_PORT" \
        policy:checkpoint --policy.config=pi05_libero --policy.dir="$CHECKPOINT"
) > "$PIGEY_ROOT/logs/policy-server-$SLURM_JOB_ID.log" 2>&1 &
echo $! > "$TMPDIR/policy-server.pid"
for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$POLICY_PORT/healthz" > /dev/null && break
    if ! kill -0 "$(cat "$TMPDIR/policy-server.pid")" 2> /dev/null; then
        tail -50 "$PIGEY_ROOT/logs/policy-server-$SLURM_JOB_ID.log" >&2
        echo "pi0.5 policy server exited during startup" >&2
        exit 1
    fi
    sleep 5
done
curl -sf "http://127.0.0.1:$POLICY_PORT/healthz" > /dev/null \
    || { tail -50 "$PIGEY_ROOT/logs/policy-server-$SLURM_JOB_ID.log" >&2; echo "pi0.5 server not ready" >&2; exit 1; }
printf 'pi0.5 policy server ready on port %s\n' "$POLICY_PORT"

failed=()
for suite in ${PIGEY_SUITES:-libero_object_swap libero_object_task libero_spatial_swap libero_spatial_task libero_goal_swap libero_goal_task}; do
    OUT_DIR="$OUT_BASE/$suite"
    mkdir -p "$OUT_DIR"

    declare -A finished=()
    while read -r task episode; do
        [[ -n "$task" ]] && finished["$task $episode"]=1
    done < <(python "$SETUP/runner/completed_trials.py" "$OUT_DIR" --suite "$suite")
    printf '\n=== suite %s -> %s (%d finished trial(s) found, skipped) ===\n' "$suite" "$OUT_DIR" "${#finished[@]}"

    for task in ${PIGEY_TASKS:-0 1 2 3 4 5 6 7 8 9}; do
        for episode in ${PIGEY_EPISODES:-$(seq -s ' ' 25 49)}; do
            if [[ -n "${finished[$task $episode]:-}" ]]; then
                printf 'suite %s task %s episode %s already finished; skipping\n' "$suite" "$task" "$episode"
                continue
            fi
            kill -0 "$(cat "$TMPDIR/policy-server.pid")" 2> /dev/null \
                || { echo "policy server is no longer running; stopping before suite $suite task $task episode $episode" >&2; exit 1; }
            printf '\n##### suite %s task %s episode %s (%s)\n' "$suite" "$task" "$episode" "$(date)"
            code=0
            python "$REPO_DIR/baselines/Pigey/sim/agent_sim.py" \
                --mode raw --suite "$suite" --task "$task" --episode "$episode" \
                --policy-host localhost --policy-port "$POLICY_PORT" \
                --out-dir "$OUT_DIR" --no-video || code=$?
            (( code == 0 )) || failed+=("suite $suite task $task episode $episode (exit $code)")
        done
    done
    unset finished
done
ls -la "$OUT_BASE" || true
date
(( ${#failed[@]} == 0 )) || { printf 'Failed: %s\n' "${failed[@]}" >&2; exit 1; }
