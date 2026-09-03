#!/bin/bash
#SBATCH --job-name=dc_phase2
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 12:00:00
#SBATCH -o logs/dc_phase2_launcher_%j.log

# ─── Design C Phase 2 launcher: source banks for Minesweeper and Wordle ───────
#
# Waits for the gpt-oss-120b server named by SERVER_JOB to publish its endpoint,
# then submits the arms in two stages so the no-bank gate exists before the bank
# it gates:
#
#   stage 1 (parallel):  baseline x 3 runs x 2 tasks   = 6 jobs
#   stage 2 (dependent): HExA     x 3 runs x 2 tasks   = 6 jobs
#
# Three runs per arm is the point, not a nicety: every ORBIT arm in this project
# so far is n=1, so no effect has an across-run spread to be judged against. The
# replicates share their pools and differ in actor sampling seed and play order,
# so they are paired on identical instances.
#
#   SERVER_JOB=63790814 sbatch scripts/run_design_c_phase2.sh
#
# Every arm runs through scripts/run_maze_arm.sh, which is task-agnostic: it only
# needs SCRIPT, OUT, ARM_ARGS, MODEL and VLLM_URL.

set -Eeuo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"

SERVER_JOB="${SERVER_JOB:?set SERVER_JOB to the gpt-oss serving job id}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
ENDPOINT="logs/vllm_endpoint_${SERVER_JOB}.json"

# Read the endpoint from the serving job's own file, never from
# vllm_endpoint_latest.json -- that symlink-by-convention has pointed at a dead
# server of a different model before, and an arm aimed at the wrong model is
# worse than one that fails.
echo "[$(date)] waiting for ${ENDPOINT}"
for _ in $(seq 1 360); do
    [[ -f "$ENDPOINT" ]] && break
    squeue -j "$SERVER_JOB" -h -o %T | grep -qE 'PENDING|RUNNING|CONFIGURING' || {
        echo "ERROR: job ${SERVER_JOB} left the queue without writing ${ENDPOINT}"; exit 1; }
    sleep 30
done
[[ -f "$ENDPOINT" ]] || { echo "ERROR: timed out waiting for ${ENDPOINT}"; exit 1; }

VLLM_URL=$(python -c "import json,sys; print(json.load(open('${ENDPOINT}'))['base_url'])")
echo "[$(date)] endpoint ${VLLM_URL}"
curl -sf "${VLLM_URL}/models" >/dev/null || { echo "ERROR: ${VLLM_URL} not answering"; exit 1; }

# run_maze_arm.sh appends `--model` and `--out` itself, so ARM_ARGS must not
# carry either; passing --out twice makes argparse take the last one silently.
submit () {  # submit <dependency-or-empty> <walltime> <script> <out> <args...>
    local dep="$1" walltime="$2" script="$3" out="$4"; shift 4
    local flags=(--time="$walltime")
    [[ -n "$dep" ]] && flags+=(--dependency="afterany:${dep}")
    sbatch "${flags[@]}" --parsable \
        --export=ALL,SCRIPT="$script",OUT="$out",MODEL="$MODEL",VLLM_URL="$VLLM_URL",ARM_ARGS="$*" \
        scripts/run_maze_arm.sh
}

BASE_IDS=()
for task in minesweeper wordle; do
    for run in 0 1 2; do
        out="results/orbit_${task}_baseline/run${run}"
        id=$(submit "" 6:00:00 scripts/run_orbit_source_baseline.py "$out" \
             --task "$task" --run-index "$run" --concurrency 8)
        echo "baseline ${task} run${run} -> job ${id}"
        BASE_IDS+=("$id")
    done
done

# The HExA arms wait on the whole baseline stage rather than on their own task's:
# 12 clients against one server is more contention than throughput, and the gate
# is read per task only after both arms of that task exist anyway.
DEP=$(IFS=:; echo "${BASE_IDS[*]}")
for task in minesweeper wordle; do
    for run in 0 1 2; do
        out="results/orbit_${task}_hexa/run${run}"
        # Wordle's train pool is capped at 132 by its 192-word vocabulary (D-C3),
        # so 20 rounds x 10 would exhaust it; 13 rounds is the honest maximum.
        rounds=20; [[ "$task" == "wordle" ]] && rounds=13
        id=$(submit "$DEP" 12:00:00 scripts/run_orbit_source_hexa.py "$out" \
             --task "$task" --run-index "$run" --rounds "$rounds" --batch-size 10 \
             --max-trajs 12 --concurrency 8)
        echo "hexa ${task} run${run} -> job ${id}"
    done
done

echo "[$(date)] all Design C Phase 2 arms submitted"
