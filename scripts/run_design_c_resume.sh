#!/bin/bash
# Restart Design C after the gpt-oss server was lost. One command, n=1 (D-C9).
#
#   SERVER_JOB=<jobid> bash scripts/run_design_c_resume.sh
#
# Submits the one thing Phase 3 is blocked on: Minesweeper HExA run0, resumed
# from its last completed round.
#
# The Mastermind baseline is NOT re-run. `results/orbit_mastermind_baseline_oss/`
# already holds it on gpt-oss-120b: n=256, `guidelines: None`, `added_tokens: 0`,
# and its instance ids are identical to `MastermindTask().eval_specs(256)` --
# checked, 256/256 -- so the transfer arms pair against it exactly. Maze's
# baseline is `results/design_c/maze_baseline/run{0,1,2}`, already at n=3.
#
# It stops there. Phase 3 is a separate command on purpose: its banks must be read
# before any eval compute is spent, and a script that chained straight into Phase 4
# would silently turn "flagged for a human read" into "passed".
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

SERVER_JOB="${SERVER_JOB:?set SERVER_JOB to the gpt-oss serving job id}"
ENDPOINT="logs/vllm_endpoint_${SERVER_JOB}.json"
MODEL="${MODEL:-openai/gpt-oss-120b}"

echo "[$(date)] waiting for ${ENDPOINT}"
for _ in $(seq 1 360); do
    [[ -f "$ENDPOINT" ]] && break
    squeue -j "$SERVER_JOB" -h -o %T | grep -qE 'PENDING|RUNNING|CONFIGURING' || {
        echo "ERROR: job ${SERVER_JOB} left the queue without writing ${ENDPOINT}"; exit 1; }
    sleep 30
done
V=$(python -c "import json; print(json.load(open('${ENDPOINT}'))['base_url'])")
curl -sf "${V}/models" >/dev/null || { echo "ERROR: ${V} not answering"; exit 1; }
echo "[$(date)] endpoint ${V}"

# 32768, not run_maze_arm.sh's 16384 default: at 16384 the evolution response is
# cut off, the JSON fails to parse, and evolve_skill_bank silently keeps the
# previous bank. That cost every round of the first Minesweeper arm.
sub () {  # sub <walltime> <script> <out> <args...>
    local t="$1" script="$2" out="$3"; shift 3
    sbatch --parsable --time="$t" \
        --export=ALL,SCRIPT="$script",OUT="$out",MODEL="$MODEL",VLLM_URL="$V",HEXA_TEACHER_MAX_TOKENS=32768,ARM_ARGS="$*" \
        scripts/run_maze_arm.sh
}

id=$(sub 8:00:00 scripts/run_orbit_source_hexa.py results/orbit_minesweeper_hexa/run0 \
     --task minesweeper --run-index 0 --rounds 20 --batch-size 10 --max-trajs 12 --concurrency 6)
echo "minesweeper hexa run0 (resumes from its last round) -> $id"

echo
echo "When it finishes, Phase 3 (6 syntheses, run0 only):"
echo "  python scripts/run_design_c_phase3_all.py --runs 0"
echo "Read every inverted-premise flag before queueing Phase 4."
