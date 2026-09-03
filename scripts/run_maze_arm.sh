#!/bin/bash
#SBATCH --job-name=maze_arm
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 6:00:00
#SBATCH -o logs/maze_arm_%j.log
# No -e: stderr is merged into the same file. Progress prints go to stdout
# while the httpx/teacher logging goes to stderr, and reading a run's story
# across two files interleaved by nothing is a waste of time.

# ─── Run one Maze arm as a batch job, not on the login node ───────────────────
#
# The experiment client is I/O bound -- it spends its time waiting on the vLLM
# server, using ~26% of one core and 190 MB. So this asks for CPU only; the GPU
# belongs to the separate `serve_qwen_vllm.sh` job it talks to.
#
# It runs here rather than on a login node for three reasons, none of them load:
# a 256-instance arm takes ~2 hours, which is long enough that an admin sweep or a
# dropped session is a normal event; batch jobs get logs and accounting; and the
# login node is shared with ~400 users.
#
#   ARM_ARGS="--n 256 --concurrency 12 --resume" \
#   VLLM_URL=http://gpu013:8000/v1 \
#   OUT=results/orbit_maze_baseline \
#     sbatch scripts/run_maze_arm.sh
#
# For the HExA arm, set SCRIPT=scripts/run_orbit_maze_hexa.py.

set -Eeuo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"

SCRIPT="${SCRIPT:-scripts/run_orbit_maze_baseline.py}"
PYTHON="${PYTHON:-$HOME/.conda/envs/phyreagent/bin/python}"
OUT="${OUT:-results/orbit_maze_baseline}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
ARM_ARGS="${ARM_ARGS:---n 256 --concurrency 12}"

# The endpoint must be passed in, not guessed: `vllm_endpoint_latest.json` can be
# stale from a previous job (one from Aug 22 pointed at a dead gpt-oss server), and
# silently aiming an arm at the wrong model would be worse than failing.
if [[ -z "${VLLM_URL:-}" ]]; then
    echo "ERROR: set VLLM_URL to the serving job's base_url."
    echo "       Read it from logs/vllm_endpoint_<jobid>.json -- NOT from"
    echo "       vllm_endpoint_latest.json, which may belong to an older job."
    exit 1
fi
export HEXA_VLLM_URL="$VLLM_URL"
# The teacher's default 32768-token output budget is right only against a server
# with room for it (gpt-oss is served at 65536). On a 32k-context server vLLM
# rejects the call outright and `evolve_skill_bank` silently keeps the old bank,
# so a whole round learns nothing. 16384 leaves ample room for prompt + thinking.
export HEXA_TEACHER_MAX_TOKENS="${HEXA_TEACHER_MAX_TOKENS:-16384}"

if ! curl -sf "${VLLM_URL}/models" >/dev/null 2>&1; then
    echo "ERROR: ${VLLM_URL} is not answering. Is the serving job still running?"
    exit 1
fi
echo "[$(date)] endpoint ${VLLM_URL} is live"
echo "[$(date)] ${SCRIPT} ${ARM_ARGS} --model ${MODEL} --out ${OUT}"

# -u so progress reaches the log as instances land rather than at process exit.
exec "$PYTHON" -u "$SCRIPT" $ARM_ARGS --model "$MODEL" --out "$OUT"
