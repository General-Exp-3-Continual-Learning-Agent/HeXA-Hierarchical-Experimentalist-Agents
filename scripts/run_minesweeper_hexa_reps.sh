#!/bin/bash
#SBATCH --job-name=ms_hexa_rep
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 12:00:00
#SBATCH -o logs/ms_hexa_rep_%j.log

# ─── Finish Minesweeper HExA runs 1 and 2 ────────────────────────────────────
#
# run0 completed 20 rounds and has an eval_frozen; runs 1 and 2 were killed
# mid-round (8 and 7) and have no run.json, so the within-level Minesweeper
# result is currently n=1 -- 1 bank against 3 baselines rather than 3x3.
#
# Resume is the default (`--no-resume` not passed): `loop.resume_state` reads
# the output dir and restarts at the first incomplete round, so the finished
# rounds are not recomputed.
#
# Settings mirror run0's manifest exactly: rounds 20, batch 10, train 400,
# max_trajs 12, reward-bin on (it is the default; --no-reward-bin is the opt-out).
#
# Teacher budget 32768, not the 16384 that silently froze banks elsewhere in
# this project: an over-long evolution response is rejected and
# `evolve_skill_bank` keeps the previous bank, so a whole round learns nothing.
#
#   RUN_INDEX=1 VLLM_URL=http://gpu017:8000/v1 sbatch scripts/run_minesweeper_hexa_reps.sh

set -Eeuo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"

PYTHON="${PYTHON:-$HOME/.conda/envs/phyreagent/bin/python}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
CONCURRENCY="${CONCURRENCY:-24}"
RUN_INDEX="${RUN_INDEX:?set RUN_INDEX to 1 or 2}"

if [[ -z "${VLLM_URL:-}" ]]; then echo "ERROR: set VLLM_URL."; exit 1; fi
export HEXA_VLLM_URL="$VLLM_URL"
export HEXA_TEACHER_MAX_TOKENS="${HEXA_TEACHER_MAX_TOKENS:-32768}"
curl -sf "${VLLM_URL}/models" >/dev/null || { echo "ERROR: ${VLLM_URL} not answering."; exit 1; }
echo "[$(date)] endpoint ${VLLM_URL} live, teacher budget ${HEXA_TEACHER_MAX_TOKENS}, run${RUN_INDEX}"

"$PYTHON" -u scripts/run_orbit_source_hexa.py \
    --task minesweeper --run-index "$RUN_INDEX" \
    --rounds 20 --batch-size 10 --train-size 400 --max-trajs 12 \
    --concurrency "$CONCURRENCY" --max-response-tokens 4096 \
    --model "$MODEL" --teacher-model "$MODEL" --base-url "$VLLM_URL"

echo "=== rounds where the bank actually changed (run${RUN_INDEX}) ==="
"$PYTHON" - "results/orbit_minesweeper_hexa/run${RUN_INDEX}" <<'PY'
import glob, hashlib, re, sys
rs = sorted(glob.glob(f"{sys.argv[1]}/round*/bank.json"),
            key=lambda p: int(re.search(r"round(\d+)", p).group(1)))
prev, changed = None, []
for p in rs:
    h = hashlib.sha1(open(p, "rb").read()).hexdigest()
    if h != prev: changed.append(int(re.search(r"round(\d+)", p).group(1)))
    prev = h
print(f"changed in {len(changed)}/{len(rs)} rounds: {changed}")
if rs and len(changed) < 0.3 * len(rs):
    print("WARNING: bank mostly frozen -- check for teacher truncation.")
PY
echo "[$(date)] done"
