#!/bin/bash
#SBATCH --job-name=mm_designb
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 10:00:00
#SBATCH -o logs/mm_designb_%j.log

# ─── Design B Mastermind, rerun on the ORBIT-paper task config ───────────────
#
# `hexa_orbit.mastermind.TASK_CONFIG` was changed to match the ORBIT paper:
# code_length 3, num_numbers 6 (120 codes), down from 4/8 (1680 codes). Every
# Mastermind artifact produced before that change is measured on a different
# task and cannot be compared against anything current:
#
#   results/orbit_mastermind_baseline_oss      4/8  stale
#   results/orbit_mastermind_hexa_v3           4/8  stale
#   results/design_c/mastermind_baseline       4/8  stale (partial)
#
# The old hexa_v3 run was compromised twice: besides the config, its bank
# changed in only rounds 0, 8 and 14 of 20 -- the teacher-truncation signature
# from HEXA_TEACHER_MAX_TOKENS=16384, where an over-long evolution response is
# rejected and `evolve_skill_bank` silently keeps the previous bank. Hence 32768
# here, which gpt-oss (served at 65536) has room for.
#
# Rounds are 8, not the old run's 20. `perm(6,3) = 120` is the whole code space
# and instance identity IS the code, so train gets the 80 codes left after the
# 40-code eval prefix -- which caps a run at 8 rounds of batch 10. Asking for 20
# does not raise; `train_specs(80)` is drawn once and the rounds recycle it,
# showing the teacher each code ~2.5 times. See MastermindTask.EVAL_POOL_SIZE.
#
#   VLLM_URL=http://gpu017:8000/v1 sbatch scripts/run_orbit_mastermind_rerun.sh

set -Eeuo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"

PYTHON="${PYTHON:-$HOME/.conda/envs/phyreagent/bin/python}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
CONCURRENCY="${CONCURRENCY:-24}"
OUT="${OUT:-results/orbit_mastermind_hexa_oss_v4}"

if [[ -z "${VLLM_URL:-}" ]]; then echo "ERROR: set VLLM_URL."; exit 1; fi
export HEXA_VLLM_URL="$VLLM_URL"
export HEXA_TEACHER_MAX_TOKENS="${HEXA_TEACHER_MAX_TOKENS:-32768}"
if ! curl -sf "${VLLM_URL}/models" >/dev/null 2>&1; then
    echo "ERROR: ${VLLM_URL} is not answering."; exit 1
fi
echo "[$(date)] endpoint ${VLLM_URL} is live, teacher budget ${HEXA_TEACHER_MAX_TOKENS}"

"$PYTHON" -u scripts/run_orbit_mastermind_hexa.py \
    --rounds 8 --batch-size 10 --train-size 80 --eval-size 40 \
    --max-trajs 12 --reward-bin \
    --concurrency "$CONCURRENCY" --max-response-tokens 4096 \
    --model "$MODEL" --teacher-model "$MODEL" --base-url "$VLLM_URL" \
    --out "$OUT"

echo "=== rounds where the bank actually changed ==="
"$PYTHON" - "$OUT" <<'PY'
import glob, hashlib, re, sys
rs = sorted(glob.glob(f"{sys.argv[1]}/round*/bank.json"),
            key=lambda p: int(re.search(r"round(\d+)", p).group(1)))
prev, changed = None, []
for p in rs:
    h = hashlib.sha1(open(p, "rb").read()).hexdigest()
    if h != prev:
        changed.append(int(re.search(r"round(\d+)", p).group(1)))
    prev = h
print(f"changed in {len(changed)}/{len(rs)} rounds: {changed}")
if len(changed) < 0.3 * len(rs):
    print("WARNING: bank mostly frozen -- check for teacher truncation.")
PY
echo "[$(date)] done"
