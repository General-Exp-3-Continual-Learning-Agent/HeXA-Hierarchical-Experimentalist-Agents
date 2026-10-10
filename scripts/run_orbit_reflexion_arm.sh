#!/bin/bash
#SBATCH --job-name=orbit_reflexion
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -o logs/orbit_reflexion_%j.log
# stderr merges into the same file, matching run_orbit_memory_arm.sh.

# ─── The Reflexion baseline on ORBIT Maze or Mastermind ─────────────────────
#
# CPU only: the client waits on a separately-served vLLM endpoint the whole
# time, exactly as run_orbit_memory_arm.sh does. Bring the GPU up first:
#
#   MODEL_DIR=/datasets/ai/qwen3/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
#   SERVED_MODEL_NAME=Qwen/Qwen3-8B REASONING_PARSER=qwen3 \
#     sbatch scripts/serve_qwen_vllm.sh
#
#   TASK=mastermind VLLM_URL=http://gpu016:8000/v1 \
#     sbatch scripts/run_orbit_reflexion_arm.sh
#
# Runs REPLICATES independently built runs back to back, because a replicate is
# a whole run: each is its own trial-1 -> reflect -> trial-2 chain, and the
# reflections are what varies. Re-evaluating one set of reflections would
# measure actor sampling only and understate the spread.
#
# NO build pass and NO calibration, unlike the memory baselines: Reflexion's
# memory is per-instance and is discarded at the instance boundary, so there is
# nothing to construct before evaluating.
#
# Sizing, from measured cost (orbit-run-compute-constraints). The episode
# variants make ONE pass -- the retry structure is ORBIT's own three episodes --
# so they cost about what the no-bank arm costs plus two short reflection calls
# per instance, roughly half what --variant trials costs:
#   episode-*  mastermind ~0.8 h, maze ~3.2 h per replicate
#   trials     mastermind ~1.5 h, maze ~6.5 h per replicate
# --resume makes a walltime kill continue from the instance it reached.

set -Eeuo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"

PYTHON="${PYTHON:-$HOME/.conda/envs/phyreagent/bin/python}"
TASK="${TASK:-mastermind}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
# episode-reset is the reported arm: reflect between ORBIT's own episodes and
# wipe the chat, so the lesson is the only thing that crosses. episode-keep is
# its ablation (transcript retained). trials replays the whole instance K times.
VARIANT="${VARIANT:-episode-reset}"
K_TRIALS="${K_TRIALS:-2}"
REPLICATES="${REPLICATES:-3}"
OUT_ROOT="${OUT_ROOT:-results_orbit/Reflexion_qwen/orbit_${TASK}_reflexion_${VARIANT}_qwen8}"

case "$TASK" in
    maze)        N="${N:-256}"; CONCURRENCY="${CONCURRENCY:-24}" ;;
    mastermind)  N="${N:-40}";  CONCURRENCY="${CONCURRENCY:-16}" ;;
    *) echo "ERROR: TASK must be maze or mastermind"; exit 1 ;;
esac

# Both Qwen3-8B tables run 16384 on BOTH tasks. The 2k default truncates ~5% of
# turns and truncation_scoring=exclude then drops them silently, so a mismatch
# here does not fail -- it quietly produces a number that cannot be paired
# against the no-bank arm this is differenced from.
MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-16384}"
# Unset by default -> run_orbit_reflexion.py's own default (ActorConfig.reasoning_effort,
# "low") is used. Override to re-run a model at a different reasoning depth --
# gpt-oss honours this as a hard cap; Qwen3's chat template does not throttle
# equivalently at "low" (qwen3-thinking-fills-the-budget), so the two models
# are not symmetric here and a cross-model comparison at mismatched effective
# effort is not paired even if this flag is set the same on both.
REASONING_EFFORT="${REASONING_EFFORT:-}"

if [[ -z "${VLLM_URL:-}" ]]; then echo "ERROR: set VLLM_URL."; exit 1; fi
if ! curl -sf "${VLLM_URL}/models" >/dev/null 2>&1; then
    echo "ERROR: ${VLLM_URL} is not answering."; exit 1
fi
export HEXA_VLLM_URL="$VLLM_URL"

# Replicate seeds. Mastermind's are 0 / 100000 / 200000 -- NOT 0/1/2 -- matching
# the no-bank arm these are paired against. Maze's no-bank replicates ran with
# no seed at all, sampling nondeterminism being the draw, so an empty entry here
# means "pass no --seed" and keeps the two arms drawn the same way.
case "$TASK" in
    mastermind) SEEDS=(0 100000 200000) ;;
    maze)       SEEDS=("" "" "") ;;
esac

echo "[$(date)] reflexion arm: task=$TASK variant=$VARIANT n=$N replicates=$REPLICATES"
echo "           model=$MODEL endpoint=$VLLM_URL max_response_tokens=$MAX_RESPONSE_TOKENS"

for ((i = 0; i < REPLICATES; i++)); do
    OUT="${OUT_ROOT}/run${i}"
    SEED="${SEEDS[$i]:-}"
    echo
    echo "================ replicate $((i + 1))/$REPLICATES -> $OUT ================"
    ARGS=(
        --task "$TASK" --n "$N" --variant "$VARIANT" --k-trials "$K_TRIALS"
        --concurrency "$CONCURRENCY"
        --model "$MODEL" --base-url "$VLLM_URL"
        --max-response-tokens "$MAX_RESPONSE_TOKENS"
        --out "$OUT" --resume
    )
    [[ -n "$REASONING_EFFORT" ]] && ARGS+=(--reasoning-effort "$REASONING_EFFORT")
    [[ -n "$SEED" ]] && ARGS+=(--seed "$SEED")

    # A dead server must not burn the remaining replicates against nothing.
    if ! curl -sf "${VLLM_URL}/models" >/dev/null 2>&1; then
        echo "ERROR: endpoint died before replicate $i; stopping so --resume can pick up."
        exit 1
    fi
    "$PYTHON" scripts/run_orbit_reflexion.py "${ARGS[@]}"
done

echo
echo "[$(date)] all $REPLICATES replicates done -> $OUT_ROOT"
