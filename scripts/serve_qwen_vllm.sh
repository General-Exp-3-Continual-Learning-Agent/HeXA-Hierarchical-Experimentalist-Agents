#!/bin/bash
#SBATCH --job-name=qwen_vllm
#SBATCH --partition=superpod-a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH -t 8:00:00
#SBATCH --mem=96G
#SBATCH -o logs/qwen_vllm_%j.out
#SBATCH -e logs/qwen_vllm_%j.err
#SBATCH -A pi_sniekum_umass_edu

# ─── Qwen2.5-7B-Instruct as an OpenAI-compatible vLLM server ─────────────────
#
# The small-scale actor/evolver for the ORBIT Maze arm (DECISIONS.md D7a). One
# A100 instead of the four gpt-oss-120b needs, which is the point: a 1-GPU job
# schedules in minutes where a 4-GPU one queued behind ~6 hours.
#
#   sbatch scripts/serve_qwen_vllm.sh
#   PORT=8123 MODEL_DIR=/path/to/other sbatch scripts/serve_qwen_vllm.sh
#
# Deliberately much leaner than serve_gpt_oss_vllm.sh, because almost everything
# that script fights with is gpt-oss-specific and does not apply here:
#
#   * No --tensor-parallel-size / --disable-custom-all-reduce. One GPU, so there
#     is no collective to misconfigure -- that script's P2P failure (job 62653168)
#     cannot occur.
#   * No --reasoning-parser. Qwen2.5-Instruct has no separate analysis channel, so
#     the whole response arrives in `content`. `hexa_core.llm.base` reads
#     `reasoning`/`reasoning_content`, finds neither, and `raw_text` collapses to
#     the final text -- which is CORRECT here, not the silent chain-of-thought loss
#     that script warns about. `hexa_orbit.actor` parses `raw_text` and stores
#     `final_text`; with no reasoning channel those are the same string.
#   * No VLLM_SYSTEM_START_DATE pin. That exists because vLLM's *harmony* renderer
#     stamps the current date into gpt-oss's system message. Qwen's chat template
#     does not, so there is no midnight non-determinism to pin.
#   * No MXFP4. Qwen2.5-7B is bf16, ~15 GB of weights on an 80 GB card.
#
# What is kept, because it is not model-specific: port collision handling, the
# endpoint JSON the client reads, a readiness probe that fails loudly, and a
# cleanup trap so a GPU process never outlives the job.

set -Eeuo pipefail

MODEL_DIR="${MODEL_DIR:-/datasets/ai/qwen2/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}"
# Served under its real name. Unlike gpt-oss there is no name-based gating to
# worry about: hexa_core's client injects reasoning_effort only for names
# containing "gpt-oss-120b", so a Qwen name automatically avoids sending an
# argument this model does not understand.
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8000}"
# Qwen2.5's native context. Our worst case is a 27-turn cumulative transcript
# plus a ~1000-token bank block, well under 16k -- but the cap costs nothing.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# Set to `qwen3` for Qwen3-*, which is a hybrid thinking model: without a parser
# its `<think>...</think>` block stays inline in `content`, so `final_text` would
# carry the whole chain of thought and `hexa_orbit.actor` would feed it back into
# the transcript on every one of up to 27 turns -- context bloat plus the model
# reading its own discarded reasoning as dialogue. With the parser, `reasoning`
# and `content` arrive split, which is exactly what the actor assumes: it parses
# `raw_text` (reasoning + content) for the action and stores only `final_text`.
# Leave EMPTY for Qwen2.5-Instruct, which has no thinking mode at all.
REASONING_PARSER="${REASONING_PARSER:-}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"
CONDA_ENV="${CONDA_ENV:-/work/pi_sniekum_umass_edu/${USER}/.conda/envs/vllm-gptoss}"

# $0 is not usable under sbatch (SLURM copies the script to its spool dir), so
# repo-relative paths must come from SLURM_SUBMIT_DIR.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p "$REPO_DIR/logs"
# HF_HUB_CACHE, not HF_HOME, is what hub resolution actually uses when both are
# set -- and this cluster presets it to a path under another PI's project dir that
# we cannot write. Override both, or a download fails with EACCES on a directory
# nobody asked for.
export HF_HOME="${HF_HOME:-${MODEL_CACHE:-$HOME}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE_OVERRIDE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1        # weights are already local; never reach out

# ─── env ─────────────────────────────────────────────────────────────────────
# `conda` is NOT on PATH inside a batch job on this cluster -- it arrives via the
# module system, so `conda info --base` fails and sourcing its profile script is a
# silent no-op. `module load conda/latest` first, exactly as
# serve_gpt_oss_vllm.sh:136 does. Job 63540256 died in one second on this.
if ! command -v vllm >/dev/null 2>&1; then
    module load conda/latest 2>/dev/null || true
    conda activate "$CONDA_ENV" || {
        echo "ERROR: could not activate ${CONDA_ENV}, and vllm is not on PATH."
        echo "       vllm lives in exactly one env on this machine; see"
        echo "       scripts/serve_gpt_oss_vllm.sh for the note on why."
        exit 1
    }
fi
command -v vllm >/dev/null 2>&1 || { echo "ERROR: vllm still not on PATH"; exit 1; }
[[ -d "$MODEL_DIR" ]] || { echo "ERROR: MODEL_DIR not found: $MODEL_DIR"; exit 1; }

# ─── cluster guards, ported verbatim from serve_gpt_oss_vllm.sh:147-177 ──────
#
# These are CLUSTER-specific, not model-specific, which is why stripping them for
# a "leaner" script was wrong: job 63540842 died after 2m32s on exactly the
# traceback that block's comment names. That script is known-good -- it served the
# CORE experiments -- so this is a copy, not a re-derivation.
#
# FlashInfer's sampler is JIT-compiled at startup and needs nvcc, which compute
# nodes do not have (only the CUDA runtime). Left on, startup dies during KV-cache
# profiling:
#   flashinfer/jit/cpp_ext.py:61 get_cuda_path ->
#   RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'
#     doesn't exist
# Turning it off uses vLLM's native Torch sampler: identical sampling semantics
# (temperature/top_p applied the same way), just an unfused kernel. Recorded in the
# endpoint JSON because a fused vs unfused sampler can differ in the last bits of
# the probability computation, so a run that flipped this is not bitwise-comparable
# to one that did not.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# Belt and braces for any OTHER FlashInfer JIT path. Best-effort: if the module is
# unavailable, the sampler opt-out above is what actually keeps startup alive, so a
# failure here must not abort the job.
if [[ -z "${CUDA_HOME:-}" ]]; then
    module load "${CUDA_MODULE:-cuda/13.1}" 2>/dev/null || true
    if command -v nvcc >/dev/null 2>&1; then
        CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
        export CUDA_HOME
    fi
fi
# JIT scratch must be writable: $HOME is quota-limited here and the default lands
# under a read-only prefix on some installs.
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$REPO_DIR/.cache/flashinfer}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$REPO_DIR/.cache/torch_extensions}"
mkdir -p "$FLASHINFER_WORKSPACE_BASE" "$TORCH_EXTENSIONS_DIR"

# ─── port ─────────────────────────────────────────────────────────────────────
REQUESTED_PORT="$PORT"
for candidate in $(seq "$REQUESTED_PORT" $(( REQUESTED_PORT + 63 ))); do
    if ! (exec 3<>"/dev/tcp/127.0.0.1/${candidate}") 2>/dev/null; then PORT="$candidate"; break; fi
    echo "[port] ${candidate} busy on $(hostname -s); trying next"
done
[[ "$PORT" != "$REQUESTED_PORT" ]] && echo "[port] using ${PORT}; read the endpoint JSON, not the default"

NODE_SHORT="$(hostname -s)"
BASE_URL="http://${NODE_SHORT}:${PORT}/v1"
echo "================================================================"
echo "  Job ID    : ${SLURM_JOB_ID:-interactive}"
echo "  Node      : ${NODE_SHORT}"
echo "  Model     : ${MODEL_DIR}"
echo "  Served as : ${SERVED_MODEL_NAME}"
echo "  Reasoning : ${REASONING_PARSER:-<none>}"
echo "  Endpoint  : ${BASE_URL}"
echo "  Sampler   : VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER} (0 = native Torch; FlashInfer needs nvcc, absent here)"
echo "  GPUs      : $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')"
echo "================================================================"

VLLM_ARGS=(
    --served-model-name "$SERVED_MODEL_NAME"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --host 0.0.0.0
    --port "$PORT"
)
[[ -n "$REASONING_PARSER" ]] && VLLM_ARGS+=(--reasoning-parser "$REASONING_PARSER")

vllm serve "$MODEL_DIR" "${VLLM_ARGS[@]}" &
SERVER_PID=$!

cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[$(date)] stopping vllm (pid $SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ─── readiness ───────────────────────────────────────────────────────────────
waited=0
until curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: vllm exited during startup; see the .err file"; exit 1
    fi
    if (( waited >= READY_TIMEOUT )); then
        echo "ERROR: not ready after ${READY_TIMEOUT}s"; exit 1
    fi
    sleep 10; waited=$(( waited + 10 ))
done
echo "[$(date)] server up after ~${waited}s"

# A real generation, not just /v1/models: a server that lists a model but cannot
# decode is the failure that wastes a whole run.
PROBE=$(curl -sf "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${SERVED_MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: \\\\boxed{up}\"}],\"max_tokens\":32,\"temperature\":0}" || true)
echo "[probe] ${PROBE:0:300}"
if ! printf '%s' "$PROBE" | grep -q '"content"'; then
    echo "ERROR: probe returned no content; refusing to advertise a broken endpoint"
    exit 1
fi

# ─── publish the endpoint ────────────────────────────────────────────────────
ENDPOINT_JSON="$REPO_DIR/logs/vllm_endpoint_${SLURM_JOB_ID:-local}.json"
cat > "$ENDPOINT_JSON" <<EOF
{
  "job_id": "${SLURM_JOB_ID:-local}",
  "node": "${NODE_SHORT}",
  "port": ${PORT},
  "base_url": "${BASE_URL}",
  "served_model_name": "${SERVED_MODEL_NAME}",
  "model_dir": "${MODEL_DIR}",
  "max_model_len": ${MAX_MODEL_LEN},
  "gpu_memory_utilization": ${GPU_MEM_UTIL},
  "reasoning_parser": "${REASONING_PARSER:-null}",
  "tensor_parallel_size": 1,
  "flashinfer_sampler": ${VLLM_USE_FLASHINFER_SAMPLER},
  "cuda_home": "${CUDA_HOME:-}",
  "written_at": "$(date -Is)",
  "written_epoch": $(date +%s)
}
EOF
cp "$ENDPOINT_JSON" "$REPO_DIR/logs/vllm_endpoint_latest.json"
echo "  endpoint written to: ${ENDPOINT_JSON}"
echo
echo "  export HEXA_VLLM_URL=${BASE_URL}"
echo "  python scripts/run_orbit_maze_baseline.py --n 5 \\"
echo "      --model ${SERVED_MODEL_NAME} --reasoning-effort '' \\"
echo "      --out results/orbit_maze_baseline"
echo

wait "$SERVER_PID"
