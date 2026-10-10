#!/bin/bash
#SBATCH --job-name=gpt_oss_120b_vllm
#SBATCH --partition=superpod-a100
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH -c 16
#SBATCH -t 3-00:00:00
#SBATCH --mem=480G
#SBATCH -o logs/gpt_oss_120b_vllm_%j.out
#SBATCH -e logs/gpt_oss_120b_vllm_%j.err
#SBATCH -A pi_sniekum_umass_edu

# ─── gpt-oss-120b as an OpenAI-compatible vLLM server ────────────────────────
#
# Serves the actor/evolver for hexa_core. Both the Phase 2 reproduction gate
# (scripts/run_repro_gate.py) and everything after it talk to this endpoint.
#
# Walltime is 3 days, not 1: job 64581745 hit the old 24h cap mid-run (SLURM
# TIMEOUT, not a crash) while `scripts/run_core_memory_baseline_arm.sh` was
# still mid-flight, aborting 31 in-flight requests and corrupting one eval
# pass with connection-error "failures" indistinguishable from real ones at
# the time. superpod-a100's own MaxTime is 30 days, so this was purely our
# own default, not a cluster limit.
#
#   sbatch scripts/serve_gpt_oss_vllm.sh          # 4x A100 (superpod-a100), TP=4
#   H100=1 bash scripts/serve_gpt_oss_vllm.sh     # 2x H100 (gpu),           TP=2
#
# GPU count: MEASURED, not inferred. An earlier version of this comment claimed
# A100 (SM80) cannot run MXFP4 so vLLM dequantizes to bf16 (~234 GB, hence 4 GPUs).
# That is WRONG for vLLM 0.26: the load log reads "Using 'MARLIN' Mxfp4 MoE backend"
# and "Model loading took 17.22 GiB memory" per rank -- ~69 GB total against a
# 60.77 GiB checkpoint. MXFP4 runs natively on SM80 via MARLIN.
#
# So TP=4 is not a memory requirement. TP=2 (~34.5 GiB/rank) fits comfortably and
# halves the allocation. Prefer it when the client can keep the server busy: this
# cluster cancels underutilised jobs (job 62661980 was killed by the `unity-data`
# account after 2h09m while the server sat at ~1 request in flight), and 2 busy
# GPUs are safer than 4 idle ones. TP=4 remains the default only because it is the
# configuration proven to serve this model here.
#
# PREREQUISITE, not installed by this script: vllm lives in ONE env on this
# machine -- /work/pi_sniekum_umass_edu/$USER/.conda/envs/vllm-gptoss (vllm 0.26.0,
# openai 2.53.0, tiktoken 0.13.0). That is the default $CONDA_ENV below. It is NOT
# `phyreagent`, which has openai and torch but no vllm at all.
# The script preflight-checks the import and exits with instructions if missing.
# It deliberately never pip-installs anything itself -- silently mutating a shared
# env from inside a batch job is how runs stop being reproducible.
#
# Override anything via env:
#   PORT=8123 MAX_MODEL_LEN=98304 GPU_MEM_UTIL=0.92 sbatch scripts/serve_gpt_oss_vllm.sh

set -Eeuo pipefail

# ─── H100 path: sbatch directives above cannot be conditional, so re-submit ──
# with CLI flags, which override the in-file #SBATCH block.
if [[ "${H100:-0}" == "1" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "ERROR: H100=1 was set inside an existing job (${SLURM_JOB_ID})."
        echo "       The #SBATCH block in this file already asked for 4x A100, so this"
        echo "       job has A100s and TP=2 would be wrong. Run instead:"
        echo "           H100=1 bash scripts/serve_gpt_oss_vllm.sh"
        exit 1
    fi
    mkdir -p logs
    # --gres=gpu:h100:2 rather than --gpus=2: the `gpu` partition is heterogeneous
    # (v100 / a40 / a100 / l40s / h100), and a bare --gpus=2 would happily land on
    # V100s, which cannot run this model at all.
    exec sbatch --job-name=gpt_oss_120b_vllm_h100 \
                --partition=gpu \
                --nodes=1 \
                --gres=gpu:h100:2 \
                -c 16 \
                --mem=240G \
                -t "${WALLTIME:-1-00:00:00}" \
                -o logs/gpt_oss_120b_vllm_h100_%j.out \
                -e logs/gpt_oss_120b_vllm_h100_%j.err \
                "$0" "$@"
fi

# ─── Configurable ────────────────────────────────────────────────────────────
GPT_OSS_120B_PATH="${GPT_OSS_120B_PATH:-/datasets/ai/gpt/hub/models--openai--gpt-oss-120b/snapshots/eabf0c518da7584a2e7dab4ab272709785a72126/}"
# $0 is NOT usable under sbatch: SLURM copies the script into its spool dir, so
# dirname "$0" is /var/spool/slurm/slurmd and every repo-relative path (tiktoken
# cache, logs, endpoint JSON) lands somewhere unwritable. SLURM_SUBMIT_DIR is the
# directory sbatch was invoked from, which is the repo root by construction of the
# documented usage; fall back to $0 only for the `bash` path, where it is correct.
if [[ -z "${REPO_DIR:-}" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        REPO_DIR="$SLURM_SUBMIT_DIR"
    else
        REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
    fi
fi
# Verify rather than trust: submitting from elsewhere would otherwise scatter run
# artifacts into an unrelated tree and only surface as "the gate can't find them".
if [[ ! -d "$REPO_DIR/hexa_core" ]]; then
    echo "ERROR: REPO_DIR=$REPO_DIR does not look like the HeXA repo (no hexa_core/)."
    echo "       sbatch from the repo root, or pass REPO_DIR=/path/to/HeXA explicitly."
    exit 1
fi
# The serving env, which is NOT `phyreagent`: phyreagent has openai + torch but no
# vllm, so the preflight below would abort on every launch and its error message
# would tell you to pip-install into a shared env. `vllm-gptoss` is the env that
# actually has vllm (0.26.0) + openai + tiktoken. `conda activate` accepts a path.
CONDA_ENV="${CONDA_ENV:-/work/pi_sniekum_umass_edu/${USER}/.conda/envs/vllm-gptoss}"
HF_HOME_DIR="${HF_HOME_DIR:-${SCRATCH:-$HOME}/hf_cache}"

# `openai/gpt-oss-120b`, not the snapshot path: CORE's client injects
# reasoning_effort="high" only for model names containing "gpt-oss-120b"
# (src/model/nvidia_client.py:102), and hexa_core's client mirrors that. Serving
# under a different name silently drops the reasoning effort.
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-openai/gpt-oss-120b}"
PORT="${PORT:-8000}"
# Must exceed --eval-tokens (32768) PLUS the prompt. ZebraLogic xlarge prompts are
# the long ones; a 32768 context would guarantee finish_reason == "length" on every
# long reasoning trace and quietly floor the measured accuracy.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"   # a 234 GB bf16 load is slow
STRICT_PROBE="${STRICT_PROBE:-1}"        # 0 = serve anyway if the probe fails

# REPRODUCIBILITY, load-bearing. vLLM's harmony renderer stamps the *current date*
# into gpt-oss's system message and says so itself:
#   vllm/entrypoints/openai/parser/harmony_utils.py:133-138
#   # NOTE(woosuk): This brings non-determinism in vLLM.
#   start_date = envs.VLLM_SYSTEM_START_DATE or datetime.now().strftime("%Y-%m-%d")
# Unpinned, the prompt every request sees changes at midnight, so a campaign that
# spans days is silently several different experiments. Pinned here, recorded in
# the endpoint JSON, and carried into every run manifest; treat a change in this
# value as a change in experimental condition, not a detail.
VLLM_SYSTEM_START_DATE="${VLLM_SYSTEM_START_DATE:-2026-01-01}"
export VLLM_SYSTEM_START_DATE
# ─────────────────────────────────────────────────────────────────────────────

# Tensor parallelism follows what the job actually got, not what it asked for.
# nvidia-smi is the source of truth: SLURM_JOB_GPUS is set by --gpus but not by
# every --gres form, so deriving TP from the env is how you end up with TP=4 on
# two GPUs.
VISIBLE_GPUS="$( (nvidia-smi -L 2>/dev/null || true) | grep -c '^GPU' || true)"
VISIBLE_GPUS="${VISIBLE_GPUS:-0}"
TP="${TP:-$VISIBLE_GPUS}"

module load conda/latest
conda activate "$CONDA_ENV"

export HF_HOME="$HF_HOME_DIR"
export HF_HUB_OFFLINE=1          # weights are a local snapshot; never phone home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
# Some vLLM builds need this for gpt-oss sink attention on SM80. Documented, not
# forced on: pinning the wrong backend breaks startup outright.
# export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1

# FlashInfer's sampler is JIT-compiled at startup and needs nvcc, which compute
# nodes do not have (only the CUDA runtime). Left on, startup dies during KV-cache
# profiling:
#   flashinfer/jit/cpp_ext.py:61 get_cuda_path ->
#   RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'
#     doesn't exist
# vLLM defaults it to True (`vllm/envs.py:49`) and reads it as an int (`:841`).
# Turning it off makes vLLM use its native Torch sampler -- same sampling semantics
# (temperature/top_p are applied identically), just an unfused kernel, and sampling
# is nowhere near the bottleneck for 32k-token reasoning traces. Recorded in the
# endpoint JSON because it is a numerics-relevant choice, not a detail: a fused vs
# unfused sampler can differ in the last bits of the probability computation, so a
# run that flipped this value is not bitwise-comparable to one that did not.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# Belt and braces for any OTHER FlashInfer JIT path that may still be reached.
# torch here is built against CUDA 13.0; `cuda/13.1` is the closest module. Loading
# it is best-effort -- if the module is unavailable the sampler opt-out above is
# what actually keeps startup alive, so a failure here must not abort the job.
if [[ -z "${CUDA_HOME:-}" ]]; then
    module load "${CUDA_MODULE:-cuda/13.1}" 2>/dev/null || true
    if command -v nvcc >/dev/null 2>&1; then
        CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
        export CUDA_HOME
    fi
fi
# JIT scratch must be writable and node-local-safe; $HOME can be quota-limited and
# the default lands under a read-only prefix on some installs.
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$REPO_DIR/.cache/flashinfer}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$REPO_DIR/.cache/torch_extensions}"
mkdir -p "$FLASHINFER_WORKSPACE_BASE" "$TORCH_EXTENSIONS_DIR"

# Fix tiktoken/harmony vocab download issue (known gpt-oss bug): the fetch fails
# on compute nodes, so pre-stage the encodings from the login node's cache.
mkdir -p "$REPO_DIR/tiktoken_encodings"
if [ ! -f "$REPO_DIR/tiktoken_encodings/o200k_base.tiktoken" ]; then
    wget -q -O "$REPO_DIR/tiktoken_encodings/o200k_base.tiktoken" \
        "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" || true
    wget -q -O "$REPO_DIR/tiktoken_encodings/cl100k_base.tiktoken" \
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken" || true
fi
export TIKTOKEN_ENCODINGS_BASE="$REPO_DIR/tiktoken_encodings"

mkdir -p "$REPO_DIR/logs"
cd "$REPO_DIR"

# ─── Validate the hardware before spending a 30-minute model load ────────────
if [ ! -d "$GPT_OSS_120B_PATH" ]; then
    echo "ERROR: weights not found at $GPT_OSS_120B_PATH"
    echo "       override with GPT_OSS_120B_PATH=..."
    exit 1
fi

if [ "$VISIBLE_GPUS" -lt 1 ]; then
    echo "ERROR: no GPU is visible. Run this via sbatch (or from a GPU allocation),"
    echo "       not on a login node."
    exit 1
fi
if [ "$VISIBLE_GPUS" -lt "$TP" ]; then
    echo "ERROR: --tensor-parallel-size $TP but only $VISIBLE_GPUS GPU(s) visible."
    exit 1
fi

COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
if awk -v c="$COMPUTE_CAP" 'BEGIN { exit !(c + 0 < 9.0) }'; then
    # MEASURED, superseding the older assumption inherited from
    # react_agent/bash_scripts/run_gpt_oss_4xa100.sh. That script's "117B x 2 bytes
    # = 234 GB BF16" premise -- that SM80 must dequantize MXFP4 -- is NOT what vLLM
    # 0.26 does. Job 62653168 on 4x A100 (SM80) logged:
    #     Checkpoint size: 60.77 GiB
    #     mxfp4.py:517 Using 'MARLIN' Mxfp4 MoE backend
    #     gpu_model_runner.py:5347 Model loading took 17.22 GiB memory
    # 17.22 GiB/rank x 4 ~= 69 GB total, i.e. weights stay quantized and MARLIN
    # handles MXFP4 on SM80. So 2 GPUs are sufficient on memory grounds; TP=4 stays
    # the default for KV-cache headroom at max-model-len 65536, not from necessity.
    if [ "$TP" -lt 2 ]; then
        echo "ERROR: TP=$TP is too small. Measured weight footprint is ~69 GB total"
        echo "       (MARLIN MXFP4, ~17.2 GiB/rank at TP=4), which leaves too little"
        echo "       room for a 65536-token KV cache on a single 80 GB card."
        exit 1
    fi
    PRECISION_NOTE="MXFP4 via MARLIN on SM${COMPUTE_CAP/./} (~17.2 GiB/rank measured, ~69 GB total)"
else
    PRECISION_NOTE="native MXFP4 (SM90+), ~61 GB"
fi

# ─── Preflight: vllm must already be installed ───────────────────────────────
if ! VLLM_VERSION="$(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"; then
    echo "ERROR: vllm is not installed in the conda env '$CONDA_ENV'."
    echo "       This script does not install packages. Either point it at the env"
    echo "       that already has vllm:"
    echo "           CONDA_ENV=/work/pi_sniekum_umass_edu/\$USER/.conda/envs/vllm-gptoss \\"
    echo "               sbatch scripts/serve_gpt_oss_vllm.sh"
    echo "       or install it once, yourself:"
    echo "           conda activate $CONDA_ENV && pip install \"vllm>=0.10.1\""
    echo "       (0.10.1 is the first release with gpt-oss / MXFP4 support.)"
    exit 1
fi

# ─── Pick a port nobody else is already holding ───────────────────────────────
# These are shared nodes and 8000 is everyone's default. A foreign OpenAI-compatible
# server on the requested port is not hypothetical -- job 62632641 hit exactly that
# on gpu018: the readiness probe passed in ~0s (impossible for a 234 GB bf16 load)
# because it was talking to somebody else's server, which then 404'd our model name.
# The dangerous version of that near-miss is a neighbour serving a model under the
# SAME name, in which case the whole gate would have scored another user's model and
# produced plausible, wrong numbers. So: refuse to reuse a live port.
port_is_taken() {
    # bash's /dev/tcp needs no python and no lsof. A successful connect means
    # something is listening.
    (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && { exec 3<&- 3>&-; return 0; }
    return 1
}

REQUESTED_PORT="$PORT"
for (( probe_offset=0; probe_offset < 64; probe_offset++ )); do
    candidate=$(( REQUESTED_PORT + probe_offset ))
    if ! port_is_taken "$candidate"; then
        PORT="$candidate"
        break
    fi
    echo "[port] ${candidate} is already in use on $(hostname -s); trying $(( candidate + 1 ))"
done
if port_is_taken "$PORT"; then
    echo "ERROR: no free port in ${REQUESTED_PORT}..$(( REQUESTED_PORT + 63 )) on $(hostname -s)."
    exit 1
fi
if [ "$PORT" != "$REQUESTED_PORT" ]; then
    echo "[port] NOTE: serving on ${PORT}, not the requested ${REQUESTED_PORT}."
    echo "[port] Read the endpoint from logs/vllm_endpoint_latest.json rather than"
    echo "[port] assuming ${REQUESTED_PORT}."
fi

NODE_SHORT="$(hostname -s)"
NODE_FQDN="$(hostname -f)"
NODE_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
BASE_URL="http://${NODE_SHORT}:${PORT}/v1"

echo "================================================================"
echo "  gpt-oss-120b via vLLM (OpenAI-compatible server)"
echo "  Job ID    : ${SLURM_JOB_ID:-interactive}"
echo "  Node      : ${NODE_SHORT} (${NODE_FQDN}) ip=${NODE_IP:-?}"
echo "  Partition : ${SLURM_JOB_PARTITION:-?}"
echo "  GPUs      : ${VISIBLE_GPUS} x compute_cap ${COMPUTE_CAP}   TP=${TP}"
echo "  Precision : ${PRECISION_NOTE}"
echo "  vllm      : ${VLLM_VERSION}"
echo "  Weights   : ${GPT_OSS_120B_PATH}"
echo "  Served as : ${SERVED_MODEL_NAME}"
echo "  Endpoint  : ${BASE_URL}"
echo "  Context   : max-model-len=${MAX_MODEL_LEN}  gpu-mem-util=${GPU_MEM_UTIL}"
echo "  Harmony   : VLLM_SYSTEM_START_DATE=${VLLM_SYSTEM_START_DATE} (pinned; unpinned"
echo "              this defaults to today and the prompt drifts daily)"
echo "  Sampler   : VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER}"
echo "              (0 = native Torch sampler; FlashInfer needs nvcc, absent here)"
echo "  CUDA_HOME : ${CUDA_HOME:-<unset; no nvcc>}"
echo "================================================================"

# --disable-custom-all-reduce is REQUIRED on these nodes. vLLM prefers its own
# CUSTOM all-reduce kernel (log: "Using ['CUSTOM', 'PYNCCL'] all-reduce backends"),
# which needs working peer-to-peer/IPC between the allocated GPUs. Under SLURM these
# 4 A100s are a partition of an 8-GPU node and P2P setup fails; job 62653168 died
# right after CUDA-graph profiling with all four ranks reporting:
#   Failed: Cuda error /workspace/csrc/custom_all_reduce.cuh:455 'invalid argument'
# then "Worker proc VllmWorker-3 died unexpectedly (exit code: None)" -- a signal
# death, so no Python traceback pointed at the real cause. Disabling it falls back
# to PyNCCL, which is correct and portable; the throughput cost is small next to
# 32k-token reasoning traces, and correctness of the collective is not optional.
#
# --reasoning-parser openai_gptoss is MANDATORY. Without it the server returns no
# `reasoning_content`, so raw_text = reasoning + final silently loses the whole
# chain of thought -- and because Tower-of-Hanoi's extractor takes the LAST
# parseable move list in the text, every score in the run is then computed on the
# wrong string. This is the single most likely cause of a Phase 2 miss.
vllm serve "$GPT_OSS_120B_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size "$TP" \
    --reasoning-parser openai_gptoss \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --disable-custom-all-reduce \
    --host 0.0.0.0 \
    --port "$PORT" &
SERVER_PID=$!

# Never leak a 4-GPU process past the end of the job.
cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[$(date)] stopping vllm (pid $SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT TERM INT

echo "[$(date)] waiting up to ${READY_TIMEOUT}s for the server to come up ..."
READY=0
for (( waited=0; waited < READY_TIMEOUT; waited += 10 )); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: vllm exited during startup. See the .err log above."
        exit 1
    fi
    # "Something answers on this port" is NOT readiness -- that is what let job
    # 62632641 declare success in 0s against a neighbour's server. Require the
    # response to actually advertise the model WE are serving.
    MODELS_JSON="$(curl -sf "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null || true)"
    if [ -n "$MODELS_JSON" ]; then
        if printf '%s' "$MODELS_JSON" | grep -qF "\"$SERVED_MODEL_NAME\""; then
            READY=1
            echo "[$(date)] server is up after ~${waited}s and advertises ${SERVED_MODEL_NAME}"
            break
        fi
        # Answering but with the wrong inventory means a foreign process owns this
        # port. Never fall through to the gate: fail loudly instead.
        echo "ERROR: something is already serving on port ${PORT} and it is not ours."
        echo "       /v1/models did not list '${SERVED_MODEL_NAME}'. Response was:"
        printf '       %s\n' "$(printf '%s' "$MODELS_JSON" | head -c 600)"
        echo "       This is a different process (ours would advertise that name)."
        echo "       Re-run; the port scan picks the next free port."
        exit 1
    fi
    sleep 10
done
if [ "$READY" -ne 1 ]; then
    echo "ERROR: server did not answer /v1/models within ${READY_TIMEOUT}s."
    exit 1
fi

# ─── Startup probe ───────────────────────────────────────────────────────────
# stdlib urllib, because `openai` is not installed in this env. Checks the two
# things that silently invalidate a whole run, and reports WHICH reasoning_effort
# transport the server accepted -- that is exactly what the client needs to know.
PROBE_OUT="$REPO_DIR/logs/vllm_probe_${SLURM_JOB_ID:-local}.json"
set +e
PROBE_RESULT="$(
PORT="$PORT" SERVED_MODEL_NAME="$SERVED_MODEL_NAME" PROBE_OUT="$PROBE_OUT" python - <<'PYEOF'
import json, os, sys, urllib.error, urllib.request

url = f"http://127.0.0.1:{os.environ['PORT']}/v1/chat/completions"
model = os.environ["SERVED_MODEL_NAME"]
messages = [{"role": "user", "content": "What is 17 * 23? Think it through."}]


def ask(body):
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


attempts = [
    ("top_level", {"model": model, "messages": messages, "max_tokens": 256,
                   "temperature": 0.6, "top_p": 0.95, "reasoning_effort": "high"}),
    # Older vLLM builds only accept it through the chat template.
    ("chat_template_kwargs", {"model": model, "messages": messages, "max_tokens": 256,
                              "temperature": 0.6, "top_p": 0.95,
                              "chat_template_kwargs": {"reasoning_effort": "high"}}),
]

evidence = {"attempts": []}
for mechanism, body in attempts:
    try:
        payload = ask(body)
    except urllib.error.HTTPError as exc:
        evidence["attempts"].append(
            {"mechanism": mechanism, "http_status": exc.code,
             "error": exc.read().decode(errors="replace")[:400]}
        )
        continue
    except Exception as exc:
        evidence["attempts"].append({"mechanism": mechanism, "error": repr(exc)})
        continue
    message = payload["choices"][0]["message"]
    # Both names, newest first -- vLLM >=0.26 returns `reasoning` on the RESPONSE
    # (ChatMessage.reasoning); only NIM and older vLLM use `reasoning_content`, and
    # vLLM's compat shim for that name applies to inbound requests only. Reading one
    # name would abort a perfectly good server, or bless a broken one.
    reasoning = ""
    reasoning_field = None
    for candidate in ("reasoning", "reasoning_content"):
        if message.get(candidate):
            reasoning = message[candidate]
            reasoning_field = candidate
            break
    evidence["attempts"].append({
        "mechanism": mechanism,
        "http_status": 200,
        "reasoning_field": reasoning_field,
        "reasoning_chars": len(reasoning),
        "final_chars": len(message.get("content") or ""),
        "finish_reason": payload["choices"][0].get("finish_reason"),
        "usage": payload.get("usage"),
    })
    if reasoning:
        evidence["reasoning_effort_mechanism"] = mechanism
        evidence["reasoning_field"] = reasoning_field
        evidence["ok"] = True
        break
    evidence["reasoning_content_missing"] = True

with open(os.environ["PROBE_OUT"], "w") as handle:
    json.dump(evidence, handle, indent=2)

if not evidence.get("ok"):
    print("FAILED")
    sys.exit(1)
print(f"{evidence['reasoning_effort_mechanism']}|{evidence['reasoning_field']}")
PYEOF
)"
PROBE_RC=$?
set -e
PROBE_MECHANISM="${PROBE_RESULT%%|*}"
PROBE_REASONING_FIELD="${PROBE_RESULT#*|}"

if [ "$PROBE_RC" -ne 0 ]; then
    echo "################################################################"
    echo "  PROBE FAILED -- do not run the gate against this server."
    echo "  No non-empty reasoning_content came back, which means either"
    echo "  --reasoning-parser openai_gptoss is not in effect, or the model"
    echo "  is not emitting a chain of thought. Every scored string would be"
    echo "  missing the CoT, so every number would be wrong."
    echo "  Evidence: $PROBE_OUT"
    echo "################################################################"
    if [ "$STRICT_PROBE" != "0" ]; then
        echo "Aborting (set STRICT_PROBE=0 to serve anyway)."
        exit 1
    fi
    PROBE_MECHANISM="unverified"
    PROBE_REASONING_FIELD="unverified"
fi

# ─── Publish the endpoint so the client can be pointed at it ─────────────────
ENDPOINT_JSON="$REPO_DIR/logs/vllm_endpoint_${SLURM_JOB_ID:-local}.json"
cat > "$ENDPOINT_JSON" <<EOF
{
  "job_id": "${SLURM_JOB_ID:-local}",
  "node": "${NODE_SHORT}",
  "node_fqdn": "${NODE_FQDN}",
  "ip": "${NODE_IP:-}",
  "port": ${PORT},
  "base_url": "${BASE_URL}",
  "served_model_name": "${SERVED_MODEL_NAME}",
  "partition": "${SLURM_JOB_PARTITION:-}",
  "tensor_parallel_size": ${TP},
  "compute_cap": "${COMPUTE_CAP}",
  "precision": "${PRECISION_NOTE}",
  "max_model_len": ${MAX_MODEL_LEN},
  "vllm_version": "${VLLM_VERSION}",
  "reasoning_parser": "openai_gptoss",
  "reasoning_effort_mechanism": "${PROBE_MECHANISM}",
  "reasoning_field": "${PROBE_REASONING_FIELD}",
  "probe_evidence": "${PROBE_OUT}",
  "weights_path": "${GPT_OSS_120B_PATH}",
  "gpu_memory_utilization": ${GPU_MEM_UTIL},
  "visible_gpus": ${VISIBLE_GPUS},
  "conda_env": "${CONDA_ENV}",
  "vllm_system_start_date": "${VLLM_SYSTEM_START_DATE}",
  "vllm_use_flashinfer_sampler": "${VLLM_USE_FLASHINFER_SAMPLER}",
  "cuda_home": "${CUDA_HOME:-}",
  "torch_cuda": "cu130"
}
EOF
cp "$ENDPOINT_JSON" "$REPO_DIR/logs/vllm_endpoint_latest.json"

echo "================================================================"
echo "  SERVER READY"
echo "  reasoning_effort transport accepted: ${PROBE_MECHANISM}"
echo "  reasoning channel arrived in field  : ${PROBE_REASONING_FIELD}"
echo "  endpoint written to: ${ENDPOINT_JSON}"
echo ""
echo "  Point the gate at it:"
echo "    python scripts/run_repro_gate.py --base-url ${BASE_URL} \\"
echo "        --model ${SERVED_MODEL_NAME} --smoke 20 --runs 1 --tasks tower-of-hanoi"
echo "    python scripts/run_repro_gate.py --base-url ${BASE_URL} --runs 3"
echo "================================================================"

# ─── optional in-job pipeline ────────────────────────────────────────────────
#
# Without this the server idles until the walltime or a reaper ends it, and every
# minute spent thinking at the login node is 4 idle A100s. PIPELINE runs the work
# inside the allocation and exits when it is done, so the GPUs are busy from
# allocation to teardown and there is no window to be cancelled in.
#
#   PIPELINE=smoke  sbatch scripts/serve_gpt_oss_vllm.sh   # ~40 calls, wiring only
#   PIPELINE=train  sbatch scripts/serve_gpt_oss_vllm.sh   # the real run: train + eval
#   PIPELINE=full   sbatch scripts/serve_gpt_oss_vllm.sh   # smoke -> gate -> pilot
#   PIPELINE=core_matched sbatch scripts/serve_gpt_oss_vllm.sh  # CORE's own runner,
#                                                    # our server, our 900-call budget
#   (unset)                                                 # serve and wait, as before
#
# PIPELINE_TASK selects the task (default zebralogic); PIPELINE_NTRAIN the regime.
# One of matchstick | mathgap | zebralogic | tower-of-hanoi -- run_pilot validates it
# against hexa_core.tasks, so a typo fails at startup rather than mid-run.
#   PIPELINE=train PIPELINE_TASK=tower-of-hanoi sbatch scripts/serve_gpt_oss_vllm.sh
#
# mathgap note: its eval split is `data[NTRAIN : NTRAIN+100]` out of a 280-row pool,
# so PIPELINE_NTRAIN changes *which* 100 items are evaluated (see RECON.md 1.2).
# Keep NTRAIN fixed across mathgap runs that are meant to be pooled, and cap it at
# 180. Its problems run long (some are ~700 words), so PIPELINE_MAXTOK=60000 as with
# the other 60k campaigns.
#
# Also honoured: PIPELINE_ROUNDS, PIPELINE_BUDGET, PIPELINE_MAXTOK, PIPELINE_RUNIDX,
# PIPELINE_OUT, PIPELINE_NBASE, PIPELINE_M1, the two 1-ablation knobs
# PIPELINE_SELECTION (top5|balanced|paired) and PIPELINE_MAXTRAJS, plus
# PIPELINE_MAXINJECT (how many skills are injected) and PIPELINE_TRUNCGUIDE=1
# (truncation-aware evolution) -- both WEAKNESSES.md 11 / 5, both off by default.
#
# `full` is a chain with a gate on each link: if the smoke run fails there is no
# point spending 900 gate calls, and if the gate fails the pilot numbers are
# uninterpretable anyway. Each step's exit status decides whether the next runs.
if [ -n "${PIPELINE:-}" ]; then
    export HEXA_TEACHER_BASE_URL="$BASE_URL"
    PIPE_LOG="$REPO_DIR/logs/pipeline_${SLURM_JOB_ID:-local}.log"
    echo ""
    echo "=== PIPELINE=${PIPELINE} starting $(date) -> ${PIPE_LOG} ==="

    run_step() {
        local label="$1"; shift
        echo ""
        echo "--- [${label}] $(date) ---"
        if "$@"; then
            echo "--- [${label}] OK ---"
            return 0
        fi
        echo "--- [${label}] FAILED (rc=$?) -- stopping the chain ---"
        return 1
    }

    PY_BIN="${CONDA_ENV}/bin/python"
    PIPE_RC=0
    PIPE_TASK="${PIPELINE_TASK:-zebralogic}"
    # 60 problems / 30 rounds, from WEAKNESSES.md 4: at 600 calls this is 30 rounds x 2
    # problems x k=10, against the 10 rounds x 20 x k=30 of the Hanoi and Matchstick
    # campaigns. Same budget, 3x the problem diversity, and k=10 still leaves every
    # problem enough attempts to yield a success/failure pair for `paired` selection.
    PIPE_NTRAIN="${PIPELINE_NTRAIN:-60}"
    PIPE_ROUNDS="${PIPELINE_ROUNDS:-30}"
    PIPE_CC="${PIPE_CONCURRENCY:-64}"
    PIPE_M1="${PIPELINE_M1:-1}"        # difficulty-weighted reward; 0 disables

    PILOT=("$PY_BIN" -u "$REPO_DIR/scripts/run_pilot.py"
           --base-url "$BASE_URL" --model "$SERVED_MODEL_NAME"
           --task "$PIPE_TASK" --concurrency "$PIPE_CC"
           --round-target "$PIPE_ROUNDS")
    [ "$PIPE_M1" = "1" ] && PILOT+=(--m1)
    [ -n "${PIPELINE_MAXTOK:-}" ]   && PILOT+=(--max-tokens "$PIPELINE_MAXTOK")
    [ -n "${PIPELINE_BUDGET:-}" ]   && PILOT+=(--budget-calls "$PIPELINE_BUDGET")
    [ -n "${PIPELINE_RUNIDX:-}" ]   && PILOT+=(--run-index "$PIPELINE_RUNIDX")
    [ -n "${PIPELINE_OUT:-}" ]      && PILOT+=(--output-dir "$PIPELINE_OUT")
    [ -n "${PIPELINE_NBASE:-}" ]    && PILOT+=(--n-base "$PIPELINE_NBASE")
    # Second eval window, in the same job. mathgap only: its held-out set is
    # data[n_train : n_train+100], so training at 60 scores data[60:160] while
    # CORE publishes the data[100:200] window. Additive -- the native eval still
    # runs -- so it costs 200 calls, not a second allocation.
    [ -n "${PIPELINE_EVALNTRAIN:-}" ] && PILOT+=(--eval-n-train "$PIPELINE_EVALNTRAIN")
    # The two 1-fixes, exposed so the ablation arms (top5 / 5 trajectories) can be run
    # from the same launcher rather than by editing defaults.
    [ -n "${PIPELINE_MAXTRAJS:-}" ] && PILOT+=(--max-trajs "$PIPELINE_MAXTRAJS")
    [ -n "${PIPELINE_SELECTION:-}" ] && PILOT+=(--contrast-selection "$PIPELINE_SELECTION")
    # WEAKNESSES.md 11 / 5: trims the injected block from its low-confidence tail. The
    # lever for truncation on tasks where the block adds per-candidate work.
    [ -n "${PIPELINE_MAXINJECT:-}" ] && PILOT+=(--max-injected-skills "$PIPELINE_MAXINJECT")
    # WEAKNESSES.md 11: truncation-aware evolution. OFF unless asked for -- with it off
    # the evolution prompt is byte-identical to the one every banked result used.
    [ "${PIPELINE_TRUNCGUIDE:-0}" = "1" ] && PILOT+=(--truncation-aware-evolution)

    # MATCHSTICK_DIAGNOSIS.md fixes 1 and 2, the two schedule levers. Both default to
    # the frozen streaming regime, so omitting them reproduces every banked result.
    #   PIPELINE_VISITS=2   each problem appears in 2 rounds instead of 1, paid for out
    #                       of attempts-per-problem (600 calls either way: 30x2x10
    #                       becomes 30x4x5). Lets a rule distilled from a failed round
    #                       be validated after the bank has changed.
    #   PIPELINE_SAMPLING=difficulty   apportions the surplus visits by measured
    #                       hardness (1 - b_q) instead of index order. Requires m1.
    #   PIPELINE_SAMPLEMIX  uniform share, CORE publishes 0.5; 0.0 is pure hardness.
    # NOTE visits must divide the budget: at 600/30 rounds, 2 is exact and 3 leaves 60
    # calls of slack, which run_pilot prints as `slack` -- a slack>0 arm is NOT
    # budget-matched against CORE's 600 and must not be reported as such.
    [ -n "${PIPELINE_VISITS:-}" ]    && PILOT+=(--visits-per-problem "$PIPELINE_VISITS")
    [ -n "${PIPELINE_SAMPLING:-}" ]  && PILOT+=(--problem-sampling "$PIPELINE_SAMPLING")
    [ -n "${PIPELINE_SAMPLEMIX:-}" ] && PILOT+=(--sampling-uniform-mix "$PIPELINE_SAMPLEMIX")

    echo "    task=${PIPE_TASK}  n_train=${PIPE_NTRAIN}  rounds=${PIPE_ROUNDS}"
    echo "    concurrency=${PIPE_CC}  m1=${PIPE_M1}"
    echo "    selection=${PIPELINE_SELECTION:-<default paired>}  max_trajs=${PIPELINE_MAXTRAJS:-<default 12>}"
    echo "    sampling=${PIPELINE_SAMPLING:-<default round_robin>}  visits=${PIPELINE_VISITS:-<default 1>}  mix=${PIPELINE_SAMPLEMIX:-<default 0.5>}"

    # 1. Smoke: ~40 calls, no eval. Gates everything else -- a broken injection join
    #    here would make the real run measure the no-memory baseline for 2000 calls.
    #    Skipped for PIPELINE=train, where the caller has already smoked this task.
    #    Also skipped for core_matched, which does not run our actor at all -- the
    #    smoke exercises run_pilot.py, so there it would only burn ~40 calls on an
    #    unrelated task before CORE's runner starts.
    if [ "${PIPELINE}" != "train" ] && [ "${PIPELINE}" != "eval_rerun" ] \
       && [ "${PIPELINE}" != "core_matched" ]; then
        run_step smoke "${PILOT[@]}" --smoke || PIPE_RC=1
    fi

    # 2. The real run: curriculum training, then held-out eval with the frozen bank
    #    AND the paired no-memory control on the same problems and the same seeds.
    if { [ "${PIPELINE}" = "train" ] || [ "${PIPELINE}" = "full" ]; } \
       && [ "$PIPE_RC" -eq 0 ]; then
        run_step train "${PILOT[@]}" --n-train "$PIPE_NTRAIN" || PIPE_RC=1
    fi

    # 3. eval_rerun: re-run ONLY the frozen paired eval against an already-trained
    #    bank, at a different max_tokens -- for checking a positive result against a
    #    larger token budget (WEAKNESSES.md 8) without redoing the training rounds.
    #    Required: RERUN_TASK, RERUN_NTRAIN, RERUN_BANK, RERUN_MAXTOK, RERUN_OUT.
    if [ "${PIPELINE}" = "eval_rerun" ]; then
        RERUN=("$PY_BIN" -u "$REPO_DIR/scripts/rerun_frozen_eval.py"
               --base-url "$BASE_URL" --model "$SERVED_MODEL_NAME"
               --task "$RERUN_TASK" --n-train "$RERUN_NTRAIN"
               --bank "$RERUN_BANK" --max-tokens "$RERUN_MAXTOK"
               --output-dir "$RERUN_OUT")
        [ -n "${RERUN_RUNIDX:-}" ] && RERUN+=(--run-index "$RERUN_RUNIDX")
        echo "    task=${RERUN_TASK}  bank=${RERUN_BANK}  max_tokens=${RERUN_MAXTOK}"
        run_step eval_rerun "${RERUN[@]}" || PIPE_RC=1
    fi

    # 4. core_matched: run CORE's OWN scripts/run_core.py against this same server,
    #    at OUR schedule. This is the only comparison that controls provider and
    #    rollout budget at once -- everything in CORE_COMPARISON.md 3 otherwise pits
    #    our 900-call runs against their published 2050-4000-call ones on a different
    #    provider, and the two confounds point in opposite directions.
    #
    #    The budget maps category-for-category. CORE's counter excludes baseline
    #    estimation (run_core.py:2487), so --max-rollouts 600 --num-baseline-train-runs 5
    #    is 600 training + 5x60 = 300 baseline = 900 total, which is exactly what our
    #    schedule spends (600 training + 300 for b_q).
    #
    #    The splits need no reconciling: hexa_core/tasks.py and run_core.py's init_task
    #    both call single_turn.get_task(name, num_train_items=N, seed=42), so at N=60
    #    CORE trains on the same 60 problems and evaluates on the same 100 as we do.
    #    --seed therefore stays 42; per-run variation goes through --llm-seed only.
    #
    #    Tokens default to 60000 rather than CORE's 32768: on Matchstick truncation is
    #    the dominant error mode, and a 32768 cap is exactly what produced our own
    #    bogus -0.250 (WEAKNESSES.md 11). Handing CORE a cap we know is too small
    #    would manufacture the result.
    if [ "${PIPELINE}" = "core_matched" ]; then
        CORE_PY="${CORE_PY:-/work/pi_sniekum_umass_edu/${USER}/envs/core-run/bin/python}"
        CORE_TASKS="${CORE_TASKS:-matchstick mathgap}"
        CORE_RUNS="${CORE_RUNS:-3}"
        # Which run indices this job covers. One job per (task, run) keeps a single
        # run well inside the 24 h wall limit and stops a timeout from taking the
        # runs after it down too; llm-seed is derived from the index, so the seeds
        # are the same whether a job does one run or three.
        CORE_RUNSTART="${CORE_RUNSTART:-0}"
        CORE_NTRAIN="${CORE_NTRAIN:-60}"
        CORE_ROLLOUTS="${CORE_ROLLOUTS:-600}"
        CORE_NBASE="${CORE_NBASE:-5}"
        CORE_TOKENS="${CORE_TOKENS:-60000}"
        # Only ever lowered for the wiring smoke; the real runs use CORE's 100.
        CORE_NEVAL="${CORE_NEVAL:-100}"
        # CORE ships with batching OFF, i.e. one API call at a time. The 62951655
        # smoke measured ~42 s/call sequential, which puts a 1150-call run near a
        # day and six of them near a week -- so this is a wall-clock decision, not
        # an algorithmic one, and it has to be made carefully in each phase:
        #
        #  * TRAIN stays at CORE's own default of 5. Bigger windows are tempting
        #    (5 -> 20 would be 4x faster) but memory is updated between windows,
        #    so a larger window means more rollouts issued against staler memory.
        #    That would slow CORE's learning and hand us a flattering result. We
        #    do not raise a knob that could depress the baseline.
        #  * EVAL goes to 25. Eval is frozen -- no bank writes, no utility
        #    updates, no registration -- so concurrency there cannot change what
        #    any single call sees. Purely a throughput win.
        #  * BASELINE goes to 30. Baseline estimation runs with no memory at all --
        #    it is the no-learning measurement, exactly what our b_q is, and we ran
        #    that at concurrency 64. Letting it inherit the train batch of 5 (the
        #    default when the baseline-specific flags are omitted) is what made job
        #    62956197 spend 6h47m on baseline alone and never reach a rollout.
        CORE_TRAIN_BATCH="${CORE_TRAIN_BATCH:-5}"
        CORE_EVAL_BATCH="${CORE_EVAL_BATCH:-25}"
        CORE_BASELINE_BATCH="${CORE_BASELINE_BATCH:-30}"
        # Two flags CORE's shell scripts set but run_core.py defaults differently.
        # Job 63064815 was cancelled at ~380/600 rollouts for missing both; see
        # results/core_matched_v1/matchstick/run0_ABANDONED_knnz10/WHY_ABANDONED.md.
        #
        #  * KNNZ. Every one of CORE's twelve published scripts sets
        #    knn-max-z = num-train-items (5_train->5, rollout_efficiency->10,
        #    100_train->100). Because knn_k = min(knn_max_z, |registered|), that
        #    convention makes the neighbourhood the WHOLE training set, so an
        #    insight's utility is a global average -- query-independent by
        #    construction. The argparse default of 10 against our n_train=60 would
        #    instead condition utility on the 10 nearest problems: a different
        #    retrieval rule, and not the one they report. Track n_train.
        #  * EVAL_PERIOD. The sample-efficiency scripts use 10000 to suppress
        #    intermediate evals and keep only the initial and final pass; the
        #    rollout-efficiency scripts use 250 because they need the Figure 2
        #    curve. We want final accuracy, so 10000. The default of 250 cost
        #    63064815 two unintended 100-problem eval passes (~1.4h). It does not
        #    touch accuracy -- eval is frozen -- but it does inflate wall-clock,
        #    which we now report.
        CORE_KNNZ="${CORE_KNNZ:-$CORE_NTRAIN}"
        CORE_EVAL_PERIOD="${CORE_EVAL_PERIOD:-10000}"
        CORE_OUTROOT="${CORE_OUTROOT:-$REPO_DIR/results/core_matched_v1}"
        # The cluster's default HF cache is read-only and has no jina copy; CORE's
        # embedder downloads to whatever this points at.
        export HF_HOME="${HF_HOME_OVERRIDE:-/work/pi_sniekum_umass_edu/${USER}/hf}"
        export HF_HUB_CACHE="${HF_HOME}/hub"
        unset TRANSFORMERS_CACHE HF_HUB_ENABLE_HF_TRANSFER

        if [ ! -x "$CORE_PY" ]; then
            echo "ERROR: CORE_PY=${CORE_PY} is not executable. CORE needs its own env"
            echo "       (sentence-transformers + transformers<5 for the jina embedder);"
            echo "       do not install those into the vLLM env."
            PIPE_RC=1
        fi

        echo "    tasks=${CORE_TASKS}  runs=${CORE_RUNS}  n_train=${CORE_NTRAIN}"
        echo "    rollouts=${CORE_ROLLOUTS} + ${CORE_NBASE}x${CORE_NTRAIN} baseline"
        echo "    tokens=${CORE_TOKENS}  knn_max_z=${CORE_KNNZ}  eval_period=${CORE_EVAL_PERIOD}"
        echo "    batch train/eval/baseline=${CORE_TRAIN_BATCH}/${CORE_EVAL_BATCH}/${CORE_BASELINE_BATCH}"
        echo "    out=${CORE_OUTROOT}"

        for ctask in ${CORE_TASKS}; do
            for cidx in $(seq "$CORE_RUNSTART" $((CORE_RUNSTART + CORE_RUNS - 1))); do
                [ "$PIPE_RC" -eq 0 ] || break 2
                cout="${CORE_OUTROOT}/${ctask}/run${cidx}"
                run_step "core:${ctask}:run${cidx}" \
                    "$CORE_PY" -u "$REPO_DIR/scripts/run_core_local.py" \
                    --base-url "$BASE_URL" --copy-to "$cout" -- \
                    --task "$ctask" \
                    --nvidia-model "$SERVED_MODEL_NAME" \
                    --num-train-items "$CORE_NTRAIN" \
                    --num-eval-items "$CORE_NEVAL" \
                    --max-rollouts "$CORE_ROLLOUTS" \
                    --num-baseline-train-runs "$CORE_NBASE" \
                    --solve-tokens "$CORE_TOKENS" \
                    --reflection-tokens "$CORE_TOKENS" \
                    --eval-tokens "$CORE_TOKENS" \
                    --use-batching-in-train true \
                    --num-train-batch-calls "$CORE_TRAIN_BATCH" \
                    --use-batching-in-baseline-train true \
                    --num-baseline-train-batch-calls "$CORE_BASELINE_BATCH" \
                    --use-batching-in-eval true \
                    --num-eval-batch-calls "$CORE_EVAL_BATCH" \
                    --knn-max-z "$CORE_KNNZ" \
                    --eval-period "$CORE_EVAL_PERIOD" \
                    --seed 42 --llm-seed $((42 + cidx)) \
                    || PIPE_RC=1
            done
        done
    fi

    echo ""
    echo "=== PIPELINE finished $(date) rc=${PIPE_RC} -- shutting the server down ==="
    exit "$PIPE_RC"
fi

wait "$SERVER_PID"
