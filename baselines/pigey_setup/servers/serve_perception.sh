#!/bin/bash
# Start Pigey's local perception servers in the background inside a GPU
# allocation, and return once both answer health checks:
#   - ER detection VLM (vLLM, OpenAI-compatible API)
#   - SAM, running Pigey's own _sam_get_mask on CUDA (sam_server.py)
# Ports are chosen free at startup (GPU nodes are shared) unless
# PIGEY_ER_PORT / PIGEY_SAM_PORT are set. PIDs are written to
# $PIGEY_ROOT/tmp/$SLURM_JOB_ID/{er,sam}-server.pid and the server URLs to
# $PIGEY_ROOT/tmp/$SLURM_JOB_ID/perception.env; source it before the runner.
set -euo pipefail
: "${SLURM_JOB_ID:?Start the perception servers inside a Slurm GPU allocation}"
SETUP="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"  # baselines/pigey_setup
PIGEY_ROOT="${PIGEY_ROOT:-/scratch/workspace/vgandhi_umass_edu-roboprobe-sim/pigey}"
PIGEY_REPO="${PIGEY_REPO:-$(dirname "$SETUP")/Pigey}"
ER_MODEL="${PIGEY_ER_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
# Leave room for the pi0.5 server when it shares the GPU.
ER_GPU_UTIL="${PIGEY_ER_GPU_UTIL:-0.4}"
RUN_DIR="$PIGEY_ROOT/tmp/$SLURM_JOB_ID"
VENV="$PIGEY_ROOT/vllm/.venv"
read -r FREE_ER FREE_SAM < <("$VENV/bin/python" "$SETUP/runner/free_ports.py" 2)
ER_PORT="${PIGEY_ER_PORT:-$FREE_ER}"
SAM_PORT="${PIGEY_SAM_PORT:-$FREE_SAM}"
mkdir -p "$RUN_DIR" "$PIGEY_ROOT/logs"
export HF_HOME="$PIGEY_ROOT/cache/huggingface"
export XDG_CACHE_HOME="$PIGEY_ROOT/cache/xdg"
export VLLM_CACHE_ROOT="$PIGEY_ROOT/cache/vllm"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

"$VENV/bin/vllm" serve "$ER_MODEL" \
    --host 127.0.0.1 --port "$ER_PORT" \
    --gpu-memory-utilization "$ER_GPU_UTIL" --max-model-len 8192 \
    --limit-mm-per-prompt '{"image": 1, "video": 0}' --seed 0 \
    > "$PIGEY_ROOT/logs/er-server-$SLURM_JOB_ID.log" 2>&1 &
echo $! > "$RUN_DIR/er-server.pid"
"$VENV/bin/python" "$SETUP/runner/sam_server.py" --port "$SAM_PORT" --pigey-repo "$PIGEY_REPO" \
    > "$PIGEY_ROOT/logs/sam-server-$SLURM_JOB_ID.log" 2>&1 &
echo $! > "$RUN_DIR/sam-server.pid"
printf 'Starting ER server (%s, port %s) and SAM server (port %s); logs in %s\n' \
    "$ER_MODEL" "$ER_PORT" "$SAM_PORT" "$PIGEY_ROOT/logs"

wait_for() {
    local name=$1 url=$2 pid
    pid=$(cat "$RUN_DIR/$name-server.pid")
    for _ in $(seq 1 120); do
        if curl -sf "$url" > /dev/null; then
            printf '%s server ready: %s\n' "$name" "$url"
            return 0
        fi
        if ! kill -0 "$pid" 2> /dev/null; then
            tail -50 "$PIGEY_ROOT/logs/$name-server-$SLURM_JOB_ID.log" >&2
            echo "$name server exited during startup" >&2
            return 1
        fi
        sleep 5
    done
    tail -50 "$PIGEY_ROOT/logs/$name-server-$SLURM_JOB_ID.log" >&2
    echo "$name server not ready after 10 minutes" >&2
    return 1
}

if ! wait_for er "http://127.0.0.1:$ER_PORT/health" || ! wait_for sam "http://127.0.0.1:$SAM_PORT/health"; then
    kill "$(cat "$RUN_DIR/er-server.pid")" "$(cat "$RUN_DIR/sam-server.pid")" 2> /dev/null || true
    exit 1
fi
printf 'export PIGEY_ER_URL=http://127.0.0.1:%s/v1\nexport PIGEY_SAM_URL=http://127.0.0.1:%s\n' \
    "$ER_PORT" "$SAM_PORT" > "$RUN_DIR/perception.env"
