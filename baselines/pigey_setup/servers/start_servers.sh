# Source this inside an sbatch script (after activate.sh, with TMPDIR set) to start
# the perception servers and the pi0.5 policy server once per job. Exports
# POLICY_PORT, PIGEY_ER_URL and PIGEY_SAM_URL, writes $TMPDIR/{policy,er,sam}-server.pid,
# and installs an EXIT trap that stops the servers. Exits the job on a failed start.
# PIGEY_POLICY_ONLY=1 starts only the pi0.5 policy (raw-VLA runs: no Claude, no perception).
: "${SLURM_JOB_ID:?start_servers.sh runs inside a Slurm job}"
: "${SETUP:?SETUP must point at baselines/pigey_setup}"
: "${PIGEY_ROOT:?source activate.sh first}"
: "${TMPDIR:?TMPDIR must be set for this job}"
CHECKPOINT="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero"

cleanup() {
    for name in policy er sam; do
        [[ -f "$TMPDIR/$name-server.pid" ]] && kill "$(cat "$TMPDIR/$name-server.pid")" 2> /dev/null || true
    done
}
trap cleanup EXIT

if [[ -z "${PIGEY_POLICY_ONLY:-}" ]]; then
    # Claude Code needs outbound HTTPS from this node.
    curl -s -o /dev/null -w 'api.anthropic.com HTTPS status: %{http_code}\n' --max-time 15 https://api.anthropic.com \
        || { echo "No outbound HTTPS from $(hostname); Claude Code cannot run here" >&2; exit 1; }

    # GPU memory split on one A100-80GB: vLLM 35%, JAX 35%, rest for SAM and EGL.
    # Perception starts first because vLLM checks free memory at startup.
    PIGEY_ER_GPU_UTIL="${PIGEY_ER_GPU_UTIL:-0.35}" bash "$SETUP/servers/serve_perception.sh"
    source "$TMPDIR/perception.env"
fi
# GPU nodes are shared, so pick a free port (after the perception servers have
# bound theirs) unless PIGEY_POLICY_PORT is set. The policy binds 0.0.0.0.
POLICY_PORT="${PIGEY_POLICY_PORT:-$(python "$SETUP/runner/free_ports.py" 1)}"
export POLICY_PORT
printf 'Policy port %s; ER %s; SAM %s\n' "$POLICY_PORT" "${PIGEY_ER_URL:-none}" "${PIGEY_SAM_URL:-none}"
(
    cd "$PIGEY_ROOT/openpi" || exit 1
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${PIGEY_POLICY_GPU_FRACTION:-0.35}"
    # --port must precede the policy:checkpoint subcommand (Pigey sim README).
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
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
