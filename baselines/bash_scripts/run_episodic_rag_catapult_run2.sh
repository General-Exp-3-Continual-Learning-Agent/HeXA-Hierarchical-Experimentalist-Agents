#!/bin/bash
# Episodic RAG on catapult, run2: identical to episodic_rag_catapult_claude
# except memory is warm-started from the offline seeds 1-5 (--initial-traj-dir),
# matching what HeXA's evolving loop gets. run1 had initial_traj_dir: null.
#
# Resume-safe: stream.py skips seeds whose trajectory file already exists.
# NOTE: episodes.jsonl is append-only, so after a resume it may contain more
# lines than seeds; the per-seed trajectory files are the source of truth.
set -u
REPO="/home/udhanuka_umass_edu/Scalar_Research/HeXA"
PY="$HOME/.conda/envs/phyreagent/bin/python"
OUT="$REPO/baselines/data/results/episodic_rag_catapult_claude_run2"
LOG="${LOG:-$REPO/baselines/data/results/episodic_rag_run2.log}"
STALL_SECS="${STALL_SECS:-1200}"
cd "$REPO"

capped() { grep -q "Claude Error" "$1" 2>/dev/null; }
stalled() {
  local age=$(( $(date +%s) - $(stat -c %Y "$1" 2>/dev/null || date +%s) ))
  [ "$age" -ge "$STALL_SECS" ]
}

echo "[$(date -u)] === episodic_rag catapult run2 (with initial trajectories) ==="
"$PY" -m baselines.runner.run_episodic_rag \
    --level catapult \
    --seeds 6-57 \
    --initial-traj-dir Initial_trajectories/catapult \
    --output-dir "$OUT" \
    --model claude \
    --verbose > "$LOG" 2>&1 &
PID=$!
while kill -0 $PID 2>/dev/null; do
  if capped "$LOG"; then
    echo "[$(date -u)] CAP DETECTED - killing run2"
    pkill -P $PID claude 2>/dev/null; kill $PID 2>/dev/null; wait $PID 2>/dev/null; break
  fi
  if stalled "$LOG"; then
    echo "[$(date -u)] STALL DETECTED (${STALL_SECS}s) - killing run2"
    pkill -P $PID claude 2>/dev/null; kill $PID 2>/dev/null; wait $PID 2>/dev/null
    echo "[$(date -u)] ABORT: stalled"; exit 3
  fi
  sleep 20
done
wait $PID 2>/dev/null
if capped "$LOG"; then echo "[$(date -u)] ABORT: cap hit."; exit 2; fi
echo "[$(date -u)] RUN2 COMPLETE"
