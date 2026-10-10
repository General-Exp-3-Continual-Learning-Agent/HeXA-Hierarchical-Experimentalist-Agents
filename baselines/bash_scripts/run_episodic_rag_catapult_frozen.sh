#!/bin/bash
# Episodic RAG, FROZEN transfer eval on held-out seeds 57-106.
#
# Memory is seeded from episodic_rag_catapult_claude (built by streaming seeds
# 6-56; 51 records, 31 success / 20 failure) and copied into this run's output
# dir, where MemoryStore.__init__ reloads it (store.py:64).
#
# --freeze-after 0 freezes from stream index 0, i.e. every eval seed. Retrieval
# still runs (stream.py:164, before the frozen check); only the write-back at
# stream.py:197 is skipped. So each of 57-106 sees the SAME fixed 6-56 memory
# and contributes nothing back -- matching HeXA's frozen-bank eval protocol.
#
# Resume-safe: seeds with an existing trajectory file are skipped.
set -u
REPO="/home/udhanuka_umass_edu/Scalar_Research/HeXA"
PY="$HOME/.conda/envs/phyreagent/bin/python"
OUT="$REPO/baselines/data/results/episodic_rag_catapult_claude_frozen57-106"
LOG="${LOG:-$REPO/baselines/data/results/episodic_rag_frozen57-106.log}"
STALL_SECS="${STALL_SECS:-1200}"
cd "$REPO"
capped() { grep -q "Claude Error" "$1" 2>/dev/null; }
stalled() { local a=$(( $(date +%s) - $(stat -c %Y "$1" 2>/dev/null || date +%s) )); [ "$a" -ge "$STALL_SECS" ]; }

echo "[$(date -u)] === episodic_rag catapult FROZEN eval, seeds 57-106 ==="
"$PY" -m baselines.runner.run_episodic_rag \
    --level catapult \
    --seeds 57-106 \
    --freeze-after 0 \
    --output-dir "$OUT" \
    --model claude \
    --verbose > "$LOG" 2>&1 &
PID=$!
while kill -0 $PID 2>/dev/null; do
  if capped "$LOG"; then
    echo "[$(date -u)] CAP DETECTED - killing frozen eval"
    pkill -P $PID claude 2>/dev/null; kill $PID 2>/dev/null; wait $PID 2>/dev/null; break
  fi
  if stalled "$LOG"; then
    echo "[$(date -u)] STALL DETECTED (${STALL_SECS}s) - killing frozen eval"
    pkill -P $PID claude 2>/dev/null; kill $PID 2>/dev/null; wait $PID 2>/dev/null
    echo "[$(date -u)] ABORT: stalled"; exit 3
  fi
  sleep 20
done
wait $PID 2>/dev/null
if capped "$LOG"; then echo "[$(date -u)] ABORT: cap hit."; exit 2; fi
echo "[$(date -u)] FROZEN EVAL COMPLETE"
