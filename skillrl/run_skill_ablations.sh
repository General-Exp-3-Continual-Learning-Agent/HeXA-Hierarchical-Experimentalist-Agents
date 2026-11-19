#!/usr/bin/env bash
# Group A - skill-bank ablations (reviewer r3fT Q3)
#
# All conditions use run4's frozen bank 17 on catapult seeds 57-106, the same
# 50 held-out seeds where full HeXA scores 92.0% (46/50, 11.5 avg iters).
# That row is the reference; nothing here retrains anything.
#
#   A1  strategies only      unmodified bank, mistakes suppressed at retrieval
#   A2  mistakes only        unmodified bank, skills suppressed at retrieval
#   A3  top-conf removed     bank minus the 3 highest-confidence skills
#   A4  shuffled conf        confidence values permuted across the 10 skills
#   A5  irrelevant skills    down_to_earth skills, original catapult mistakes
#
# Runs sequentially, one condition at a time. Resume-safe: run_batch skips any
# seed that already has a trajectory file, so re-invoking after a cap resumes.
#
#   RUN_LIST="A1 A2 A3" ./skillrl/run_skill_ablations.sh
set -u

cd "$(dirname "$0")/.." || exit 1

PY="${PY:-python}"

RUN_LIST="${RUN_LIST-A1 A2 A3}"
SEEDS="$(seq 57 106 | tr '\n' ' ')"
LEVEL=catapult
MODEL=claude

ROOT=skillrl/RESULTS/catapult_claude/skill_ablations
BANKS=$ROOT/banks
REF_BANK=skillrl/RESULTS/catapult_claude/HeXA_catapult/HeXA_catapult_claude_run4/skill_bank_evolving_17.json

STALL_SECS="${STALL_SECS:-1200}"
RETRY_WAIT="${RETRY_WAIT:-900}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-40}"

# condition -> "bank|general|specific|mistakes"
spec_for() {
  case "$1" in
    A1) echo "$REF_BANK|0|6|0" ;;
    A2) echo "$REF_BANK|0|0|4" ;;
    A3) echo "$BANKS/bank_A3_drop_top3conf.json|0|6|4" ;;
    A4) echo "$BANKS/bank_A4_shuffled_conf.json|0|6|4" ;;
    A5) echo "$BANKS/bank_A5_irrelevant.json|0|6|4" ;;
    *)  echo "" ;;
  esac
}

capped()  { grep -qi "usage limit\|Claude Error\|rate.limit" "$1" 2>/dev/null; }
n_done()  { ls "$1"/*/trajectory_seed*_skillrl.json 2>/dev/null | wc -l; }
stalled() {
  local age
  age=$(( $(date +%s) - $(stat -c %Y "$1" 2>/dev/null || date +%s) ))
  [ "$age" -ge "$STALL_SECS" ]
}

for COND in $RUN_LIST; do
  SPEC="$(spec_for "$COND")"
  if [ -z "$SPEC" ]; then echo "[$COND] unknown condition, skipping"; continue; fi
  IFS='|' read -r BANK GEN SPEC_N MIS <<<"$SPEC"

  if [ ! -f "$BANK" ]; then echo "[$COND] bank missing: $BANK"; continue; fi

  OUT=$ROOT/$COND
  LOG=$ROOT/${COND}.log
  mkdir -p "$OUT"

  echo "=============================================================="
  echo "[$COND] bank=$(basename "$BANK")  general=$GEN specific=$SPEC_N mistakes=$MIS"
  echo "[$COND] out=$OUT"
  echo "=============================================================="

  attempt=0
  while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
    attempt=$(( attempt + 1 ))
    have=$(n_done "$OUT")
    if [ "$have" -ge 50 ]; then
      echo "[$COND] complete: $have/50"
      break
    fi
    echo "[$COND] attempt $attempt, have $have/50 -> launching"

    "$PY" -m skillrl.runner.run_skill_agent \
      --level "$LEVEL" \
      --seeds $SEEDS \
      --model "$MODEL" \
      --skill-bank "$BANK" \
      --eval-dir "$OUT" \
      --max-general-skills "$GEN" \
      --max-specific-skills "$SPEC_N" \
      --max-mistakes "$MIS" \
      >> "$LOG" 2>&1 &
    pid=$!

    # watch for cap / stall while it runs
    while kill -0 "$pid" 2>/dev/null; do
      sleep 60
      if capped "$LOG"; then
        echo "[$COND] usage cap detected -> killing, waiting ${RETRY_WAIT}s"
        kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
        sleep "$RETRY_WAIT"
        continue 2
      fi
      if [ -f "$LOG" ] && stalled "$LOG"; then
        echo "[$COND] no log activity for ${STALL_SECS}s -> killing, will retry"
        kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
        break
      fi
    done
    wait "$pid" 2>/dev/null
  done

  final=$(n_done "$OUT")
  echo "[$COND] finished with $final/50 trajectories"
done

echo
echo "=== Group A summary ==="
for COND in $RUN_LIST; do
  OUT=$ROOT/$COND
  python3 - "$OUT" "$COND" <<'PY'
import json,glob,sys,os
out,cond=sys.argv[1],sys.argv[2]
fs=glob.glob(os.path.join(out,'**','trajectory_seed*_skillrl.json'),recursive=True)
ok=0; it=[]
for f in fs:
    try: d=json.load(open(f))
    except Exception: continue
    ok+=bool(d.get('success'))
    v=d.get('iterations') or d.get('num_iterations')
    if isinstance(v,(int,float)): it.append(v)
n=len(fs)
if n:
    print(f"{cond}: {ok}/{n} = {100*ok/n:.1f}%   avg_iter={sum(it)/len(it):.1f}" if it
          else f"{cond}: {ok}/{n} = {100*ok/n:.1f}%")
else:
    print(f"{cond}: no trajectories")
PY
done
echo "reference (full HeXA, run4 bank 17, same 50 seeds): 46/50 = 92.0%  avg_iter=11.5"
