#!/bin/bash
# ─── HeXA with the Qwen2.5-3B configuration ──────────────────────────────────
#
# HeXA uses the actor model as its own teacher, so a 3B run asks one small model
# to both solve episodes and maintain the skill bank. The loop defaults were
# tuned against larger teachers; three settings change for a 3B actor. The
# METHOD is unchanged -- same loop, same retriever, same update rule, same
# frozen-bank evaluation -- this is configuration only.
#
# 1. SLIM EVOLUTION PROMPT (automatic, no flag).
#    The default SKILL_BANK_EVOLUTION_PROMPT is ~8.2k characters and is resent
#    every round from round 2 on, on top of the bank and the trajectory block. A
#    3B teacher loses the task in that much preamble: it returns flat
#    confidences, stops merging, and emits near-duplicate skills until the cap.
#    skillrl/distillation/teacher_prompts_qwen_slim.py keeps every evolution
#    rule -- complete-bank output, hard caps, is_new/source_seeds, confidence
#    moving in both directions, numeric constants copied through, duplicate and
#    dead-skill removal, contrastive mistakes, self-consistency -- and cuts the
#    worked examples, the per-rule rationales and the optional audit fields that
#    evolving_distill already reads with a default. Placeholders are identical,
#    so it is a drop-in rebind. evolving_loop applies it on its own when the
#    teacher model name looks like a 3B Qwen, so NO OTHER MODEL IS AFFECTED: 7B,
#    Qwen3-8B, gpt-oss and Sonnet all keep the full template. Nothing below
#    switches it on; --teacher-model is what selects it.
#
# 2. BANK CAPACITY 5 SKILLS / 3 MISTAKES (default 10/5).
#    At the default the 3B teacher filled the bank with near-paraphrases that
#    all carried the same uninformative confidence, which leaves the retriever
#    choosing among them arbitrarily. A tighter cap forces the evolver to merge
#    or drop rather than accumulate. MAXSK/MAXMI also drive what the frozen eval
#    injects, so capacity and injection stay consistent -- asking for 6 skills
#    out of a 5-skill bank just injects the whole bank.
#
# 3. SIX SEEDS PER ROUND (default 3).
#    Evolution is contrastive: it needs a success AND a failure from the same
#    round to say what the success did differently. At a small actor's solve
#    rate, 3 seeds/round frequently produced a round with no success at all, so
#    the evolver had nothing to contrast and kept rewriting the same few ideas.
#    Wider rounds, same number of rounds.
#
# The actor prompt, tool set, retriever, update strategy and evaluation protocol
# (frozen bank, --no-evolver, held-out seeds) are exactly those used for every
# other model.
#
# Usage -- train, then evaluate the frozen final bank on held-out seeds:
#
#   LEVEL=two_body_problem INIT_TRAJ=Initial_trajectories/two_body_problem \
#   OUT=runs/tbp_qwen3b ./scripts/run_hexa_qwen3b.sh
#
#   ROUNDS=9 SEED_LO=60 SEED_HI=110 ... ./scripts/run_hexa_qwen3b.sh
#   TRAIN_ONLY=1 ... ./scripts/run_hexa_qwen3b.sh      # skip the eval stage
#
# Point OPENAI_BASE_URL at a vLLM server for the model (scripts/serve_qwen_vllm.sh),
# or override MODEL/TEACHER_MODEL for a different backend.
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

PYTHON="${PYTHON:-python}"
LEVEL="${LEVEL:?set LEVEL=<level name, e.g. two_body_problem>}"
INIT_TRAJ="${INIT_TRAJ:?set INIT_TRAJ=<dir of warm-start trajectories>}"
OUT="${OUT:?set OUT=<output dir>}"

MODEL="${MODEL:-openai:Qwen/Qwen2.5-3B-Instruct}"
# Teacher = actor, as everywhere in HeXA. This string is also what selects the
# slim evolution prompt, so keep the "3B" in it when pointing at a local server.
TEACHER_MODEL="${TEACHER_MODEL:-$MODEL}"

ROUNDS="${ROUNDS:-17}"
SEEDS_PER_ROUND="${SEEDS_PER_ROUND:-6}"   # (3) above
START_SEED="${START_SEED:-6}"
MAXSK="${MAXSK:-5}"; MAXMI="${MAXMI:-3}"  # (2) above

# Held-out eval window. Default starts one past the last training seed so the
# two cannot overlap; override for a window further out.
LAST_TRAIN_SEED=$(( START_SEED + ROUNDS * SEEDS_PER_ROUND - 1 ))
SEED_LO="${SEED_LO:-$(( LAST_TRAIN_SEED + 1 ))}"
SEED_HI="${SEED_HI:-$(( SEED_LO + 49 ))}"
EVAL_SEEDS="${EVAL_SEEDS:-$(seq "$SEED_LO" "$SEED_HI" | tr '\n' ' ')}"

[ -d "$INIT_TRAJ" ] || { echo "ERROR: warm-start dir missing: $INIT_TRAJ"; exit 1; }
mkdir -p "$OUT"

echo "=== HeXA (Qwen2.5-3B config) level=$LEVEL ==="
echo "    actor = teacher = $MODEL"
echo "    train  ${ROUNDS} rounds x ${SEEDS_PER_ROUND} seeds from ${START_SEED} (through ${LAST_TRAIN_SEED})"
echo "    bank   ${MAXSK} skills / ${MAXMI} mistakes"
echo "    eval   seeds ${SEED_LO}-${SEED_HI}, frozen bank, --no-evolver"

"$PYTHON" scripts/run_hexa.py \
  --level "$LEVEL" \
  --initial-traj-dir "$INIT_TRAJ" \
  --output-dir "$OUT" \
  --num-rounds "$ROUNDS" \
  --seeds-per-round "$SEEDS_PER_ROUND" \
  --start-seed "$START_SEED" \
  --model "$MODEL" \
  --teacher-model "$TEACHER_MODEL" \
  --max-skills "$MAXSK" \
  --max-mistakes "$MAXMI"

[ "${TRAIN_ONLY:-0}" = "1" ] && { echo "=== TRAIN_ONLY=1: skipping eval ==="; exit 0; }

FINAL="$OUT/skill_bank_evolving_${ROUNDS}.json"
[ -f "$FINAL" ] || { echo "ERROR: no final bank at $FINAL"; exit 1; }

echo "=== frozen eval of $(basename "$FINAL") on seeds ${SEED_LO}-${SEED_HI} ==="
"$PYTHON" -m skillrl.runner.run_skill_agent \
  --level "$LEVEL" --seeds $EVAL_SEEDS \
  --model "$MODEL" --skill-bank "$FINAL" \
  --eval-dir "$OUT/eval_seeds${SEED_LO}-${SEED_HI}" \
  --max-iterations 25 \
  --max-general-skills 0 --max-specific-skills "$MAXSK" --max-mistakes "$MAXMI" \
  --no-evolver
