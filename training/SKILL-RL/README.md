# SKILL-RL — Evolving Skill Bank Pipeline

This directory contains the skill bank distillation and evolution pipeline for the Interphyre GRPO training project. It is used to build and iteratively refine physics skill banks from agent trajectories, which are then injected into the LLM's system prompt during training.

---

## Three Training Variants

| Variant | Skill bank | Training data | Scripts |
|---------|-----------|---------------|---------|
| **Non-skilled** | None | Fixed 50-seed parquet, all epochs | `slurm_train_3b_a100_1gpu_dte_bsz1.sh` |
| **Skilled (static)** | Fixed at data-gen time | Fixed 50-seed parquet with skills baked in | `slurm_train_3b_a100_1gpu_dte_bsz1_skilled.sh` |
| **Evolving-skilled** | Grows each round via 7B teacher | 3 new seeds per round, skill bank evolves | `slurm_evolving_grpo_dte.sh` |

All variants train the same **Qwen 2.5 3B Instruct** model via GRPO (batch_size=1, n=4 rollouts, lr=1e-6).

---

## Evolving Skill Bank Setup (continuous-process, 2026-05-04 → 2026-05-05)

### Overview

**One verl process for the whole run.** The skill bank lives in `skill_bank_live.json` on disk; the dataset reads it at every `__getitem__` and injects the level's skill block into the system prompt; a synchronous trainer hook fires every K steps to call the teacher and atomically rewrite the bank file. No bash round loop, no Adam reset, no Ray/FSDP/vLLM-rollout re-init.

The teacher's input pipeline went through two redesigns within 24 hours after several bugs were found in inspection — see "Hook input pipeline (Option B)" below for the current design and "Engineering issues found during run43–run47" for what was wrong with the older designs.

```
Phase 0 (once, before verl starts):
  5 base-model trajectories (FirstFiveTrajs/)
      → Qwen 7B teacher (vLLM, OpenAI-compatible HTTP)
      → skill_bank_round_0.json
  Copied to → skill_bank_live.json
  Snapshotted to logs/run{N}/skill_evolution/step_0/{level}/bank_out.json

verl process (single, continuous, ~500 steps):
  Each gradient step:
    rl_dataset.__getitem__ reads skill_bank_live.json fresh,
        appends "## Learned Physics Skills" block to system message,
        retokenizes prompt
    GRPO step (n=4 rollouts at temp=1.0)
    reward_manager writes:
      - trajectories.jsonl (compact inspection log, last 2000 chars)
      - $FULL_TRAJ_DIR/step_{S}/{phase}/seed{N}_w{w}.json  (full ReAct text — canonical hook input)
  Every SKILL_EVOLVE_FREQ steps (default 3), trainer.fit() calls run_skill_evolution():
    scan $FULL_TRAJ_DIR/step_*/train/ for new files (vs _processed_files set)
    group by seed, pick highest-scoring rollout per seed (1 trajectory per seed)
    parse ReAct → write trajectory_seed{N}_step{S}.json (matches teacher's glob)
    evolve_skill_bank(prev_bank, new_trajs) → teacher HTTP call
    atomic os.replace(tmp, skill_bank_live.json)
    snapshot bank to step_{S}/{level}/bank_out.json
  Continue training. Optimizer state, FSDP shards, vLLM rollout engine all stay warm.
```

**Key parameters (all overridable via env var):**

| Variable | Default | Meaning |
|----------|---------|---------|
| `SKILL_EVOLVE_FREQ` | 3 | Training steps between teacher calls |
| `TRAIN_SEED_START` | 1 | First training seed |
| `NUM_TRAIN_SEEDS` | 50 | Number of training seeds |
| `VAL_SEED_START` | 51 | First val seed |
| `NUM_VAL_SEEDS` | 50 | Number of val seeds |
| `MAX_SKILLS` | 10 | Max skills per level in the bank |
| `MAX_MISTAKES` | 5 | Max mistake patterns per level |

Phase 0 uses seeds 1-5 (from `FirstFiveTrajs/`). GRPO training uses seeds 1-50 (these are the dataset seeds; Phase 0 just bootstraps the bank, training reuses them). Val: seeds 51-100.

### Log / Output Structure

```
logs/run{N}/
├── trajectories.jsonl                                  (compact inspection log; last-2000-chars snippet)
├── metrics.csv                                         (live training metrics, parse_metrics.py)
├── full_trajectories/                                  (canonical hook input — full ReAct text per rollout)
│   └── step_{S}/{train|val}/seed{N}_w{w}.json
│       (planned: seed{N}_pid{P}_w{w}.json once PID fix lands)
├── skill_evolution/
│   ├── step_0/{level}/bank_out.json                    (Phase-0 starting bank)
│   └── step_{K}/{level}/                               (per fire — every K=3 steps)
│       ├── trajectory_seed{N}_step{S}.json             (one per seed; matches "trajectory_seed*.json" glob)
│       └── bank_out.json                               (post-evolve bank snapshot)
└── tool_server.log
```

The live bank lives outside the run dir at `SKILL-RL/Skill_banks/evolving_grpo_dte/skill_bank_live.json` (and `skill_bank_round_0.json` from Phase 0). It's mutated in place via atomic `os.replace`; per-step `bank_out.json` snapshots in `skill_evolution/` give a full history.

### Components

#### Teacher model — Qwen 2.5 7B Instruct (vLLM server)
- Runs as a separate SLURM job on a dedicated GPU. Stays idle between hook calls.
- Served via vLLM OpenAI-compatible API on port 8000
- Called only by the trainer hook (synchronously, ~10–60 s per call)
- Script: `examples/train/interphyre/slurm_vllm_7b_teacher.sh`
- Model path: `/scratch4/workspace/svaidyanatha_umass_edu-phyre/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28`
- Writes its endpoint to: `logs/vllm_teacher_endpoint.txt`

#### Orchestrator
- Single SLURM job. Runs Phase 0 distillation, generates the parquet (with `--runtime_skill_bank_path`, no skills baked in), starts the tool server, then launches **one** verl invocation that runs all 500 steps.
- Script: `examples/train/interphyre/slurm_evolving_grpo_dte.sh` (DTE)
- Script: `examples/train/interphyre/slurm_evolving_grpo_tbp.sh` (TBP)
- Passes hook config via `+trainer.skill_evolve_freq`, `+trainer.skill_bank_path`, `+trainer.skill_evolve_teacher_endpoint`, etc.

#### Skill-evolution hook (`verl_tool/hooks/skill_evolution.py`) — rewritten 2026-05-05
- Inserted into `verl/trainer/ppo/ray_trainer.py:fit()` after the checkpoint-save block.
- Module-level `_processed_files: set` tracks which files in `$FULL_TRAJ_DIR` are already consumed; each fire processes only new files.
- `_select_new_train_files()` globs `step_*/train/*.json` (val files in `step_*/val/` are ignored — written for inspection only).
- `_stage_one_per_seed()` groups by seed and picks the **highest-scoring rollout per seed**, then parses ReAct via `_parse_react_steps` and writes `trajectory_seed{N}_step{S}.json` (matches `distill.py:167`'s `trajectory_seed*.json` glob).
- Calls `evolve_skill_bank(...)` with `teacher_model="vllm:<endpoint>"`. On success: atomic `os.replace()` rewrites `skill_bank_live.json` and snapshots `bank_out.json` into the per-step folder. On any failure (parse error, teacher down, glob empty): logged + swallowed, previous bank kept, training continues.

#### Reward manager (`verl_tool/workers/reward_manager/interphyre.py`)
- Two output paths, both gated on env vars (so non-evolving variants are unaffected):
  - **`TRAJ_LOG_FILE`** → compact JSONL inspection log. Each entry: `step, phase, level, seed, rollout_idx, score, response_length, tool_calls, n_turns, response_snippet (last 2000 chars)`. With `TRAJ_LOG_ALL=1` all n=4 rollouts per step are logged.
  - **`FULL_TRAJ_DIR`** → per-rollout JSON with the **full `raw_response`** at `step_{S}/{phase}/seed{N}_w{w}.json`. Canonical input for the skill-evolution hook.
- `phase: "train"|"val"` tag added (read from `data.meta_info["validate"]`) so train/val routing is unambiguous.

#### Trajectory format (no separate eval pass)
- The training rollouts at temp=1.0 ARE the teacher's input. n=4 rollouts per step give the teacher contrastive success+failure signal on the same seed.
- Set by `TRAJ_LOG_ALL=1` and `FULL_TRAJ_DIR` exports in the orchestrator script.
- **Current selection: 1 trajectory per seed** (best score). Hook receives 12 rollouts per K=3 fire (3 seeds × 4 rollouts) but stages 3 (one per seed) for the teacher, to keep the prompt within Qwen 7B's 8K context budget. Note: this loses the within-seed contrastive signal — could be revisited.

#### Trajectory converter (Phase 0 only)
- Script: `SKILL-RL/tools/convert_training_trajs.py` (unchanged)
- Used once, at the top of the orchestrator, to convert `FirstFiveTrajs/*.jsonl` → per-seed JSON for the initial distillation.

### How to run

```bash
# 1. Start Qwen 7B teacher server (on a separate node)
sbatch --reservation=sniekum2 examples/train/interphyre/slurm_vllm_7b_teacher.sh

# 2. Start evolving orchestrator (preferably different node from teacher; OK to co-locate
#    since the teacher is a vLLM server, not a Ray job)
sbatch --reservation=sniekum examples/train/interphyre/slurm_evolving_grpo_dte.sh

# Override parameters:
SKILL_EVOLVE_FREQ=5 sbatch --reservation=sniekum examples/train/interphyre/slurm_evolving_grpo_dte.sh
```

The orchestrator polls for `logs/vllm_teacher_endpoint.txt` automatically (up to 5 min file wait + 10 min health check). Submission order does not matter.

---

## Directory Structure

```
SKILL-RL/
├── FirstFiveTrajs/           # Epoch-0 (base model) trajectories for initial seeding
│   ├── dte_epoch0_trajectories.jsonl   # 52 DTE trajectories from run35 step 0
│   └── tbp_epoch0_trajectories.jsonl   # 52 TBP trajectories from run34 step 0
│
├── Skill_banks/              # Built skill bank JSON files
│   ├── skill_bank_evolving_DTE.json        # Used for static-skilled DTE training (run35)
│   ├── skill_bank_evolving_two_body.json   # Used for static-skilled TBP training (run34)
│   └── evolving_grpo_dte/                  # Per-round banks for the evolving variant
│       ├── skill_bank_round_0.json         # Distilled from FirstFiveTrajs
│       ├── skill_bank_round_1.json         # Round 1 (initial bank copy)
│       ├── skill_bank_round_2.json         # Evolved after round 1 trajectories
│       └── ...                             # Global numbering across epochs
│
├── data/                     # Converted trajectory files (SKILL-RL JSON format)
│   ├── initial/down_to_earth/            # Seeds 1-5, converted from FirstFiveTrajs
│   └── epoch_{e}/round_{r}/down_to_earth/ # Seeds collected after each round
│
├── tools/
│   ├── convert_training_trajs.py  # Converts training JSONL → SKILL-RL trajectory JSON
│   └── vllm_eval.py               # Fast batched trajectory collection via vLLM
│
├── core/
│   ├── skill_bank.py     # SkillBank / Skill / Mistake dataclasses + JSON persistence
│   ├── config.py         # Constants: TEACHER_MODEL, MAX_SKILLS_PER_LEVEL, level defs
│   └── retriever.py      # Skill retrieval logic (cross-level transfer map)
│
├── distillation/
│   ├── distill.py            # Three-pass distillation; call_teacher() with vLLM support
│   ├── evolving_distill.py   # Incremental skill bank evolution
│   ├── teacher_prompts.py    # Prompt templates + trajectory formatter
│   └── evolution_prompts.py  # Prompts for bank evolution
│
├── loops/
│   ├── evolving_loop.py      # Standalone evolving loop (Claude agent, not GRPO)
│   └── iterative_loop.py     # Full re-distillation per round (v1, for reference)
│
└── runner/
    ├── run_skill_agent.py    # CLI for running Claude agent with skill bank
    └── augmented_runner.py   # Skill injection into agent prompts
```

---

## Key Implementation Notes

### Skill injection into training prompts
Two paths share the same `format_skills_block()` helper (now lives in `verl_tool/utils/skill_injection.py`):

- **Static-skilled** (`interphyre_data.py --skill_bank`): bank is baked into the parquet's system prompt at gen-time.
- **Evolving** (`interphyre_data.py --runtime_skill_bank_path`): parquet stores only the path in `extra_info`; `verl/utils/dataset/rl_dataset.py:_build_messages()` reads the file fresh on every `__getitem__` and appends the skill block before the chat template is applied. n=4 rollouts in one GRPO step share the same bank because the dataloader returns one sample per step.

Skill-augmented prompts are ~2200 tokens, so `max_prompt_length=3072` is required (vs 2048 for non-skilled).

### Val timing
With the continuous-process design, val timing matters less because the model is always trained against whatever bank is currently in `skill_bank_live.json`. The orchestrator runs val on seeds 51-100 at `test_freq=50` (epoch boundaries) using the live bank. The "val must use the same bank as training" concern of the bash-loop variant is automatic here — val and training both read the same file.

### Python import path
The `SKILL-RL/` directory uses `skillrl.*` imports internally. A symlink `skillrl -> SKILL-RL` exists at the repo root so `sys.path.insert(0, '.')` resolves these correctly.

### Teacher model dispatch (`distill.py: call_teacher()`)
- `"claude-*"` → Claude CLI subprocess
- `"vllm:<url>"` → OpenAI-compatible HTTP POST to vLLM server (e.g. `vllm:http://gpu013:8000`)
- anything else → local HuggingFace model

### Trajectory format

**Three formats coexist in this repo — don't confuse them:**

`trajectories.jsonl` (compact inspection log written by reward manager):
- One line per rollout. Fields: `step`, `phase`, `level`, `seed`, `rollout_idx`, `score`, `response_length`, `tool_calls` (compact summary), `n_turns`, `response_snippet` (**last 2000 chars** — truncated, not for teacher input).
- Used for grep/inspection. Not the teacher's input.

`$FULL_TRAJ_DIR/step_{S}/{phase}/seed{N}_w{w}.json` (per-rollout full text, written by reward manager):
- One file per rollout. Fields: `step, phase, level, seed, rollout_idx, write_idx, score, response_length, n_turns, raw_response (FULL text, no cap)`.
- This is what the hook reads to drive evolution.

`logs/run{N}/skill_evolution/step_{S}/{level}/trajectory_seed{N}_step{S}.json` (SKILL-RL teacher input format, written by hook):
- One file per seed (the best-scoring rollout per seed). Fields: `success, seed, iterations, level, trajectory (parsed list of {thought, action, observation}), raw_response`.
- Filename starts with `trajectory_seed` to match `distill.py:167`'s glob `trajectory_seed*.json`.
- The `raw_response` field is used as fallback by the teacher formatter when parsed steps are empty.

### vllm_eval.py vs interphyre_eval.py
Neither is used by the continuous evolving loop — training rollouts ARE the teacher's input. `vllm_eval.py` is still available for ad-hoc evaluation of any checkpoint; `interphyre_eval.py` (HF inference) is used by the standalone eval scripts.

### Why "rounds" is no longer a structural concept
In the bash-loop variant, "round" was a real boundary in the system: verl exited and re-entered. Now there is no such boundary — gradient steps run continuously and the hook fires every K steps. The bank file is mutated in place via `os.replace()` between steps; the next `__getitem__` picks up the new version. To reproduce a run, log the bank hash into `trajectories.jsonl` per step.

---

## Engineering issues found during run43–run47 (2026-05-04 → 2026-05-05)

The continuous-process design landed working in concept on 2026-05-04, but several pipeline bugs were uncovered in subsequent inspection. In rough order of discovery:

1. **`load_trajectories` glob mismatch** (run43, run45 affected): `distill.py:167` globs `trajectory_seed*.json` but the original hook wrote `trajectory_step{S}_r{i}_seed{N}.json`. Empirically zero matches → teacher saw `0 successes, 0 failures` every fire → bank "evolved" via teacher hallucination from the static prompt, with no GRPO signal. Fixed in run46 by renaming hook output to `trajectory_seed{N}_step{S}_r{i}.json`.

2. **Validation rollouts polluting train evolution input**: `TRAJ_LOG_FILE` was shared between train and val; the hook drained both blindly, so at every val boundary (steps 0, 50, 100, ...) the next K-step fire absorbed all 50 val seeds as if they were training rollouts. Fixed: reward manager tags each entry with `phase: "train"|"val"` (read from `data.meta_info["validate"]` at `ray_trainer.py:614`); val rollouts are now written to `$FULL_TRAJ_DIR/step_{S}/val/` and ignored by the hook.

3. **Response truncation hiding 95% of trajectory text**: `response_snippet` stored only the last 2000 chars. For 25-turn ReAct loops (~10–20 KB) that's the tail of repeated retries — early thoughts and the `[FINAL_RESULT]` block are missing. Combined with #1 the teacher saw nothing useful. Fixed via the **`FULL_TRAJ_DIR` redesign** ("Option B"): per-rollout JSON with full `raw_response`. `trajectories.jsonl` retained as the compact inspection log only.

4. **1-of-4 rollout collision in FULL_TRAJ_DIR** (open in run47): verl's async ReAct invokes the reward manager per-rollout with `len(data)==1`, with multiple worker processes each having their own reward manager instance. Each process started `_write_counter=0` and wrote `seed{N}_w0.json`, so all 4 rollouts of a step collided on disk and only the last write survived. Fix: include `os.getpid()` in filename (`seed{N}_pid{P}_w{w}.json`). Source-edited but **not yet deployed in run47** — would require scancel + resubmit.

5. **ReAct parser undercounts steps**: `_parse_react_steps` splits on `Thought:` and runs a single `re.search` per chunk, so multi-action-per-thought sequences yield only the first Action+Observation pair. Example: `step_270/trajectory_seed33_step270.json` has `iterations: 20` (truth) but `len(trajectory): 9`. `format_trajectory_for_teacher` also caps at `max_steps=10`, compounding the loss. Open.

6. **Tool-output hallucination by the model** (observed in run47): step 220 seed 43, the model wrote its own `Observation: SUCCESS` block (with a fake markdown image link!) instead of waiting for the tool server, then started `Action: finish` and ran out of tokens. Reward correctly scored 0.0. Cause unclear — may be reinforced by feeding noisy "last-written" rollouts back through evolution. Watch this if val drops on long ReAct loops.

---

## Active Runs (as of 2026-05-05)

| Job | ID | Node | Description |
|-----|----|------|-------------|
| `vllm_7b_teacher` | 56685118 | gpu013 | Qwen 7B teacher server. Up since 2026-05-04 16:28; reused across run43/45/46/47. Endpoint at `logs/vllm_teacher_endpoint.txt`. `max_model_len=8192`, `max_tokens=2048` for completion. |
| `evolving_grpo_dte` | 56694542 | gpu015 | Run47 — Option B pipeline. Step ~280/500 last check. Val collapsed to 0.00 at step 200; train plateaued ~0.24 in epoch 3. Worse than run45 (which had broken evolution via #1). Likely fixable by deploying the PID fix (#4). |

### Recent run history

| Run | Job | Steps | Status | Notes |
|-----|-----|-------|--------|-------|
| run43 | 56685119 | 400 | cancelled 2026-05-04 | All 132 evolve calls fed empty input due to #1. Bank hallucinated. |
| run44 | 56685339 | 500 (full) | **completed** | Static-skilled DTE (different orchestrator). 100% val. Strong baseline; checkpoints under `interphyre/.../skilled-dte/run44/global_step_*`. |
| run45 | 56689677 | ~284 | cancelled 2026-05-04 | Phase tag, val routing, bank_out snapshots; #1 still present. Despite empty teacher input, val outpaced run44 through step 250 (likely noise). |
| run46 | (brief) | <10 | cancelled | Glob-fix only; cancelled to add Option B. |
| run47 | 56694542 | active | running | Option B deployed: full ReAct text, dir-scan hook, best-per-seed staging. Issue #4 still open → only 1 of 4 rollouts survives; teacher selection has nothing to choose from. |
