# Pigey on Unity, with HeXA's skill bank

This directory runs [Pigey](../Pigey) (a frozen pi0.5 policy driven by an LLM
orchestrator) on the Unity cluster, with Claude Code as the orchestrator and
open-weight perception, and adds [HeXA](../../README.md)'s evolving skill bank on
top of it. The Pigey submodule and HeXA's `skillrl/` are used as released; every
adaptation lives here.

Two things are worth knowing before anything else:

* **The experiment protocol is one script.** `experiments/libero_goal_task/submit.sh`
  trains a bank for 15 rounds, evaluates Pigey with and without it on 250 held-out
  episodes each, and compares the two arms. `protocol.env` next to it is the whole
  design, and `initial_bank.json` is the bank it starts from.
* **This is not the paper's configuration.** The orchestrator, detector and verifier
  are substitutes (table below), and the paper is zero-shot while the HeXA arm
  learns from earlier episodes. Label results accordingly.

## Layout

```text
pigey_setup/
├── README.md
├── activate.sh                  environment variables + the simulator venv; source it first
├── run_claude.sh                one Pigey episode through Claude Code (the entry point)
├── runner/                      the Claude Code-backed runner
│   ├── run_claude.py            wraps Pigey's agent_sim.py: Claude bridge, local perception, skill injection
│   ├── claude_code.py           `claude -p` with native images and structured tool selection
│   ├── perception.py            Qwen3-VL detector + SAM client + Claude VerifyCandidate, routed in memory
│   ├── sam_server.py            Pigey's own _sam_get_mask served on CUDA
│   ├── perception_check.py      detector accuracy against simulator ground truth
│   ├── completed_trials.py      lists finished (task, init state) pairs: how jobs resume
│   └── free_ports.py
├── hexa/                        HeXA's skill bank over Pigey (nothing in skillrl/ is edited)
│   ├── trajectory.py            trial.json -> HeXA trajectory, reward, teacher formatting
│   ├── teacher.py               prompts, trajectory selection, Claude teacher, prompt-block rendering
│   ├── loop.py                  the evolving loop: bank -> round of episodes -> evolved bank -> ...
│   └── compare.py               paired with-bank vs no-bank comparison of two eval arms
├── servers/
│   ├── serve_perception.sh      vLLM (detector) + SAM servers on free ports, health-checked
│   └── start_servers.sh         sourced by the Slurm jobs: perception + pi0.5 policy server, cleanup trap
├── slurm/
│   ├── run_trial.sbatch         Pigey episodes on one A100 (eval arms, baselines); resumable
│   ├── run_hexa.sbatch          the HeXA training loop on one A100; resumable
│   ├── run_raw.sbatch           the paper's pi0.5-alone baseline (no LLM); resumable
│   ├── run_hexa_arm.sbatch      seed episodes (optional) + training + with-bank eval in one job
│   └── check_perception.sbatch  detector smoke test
├── experiments/
│   ├── submit.sh                submits one directory's protocol: [seed ->] train -> eval-bank,
│   │                            eval-nobank, compare
│   ├── libero_goal_task/
│   │   ├── protocol.env         the split (init states 0-9 / 10-24 / 25-49), models, walltimes
│   │   ├── initial_bank.json    the bank training starts from (8 skills, 4 mistakes)
│   │   └── submit.sh            shortcut for ../submit.sh on this directory
│   └── libero_{object_swap,object_task,spatial_swap,spatial_task,goal_swap}/
│       └── protocol.env         same splits; no initial bank, so a seed job (init states 0-9,
│                                no bank) runs first and training distills round 1's bank from it
├── install/                     one-time setup jobs (environments, models, LIBERO-PRO assets, validation)
└── tests/                       offline tests, no GPU or API: python -m pytest tests
```

Python entry points put `pigey_setup/` on `sys.path` and import `runner.*` and
`hexa.*`; importing `hexa` also adds the repository root so `skillrl` and
`hexa_core` resolve.

## Quick start

Everything heavy runs under Slurm; the login node only edits files and submits.
Paths default to this checkout and the `roboprobe-sim` scratch workspace
(`PIGEY_ROOT`, set in `activate.sh`); override `REPO_DIR`, `PIGEY_ROOT` and the
`#SBATCH --account/--partition` lines for another cluster.

```sh
cd /project/pi_sniekum_umass_edu/vgandhi/HeXA-Hierarchical-Experimentalist-Agents
source baselines/pigey_setup/activate.sh        # PIGEY_ROOT, caches, simulator venv

# 1. Install (once; cpu jobs): simulator + policy venvs, then vLLM + models (~1 h)
sbatch -o "$PIGEY_ROOT/logs/%x-%j.out" -e "$PIGEY_ROOT/logs/%x-%j.err" baselines/pigey_setup/install/install.sbatch
sbatch -o "$PIGEY_ROOT/logs/%x-%j.out" -e "$PIGEY_ROOT/logs/%x-%j.err" baselines/pigey_setup/install/install_models.sbatch
PIGEY_ER_MODEL=Qwen/Qwen3-VL-32B-Instruct-FP8 sbatch --export=ALL baselines/pigey_setup/install/download_er_models.sbatch Qwen/Qwen3-VL-32B-Instruct-FP8

# 2. Check the Claude Code login (no inference) and the offline tests
bash baselines/pigey_setup/run_claude.sh --check-cli
python -m pytest baselines/pigey_setup/tests

# 3. A few plain Pigey episodes on one A100 (task 0 and 8, init state 2)
PIGEY_TASKS="0 8" PIGEY_EPISODES="2" PIGEY_ER_MODEL=Qwen/Qwen3-VL-32B-Instruct-FP8 PIGEY_ER_GPU_UTIL=0.55 \
  sbatch --export=ALL -o "$PIGEY_ROOT/logs/%x-%j.out" -e "$PIGEY_ROOT/logs/%x-%j.err" \
  baselines/pigey_setup/slurm/run_trial.sbatch --model claude-sonnet-5-5

# 4. The whole HeXA protocol (train, both eval arms, comparison)
bash baselines/pigey_setup/experiments/libero_goal_task/submit.sh --dry-run   # shows the four sbatch commands
bash baselines/pigey_setup/experiments/libero_goal_task/submit.sh
# Other LIBERO-PRO suites: seed -> train -> eval arms -> comparison (EXCLUDE_NODES=... skips bad nodes)
bash baselines/pigey_setup/experiments/submit.sh baselines/pigey_setup/experiments/libero_object_swap --dry-run
# --one-job: seed + training + with-bank eval as one GPU job (slurm/run_hexa_arm.sbatch, 36 h),
# so no phase waits in the queue again; the no-bank arm and the comparison stay separate
bash baselines/pigey_setup/experiments/submit.sh baselines/pigey_setup/experiments/libero_object_swap --one-job
```

Results go under `$PIGEY_ROOT/results/`, logs under `$PIGEY_ROOT/logs/`. The
scratch workspace expires (check `ws_list`); copy results to
`/work/pi_sniekum_umass_edu/vgandhi` before then.

## The runner: Pigey through Claude Code

`run_claude.sh` runs Pigey's unmodified `sim/agent_sim.py` in harness mode with
three in-memory substitutions, so no Anthropic or Gemini API key is needed:

| Role | Pigey paper | Here | Flag |
|---|---|---|---|
| Orchestrator | Claude Sonnet 4.6 (API) | Claude Code CLI, `--model` (default `claude-haiku-4-5`; experiments use `claude-sonnet-5-5`), native image inputs, schema-validated tool selection | `--model` |
| Detector (`Perceive` labels + boxes) | Gemini Robotics-ER | `Qwen/Qwen3-VL-32B-Instruct-FP8` served by vLLM; output converted to Gemini's box format, Pigey's parsing and 3D unprojection unchanged | `--perception local` (or `gemini`), `--er-model` |
| Verifier (`VerifyCandidate` YES/NO/UNSURE) | Gemini Flash-Lite | Claude Code, `claude-haiku-4-5` | `--verifier claude` (or `gemini`), `--verify-model` |
| SAM masks | `facebook/sam-vit-base` in-process on CUDA | the same model and Pigey's own `_sam_get_mask`, served by `sam_server.py` (the simulator venv has CPU Torch 1.11) | `--sam service` (or `inprocess`) |
| Phase 0 (silent pi0.5 attempt before the agent) | skipped in the paper's Algorithm 1 | skipped (`CORTEX_NO_PHASE0=1`) | `--phase0 skip` (or `run`) |

The Claude bridge replaces `call_claude`: each call sends the upstream robot system
prompt plus the conversation (camera images as image blocks, including those inside
earlier tool results) to `claude -p` in a scratch working directory with coding
tools, MCP servers, hooks and session persistence off, and gets a structured
`{reasoning, tool, input}` back. A "Context to ignore" preamble tells the model to
disregard the CLAUDE.md and account context the CLI injects. `--claude-timeout`
(300 s), `--claude-max-turns` (3, for the structured-output step) and
`CLAUDE_BINARY` tune it. Per-call usage and cost are kept in `trial.json`.

`trial.json` (written by Pigey under `<out-dir>/<model>/<suite>_t<task>_e<episode>/<timestamp>/`)
gains `backend`, `claude_code_calls`, `er_calls`, `sam_calls`, `verify_calls`,
`perception`, `verifier`, `sam`, `phase0` and, with a bank, `skill_injection`.
A CLI, detector, SAM or verifier failure ends the episode with
`status: backend_error` and `success: null`; exclude those from statistics
(`completed_trials.py` and `compare.py` already do). Pigey's own `success` is the
simulator's predicate, untouched.

Live checks that need a Slurm allocation but no simulator:
`run_claude.sh --cli-smoke-test` (one image + tool-selection call) and
`slurm/check_perception.sbatch` (detector accuracy on 30 LIBERO frames; writes
annotated PNGs and `perception_check.json`). With Qwen3-VL-32B-FP8 the detector
found 112/120 objects at a median box IoU of 0.85 and 3.4 cm median 3D error, 0
detection loops; the 8B model was much worse (IoU 0.53, 236 junk detections).

## Servers and Slurm jobs

`servers/start_servers.sh`, sourced by both GPU jobs, checks outbound HTTPS (Claude
Code needs it), starts the detector and SAM (`serve_perception.sh`) and the pi0.5
policy server from the downloaded `pi05_libero` checkpoint, waits for their health
endpoints, exports `POLICY_PORT`, `PIGEY_ER_URL`, `PIGEY_SAM_URL`, and installs an
exit trap that stops them. Ports are chosen free at startup because GPU nodes are
shared (`PIGEY_POLICY_PORT`, `PIGEY_ER_PORT`, `PIGEY_SAM_PORT` pin them). One
A100-80GB holds everything: vLLM at `PIGEY_ER_GPU_UTIL=0.55` for the 32B detector
(0.4 default suits the 8B), pi0.5 at `XLA_PYTHON_CLIENT_MEM_FRACTION=0.35`, SAM and
EGL in the rest.

`slurm/run_trial.sbatch` (1 A100, 8 CPUs, 64 GB; set `--time` for about 2-3 min per
episode plus 10 min of startup) runs every pair in `PIGEY_TASKS` x `PIGEY_EPISODES`
of `PIGEY_SUITE` (default `libero_goal_task`) with 35 LLM steps and 500 steps per
rollout, Pigey's README settings. `PIGEY_OUT_DIR` names the results directory
(default `results/claude-code/<job id>`); an existing one is **resumed**, skipping
pairs that already finished. `PIGEY_SKILL_BANK` makes it the with-bank arm. Extra
arguments go to `run_claude.sh`. A failed episode is recorded and the rest still run.

`slurm/run_hexa.sbatch` runs `hexa/loop.py` the same way: `PIGEY_HEXA_INITIAL_BANK`
(a finished bank) or `PIGEY_HEXA_INITIAL` (trial directories to distill one from)
seed round 1; `PIGEY_HEXA_EPISODES` is the pool of init states consumed
`PIGEY_HEXA_EPISODES_PER_ROUND` at a time; `PIGEY_HEXA_OUT` names the run directory
and resumes it. `--rounds`, `--model` and other options pass through.

`slurm/run_raw.sbatch` is the paper's "pi0.5 alone" baseline: Pigey's
`agent_sim.py --mode raw` gives pi0.5 the task string once and runs it closed-loop
until success or Pigey's step limit for the suite (`SUITE_MAX_STEPS`; 400 for
`libero_goal_task`, which has no entry). No Claude, no detector: `start_servers.sh`
starts only the policy (`PIGEY_POLICY_ONLY=1`) and `HARNESS_PERCEPTION=none` keeps
`agent_sim` from calling Gemini. Same `PIGEY_TASKS`/`PIGEY_EPISODES`/`PIGEY_SUITE`/
`PIGEY_OUT_DIR` variables and resume as `run_trial.sbatch`; each trial also saves
`rollout.mp4`.

## HeXA's skill bank over Pigey

### The idea in plain terms

Pigey's orchestrator starts every episode from scratch: nothing it learned in one
episode reaches the next. HeXA adds a memory. A teacher model reads finished
episodes (what the orchestrator reasoned, which tool it called, what came back) and
writes a short list of **skills** ("after one centre-grasp miss on a plate, switch to
a rim bbox") and **mistakes** ("`HELD` with gripper width 0.001 means nothing is
held"). That list is appended to the orchestrator's system prompt for the next
episodes. After each round of episodes the teacher reads the new episodes and the
current list, and keeps, rewrites, adds or drops entries, capped at 10 skills and 5
mistakes so it has to choose. The orchestrator model itself never changes.

### What is held out

Each LIBERO goal task is a *question* (open the drawer, put the bottle on the plate,
turn off the stove, ...). Each question comes in 50 *versions*, the init states,
which differ only in where the objects start. All 10 questions appear in every
phase; the split is on versions, so the question answered is "after learning from
some arrangements of these tasks, does the agent do better on arrangements it has
not seen?". Holding out whole tasks instead (`--tasks 0-4` for training, eval on
`5-9`) would ask about transfer to unseen goals; possible, but with 10 tasks the
eval would be small.

### How the first bank is built

The first bank comes from two teacher calls over the 100 seed episodes (init states
0-9, run without a bank). This is HeXA's offline seeding, as in InterPhyre
(`skillrl/loops/evolving_loop.py`, seed runs in `Initial_trajectories/`); the
`hexa_core` and `hexa_orbit` ports have no seed set, their round 1 runs without a
bank and the first bank is distilled from it, which `hexa/loop.py` does when neither
`--initial-bank` nor `--initial-trials` is given.

1. **Skills call** (`distill_contrastive`): up to `--seed-trajs` (6) highest-reward
   successes and 6 lowest-reward failures, each set spread round-robin across tasks
   so no task fills the block; the teacher writes strategies that separate them.
2. **Mistakes call** (`distill_mistakes`): the same 6 failures; the teacher writes
   recurring errors, plus skills implied by the failures.

So about 12 of the 100 seed episodes reach the teacher; the rest are unused, which
is HeXA's design (the InterPhyre seed set has 5 episodes). Each episode is shown as
its last 12 steps of reasoning, tool call and result, with detection lists stripped.
A new entry's confidence is the mean reward of the episodes it cites, mapped from
[-1, 1] onto [0.1, 1.0]. The bank is capped at 10 skills and 5 mistakes.
`experiments/libero_goal_task/initial_bank.json` is this bank (job 66525522, 45 s,
$0.22; 8 skills, 4 mistakes; sha256 `4d31956f...`), and the loop copies it in as
`bank_round_1.json` and records the copy in `initial_bank_source.json`.

### A round, step by step

1. `hexa/loop.py` has `bank_round_<r>.json` (round 1: the initial bank; later
   rounds: evolved from the previous round's episodes).
2. For every task it runs one `run_claude.sh` episode with `--skill-bank`, which
   appends the rendered block (confidence-ordered, within `--injection-budget-tokens`
   = 4000, enough for a full bank of about 3400 tokens) to Pigey's system prompt.
   Each `trial.json` records exactly which entries it saw (`skill_injection`), and
   `parent_overlap` names any injected entry whose `source_seeds` include this very
   episode: expected during training when an init state repeats, never in the eval.
3. The teacher (`claude-sonnet-5-5` through `claude -p`, `--teacher-model`) gets the
   round's episodes (up to `--max-trajs` = 10, same-task success/failure pairs first)
   plus the current bank and returns the complete new bank; new entries get
   confidence from the rewards of the episodes they cite, kept entries keep the
   teacher's updated confidence. Every prompt and response is saved under `teacher/`
   as `teacher_NN_{prompt,response}.txt`.
4. `progress.json` gains one record: successes per task, actor cost, teacher cost.

After the last round the loop evolves once more, so `bank_round_<rounds+1>.json`,
the bank that has seen every training episode, is the one the frozen eval uses.

**Reward.** HeXA's shaped reward with Pigey's LLM steps as the iteration count:
success 1.0 (at most 3 steps), 0.75 (8), 0.5 (15), else 0.25; failure -0.5 when a
budget ran out (the 2000-step simulator horizon or the 35-step LLM cap), -0.75 when
the episode ended early. In practice every seed failure hit the horizon, so failures
all score -0.5; the reward mostly ranks successes by speed.

**Frozen eval.** `run_trial.sbatch` with and without `PIGEY_SKILL_BANK` on the same
held-out init states; nothing is written back to the bank. `hexa/compare.py` pairs
the two arms episode by episode and reports success per arm, the difference,
discordant episodes with an exact McNemar p-value, per-task counts, cost, and an
audit that every with-bank episode received the full bank with no parent overlap.

| File | Role |
|---|---|
| `hexa/trajectory.py` | A Pigey `trial.json` as a HeXA trajectory: steps of reasoning -> `Tool(input)` -> result, seed `t<task>_e<episode>`, the reward, the teacher's trajectory text. `load_trial_trajectories` replaces `distill.load_trajectories`. |
| `hexa/teacher.py` | `install()` rebinds reward, formatter, loader, the three teacher prompts (facts-only suite description plus Pigey's own system prompt quoted so the teacher does not restate it) and the teacher call. `render_block()` renders the injected block within a token budget. |
| `runner/run_claude.py --skill-bank` | Appends the rendered block to Pigey's system prompt by rebinding `get_system_prompt`; records `skill_injection`. Also `PIGEY_SKILL_BANK`, `--skill-level`, `--skill-budget-tokens`. |
| `hexa/loop.py` | The evolving loop (resumable: finished rounds and banks are reused). `--distill-only` builds or copies round 1's bank and stops. |
| `hexa/compare.py` | The paired comparison above; `--json` keeps the full result. |

## The libero_goal_task protocol

`experiments/libero_goal_task/protocol.env`:

| Init states | Episodes | Used for |
|---|---|---|
| 0-9, all tasks | 100 | Seed: plain Pigey (28/100). `initial_bank.json` was distilled from them. |
| 10-24, all tasks | 150 | Training: 15 rounds, one init state of every task per round, the bank evolving after each; the final bank is `bank_round_16.json`. |
| 25-49, all tasks | 250 per arm | Frozen eval: once with the final bank, once without, never seen by the teacher. |

`submit.sh` submits four jobs and writes their ids to `results/eval/goal_task_jobs.txt`:

| Job | Script | Starts | Output |
|---|---|---|---|
| `goal_task-train` (12 h) | `run_hexa.sbatch --model claude-sonnet-5-5 --rounds 15` | now | `results/hexa/goal_task_train/{bank_round_1..16.json, round_<r>/, teacher/, progress.json}` |
| `goal_task-eval-bank` (14 h) | `run_trial.sbatch` with `PIGEY_SKILL_BANK=.../bank_round_16.json` | after train succeeds | `results/eval/goal_task_bank/` |
| `goal_task-eval-nobank` (16 h) | `run_trial.sbatch`, no bank | now | `results/eval/goal_task_nobank/` |
| `goal_task-compare` (cpu) | `hexa/compare.py` | after both arms end | `results/eval/goal_task_comparison.{txt,json}` |

Walltimes come from the pilot (50 episodes: 1 h 46 min with the bank, 2 h 22 min
without; 5 training rounds: 1 h 45 min). If a job hits its limit, run `submit.sh`
again: finished rounds, banks and episodes are skipped, and the dependencies are
re-created. `--run-name NAME` keeps a second run apart.

**Against the paper.** Pigey reports 10 trials per LIBERO-PRO task (about 100 episodes
per suite number, init states unstated) with no learning across episodes; its
goal-task number is 22% with Claude Opus 4.7, 30% with Sonnet 4.6. The no-bank arm
here is that method with the substitutions above; the with-bank arm adds HeXA.

**Pilot (states 10-14 for training, 40-44 for eval, 50 episodes per arm, same
code):** training 23/50 over 5 rounds; eval 25/50 with the bank vs 17/50 without
(+16 points; 11 episodes solved only with the bank, 3 only without, exact McNemar
p = 0.057). Gains concentrated on cream cheese into the drawer, plate on the drawer
and cream cheese on the rack; Claude cost $13.77 vs $23.33.

## Install

`install/install.sbatch` (cpu, 30 min) clones LIBERO-PRO and openpi into
`$PIGEY_ROOT`, builds the Python 3.10 simulator venv (LIBERO-PRO's legacy pins
except `transformers`, which LiteLLM needs newer; MuJoCo 2.3.7; CPU Torch 1.11) and
openpi's locked Python 3.11 venv for the policy server, downloads LIBERO-PRO's BDDL
and init-state files (`prepare_tasks.py`, hash-verified), and runs
`validate.sbatch` (simulator import, reset and step under OSMesa; policy-server
help). `install/install_models.sbatch` (cpu, 2 h) builds the vLLM venv and downloads
the detector, SAM ViT-B and the `pi05_libero` checkpoint; `download_er_models.sbatch`
fetches further detector candidates. Package snapshots are kept under `install/locks/`.
LIBERO-PRO's package metadata discovers no packages, so `activate.sh` puts its source
root on `PYTHONPATH` and validation adds a `.pth`; the upstream checkout is unchanged.

## Tests

```sh
python -m pytest baselines/pigey_setup/tests          # or: python -m unittest discover -s baselines/pigey_setup/tests
```

43 offline tests: the Claude bridge, perception routing and SAM client, trajectory
conversion and reward, teacher selection and installation (distillation and
evolution against a fake teacher), block rendering within the budget, skill
injection into the runner, teacher logging, initial-bank seeding, resume listing and
the paired comparison. No GPU, simulator or API call.

## Teammate resume handoff (2026-11-08)

See [RESUME_TEAMMATE.md](RESUME_TEAMMATE.md) for shared scratch access, own-account
authentication, and the eval-only `experiments/resume_eval.sh` commands for the
remaining goal-swap and spatial-task episodes.
