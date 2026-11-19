# HeXA on DiscoverPhysics (`hexa_discovery`)

Runs HeXA's skill-evolution (distil → evolve a curated bank the agent reads at
solve time) on the DiscoverPhysics benchmark, focused on **Axis B: cross-world
skill evolution** — a single shared bank of *experimentation strategies* grown
across worlds and transfer-tested on unseen worlds.

## Why cross-world (not cross-seed)

A DiscoverPhysics seed only reshuffles observation noise on a fixed world with a
single fixed hidden law (seed → `--noise-seed`). So the InterPhyre notion of
"a seed is a new instance" maps to **a world is a new instance** here. Skills are
therefore method-level ("extend the time window before ruling out a static law",
"sweep one variable per experiment", "fit parameters before submitting"), which
transfer across worlds and cannot encode any single world's answer.

## What this reuses vs. replaces

HeXA's distil/evolve engine (`skillrl.distillation.*`, `skillrl.core.*`) is
reused **unchanged**. Only the four InterPhyre-specific seams are replaced:

| Seam | HeXA (InterPhyre) | Here |
|---|---|---|
| Reward | `success × iterations` | `reward.py`: `0.5·expl_score + 0.5·clip(1 − norm_mse/0.1)` ∈ [−0.5, 1.0] |
| Trajectory JSON | ReAct `{thought,action,observation}`, final `[x,y,r]` | `traj_adapter.py`: DiscoveryAgent `conversation_log` → same schema, `action=None`, `norm_mse`/`expl_score`/`reward` stamped |
| Actor runner | `run_skill_augmented` (InterphyreToolkit/ReactAgent) | `discovery_runner.py`: `DiscoveryAgent` + skills appended to `agent._system` |
| Teacher prompts | ball-placement, per-level | `teacher_prompts_discovery.py`: cross-world experimentation framing + answer-redaction + DiscoverPhysics trajectory formatter |

Seams install via `reward.patch()` then `teacher_prompts_discovery.patch()`
(order matters) — monkey-patching module globals, no edits to `skillrl`.

The whole cross-world bank lives under one synthetic level, `discovery`, so the
level-keyed distil/evolve code works as-is. The current world's **answer-free**
facts (`config.WORLD_FACTS`) are injected as the teacher's `level_description`.

## Leakage guard

`redact=True` (default) masks numeric literals in law/fit sources and fit
**parameter values** shown to the teacher (e.g. `alpha=1.5` → `<n>`), while
keeping raw trajectory observations and qualitative fit loss. Prompts also
explicitly forbid recording constant values in skills.

## LLM routing: Claude CLI by default (no API key)

Like HeXA, all model calls go through the Claude Code CLI (`claude -p`) — no
API key needed. **There are two independent dispatchers here, with different
rules for what model strings they accept — do not mix them up:**

1. **Actor / judge / critic** (`--cli-model`): `cli_client.patch_llm_client()`
   swaps `scienceagent.llm_client.complete` for a CLI backend, so
   `DiscoveryAgent`, `ExplanationJudge` and `CriticAgent` all route through
   it. This shim forwards the model string as-is to `claude --model`, so bare
   CLI aliases work: `--cli-model sonnet` / `opus` / `haiku`.
2. **Teacher** (`--teacher-model`): this goes through HeXA's own
   `skillrl.distillation.distill.call_teacher`, which is untouched by our
   patch and has its own routing rule: `model.startswith("claude-")` → Claude
   CLI subprocess, **anything else silently falls through to an HF/Qwen
   loader** (it will literally try to download `Qwen/<your string>`, fail, and
   `call_teacher` returns `""` — which then parses into a bank with **0
   skills and no visible error**). So `--teacher-model` MUST be a full
   `claude-*` string, e.g. `claude-sonnet-4-6` (the default) or
   `claude-opus-4-6` — never a bare alias like `sonnet`.
   `cross_world_loop.py` now fails fast with a clear `ValueError` if
   `--teacher-model` doesn't start with `claude-`, instead of silently
   producing an empty bank.

`--no-cli` uses API providers instead of the CLI for the actor/judge/critic
path (then the relevant API key is required); it does not affect the teacher.

## Run

All commands run from the HeXA root (`hexa_discovery/` lives there). The
upstream DiscoverPhysics code is the `discovery-agents/` git submodule
(`git submodule update --init discovery-agents`); scienceagent loads prompts by
relative path, so the runner switches into `discovery-agents/` itself while
episodes run. Needs the `claude` CLI and
`pip install -e discovery-agents/ScienceAgent/ discovery-agents/PhysicsSchool/`.

```bash
# 1. Offline bootstrap: seed trajectories for the FIRST train world (CLI actor)
python -m hexa_discovery.run_offline \
    --world gravity --seeds 0 1 2 3 4 5 \
    --output-dir hexa_discovery/data/offline/gravity \
    --cli-model sonnet --noise-std 0.075

# 2a. Sequential cross-world evolution, leave-one-WORLD-out (held out = oscillator)
python -m hexa_discovery.cross_world_loop \
    --offline-traj-dir hexa_discovery/data/offline/gravity/discovery \
    --output-dir hexa_discovery/data/runs/loo_oscillator \
    --held-out-world oscillator \
    --seeds-per-world 3 --start-seed 100 \
    --cli-model sonnet --teacher-model claude-sonnet-4-6

# 2b. Leave-one-GROUP-out (held out = the whole central-force cluster)
python -m hexa_discovery.cross_world_loop \
    --offline-traj-dir hexa_discovery/data/offline/three_species/discovery \
    --output-dir hexa_discovery/data/runs/logo_central \
    --held-out-group central_force --seeds-per-world 3 --cli-model sonnet

# 2c. Pooled baseline (distil once from pooled offline trajs, no online evolve)
python -m hexa_discovery.cross_world_loop \
    --offline-traj-dir hexa_discovery/data/offline/pooled/discovery \
    --output-dir hexa_discovery/data/runs/pooled --mode pooled \
    --held-out-world oscillator --cli-model sonnet
```

The loop writes `progress.json` with per-world online pass-rates during training
and, at the end, a `transfer` block comparing **bank vs. plain baseline**
pass-rate on each held-out world (same actor) — the headline result.

## Tests (no API / no scienceagent needed)

```bash
python -m hexa_discovery.tests.test_reward           # blended reward
python -m hexa_discovery.tests.test_traj_adapter     # schema + redaction
python -m hexa_discovery.tests.test_discovery_runner # skill retrieval/injection bookkeeping
```

## Inspecting skill injection

Every trajectory JSON records exactly which bank entries were shown to the
agent that episode (not just whether injection happened):
`skills_injected` (bool), `skill_ids_injected` / `skill_titles_injected`,
`mistake_ids_injected`. Compare these against the agent's early-round
reasoning/actions to confirm the injected skills are actually shaping behavior
(they get absorbed into the agent's own voice, not quoted verbatim, so check
behavior — e.g. adaptive time-window scaling matching a skill's recipe — rather
than looking for literal skill titles in the thoughts).

## Status / TODO

- Done + verified offline: config, reward, adapter, teacher prompts, runner,
  cross-world loop, offline bootstrap, CLI client; unit tests + mock-teacher
  end-to-end distil→evolve→inject; live CLI smoke test of `cli_complete`.
- Not yet run against a live full episode (needs `pip install -e ScienceAgent/
  PhysicsSchool/` so `scienceagent` + jax/box2d import).
- Not yet added: per-round resume in `cross_world_loop` (offline round-1 + each
  world's bank file is reused if present, but partial within-world seed resume
  relies on `run_batch_discovery`'s per-file skip).
- Axis A (within-world, noise-seed robustness) and Axis C (parametric world
  randomization) are separate follow-ups.
