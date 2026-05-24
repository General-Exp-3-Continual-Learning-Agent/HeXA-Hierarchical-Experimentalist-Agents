# Baselines

Three ReAct-family baselines are included for comparison with HeXA. All three live in [react_agent/](../react_agent/) and share the same actor model (Claude Sonnet via the `claude -p` CLI subprocess for the Claude variants; Qwen / vLLM / mock for the OSS variant).

## ReAct (single shot)

**Wrapper:** `scripts/run_react.py` (Qwen) or `scripts/run_react_claude.py` (Claude)
**Module:** [react_agent.run_react](../react_agent/run_react.py) / [react_agent.run_react_claude](../react_agent/run_react_claude.py)

Standard ReAct: the agent loops `Thought → Action → Observation` for up to `--max-iterations` (default 25) with no extra memory, no skill bank, no reflection. Each seed is independent.

```bash
python scripts/run_react_claude.py \
    --level pass_the_parcel \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --max-iterations 25 \
    --eval-dir results/react_claude_pass_the_parcel
```

**Pass-the-parcel result:** 24.0 % (12/50 seeds), avg 21.3 iterations, 5.08 h.

## Reflexion (K=2 trials)

**Wrapper:** `scripts/run_reflexion.py`
**Module:** [react_agent.run_react_claude_reflexion](../react_agent/run_react_claude_reflexion.py)
**Subpackage:** [react_agent/reflexion/](../react_agent/reflexion/) — `runner.py`, `reflect.py`, `memory.py`

Per seed, runs up to **K=2** ReAct trials. After each *failed* trial, a separate Claude call ingests the trajectory and writes a 3–5 sentence verbal self-reflection grounded in the failure mode (which strategy was attempted, why it failed kinematically, what to try next). The reflection is injected into the *next* trial's system prompt under a `## Past Lessons` block.

Stops early on success; reflections do not transfer across seeds.

Both the actor and the reflection step use the same `claude -p` subprocess pattern as the ReAct baseline — no SDK, no API key. Per-trial trajectories and per-seed summaries are saved to disk so an interrupted run can be **resumed mid-seed**:

```bash
# Initial run
python scripts/run_reflexion.py \
    --level catapult --seeds 0 1 2 3 4 5 6 7 8 9 \
    --k-trials 2 --eval-dir results/reflexion_catapult --verbose

# If killed, resume — completed trials/seeds are skipped
python scripts/run_reflexion.py \
    --level catapult --seeds 0 1 2 3 4 5 6 7 8 9 \
    --k-trials 2 --eval-dir results/reflexion_catapult --resume
```

**Pass-the-parcel result:** 17.1 % (6/35 seeds; partial 35-of-50 run), avg 17.9 iterations across both trials, 12.50 h. The partial number is unusually low for Reflexion-style baselines because PtP rewards persistence on the first trial — re-run with `scripts/run_reflexion.py` (and `--resume` if interrupted) to regenerate per-seed summaries with the trial-of-success breakdown.

The Reflexion plan and reproduction notes are in [skillrl/SKILLRL_SUMMARY.md](../skillrl/SKILLRL_SUMMARY.md) and the in-package docstring in `react_agent/run_react_claude_reflexion.py`.

## Direct-answer (1-shot, 2-iteration ceiling)

**Wrapper:** `scripts/run_direct.py`
**Module:** [react_agent.run_react_claude_direct](../react_agent/run_react_claude_direct.py)

The agent is given the same tools as the ReAct baseline but `--max-iterations 2` — one peek at the scene, one action commit. No `simulate_action` retries, no `predict_first_contact`, no `describe_scene_geometry` chaining. This is the floor: it bounds how hard the level is when the agent is denied the simulator-feedback loop.

```bash
python scripts/run_direct.py \
    --level pass_the_parcel \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --max-iterations 2 \
    --eval-dir results/direct_pass_the_parcel
```

**Pass-the-parcel result:** 0.0 % (0/50 seeds), avg 1.0 iterations, 0.13 h.

## Comparison

| Baseline | Trials/seed | Iterations | Memory across attempts | PtP acc. |
|---|---:|---:|---|---:|
| Direct | 1 | 2 | none | 0 % |
| ReAct (Claude) | 1 | up to 25 | none | 24 % |
| Reflexion (K=2) | up to 2 | up to 2×25 | verbal reflection between trials | 17.1 %\* |
| HeXA | n/a (per-round eval) | up to 25 | distilled skill bank evolved across rounds | **60 %** |

\* on 35/50 seeds (the rest will be filled in once the run completes).
