# HeXA: the headline method

HeXA solves InterPhyre puzzles by maintaining a small, curated bank of physics **skills** (strategy patterns) and **mistakes** (counterfactual lessons). The bank is updated each round by a teacher LLM that sees both the current bank and a fresh batch of agent rollouts. The agent reads the bank at solve time as part of its system prompt.

## Anatomy

```
                Round 1                 Round 2                 Round 3
              ┌─────────┐              ┌─────────┐              ┌─────────┐
   offline    │ Distill │   bank₁      │ Evolve  │   bank₂      │ Evolve  │
   trajs ───▶ │ (T)     │ ─────▶ run ─▶│ (T)     │ ─────▶ run ─▶│ (T)     │ ─▶ ...
              └─────────┘   on x=3     └─────────┘   on x=3     └─────────┘
                            new seeds                new seeds
```

- **T** = teacher LLM (Claude Sonnet by default; configurable via `--teacher-model`)
- **x** = `--seeds-per-round` (default `3`); each round runs the agent on `x` fresh seeds
- **bank capacity** = `MAX_SKILLS_PER_LEVEL` in [skillrl/core/config.py](../skillrl/core/config.py) (default `10`)

The full evolving loop is in [skillrl/loops/evolving_loop.py](../skillrl/loops/evolving_loop.py); the round-1 distillation step is `run_distillation` in [skillrl/distillation/distill.py](../skillrl/distillation/distill.py); the rounds-2+ evolution step is `evolve_skill_bank` in [skillrl/distillation/evolving_distill.py](../skillrl/distillation/evolving_distill.py).

## Skill format

Each skill carries enough metadata to be sorted, retrieved, and audited:

```json
{
  "skill_id": "pas_sk_000",
  "title": "Ramp Launch: Upper-Mid Zone Placement",
  "principle": "Place ball at upper-mid zone on ramp to achieve optimal arc...",
  "when_to_apply": "When the ramp is in launching position and basket is 2+ units away",
  "example": "(x=3.5, y=4.0, r=0.5)",
  "source_seeds": [1, 5, 7],
  "confidence": 0.95,
  "generation": 0,
  "is_new": false
}
```

Mistakes share the same envelope but with `description / why_it_happens / how_to_avoid` instead of principle/example. Confidence is calibrated from the rewards of the trajectories in `source_seeds` — see [skillrl/SKILLRL_SUMMARY.md](../skillrl/SKILLRL_SUMMARY.md#confidence-scores--reward-calculation) for the exact formula.

## Round 1 — distillation

The teacher reads the top-5 successes and top-5 failures (ranked by reward) and emits a fresh bank. Phases inside the teacher prompt:

1. **From successes → strategy skills** ("On a catapult, drop the red ball on the longer lever arm…")
2. **From failures → counterfactual lessons** ("Reducing red ball radius when the arc is too flat makes things *worse*, not better…")

The lessons are usually higher value — they encode the specific broken causal beliefs that waste the agent's iteration budget.

## Rounds 2+ — evolution

Round 2 onwards, the teacher is shown the **current bank with confidence values** plus the new round trajectories, and asked to:

- **Keep** skills the new evidence still supports (preserve confidence)
- **Update** skills that need re-grounding
- **Add** novel skills from new patterns
- **Remove** skills the new evidence contradicts
- **Stay under the budget**: `MAX_SKILLS_PER_LEVEL` (default 10)

This is the "evolve" rule. Each emitted skill carries `is_new: bool` and a `generation` number, giving you a lineage you can plot. Compare with the **iterative** rule (re-distill from scratch each round, ignore the existing bank) — see [docs/CONFIGS.md](CONFIGS.md).

## At solve time

`augmented_runner.py` retrieves the top-K specific skills (sorted by confidence), the top-N general skills, and up to M mistakes, and injects them into the agent's system prompt under a `LEARNED PHYSICS SKILLS` block (and a `MISTAKES TO AVOID` block). The agent itself (`ReactAgent.solve()` in [react_agent/react_agent.py](../react_agent/react_agent.py)) is otherwise unchanged.

The retrieval defaults are configurable on the loop CLI:
- `--max-specific-skills` (default 6)
- `--max-general-skills` (default 0; HeXA disables cross-level skills by default for the single-level setting)
- `--max-mistakes-agent` (default 4)

## Why it works

1. **Memory across episodes** — base ReAct restarts from scratch each seed. HeXA carries forward distilled lessons.
2. **Counterfactuals beat raw replay** — "don't shrink the ball, move x_drop" is more compact and more actionable than the noisy 25-step trajectory it came from.
3. **Bounded growth** — capping the bank at 10 skills forces the teacher to choose, which removes redundancy and shortens the agent's prompt.
4. **Off2On vs pure online** — starting from a small offline trajectory set (we use 6 seeds in `Initial_trajectories/`) gives round 1 a meaningful bank to work with; pure online configurations spend round 1 collecting raw data.

## See also

- [docs/CONFIGS.md](CONFIGS.md) for the variant matrix (Init × Update rule)
- [docs/REPRO.md](REPRO.md) for the headline command and seed lists
- [skillrl/SKILLRL_SUMMARY.md](../skillrl/SKILLRL_SUMMARY.md) — long-form notes including the v1/v2 comparison and confidence-calibration walkthrough
