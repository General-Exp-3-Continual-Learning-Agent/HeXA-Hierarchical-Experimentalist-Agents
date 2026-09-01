# Minesweeper HExA run2 — bank invented a nonexistent action

Archived, not deleted. Superseded by a rerun on a corrected teacher description.

## What happened

The evolved bank contained a skill about a **`terminate` action that does not
exist** in `game:Minesweeper-v0-only-reveal`:

> [Strategy: Terminate only when board solved] The agent should only issue the
> terminate action when all safe cells have been revealed.

The actor emitted batched actions ending in `\boxed{terminate}`. Only the LAST
`\boxed{...}` is read, so every such turn resolved to `terminate`, the env
answered *"you did not provide a valid guess"*, and the episode ended.

| | run0 | run1 | **run2** |
|---|---|---|---|
| turns whose last box is `terminate` | 0 | 0 | **461 / 2267 (20.3%)** |
| env "did not provide a valid guess" | 18 | 47 | **461** |
| win messages | 256 | 290 | **20** |
| `any` | .551 | .562 | **.051** |
| steps / instance | 12.90 | 13.50 | **8.86** |

461 ↔ 461 exactly.

## Why nothing caught it

`unparsed_rate` for run2 was **0.0000** — *lower* than both healthy runs (.0048,
.0130). The metric answers "did we extract a `\boxed{}`", not "was the action
accepted". `\boxed{terminate}` parses perfectly; the environment rejects it.
So unparsed_rate is not a usable health check for invented actions.

This is D-C7's failure mode again — an invalid action silently ending the
episode — but caused by the bank hallucinating an action rather than by missing
instructions in the prompt.

## Root cause and fix

`TASK_DESCRIPTIONS["minesweeper"]` in `hexa_orbit/source_teacher.py` described
the mechanics but **never enumerated the valid action set**, so the teacher had
no way to know `terminate` was not available. It now states the actions
exhaustively, that there is no terminate/quit/stop/done/submit/unflag, that an
invalid action ends the episode, and that only the last `\boxed{}` is read.
Guarded by `test_minesweeper_description_enumerates_the_action_set` and
`test_detects_a_bank_that_invents_terminate`.

## Scope note, important

The fix changes the teacher prompt. Runs 0 and 1 were produced under the OLD
description, so a rerun of run2 alone yields three runs from two different
prompts. Runs 0 and 1 should be rerun on the corrected description before the
three are reported together.
