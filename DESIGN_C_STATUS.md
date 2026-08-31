# Design C — Cross-Task Skill Transfer (ORBIT)

`K_ℓ* = π_ev(K_ℓ1, …, K_ℓS, description(ℓ*))` — synthesise a bank for a held-out
target from source banks alone, with **zero target trajectories**.
Actor and teacher are the same model (gpt-oss-120b). This is self-improvement,
not distillation.

Last updated 2026-08-31 (end of Phase 3).

## Decisions

| id | decision |
|---|---|
| D-C1 | Pin the Minesweeper opening probe at (2,2). Only 8.3% of openings reproduce a given seed's layout (12.8 layouts/seed); pinning makes the layout stable across episodes, which is what T=3 adaptation requires. |
| D-C2 | nltk 3.9.1 installed for the Wordle corpus. `ensure_wordle_corpus()` no-ops after first load. |
| D-C3 | Wordle vocabulary is 192 words with 96.5% eval/train secret overlap — that is D1 again. Partition by secret word: 60 eval / 132 train. |
| D-C6 | Keep Minesweeper + Wordle-5 as sources only. Wordle-10 saturates (`any`=.994, zero failures per round) so it carries no learning signal. |
| D-C7 | `_reset_inner` discarded the env's reset text, the only statement of the `\boxed{reveal R C}` format. 760/768 episodes died on the agent's first move. Fixed by concatenating instructions with the post-opening observation. |
| D-C9 | n=1 for treatment arms (user decision, 2026-08-30). Variance comes from the already-replicated baselines. Replicate only if Design C works. |

## Phase 2 — source gates (both PASS)

Sanity checks, not headlines.

| env | ep1 | ep3 | any |
|---|---|---|---|
| Minesweeper (n=256) | .208 → **.301** (+9.3, p=.0005/.0092/.045 vs all 3 baseline runs) | .301 → .359 | .432 → **.551** (+14.8/+9.8/+10.9, all p<.005) |
| Wordle-5 (n=60×3) | −2.8 (null) | **+10.0** (p=.0021) | +2.2 (null) |

Minesweeper is the first bank in this project to move **episode 1**.

Measured noise bands (baseline run-to-run spread): Maze ep1 ±.008 / any ±.031;
Minesweeper ep1 ±.043 / any ±.051; Wordle-5 ep1 ±.050.

## Design B targets, for reference

| env | ep1 | ep3 | any | verdict |
|---|---|---|---|---|
| Mastermind | .031 → .027 | .367 → **.543** | .379 → **.543** | positive, ep3/any only |
| Maze | .250 → .219 | .637 → .617 | .703 → .641 | **null** (−2.3 vs the n=3 baseline mean .660, inside the ±.031 band) |

## Phase 3 — synthesis lattice (complete, all 6 gates pass)

| target | arm | sources | gate |
|---|---|---|---|
| maze | near | minesweeper | pass |
| maze | far | wordle | pass |
| maze | both | minesweeper+wordle | pass (soft flag: "revealed", ordinary English) |
| mastermind | near | wordle | pass |
| mastermind | far | minesweeper | pass **after a recorded manual de-leak** |
| mastermind | both | minesweeper+wordle | pass |

### Gate fix, 2026-08-31

The first pass blocked 2 banks. Two of the three hits were substring false
positives — `reveal` matched "**reveal**ed", `mine` matched "exa**mine**".
`inspect_cross_task_bank.py` now matches on word boundaries with an inflection
suffix, and splits the vocabulary:

- **HARD** (no innocent sense in the target; blocks): minesweeper `mine, flag,
  cascade, adjacent count`; wordle `vowel, consonant, green, yellow, g/y/x, letter`
- **SOFT** (also ordinary English; reports only): minesweeper `board, grid,
  reveal, probe`; wordle `word, spell`

One real leak survived and was hand-edited (`mastermind_from_minesweeper`,
`mas_xl_err_003.why_it_happens` referenced Minesweeper's flag action). Edited
rather than re-synthesised so the arm keeps a single draw not selected on the
gate. Original preserved as `skill_bank_xl_mastermind_PRE_DELEAK.json`;
the edit is recorded in that run's `provenance.json`.

### Inverted premises to hold against Phase 4

Flagged for reading, not rejected:

- `maze ← minesweeper`: **Probabilistic Direction Guessing** (conf .60) picks the
  direction with "highest expected gain (i.e. lowest estimated wall probability)".
  Frontier exploration is sound for Maze; equating it with wall probability is
  Minesweeper residue, and Maze returns no counts to estimate from.
- `maze ← minesweeper+wordle`: **Probabilistic Safe Move Selection** says
  "assign each a simple risk estimate… choose the direction with the lowest
  estimated risk." That is the stall pattern that cost the Design B maze arm
  15.2 points, transplanted whole. Worst inverted-premise exposure in the lattice.
- The other flags (`guarantee`, `safe`) fire on statements that are legitimately
  true in Maze — the goal really does guarantee success. Not inversions.

## Pre-registered prediction (recorded before Phase 4)

**Minesweeper → Mastermind (the "far" arm) will beat Minesweeper → Maze (the
"near" arm).** Minesweeper's *Deterministic Logical Inference* maps onto peg
arithmetic and onto nothing in Maze. Skill-by-skill:

- Wordle → Mastermind: 4 of 6 transfer directly (Cumulative Constraint
  Management *is* Mastermind's winning rule verbatim), 1 inverted (Wordle names
  positions, Mastermind returns counts only), 1 inapplicable (duplicate letters —
  our config forbids duplicate digits).
- Minesweeper → Maze: only 2 of 5 transfer, 2 have no analogue, 1 inverted.

If the prediction holds, the near/far axis is really a **mechanism-overlap**
axis, and the plan paired the arms on surface similarity.

## Are the synthesised skills useful? (bank audit, 2026-08-31)

Each bank's advice measured against what the baseline actor **actually does
wrong**. A skill is only worth prompt tokens if it targets a real error rate.

### Mastermind — yes. The banks target the dominant failure.

| baseline behaviour | rate | addressed by |
|---|---|---|
| guess **logically inconsistent** with feedback already received | **53.9%** (1143/2120) | Deterministic Constraint Propagation, Global Constraint Accumulation, Enforce Known Blacks and Whites, Positional Consistency Check |
| repeats an earlier guess | **14.5%** (307/2120) | Avoid Redundant Guesses, Persist Solution Across Episodes |
| illegal duplicate-digit guess | 0.0% | — already solved |

Over half of all guesses contradict information the agent already has. That is a
large, real, and precisely addressable gap, and it is what three of the top four
skills in every Mastermind bank aim at. This also explains Design B's Mastermind
result (ep3 +17.6, ep1 flat): constraint discipline can only pay once
constraints exist, which is episode 2 onward.

**One defect:** `mastermind ← minesweeper+wordle` carries *Duplicate-Digit Count
Reasoning* at conf 0.85 — but the env states "Duplicate numbers are not allowed"
and the baseline emits zero duplicate guesses. Dead weight in the highest-value
bank, and mildly misleading. Wordle's duplicate-letter skill survived
re-grounding it should not have survived.

### Maze — largely no. The banks target an error the actor does not make.

| baseline behaviour | rate | addressed by |
|---|---|---|
| moves into a **known** wall | **0.9%** (55/6275) | appears in **all three** banks (conf .85/.68/.68) *and* as a common mistake in all three |
| adjacent cell reported as `goal` | 1.5% of labels (87/5880) | Immediate Goal Capture (.8), Goal-Adjacent Immediate Move (.65/.85) |
| ep2 already optimal when ep1 succeeded | 57% (35/61), mean +0.70 steps over optimal | Cross-Episode Path Reuse (.68/.85) |
| immediate reversal (move, then move back) | 9.8% (635/6510) | Avoid Redundant Backtracking — the one skill aimed at a live error |

The single most repeated instruction across all three maze banks addresses a
0.9% failure rate. The skills are *correct*; they are not *binding*.

**What the maze banks never say, and it is the actual bottleneck:** the
per-episode cap is 9 steps against a mean optimal path of 7.46. Never-solved
instances (83/256) run 9.0/9.0/9.0 — every episode capped — with mean
`best_progress` 0.507 and **zero improvement across episodes**. Winning the hard
instances requires treating episode 1 as a scouting run that is expected to
fail, mapping aggressively, then exploiting in episodes 2-3. No source bank
suggests this because neither source has the budget structure that makes it
necessary: Minesweeper's episode *ends* on a wrong probe, and Wordle gets 5
guesses against a 192-word vocabulary. The premise that generates the right
maze strategy does not exist in either source, so synthesis could not have
produced it.

### Consequence for the prediction

The **target** axis dominates the near/far axis. Expect the Mastermind arms to
move and the Maze arms to be null *regardless of source*. This subsumes and
sharpens the earlier prediction (Minesweeper→Mastermind beats
Minesweeper→Maze): the reason is not source-target distance but whether the
target's dominant failure is one the source banks have vocabulary for.

## Next

Phase 4 — zero-shot eval, 6 arms, both targets. **Episode-1 delta over the
no-bank baseline is the number that matters most**, same as Design B.
Open: whether to add the 2 raw-dump ablation arms, and whether to raise
concurrency to 24–32 (the server was underutilised at 6 — ~650 tok/s aggregate,
19 s/instance).
