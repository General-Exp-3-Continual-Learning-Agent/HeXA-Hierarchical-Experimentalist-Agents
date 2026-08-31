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

**Maze** (config unchanged, still valid):

| env | ep1 | ep3 | any | verdict |
|---|---|---|---|---|
| Maze | .250 -> .219 | .637 -> .617 | .703 -> .641 | **null** (-2.3 vs the n=3 baseline mean .660, inside the ±.031 band) |

**Mastermind: RETRACTED.** `hexa_orbit.mastermind.TASK_CONFIG` was changed to
match the ORBIT paper -- `code_length` 3, `num_numbers` 6 (**120 codes**), from
4/8 (1680 codes). Every Mastermind number produced before that change is
measured on a different task. The "+17.6 ep3 / +16.4 any" result is withdrawn;
it was also compromised independently, its bank having changed in only rounds
0, 8 and 14 of 20 (the `HEXA_TEACHER_MAX_TOKENS=16384` truncation signature).
See `results/_STALE_mastermind_4digit_config/README.md`. Rerunning via
`scripts/run_orbit_mastermind_rerun.sh`.

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

## Pre-registration amendment (2026-08-31, BEFORE any Phase 4 eval)

Design B's headline was the **episode-1 delta over the no-bank baseline**.
Carrying that unchanged to both Design C targets would be a mistake, and the
reason is structural, not result-driven — it is recorded here before any Phase 4
arm has been run.

**Mastermind: SUSPENDED, pending the rerun.** The argument below rests on a
number that has since been retracted, and the task it was measured on no longer
exists. Re-derive from the matched no-bank baseline
(`results/design_c/phase4/mastermind_nobank/run{0,1,2}`) before fixing a primary
metric. Episode 1 is very likely **live** on the new config: 3 guesses with
feedback against 120 candidates is winnable, where 1680 was not, and the
rawdump arm already shows ep1 .125 against an old baseline of .031.

*Superseded reasoning, kept for the record:*
**Mastermind: primary becomes ep3 and `any`; ep1 is a declared non-mover.**
Episode 1 is a blind guess at 1 of 1680 codes with zero feedback available. No
bank can help before any constraint exists. The evidence: Design B's Mastermind
bank, evolved on Mastermind's own trajectories, moved ep1 **.031 -> .027** while
moving ep3 **.367 -> .543**. A metric that a target-trained bank cannot move is
not a test of transfer.

**Maze: primary stays ep1.** Baseline ep1 is .250 with a measured ±.008
run-to-run band, so there is real room and the metric is live.

Secondary on both targets: the full ep1/ep2/ep3/`any` profile, reported whatever
it shows. Paired McNemar exact tests against each baseline run, with the
run-to-run spread as the noise band.

### Predictions, recorded before the runs

| arm | prediction | confidence |
|---|---|---|
| mastermind ← wordle | beats baseline, ep3/any +6 to +14 | ~75% |
| mastermind ← minesweeper | beats baseline, ep3/any +6 to +14 | ~70% |
| mastermind ← minesweeper+wordle | beats baseline, ep3/any +6 to +14 | ~70% |
| maze ← any source | **null**, ep1 within ±.03 | ~75% null / 15% negative / 10% positive |

Basis: coverage of the Design B Mastermind bank (which earned +17.6 ep3) by each
synthesised bank is 4-5 of 5 skills. The one systematic gap is Design B's
highest-confidence skill, *Total Peg Count Pruning* (conf 1.00) — using the
**sum** of black+white pegs to infer digit membership. Every Design C bank
covers only the degenerate sum=0 case. That is the piece that required seeing
real Mastermind trajectories, and synthesis could not invent it.

Note against the plan's framing: `mastermind ← minesweeper` — the **"far"** arm —
has the best structural coverage of the three (5/5, one partial). If it performs,
the near/far axis was mis-specified.

## Phase 4 results — Maze (complete, valid)

Paired on the 243 instances common to all three baseline runs and every arm.
Deltas are min/max across the three baseline runs; p is the best (smallest)
paired McNemar exact p of the three.

| arm | ep1 (primary) | ep2 | ep3 | any |
|---|---|---|---|---|
| baseline run0/1/2 | .239 / .247 / .247 | .391 / .416 / .370 | .621 / .617 / .593 | .658 / .650 / .634 |
| near ← minesweeper | .263 (+1.6/+2.5) p=.54 | +3.7/+8.2 p=.047 | −2.1/+0.8 | −3.7/−1.2 |
| far ← wordle | .300 (+5.3/+6.2) p=.105 | +6.6/+11.1 **p=.008** | +0.8/+3.7 | +2.9/+5.3 |
| **both** | .272 (+2.5/+3.3) p=.41 | **+11.5/+16.0 p<.001** | +2.9/+5.8 | **+6.6/+9.1 p=.020** |
| rawdump ablation | .210 (**−3.7/−2.9**) p=.22 | +3.3/+7.8 p=.045 | −0.8/+2.1 | −1.6/+0.8 |

**On the pre-registered primary (ep1) every Maze arm is null.** That part of the
prediction held. Two parts did not:

1. Maze was predicted null *across the board*. The multi-source arm moves ep2 by
   +11.5/+16.0 (p<.001) and `any` by +6.6/+9.1 (p=.020), and beats both single
   sources on every metric. Multi-source synthesis does something neither source
   does alone -- the strongest argument in the lattice for the Design C
   mechanism, and it was not predicted.
2. The near/far ordering held (wordle > minesweeper on maze) but not only for
   the predicted reason: `maze_near`'s injection was budget-truncated and lost
   *Cross-Episode Map Retention* outright.

**The ablation is the load-bearing result.** Raw-dumping the Minesweeper source
bank gives ep1 −3.7/−2.9; the synthesised bank from the same source gives
+1.6/+2.5. Re-grounding earns its keep, and injecting source skills verbatim
actively hurts. Phase 3's contribution is not unearned.

### Confound to fix before the next Maze pass: the injection budget

Every arm came back `budget_bound`, and **all mistakes were dropped in all six
arms** -- the common-mistakes half of every bank never reached the actor.
Skills are cut lowest-confidence-first, and teacher confidence is
anti-correlated with utility here:

| arm | tokens | skills kept | dropped |
|---|---|---|---|
| maze near | 854 | 5/8 | Cross-Episode Map Retention (.65), Avoid Redundant Moves (.68), Probabilistic Direction Guessing (.60) |
| maze far | 956 | 7/8 | Duplicate Observation Reconciliation (.60) |
| maze both | 972 | 8/8 | — |

`maze_near` lost the one skill aimed at the real bottleneck while keeping the
vacuous *Zero-Propagation Analogy* (.80). These arms tested a top-N-by-confidence
subset, not the banks. Note the multi-source arm is the only one injected whole,
which is a plausible partial explanation of its win and should be controlled for.


## Next

- **Mastermind, everything, is rerunning on the ORBIT-paper config** (3-digit /
  1-6 / 120 codes). Three jobs: matched no-bank baseline x3
  (`run_design_c_mastermind_baseline.sh`), Design C Phase 3+4
  (`run_design_c_mastermind_redo.sh`), Design B re-evolution
  (`run_orbit_mastermind_rerun.sh`, teacher budget 32768).
- Re-derive the Mastermind primary metric from the new baseline; do not carry
  the suspended amendment forward.
- Decide whether to re-run the Maze arms with a larger `--injection-budget-tokens`
  so the banks are tested whole. As run, the budget is a confound that favours
  the multi-source arm.
- Mastermind eval is capped at `MastermindTask.EVAL_POOL_SIZE = 40`. That is thin
  for the ~10pp effects at stake; consider whether the ORBIT paper's held-out
  size should be matched here too.
