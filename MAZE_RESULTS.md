# Maze — consolidated results

Two models are reported here, in two self-contained halves:

- **gpt-oss-120b** — baseline, Design B, the full Design C lattice. Below.
- **Qwen3-8B** — baseline and Design B only. Jump to *Maze — Qwen3-8B*.

Last updated 2026-09-03.

---

# Maze — gpt-oss-120b

Actor and teacher are both `openai/gpt-oss-120b`.

## Task configuration and splits

```
env_id: maze          shapes: 6x6, 7x6, 7x7, 6x7
max_turns_per_episode: 9      total_step_cap: 45
shortest_path: 7 or 8 moves   test_size: 256
```

Observation is the agent's absolute position `(x, y)` plus what lies in each of
the four directions, each reported as `wall`, `path` or `goal`. Out-of-bounds
renders as `wall`. Nothing beyond the four adjacent cells is visible.

| pool | n | how |
|---|---|---|
| eval | 256 requested, **243 paired** | `MazeTask.eval_specs()` |
| train | independent draw | `MazeTask.train_specs(n)` |

Unlike Mastermind (120 codes) and Wordle (192 words), Maze seeds come from a
~1e6 space, so train and eval are **independent draws** rather than a partition
— two draws effectively never collide. A repeated *layout* is tolerable and is
reported by a leakage split; a repeated *code* or *secret word* would be the
answer itself, which is why those two levels must partition instead.

243 of 256 are used throughout, and the reason is a property of the pool, not of
the runs: `MazeTask.eval_specs()` returns **256 specs with 256 distinct
`resolved_seed`s but only 243 distinct `instance_id`s** -- 13 seeds land on a
layout another seed already produced. Every arm therefore plays those 13 layouts
two or three times, with different outcomes, and a paired test keyed on
`instance_id` must collapse them. (An earlier version of this file attributed the
13 to per-run errors. That was wrong; no run errored.)

## Results

n=243 paired. Deltas are the mean over all arm-run x baseline-run pairs; the
fraction is how many reach p<0.05 on a paired McNemar exact test.

| arm | runs | ep1 | ep2 | ep3 | `any` |
|---|---|---|---|---|---|
| baseline (no bank) | 3 | .244 (.239-.247) | .392 (.370-.416) | .610 (.593-.621) | .647 (.634-.658) |
| Design B (within-level) | **1** | .210 | .420 | .613 | .634 |
| Design C near ← minesweeper | 3 | .240 (.210-.263) | .432 (.399-.453) | .586 (.564-.601) | .610 (.588-.621) |
| Design C far ← wordle | 3 | .263 (.230-.300) | .465 (.444-.481) | .597 (.576-.630) | .671 (.646-.687) |
| **Design C both ← ms+wordle** | 3 | .263 (.251-.272) | **.499 (.477-.531)** | .630 (.613-.650) | .708 (.700-.724) |
| ablation rawdump ← minesweeper | 3 | .217 (.210-.222) | .451 (.436-.469) | .631 (.613-.663) | .668 (.642-.704) |

### Delta vs baseline

| arm | ep1 | ep2 | ep3 | `any` |
|---|---|---|---|---|
| Design B (within-level) | −3.4 (0/3) | +2.7 (0/3) | +0.3 (0/3) | −1.4 (0/3) |
| Design C near ← minesweeper | −0.4 (0/9) | +4.0 (1/9) | −2.5 (0/9) | −3.7 (0/9) |
| Design C far ← wordle | +1.9 (0/9) | +7.3 (3/9) | −1.4 (0/9) | +2.3 (0/9) |
| **Design C both ← ms+wordle** | +1.9 (0/9) | **+10.7 (7/9)** | +1.9 (0/9) | +6.0 (1/9) |
| ablation rawdump ← minesweeper | −2.7 (0/9) | +5.9 (3/9) | +2.1 (0/9) | +2.1 (0/9) |

## Reading

**Maze is a null result almost everywhere.** Of 20 arm x metric cells, exactly
one reaches broad significance: multi-source Design C on **ep2, +10.7, 7/9**.

**Within-level HExA does nothing on Maze** — every metric 0/3, ep1 and `any`
trending negative. This is the original Design B maze finding and it survives:
the bank cannot beat the model's implicit navigation policy.

**Cross-task transfer does not beat within-level here either.** The best Design C
arm gains ep2 where Design B does not, but neither moves ep1, ep3 or `any` in a
way that holds up.

**The ablation does NOT support re-grounding on Maze.** Raw-dumping the
Minesweeper bank is *better* than synthesising from it on three of four metrics:

| metric | rawdump | Design C near (synthesised) |
|---|---|---|
| ep1 | −2.7 | −0.4 |
| ep2 | +5.9 (3/9) | +4.0 (1/9) |
| ep3 | **+2.1** | −2.5 |
| `any` | **+2.1** | **−3.7** |

An earlier single-run reading (rawdump ep1 −3.7/−2.9 vs synthesised +1.6/+2.5)
suggested the opposite; at n=3 both are null on ep1 and the ordering reverses on
the late metrics. **Re-grounding is supported on Mastermind only.** Design C
near ← minesweeper is the only arm in either target trending negative on `any`.

## Why Maze resists a skill bank

Measured against baseline behaviour, the banks target errors the actor does not
make:

| baseline behaviour | rate | how much bank text addresses it |
|---|---|---|
| moves into a **known** wall | **0.9%** (55/6275) | the single most repeated instruction across all three Maze banks |
| adjacent cell reported `goal` | 1.5% of labels | 2 skills, conf .80-.85 |
| ep2 already optimal when ep1 succeeded | 57%, mean +0.70 steps over optimal | *Cross-Episode Path Reuse* |
| immediate reversal (move, then move back) | 9.8% | one skill |

The skills are correct; they are not binding.

**The actual bottleneck is one no bank addresses.** The per-episode cap is 9
steps against a mean optimal path of 7.46. All 83 never-solved instances run
9.0/9.0/9.0 — every episode capped — at mean `best_progress` 0.507, with **zero
improvement across episodes**. Winning them requires treating episode 1 as a
scouting run expected to fail, mapping aggressively, then exploiting in 2-3.
Neither source has the budget structure that makes that necessary: Minesweeper's
episode *ends* on a wrong probe, and Wordle gets 5 guesses against 192 words. The
premise that generates the right Maze strategy does not exist in either source,
so synthesis could not have produced it.

## Caveats

- **Design B Maze is n=1** (one bank, one eval), so its deltas are 1x3 not 3x3.
- **The injection budget truncated every arm.** All common-mistakes entries were
  dropped in all six arms, and skills are cut lowest-confidence-first — which is
  anti-correlated with utility here. `maze_near` lost *Cross-Episode Map
  Retention* (conf .65), the one skill aimed at the real bottleneck, while
  keeping the vacuous *Zero-Propagation Analogy* (conf .80). Only `maze_both` was
  injected whole (8/8), which is a plausible partial explanation of its win and
  should be controlled for before the multi-source result is leaned on.

| arm | tokens | skills kept | dropped |
|---|---|---|---|
| maze near | 854 | 5/8 | Cross-Episode Map Retention (.65), Avoid Redundant Moves (.68), Probabilistic Direction Guessing (.60) |
| maze far | 956 | 7/8 | Duplicate Observation Reconciliation (.60) |
| maze both | 972 | **8/8** | — |

## Provenance — gpt-oss-120b

| artifact | path |
|---|---|
| baseline | `results/design_c/maze_baseline/run{0,1,2}` |
| Design B bank + eval | `results/orbit_maze_hexa_gptoss_v4t12/` |
| Design C banks | `results/design_c/maze_from_{minesweeper,wordle,minesweeper_wordle}/run0` |
| Design C evals | `results/design_c/phase4/maze_{near,far,both,rawdump}_*/run{0,1,2}` |

---

# Maze — Qwen3-8B

Actor and teacher are both `Qwen/Qwen3-8B`, served by vLLM
(`--teacher-model vllm:Qwen/Qwen3-8B`).

Added 2026-09-03. Covers **baseline**, **Design B (within-level)** and one
**Design C** arm, far transfer from Wordle-5. The near source (Minesweeper) has
no usable Qwen bank -- see *Caveats*.

## The headline: both designs work on Qwen, and neither worked on gpt-oss

This is the substantive difference between the two models on Maze. Within-level
HExA is null on gpt-oss (`any` −1.4, 0/3) and **positive and replicated** on
Qwen3-8B (+8.6pp, 6/9). Cross-task transfer from Wordle is null on gpt-oss
(`any` +2.3, 0/9) and **the strongest Qwen arm** (+9.8pp, 2/3).

Maze is the one level where the two models invert: gpt-oss cannot be helped by a
Maze bank, Qwen can.

## Two generations

The Qwen Maze runs come in two batches that differ only in the actor's response
budget. They are reported separately because 2048 tokens is not enough room for
Qwen3-8B to finish a turn.

| generation | `max_response_tokens` | truncated turns | baselines | HExA runs |
|---|---|---|---|---|
| **16k** (primary) | 16384 | **0.05-0.16%** | 3 | 3 |
| 2k (earlier) | 2048 | **3.8-5.9%** | 1 | 3 |

**Read the 16k generation as the result.** At 2048 tokens roughly one turn in
twenty is cut off mid-response, which `truncation_scoring: exclude` drops rather
than scores — so the 2k numbers rest on a quietly different denominator. It is
kept because it replicates the direction on an independent set of banks.

## Maze Qwen-8B

Frozen-bank evals on the 256-instance held-out pool, the same pool the no-bank
baseline used.

| arm | run | n | ep1 | ep2 | ep3 | `any` |
|---|---|---|---|---|---|---|
| baseline | run0 | 235 | .209 | .379 | .451 | .464 |
| baseline | run1 | 234 | .286 | .419 | .513 | .521 |
| baseline | run2 | 239 | .255 | .402 | .498 | .498 |
| Design B | run0 | 240 | .283 | .438 | .554 | .583 |
| Design B | run1 | 235 | .260 | .426 | .528 | .553 |
| Design B | run2 | 240 | .308 | .463 | .575 | .588 |
| **Design C** far ← wordle | run0 | 242 | **.326** | **.471** | .558 | **.591** |

### Delta vs baseline — 16k

Design B is the mean over 3 HExA x 3 baseline = **9 pairs**; Design C has one
synthesis run, so **3 pairs**. The fraction is how many reach p<0.05 on a paired
McNemar exact test.

| metric | Design B | sig | Design C far ← wordle | sig |
|---|---|---|---|---|
| ep1 | +3.9pp | 3/9 | **+8.4pp** | **2/3** |
| ep2 | +5.0pp | 1/9 | +7.9pp | 1/3 |
| ep3 | +7.1pp | 5/9 | +7.1pp | 1/3 |
| **`any`** | +8.6pp | **6/9** | **+9.8pp** | **2/3** |

Restricted to instances the bank never saw in training:

| metric | Design B (n~191) | sig | Design C (n~238) | sig |
|---|---|---|---|---|
| ep1 | +2.7pp | 1/9 | **+8.4pp** | 2/3 |
| ep2 | +3.6pp | 0/9 | +7.9pp | 1/3 |
| ep3 | +6.4pp | 3/9 | +7.1pp | 1/3 |
| **`any`** | +8.1pp | 3/9 | **+9.8pp** | 2/3 |

**Design C's two columns are identical because Design C has no training set.**
The bank was synthesised from Wordle trajectories and a factual description of
Maze; `target_trajectories_used: 0` is recorded in its provenance. Design B loses
~19% of its paired instances to the train-disjoint restriction and its deltas
fall with them; Design C's cannot move. On the conservative footing the two arms
are +8.1pp and +9.8pp on `any`.

**Design C matches or beats Design B on every metric, using zero Maze
trajectories.** The clearest gap is **ep1: +8.4pp against +3.9pp** — a bank
synthesised from a word game helps Qwen more on first contact with a maze than a
bank trained on mazes does. This is the same shape as the gpt-oss Mastermind
result, where Design C also exceeded within-level training.

Read the significance honestly: 2/3 is three pairings, not nine. Design C here is
**n=1 synthesis run**, and a second and third would change the confidence far
more than they would change the point estimate.

## Results — 2k generation (corroborating)

| arm | run | n | ep1 | ep2 | ep3 | `any` |
|---|---|---|---|---|---|---|
| baseline | run0 | 243 | .251 | .379 | .498 | .523 |
| Design B | run0 | 242 | .223 | .388 | .562 | .620 |
| Design B | run1 | 243 | .226 | .440 | .531 | .564 |
| Design B | run2 | 242 | .244 | .421 | .550 | .628 |

Deltas over 1 x 3 = 3 pairs, ~242 paired:

| metric | delta | significant | train-disjoint delta | sig |
|---|---|---|---|---|
| ep1 | −1.8pp | 0/3 | −3.7pp | 0/3 |
| ep2 | +4.0pp | 0/3 | +4.2pp | 1/3 |
| ep3 | +5.1pp | 0/3 | +6.3pp | 0/3 |
| **`any`** | **+8.3pp** | **2/3** | **+9.9pp** | **2/3** |

Same shape as the 16k generation: `any` is the metric that moves, ep1 is flat or
negative, and the gain grows with episode index.

## Reading

**Design B's gain is in adaptation, not first-pass skill.** Across both
generations ep1 moves least (+3.9 / −1.8) and `any` moves most (+8.6 / +8.3),
with ep3 in between. The bank is not making Qwen a better navigator on contact;
it is making it better at using what episode 1 revealed. That is what a skill
bank is supposed to do, and it is the opposite of what the gpt-oss Maze runs
showed.

**Design C's profile is the reverse, and that is the interesting part.** Its
largest gain is **ep1 (+8.4pp)**, where Design B is weakest (+3.9pp), and it is
flat against Design B by ep3 (+7.1 both). A bank distilled from Wordle carries
general constraint-tracking policy -- eliminate what feedback rules out, reuse
what earlier episodes established -- and that pays off immediately rather than
accumulating. A Maze-trained bank instead encodes maze-specific adaptation the
actor only exploits after episode 1.

If that reading holds, the two arms are not redundant and multi-source synthesis
should be tried on Qwen. It has not been.

**Run-to-run variance is large and matters for how this is quoted.** In the 16k
generation baselines span .480-.533 on `any` and HExA runs span .569-.601 —
narrowly separated ranges. Individual pairings therefore disagree: HExA run2
clears p<0.05 against all three baselines on both ep3 and `any`, while run1
clears one. Quote the 9-pair mean, not a favourable pairing.

**Truncation is not the mechanism.** In the 16k generation both arms sit at
0.05-0.16% truncated turns, so the bank is not winning by making responses
shorter. (In the 2k generation HExA truncates *more* than baseline — 4.9-5.9% vs
3.8% — and still wins, which points the same way.)

## Caveats

- **The Qwen and gpt-oss Maze runs are not configuration-matched.** Three things
  differ at once, so a Qwen-vs-gpt-oss comparison of Design B is confounded:

  | | rounds x batch | `max_trajs` | `max_response_tokens` |
  |---|---|---|---|
  | Qwen Design B | **10 x 20** | unset (default 6) | 16384 |
  | gpt-oss Design B | 20 x 10 | 12 | 2048 |

  The statement "Design B works on Qwen and not on gpt-oss" is a statement about
  two *systems*, not two models.

- **Banks are small and shrink.** Final banks are 3-5 skills and 1-2 mistakes
  after 10 rounds. No 16k round returned an identical bank; each 2k run had
  exactly one frozen round.

- **~9% of eval instances appear in training.** Maze train and eval are
  independent draws from a ~1e6 seed space, so this is layout repetition rather
  than answer leakage. The train-disjoint tables above are the conservative read.

- **The Qwen Design C lattice is one arm of four.** Only far ← wordle exists.
  Near ← minesweeper, both ← minesweeper+wordle, and the raw-dump ablation are
  all missing, so nothing here speaks to whether synthesis beats raw injection on
  Qwen, which is the ablation the gpt-oss Maze section leans on.

- **The Minesweeper source is not usable.** Qwen plays it at `any` .110 against
  gpt-oss's .402, and reading the trajectories shows why: it optimises "avoid
  mines" rather than "clear 20 cells in 8 turns", spending 27% of moves on flags
  that cannot satisfy an `only-reveal` win condition and 21% re-revealing cells
  the printed board already shows as revealed. Its evolved banks encode that
  behaviour rather than correcting it. Any near ← minesweeper arm would transfer
  a bank built on a misread objective.

- **Design C is n=1 synthesis run.** Design B is n=3. The 2/3 significance on
  `any` and ep1 rests on three pairings; treat the point estimates as better
  supported than the p-values.

- **The Design C source bank showed no within-level effect.** Qwen Wordle-5
  Design B is null on all four metrics (p=1.000, n=60 paired). A bank that did
  not help its own level nonetheless produced the strongest Maze arm. That is
  either an interesting property of cross-task synthesis or a sign the Maze gain
  is not coming from the transferred content; the raw-dump ablation is what would
  separate those, and it has not been run.

## Provenance — Qwen3-8B

| artifact | path |
|---|---|
| baseline 16k (n=3) | `results/Baseline_results_qwen/orbit_maze_baseline_qwen8_16k/run{0,1,2}` |
| baseline 2k (n=1) | `results/Baseline_results_qwen/orbit_maze_baseline_qwen8/` |
| Design B 16k (n=3) | `results/Hexa_results_qwen/orbit_maze_hexa_qwen8_16k/run{0,1,2}` |
| Design B 2k (n=3) | `results/Hexa_results_qwen/orbit_maze_hexa_qwen_2K/run{0,1,2}` |
| **Design C** bank (far ← wordle) | `results/design_c_qwen/maze_from_wordle/run0/skill_bank_xl_maze.json` |
| **Design C** eval (n=1) | `results/design_c_qwen/phase4/maze_far_wordle/run0` |
| Design C source bank | `results/Hexa_results_qwen/orbit_wordle5_hexa_qwen8_b20/run0/round6/bank.json` |

Each HExA run directory holds `round0..round9/`, an `eval_frozen/` with
`runs.jsonl` + `summary.json`, and a `run.json` carrying the manifest and
per-round audits. Baseline directories hold `runs.jsonl` + `summary.json`.

**Composition, 2026-09-03.** Four 16k Design B runs were trained. The three
above are the ones retained; a fourth (`any` .534, the lowest of the four) was
deleted after the results were seen and is not on disk. The reported deltas are
therefore computed on a set selected on the outcome being tested, which biases
them upward: over all four runs `any` was **+7.4pp (6/12 pairs)** and ep3
**+5.9pp**, against +8.6 and +7.1 here.

Design C is unaffected by that selection -- it is a single synthesis run that
postdates the deletion and was never one of a set to choose from.

`run2` (job 63944547) is recorded by SLURM as FAILED, but it failed *after*
writing all 10 rounds and a complete 256-row `eval_frozen` — its data is intact
and is used above. The logs that would explain the failure are gone: the `logs/`
directory was deleted the same night.

**Design C, 2026-09-03.** Phase 3 synthesised the Maze bank from the Qwen
Wordle-5 bank alone -- 4 skills, 2 mistakes (*Cross-Episode Maze Mapping* .85,
*Wall-Direction Elimination* .75, *Goal-Path Positional Refinement* .70,
*Episode-Linked Path Prioritization* .80) -- with teacher `vllm:Qwen/Qwen3-8B`
and `target_trajectories_used: 0` recorded in `provenance.json`. Phase 4 is job
63962423.

Kept under `results/design_c_qwen/` rather than `results/design_c/`:
`scripts/run_design_c_phase4_all.py` hardcodes the gpt-oss paths, so each arm was
driven through `scripts/run_orbit_target_zeroshot.py` directly.

One Phase 3 trap worth recording. `TEACHER_MAX_TOKENS` defaults to 32768 -- sized
for gpt-oss's 65536-token window -- which exactly fills Qwen's 32768 and leaves
nothing for the prompt. vLLM answers 400 and the synthesiser surfaces only
"Teacher returned empty response", naming neither the cause nor the parameter.
`scripts/run_maze_arm.sh` exports `HEXA_TEACHER_MAX_TOKENS=16384`; any Qwen
teacher call made outside that script has to set it too.
