# Mastermind — consolidated results

Two base models, reported separately: `openai/gpt-oss-120b` (most of this
document) and `Qwen/Qwen3-8B` (final section). Within each, actor and teacher are
the **same** model — HExA is self-improvement, not distillation, so these are
never different models.

Both share the task configuration and 40/80 split below.

Last updated 2026-09-01.

## Task configuration

ORBIT's published eval parameters, read off `configs/eval_config.yaml` in the
ORBIT checkout. The prompt this produces is character-identical to Appendix A of
the paper.

```yaml
env_id: "game:Mastermind-v0-random"
code_length: 3          # \boxed{d d d}
num_numbers: 6          # digits 1-6
duplicate_numbers: False
max_turns_per_episode: 3
total_step_cap: 9       # 3 episodes x 3 turns
```

All three of `code_length`/`num_numbers`/`max_turns` must be pinned. GEM's
`-random` variant sets `_is_random` if *any* is `None` and then overwrites all
three from `[(2,4,10), (4,6,20), (4,8,30)]`. Against `total_step_cap: 9` an
inner cap of 10-30 means the outer loop always ends first, so ORBIT's config as
published yields **one episode, never three**.

## Train / eval split

`perm(6, 3) = 120` codes is the entire space, and **instance identity IS the
code**, so the pools partition those 120 rather than being drawn independently.

| pool | n | how |
|---|---|---|
| **eval** | **40** | `MastermindTask.eval_specs()` — the front of the stream, so membership never depends on the training size |
| **train** | **80** | `MastermindTask.train_specs(n)` — everything after the eval prefix |
| total | 120 | `perm(6, 3)`, the whole space |

Verified disjoint (0 shared instance ids). `train_specs(81)` raises rather than
silently shrinking.

Two consequences:

- **80 train codes cap a Design B run at 8 rounds of batch 10.** Asking for more
  does not raise — `train_specs(80)` is drawn once and further rounds recycle
  it, showing the teacher each code repeatedly.
- **ORBIT's `test_size: 256` is not reproducible here.** Only 120 codes exist, so
  their test set necessarily repeats each code ~2x. Harmless for a trained
  policy; fatal for HExA, where the bank is text carried between instances and a
  code in both pools puts the answer in the prompt.

Maze partitions differently (independent draws from a 1e6 seed space) because a
repeated *layout* is tolerable where a repeated *code* is the answer itself.

## Results

n=40 paired, 3 eval-seed replicates per arm. Within an arm the bank is fixed and
only the actor seed varies, so the spread is eval noise, not bank noise. All
numbers come through `run_orbit_target_zeroshot.py` — see the code-path note
below.
## Mastermind GPT-OSS Results 
Design B - same level evolution 
Design C - Cross level Transfer 
| arm | ep1 | ep2 | ep3 | `any` |
|---|---|---|---|---|
| baseline (no bank) | .100 (.075-.150) | .317 (.275-.350) | .650 (.625-.675) | .683 (.650-.725) |
| **Design B** — target-trained | .058 (.025-.075) | .408 (.350-.450) | **.875 (.850-.900)** | .883 (.875-.900) |
| **Design C near** ← wordle | .092 (.050-.125) | .467 (.450-.475) | **.933 (.900-.975)** | **.933 (.900-.975)** |
| Design C far ← minesweeper | .058 (.025-.075) | .267 (.150-.375) | .825 (.725-.900) | .833 (.725-.925) |
| Design C both ← ms+wordle | .083 (.075-.100) | .300 (.225-.375) | .808 (.800-.825) | .808 (.800-.825) |
| ablation rawdump ← wordle | .125 (.100-.150) | .358 (.275-.425) | **.575 (.500-.650)** | .650 (.600-.700) |

### Paired McNemar vs baseline (mean delta, significant/total run-pairs)

| arm | ep1 | ep2 | ep3 | `any` |
|---|---|---|---|---|
| Design B | −4.2 (0/9) | +9.2 (0/9) | **+22.5 (6/9)** | +20.0 (4/9) |
| **Design C near ← wordle** | −0.8 (0/9) | +15.0 (0/9) | **+28.3 (9/9)** | **+25.0 (8/9)** |
| Design C far ← minesweeper | −4.2 (0/9) | −5.0 (0/9) | +17.5 (4/9) | +15.0 (3/9) |
| Design C both | −1.7 (0/9) | −1.7 (0/9) | +15.8 (0/9) | +12.5 (0/9) |
| ablation rawdump | +2.5 (0/9) | +4.2 (0/9) | **−7.5 (0/9)** | −3.3 (0/9) |

### Design C vs Design B

+5.8 on ep3, **0/9 significant**. The two are indistinguishable at n=40.

## What the table says

**Design C near ← wordle is the result.** +28.3 ep3, significant in all nine
pairs, and the only arm that clears significance everywhere. It matches the
target-trained ceiling using **zero target trajectories** — the bank was
synthesised from Wordle experience plus a 15-line statement of Mastermind's
rules, with `target_trajectories_used: 0` recorded in its provenance sidecar.

**The ablation carries the mechanism claim.** The same Wordle bank injected
verbatim scores .575 — *below* the no-bank baseline — against .933 synthesised.
A 36-point gap from re-grounding alone on identical source material. Cross-task
transfer is not "relevant text in the prompt helps"; the raw dump hurts.

**Source matters more than the plan assumed.** `far ← minesweeper` reaches only
4/9 and `both` reaches 0/9. One of three Design C sources works robustly.

**Episode 1 is dead in every arm**, target-trained included (0/9 everywhere,
Design B trending −4.2). The story is cross-episode adaptation, not first-shot
competence.

## Caveats

- **n=40 is the whole eval universe.** Each instance is 2.5 points. It cannot be
  enlarged without breaking eval/train disjointness.
- **.933 is near saturation** from a .650 baseline. Little headroom remains.
- **Every arm reports on a truncated bank.** The injection budget dropped all
  common-mistakes entries in all arms.
- **Code path matters, unexplained.** The same Design B bank scores .825 through
  the training script's internal `eval_frozen` and .875 mean through
  `run_orbit_target_zeroshot.py`. Whether that is seed variance (.825 is inside
  the replicate spread) or a real difference in prompt assembly has not been
  checked, and it affects any Design B number in the project.
- **Wordle→Mastermind may be the easiest available transfer pair.** Both are
  constraint-satisfaction guessing games with per-guess feedback, and the bank
  audit found Wordle's *Cumulative Constraint Management* is Mastermind's winning
  rule nearly verbatim. The weaker `far ← minesweeper` result is consistent with
  that reading.

## Retracted

Everything measured on the pre-ORBIT-paper 4-of-8 config (1680 codes) — the old
`orbit_mastermind_baseline_oss` (ep1 .031 / ep3 .367 / any .379) and
`orbit_mastermind_hexa_v3` (the withdrawn "+17.6"). Both are on a different task
and share zero instance ids with anything current. `hexa_v3` was additionally
compromised: its bank changed in only rounds 0, 8 and 14 of 20, the
`HEXA_TEACHER_MAX_TOKENS=16384` truncation signature. See
`results/_STALE_mastermind_4digit_config/README.md`.

The Design B rerun at 32768 changed the bank in **8/8 rounds with zero
truncations**, confirming the diagnosis.

An earlier reading that Design C *beat* Design B (+10.8) came from Design B's
single `eval_frozen` run at .825, the bottom of its own spread. Corrected above.

# Within-level HExA (Design B) — Minesweeper, Wordle-5, Mastermind

The bank is evolved on the level's **own** trajectories and evaluated on that
level's held-out pool. Actor and teacher are both `openai/gpt-oss-120b`.
Deltas are the mean over all HExA-run x baseline-run pairs; the fraction is how
many of those pairs reach p<0.05 on a paired McNemar exact test.

| level | eval n | runs (base/HExA) | metric | baseline | HExA | delta |
|---|---|---|---|---|---|---|
| **Minesweeper** | 255 | 3 / 3 | ep1 | .209 (.184-.227) | **.333 (.302-.357)** | **+12.4 (9/9)** |
| | | | ep3 | .302 (.290-.318) | .367 (.341-.400) | +6.5 (4/9) |
| | | | `any` | .434 (.404-.455) | **.567 (.553-.584)** | **+13.3 (9/9)** |
| **Wordle-5** | 59 | 3 / 3 | ep1 | .559 (.525-.576) | .531 (.475-.610) | −2.8 (0/9) |
| | | | ep3 | .825 (.797-.847) | **.932 (.898-.983)** | **+10.7 (3/9)** |
| | | | `any` | .927 (.898-.966) | .955 (.898-.983) | +2.8 (0/9) |
| **Mastermind** | 40 | 3 / 3 | ep1 | .100 (.075-.150) | .058 (.025-.075) | −4.2 (0/9) |
| | | | ep3 | .650 (.625-.675) | **.875 (.850-.900)** | **+22.5 (6/9)** |
| | | | `any` | .683 (.650-.725) | **.883 (.875-.900)** | +20.0 (4/9) |

Full ep2 figures: Minesweeper .251→.357 (+10.6, 7/9); Wordle-5 .740→.825 (+8.5,
3/9); Mastermind .317→.408 (+9.2, 0/9).

Minesweeper per-run HExA `any`: **.553 / .565 / .584** — every HExA run above
every baseline run, with no overlap between the two spreads.

## Reading

**Minesweeper is the only level where episode 1 moves** — +12.4 on ep1 and
+13.3 on `any`, both significant in all nine run-pairs. It is the sole arm
anywhere in this project to improve first-episode performance, and at n=3 it no
longer rests on a single bank.

**Wordle-5 and Mastermind gain only late.** Both are flat-to-negative on ep1 and
positive on ep3. Consistent with banks that help accumulate and exploit
constraints rather than play a better opening.

**Mastermind gains most (+22.5 ep3)** and has the largest headroom: its .650
baseline is the lowest ep3 of the three, while Wordle-5 starts at .825 and is
close to saturated.

## Caveats specific to this table

- **Minesweeper run2 was rerun after a bank defect; the original is archived.**
  The first run2 evolved a bank invoking a `terminate` action that does not exist
  in `game:Minesweeper-v0-only-reveal`. Only the last `\boxed{...}` in a response
  is read, so 461 of 2267 turns (20.3%) resolved to it, were rejected with "you
  did not provide a valid guess", and ended the episode; `any` was .051. The
  cause was `TASK_DESCRIPTIONS["minesweeper"]` never enumerating the valid
  actions, so the teacher could not have known. It now does. The rerun shows
  **0 of 3422** turns ending in `terminate` and `any` .586. Original kept at
  `results/_BROKEN_minesweeper_hexa_run2_terminate/` with the full diagnosis.
  Note runs 0 and 1 predate the description fix, so the three runs are not all
  on an identical teacher prompt.
- **`unparsed_rate` is not a health check.** The broken run2 reported
  `unparsed_rate` 0.0000 — *lower* than both healthy runs — because
  `\boxed{terminate}` parses fine as a boxed expression; it is the environment
  that rejects the action. The metric answers "did we extract a `\boxed{}`", not
  "was the action accepted".
- **No checkpoint selection.** `eval_frozen` takes the final round
  unconditionally. Broken run2 held a healthy 9-skill bank through round 16, was
  rewritten to a single `terminate` skill at round 17, and scored **0.00** train
  accuracy at round 19 — which is nonetheless the bank that was evaluated.
  Selecting by held-out accuracy would have rejected it and cost nothing.
- **One baseline instance excluded.** Line 3 of
  `results/orbit_minesweeper_baseline/run1/runs.jsonl` was corrupted by an
  editor edit (`"metrics": {"de,`) and the file is untracked, so git cannot
  restore it. The line is skipped rather than repaired, giving n=255 for
  Minesweeper throughout.
- **Eval pools differ in size and are not comparable across levels**: 255
  (Minesweeper, after excluding one corrupted line), 59 (Wordle-5, after
  excluding one errored instance), 40 (Mastermind).
- **One Wordle-5 instance excluded**: `2540b53a7f515c84` in `hexa/run2` returned
  `GenerationFailed: openai/gpt-oss-120b returned no generation after retries`.
  That is an infrastructure failure, not a task failure, so it is dropped from
  the paired set rather than scored as a loss. n=59 for Wordle-5 throughout.
- **Two code paths.** Minesweeper and Wordle-5 numbers come from each run's
  `eval_frozen`; Mastermind's come from `run_orbit_target_zeroshot.py`, so that
  it pairs with the Design C arms. On the one bank measured both ways the paths
  disagree (.825 vs .875 mean), and that discrepancy is unexplained — see the
  Caveats section above.


# Qwen3-8B — Design B (within-level)

Actor and teacher are both `Qwen/Qwen3-8B`, served on vLLM with
`--reasoning-parser qwen3`. Sampling is ORBIT's published eval setting for both
arms: temperature 0.6, top_p 0.95, **unseeded**, `--max-response-tokens 16384`.

Same task config and same 40/80 partition as above. Training ran **8 rounds x
batch 10 = 80 instances**, which is the whole training pool.

## Results for Qwen 3B on Mastermind

n=40 paired. Ranges are min-max across runs; only the actor seed varies within
an arm.

| arm | runs | ep1 | ep2 | ep3 | `any` |
|---|---|---|---|---|---|
| baseline (no bank) | 3 | .092 (.075-.100) | .142 (.125-.175) | .283 (.225-.400) | .367 (.275-.450) |
| **Design B** — target-trained | 4 | .087 (.025-.175) | .225 (.150-.350) | **.438 (.400-.525)** | **.569 (.525-.600)** |

### Delta vs baseline, two tests

| metric | delta | 95% CI | run-level (Welch t) | per-instance (sign test) |
|---|---|---|---|---|
| ep1 | −0.004 | [−0.068, +0.060] | p=0.905 | 5 up / 4 down, p=1.000 |
| ep2 | +0.083 | [−0.008, +0.174] | p=0.150 | 14 up / 10 down, p=0.541 |
| ep3 | **+0.154** | [+0.026, +0.283] | p=0.099 | 21 up / 17 down, p=0.627 |
| `any` | **+0.202** | [+0.096, +0.308] | **p=0.044** | 24 up / 13 down, p=0.099 |

The run-level test compares the arms' run means (df≈3, so the p-values do little
work). The per-instance test averages each arm's runs per code and asks whether
individual codes get easier — the more conservative reading, and the one that
should be quoted if only one is.

### Against ORBIT's published numbers

Same game, same base model, from Table 2 of the paper:

| | ep3 |
|---|---|
| ORBIT base Qwen3-8B (published) | .27 |
| **HExA Design B, frozen bank** | **.438** |
| ORBIT + 100 GRPO steps (published) | .59 |

HExA lands between the two with **no weight updates**. This is the only
comparison here that is immune to the local baseline-composition problem noted
below, since it uses the paper's own baseline.

### Training

| round | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| `any` /10 | 6 | 3 | 3 | 6 | 5 | 6 | 5 | 8 |

Pooled over 80 training instances: `any` .525, ep3 .388. The bank grew 3 skills
(round 0) to 4 (round 5) and held, with **zero citations dropped as fabricated in
all 8 rounds** — against 50-65% drop rates for the same model on Maze. Final
bank: `Cross-Episode Feedback Utilization`, `White Peg Inference`, `Progressive
Constraint Narrowing`, `Feedback-Driven Guess Adjustment`, plus 1 mistake.

## What the table says

**`any` is the only metric positive under both tests.** ep3 is +15.4 points at
the run level but 21-vs-17 per instance — the bank helps and hurts on nearly as
many codes as it helps. The gain is in *whether* a code is ever solved, not in
solving it by episode 3.

**Episode 1 is dead**, as in every gpt-oss arm above (−0.4, p=0.905).

**The bank reduces variance more clearly than it raises accuracy.** HExA's `any`
sd across runs is 0.037 against the baseline's 0.088, and its ep3 sd is 0.060
against 0.101. Both HExA sds are below the binomial expectation at n=40
(0.078), the normal signature of re-running fixed instances; the baseline's are
above it.


## Retracted (Qwen)

- **"+0.175 on ep3, from the first eval pair."** That compared a single HExA run
  against a single baseline run that drew `any` .375 — the low end of a spread
  later measured at .275-.625. Replication removed it.
- **Everything on the 4-of-8 config** (1680 codes): baseline ep3 .077, and the
  33.6%-truncation runs at `--max-response-tokens 4096`. Archived under
  `results/orbit_mastermind_{baseline,hexa}_qwen8_ABANDONED_4of8/` with a
  `WHY_ABANDONED.md`. Those runs remain useful only as evidence about the harder
  variant: Qwen considered a median of **13 candidate codes per step** (0.77% of
  1680, but 11% of 120), **66.2% of its guesses contradicted feedback already in
  context**, and in 26 of 60 instances the code was uniquely determined by that
  feedback while the model still failed 23 of them.

## Provenance

| artifact | path |
|---|---|
| baseline | `results/design_c/phase4/mastermind_nobank/run{0,1,2}` |
| Design B bank | `results/orbit_mastermind_hexa_oss_v4/round7/bank.json` |
| Design B evals | `results/design_c/phase4/mastermind_designb_bank/run{0,1,2}` |
| Design C banks | `results/design_c/mastermind_from_{wordle,minesweeper,minesweeper_wordle}/run0` |
| Design C evals | `results/design_c/phase4/mastermind_{near,far,both,rawdump}_*/run{0,1,2}` |
| Minesweeper within-level | `results/orbit_minesweeper_{baseline,hexa}/run*` |
| Wordle-5 within-level | `results/orbit_wordle5_{baseline,hexa}/run*` |
| **Qwen** baseline | `results/orbit_mastermind_baseline_qwen8{,_rep1,_rep2}` |
| **Qwen** Design B training + bank | `results/orbit_mastermind_hexa_qwen8/round{0..7}` |
| **Qwen** Design B evals | `results/orbit_mastermind_hexa_qwen8/eval_{frozen,rep3,rep4,rep5}` |
| **Qwen** replicate driver | `scripts/replicate_mastermind_eval.py` |
| **Qwen** retracted 4-of-8 runs | `results/orbit_mastermind_{baseline,hexa}_qwen8_ABANDONED_4of8/` |
