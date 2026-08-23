# Session log — HeXA vs CORE on CORE-benchmark tasks

Rolling record of what was decided, measured, and left open. Newest section last.
Companion files: `CORE_COMPARISON.md` (the head-to-head numbers),
`MATCHSTICK_DIAGNOSIS.md` (why HeXA loses on Matchstick), `WEAKNESSES.md`,
`RECON.md`, `PHASE3_DESIGN.md`.

Last updated: 2026-08-23.

---

## 1. Standing constraints

These are binding on every run and were restated by the user more than once.

- **Local vLLM only. Never a hosted API key.** `gpt-oss-120b` served on SLURM.
- **`../core-reasoning/` is read-only.** Verified before and after every
  session with `git -C ../core-reasoning status --porcelain | wc -l` → `0`.
- **Do not disturb Hanoi or ZebraLogic reproducible results.** Those arms are
  frozen; no code path they touch may change.
- **At most 4 GPUs total** — one 4-GPU job at a time. `squeue` is checked empty
  before every submission.

## 2. Experimental setup, as actually run

- Actor and evolver are the same frozen `gpt-oss-120b`, matching CORE.
- Eval splits are CORE's own, recovered deterministically (see `RECON.md`).
  Matchstick eval = all 100 rows of `matchstick_test.jsonl`.
- Paired frozen eval: identical held-out problems and seeds in both arms;
  significance by McNemar's exact test on discordant pairs.
- `b_q` = per-problem no-memory baseline, `n_base=5` (5 x 60 = 300 calls),
  used as the M1 reward offset. `attempt_utility = reward - b_q`;
  `confidence = (mean_utility + 1) / 2`.
- Standard Matchstick training shape: 30 rounds x 2 problems x 10 attempts
  = 600 calls. The v4 arms reshape this (see §6).

## 3. Matchstick head-to-head (settled at n=3 per arm)

| arm | runs | mean |
|---|---|---|
| CORE | 0.790 / 0.890 / 0.810 | **0.830 ± 0.031** |
| HeXA v3 | 0.800 / 0.780 / 0.700 | **0.760 ± 0.031** |

Difference +0.070 ± 0.043 — **1.62σ, Welch p ≈ 0.11, not significant.**
Binomial SE at p≈0.75, n=100 is 0.043, so any single-run gap below ~0.08 is noise.

The robust, offset-immune finding is the **within-arm** asymmetry: CORE's gain
over its own no-memory baseline is significant (pooled 39/17 discordant,
p = 0.0046); HeXA's is not (28/18, p = 0.184).

**Reporting decision (user's call): report best 3 of 4 HeXA runs = 0.787.**
Constraints recorded in `CORE_COMPARISON.md` §7 and repeated here because they
are easy to lose: the phrase "best three of four runs" must travel with the
number; quote **no** error bar with it; never pair it with "n=3", with a ±, or
with "over three independent runs" — all three are false of it. Cherry-picking
buys only 0.007 over v4's honest 0.780. Selection bias was simulated
(200k trials, p=0.76, n=100): best-3-of-4 inflates the mean by +0.0148 and
shrinks the SEM from 0.022 to 0.007.

## 4. Cost

Over n=3, CORE train+eval = **22.28 h** (24.27 / 19.53 / 23.03) against HeXA's
**6.13 h** — about **3.6x**. Whole-job 25.45 h vs ~7.5 h. CORE logs **135**
reflections (167 / 97 / 140) and final memory sizes 51 / 56 / 47.

## 5. Why HeXA loses on Matchstick

Full argument in `MATCHSTICK_DIAGNOSIS.md`; the short version:

- Matchstick eval is **5 subtypes x 20 items**: `hybrid`, `operator`,
  `value-loose-chunk`, `value-tight-chunk`, `tautological`. Four sit at
  0.90–1.00 unaided. **All the headroom is `tautological`.**
- `tautological` is gated by exactly one rule — **chain equality**: the gold
  answer needs a second `=` (`XIX = XX - XX` → `XX = XX = XX`). Across all 36
  tautological problem-rounds in v3, `correct` ⟺ chain-form answer, exactly.
- Accuracy tracks truncation: ≈ `(1 - truncation) x 0.96`, and truncation is
  concentrated in `tautological` (51 of HeXA's 60 no-memory truncations).
- HeXA discovers the chain rule and then **evicts** it. Tautological-sourced
  skills have median lifetime 2 rounds vs 5 (other subtypes) vs 25 (generic),
  at *flat* confidence — eviction is for irrelevance to the current round, not
  for measured failure.
- CORE wins the subtype for two structural reasons. Its
  `--enable-biased-problem-sampling` put **71 of 97 reflections (73%)** on
  tautological failures. And its `historical_fallback` contrast pairs a failure
  with a correct response from a **different** problem (25/97 pairs are
  `different_problem`), so a rule can be bootstrapped before the model has ever
  solved that problem itself.
- Neither method retrieves per-problem: both inject a byte-identical ~1k-token
  block into all 100 eval prompts (CORE because `knn_max_z = n_train`).

Four ranked causes: (a) round-robin spends 80% of budget where there is no
headroom; (b) one visit per problem, so a new rule is never validated;
(c) relevance-based eviction kills young subtype-specific skills;
(d) the bank saturates at round 3 and the injected format costs 303 chars/item
against CORE's 182.

## 6. The v4 fixes (fixes 1 and 2), and what they did

Implemented in `hexa_core/passive_loop.py`, exposed through
`scripts/run_pilot.py` and `scripts/serve_gpt_oss_vllm.sh`:

- **Fix 1 — difficulty-weighted problem sampling.** `apportion_visits(...)`
  guarantees every problem ≥1 visit (coverage preserved), then apportions the
  surplus by hardness `1 - b_q` mixed 50/50 with uniform by default. Unmeasured
  `b_q` falls back to *mean* hardness, not 0.
- **Fix 2 — multiple visits per problem.** `Schedule.visits_per_problem`;
  `problems_per_round = ceil(n_train * visits / rounds)` **widens rounds rather
  than adding them**, holding the round count (and so the evolver call count)
  fixed.
- **These two do not ablate separately** (an earlier draft claimed they did).
  Because coverage is preserved, at `visits=1` there is no surplus to
  reallocate and `apportion_visits` returns `[1]*n` — fix 1 is inoperative
  without fix 2.
- Budget hazard, now a test: `visits=2` → 600 calls, slack 0; **`visits=3` →
  540 calls, slack 60** (floor division), i.e. not budget-matched.
- 16 tests in `hexa_core/tests/test_problem_sampling.py`, including
  `test_default_plan_is_identical_to_chunk_problems` (the frozen v3 path is a
  literal no-op, not merely equivalent).

**v4 run0 (job 63449289, 10h01m): 0.730 → 0.780 (+0.050, p = 0.180). The fixes
worked, on the wrong subtype.**

- `tautological` went **0.05 → 0.00**; chain-form predictions 1 → 0.
- `value-tight-chunk` went **0.70 → 1.00**, truncation 6/20 → 0/20 — the best
  subtype result of any arm.
- Mechanism of the tautological collapse: 235 tautological training attempts
  produced **2 correct (0.9%)**, both in round 0, then 215 consecutive failures
  with zero chain-form attempts. Round 0 banked two correct chain-equality
  skills; **round 2 banked the anti-rule** `mat_ev_3_009` — *"The equation must
  contain exactly one '=' sign after the move"* — which recurs as
  `mat_ev_19_009` at round 24 and **is in the final bank**, injected into all
  100 eval prompts. Contrastive distillation over a stream of pure failures
  generalized the shared surface feature.
- My pre-submission estimate of paired contrasts was wrong: I projected
  0.42 → 1.22 expected paired visits from v3's 42% per-visit pairing rate, but
  that rate was itself conditional on 10 attempts per visit.

## 7. Bugs found in my own new code, and how

Worth keeping because two of the three would not have thrown.

1. `apportion_visits` could give a problem **more visits than there are
   rounds** (counts `[1,3,1,3]` at rounds=2). Killed job 63447401 at the smoke
   gate. Fixed with `max_per_problem=rounds` plus redistribution.
2. **Placement greedy wrong.** Nearest-ideal-round-with-space stranded the last
   visit on **46 of 168** shapes.
3. **Found by measurement, not by an assertion.** Most-remaining-capacity-first
   was feasible on 160/160 shapes but **clustered** — all rounds start equally
   empty, so ties resolved to rounds 0,1,2,3 and the median revisit gap was 1.
   This would have run to completion and produced a meaningless null. Replaced
   with a round-major greedy (min gap 4, median 7 tautological / 12 overall).

Also corrected earlier in the session: the CORE stats loader (results are nested
under `detailed_results[].details[]`, not flat), and hardcoded v3 accuracies
(`[0.800, 0.740, 0.740]` → actual `[0.800, 0.780, 0.700]`; same mean, sem 0.031
not 0.020).

## 8. In flight

**Job 63471573 — v4 run1**, byte-identical to run0 except `RUNIDX=1` /
`PIPELINE_OUT=results/matchstick_v4_sched/run1`:

```
PIPELINE=full PIPELINE_TASK=matchstick PIPELINE_BUDGET=600 PIPELINE_MAXTOK=60000
PIPELINE_VISITS=2 PIPELINE_SAMPLING=difficulty PIPELINE_SAMPLEMIX=0.25
```

Verified allocation from run1's own `b_q.json`:

| subtype | b_q | visits | share |
|---|---|---|---|
| hybrid | 1.000 | 20 | 17% |
| operator | 0.983 | 18 | 15% |
| value-loose-chunk | 0.767 | 24 | 20% |
| value-tight-chunk | 0.571 | 15 | 12% |
| **tautological** | **0.117** | **43** | **36%** |

Tautological 3–4 visits each, revisit gap min 4 / median 9, present in **16 of
30 rounds** (run0: 16/30; v3: 10/30). 600 calls, slack 0. Third independent
`b_q` draw putting tautological at 0.05–0.12 while every other subtype sits at
0.57–1.00, so the ~10x difficulty gap is established rather than a one-off.

Note for the analysis: run1's `value-tight-chunk` baseline is **0.571**, below
run0's 0.657 and v3's 0.686 — more headroom. If run0's +0.30 there was real it
should reappear clearly; a much smaller gain from a *lower* base would mark
run0 as a favourable draw.

Three things to report when it lands: run1's paired accuracy against run0's
0.780; whether `value-tight-chunk` replicates 0.70 → 1.00; and **whether the
"exactly one '='" anti-rule is in the final bank again**. The last matters most
— recurrence in an independent run makes it a systematic failure of contrastive
distillation on low-success subtypes, which is a reportable finding on its own.

## 9. Open items

- Optionally a third v4 run (`RUNIDX=2`) to reach an unselected n=3 and let the
  Matchstick row drop the "best 3 of 4" qualifier.
- **Fix 3**: protect young / subtype-specific skills from relevance-based
  eviction. And/or cross-problem contrast mirroring CORE's
  `historical_fallback` — I argued the latter is the real fix, since it attacks
  the 0.9%-success problem directly rather than its symptom.
- Decomposition arm `PIPELINE_VISITS=2 PIPELINE_SAMPLING=round_robin` to
  isolate fix 1's contribution.
- **MathGAP is the weakest row**: +0.020 margin sits inside its +0.038 measured
  provider offset. The CORE-on-MathGAP run is still not done.
- Bookkeeping defect: ~11% of `source_seeds` are out of range.
- Instrument HeXA's real evolver call count (currently *estimated* ~30 against
  CORE's logged 135).
- Ablations `PIPELINE_SELECTION=top5`, `PIPELINE_MAXTRAJS=5`; update
  `WEAKNESSES.md` §12.
- Regenerate the MathGAP `data[100:200]` window to resolve the mtime anomaly;
  reconcile the Hanoi provider-gap discrepancy.
