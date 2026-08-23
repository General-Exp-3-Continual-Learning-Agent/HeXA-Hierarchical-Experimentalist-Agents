# Why HeXA does not learn on Matchstick

Diagnosis over `results/matchstick_v3/run{0,1,2}` (HeXA, n=3) against
`results/core_matched_v1/matchstick/run{0,1,2}` (CORE, n=3), same 100 held-out items,
same weights, same server, matched 600-rollout budget. Every number below is measured,
not estimated.

**Summary.** HeXA's Matchstick null is not a broad weakness. It is one of five problem
subtypes, `tautological`, which holds 100% of the available headroom. HeXA's memory
already works on the other four. On that one subtype the required knowledge is a single
rule; HeXA discovers it and then deletes it before evaluation.

---

## 1. All the headroom is in one subtype

The eval set is 5 subtypes × 20 items. Paired, pooled over 3 runs each:

| Subtype | HeXA no-mem → with-mem | CORE no-mem → with-mem | HeXA trunc (no-mem → with-mem) |
|---|---|---|---|
| operator | 0.983 → 1.000 (+0.017) | 1.000 → 1.000 (0.000) | 2% → 0% |
| hybrid | 0.983 → 0.933 (−0.050) | 1.000 → 0.950 (−0.050) | 2% → 2% |
| value-loose-chunk | 0.900 → 0.850 (−0.050) | 1.000 → 0.950 (−0.050) | 10% → 13% |
| value-tight-chunk | 0.700 → **0.917 (+0.217)** | 0.800 → 0.900 (+0.100) | 25% → **5%** |
| **tautological** | 0.067 → **0.100 (+0.033)** | 0.150 → **0.650 (+0.500)** | **85% → 77%** |

Three facts follow.

1. **Four of five subtypes have no headroom.** They sit at 0.90–1.00 before any memory,
   with 0–2% truncation. Nothing can be won there, and both methods lose a little to
   noise on `hybrid` and `value-loose-chunk`.
2. **HeXA beats CORE where both can learn.** On `value-tight-chunk` HeXA gains +0.217
   against CORE's +0.100, and cuts truncation 25% → 5%. The distillation pipeline is not
   the problem.
3. **`tautological` is the entire gap.** CORE's +0.500 on 20 items is +0.100 overall —
   which is its whole reported gain (+0.073 ± 0.018). HeXA's +0.033 on the same items is
   +0.007 overall. Everything else cancels.

`tautological` also carries essentially all of the truncation this task is famous for:
51/60 of HeXA's no-memory truncations, against 1–15/60 for every other subtype. The
"Matchstick accuracy is a truncation proxy" finding is really "Matchstick accuracy is a
*tautological-subtype* truncation proxy".

## 2. The subtype is gated by exactly one rule

A `tautological` item is one whose only valid answer is a **chain equality**:

```
problem:     XIX = XX - XX
gold answer: XX = XX = XX
```

20/100 eval items admit a chain-equality gold answer — precisely the 20 `tautological`
ones. And in HeXA's training logs, **`correct` and `produced a chain-form answer` match
exactly, item for item, in all 36 tautological problem-rounds across the three runs.**
There is no partial credit and no other route: either the model knows a second `=` is
legal, or it searches a two-term solution space that contains no answer until it hits the
60,000-token ceiling. Hence 85% truncation and ~0.07 accuracy.

So the entire question is whether the memory ends up containing that one rule.

## 3. CORE gets the rule; HeXA discovers it and evicts it

**CORE.** Its `--enable-biased-problem-sampling` concentrates effort where the model
fails: **71 of 97 reflections (73%) are on `tautological` failures**, against that
subtype's 20% share of the training set. 46 of those 97 reflections mention chain
equality — all 46 from tautological failures. Six chain-equality lessons survive into its
final 25-lesson block (items 7, 8, 9, 13, 15, 19). Chain-form predictions at eval go
**3 → 13**. Accuracy on the subtype goes 0.15 → 0.65.

Its reflection also pairs a failure with a correct response from a *different* problem
(`contrast_source: historical_fallback`, 25/97 `different_problem`, e.g. failure on
`matchstick_027` paired with a success on `matchstick_058`). That is how it bootstraps the
rule before it has ever solved the problem in front of it.

**HeXA.** The rule *is* discovered — the wording is exactly right:

| Run | Skill | Rounds present | Wording |
|---|---|---|---|
| run1 | `mat_ev_14_009` | **21 only** | "turn an operator into a second '=' and make all terms identical → trivially true chain equality (e.g. A = A = A)" |
| run1 | `mat_ev_12_009` | **19 only** | "if you can move a single stick so that all terms become the same, the equation is trivially true" |
| run0 | `mat_ev_4_006` | 3–13 | *partial*: turn '-' into '=' "creating a chain equality" — never states that identical terms make it true |
| run2 | — | — | never discovered |

**None survived to evaluation.** The complete statement of the rule existed in exactly one
bank snapshot, in one of three runs, and was gone the next round.

## 4. Four causes, ranked

**(a) Round-robin sampling spends 80% of the budget where there is no headroom.**
HeXA visits all 60 training problems once, in index order: 48 of them are subtypes already
at 0.90–1.00. Only 12 are `tautological`. CORE puts 73% of its learning effort on that
subtype; HeXA puts 20%. This is the single largest cause and the cheapest to fix — `b_q`
is already computed per problem and already on disk (`b_q.json`), so difficulty weighting
needs no new rollouts.

**(b) One visit per problem, so a new rule is never validated.**
`problems_per_round: 2, attempts_per_problem: 10` over 30 rounds = each problem seen once,
never again. CORE runs 350 episodes over the same 60 problems (~6 visits each). When HeXA
discovers chain equality in round 21, no tautological problem appears again except round
25 — and by then the skill is gone. A rule distilled from a failing round has no
opportunity to earn its keep.

**(c) The evolver evicts subtype-specific skills for being irrelevant to the round it just saw.**
The evolver rewrites the whole bank each round while seeing only that round's 2 problems,
under rules that explicitly instruct removal of skills "too instance-specific to reuse"
and skills whose principle cannot "name an error from the attempts above"
(`prompts_core_tasks.py:428-438`). A chain-equality skill satisfies both removal criteria
in any round that is not tautological — and round-robin guarantees the next round is not.
Measured over all 98 skills across the three runs:

| Skill's source problems | n | mean confidence | median lifetime |
|---|---|---|---|
| all tautological | 14 | 0.500 | **2 rounds** |
| mixed | 5 | 0.435 | 4 rounds |
| no tautological | 72 | 0.520 | **5 rounds** |
| generic / no seed match | 7 | 0.521 | **25 rounds** |

Confidence is flat across the groups (0.435–0.521), so **confidence is not what
discriminates — LLM-judged relevance to the current round is.** Generic skills
("verify the arithmetic after the move") live 25 rounds because they are relevant to every
round; the one rule that unlocks 20% of the eval set lives 2.

**(d) The bank saturates at round 3 of 30, and the injected format wastes the budget.**
`max_skills: 10` is reached by round 3-4 in all three runs and `injection_budget_tokens:
1000` is saturated from round 4 on (added_tokens 944–1026 for the remaining 26 rounds).
So 26 of 30 rounds are pure churn under a hard cap. Worse, at equal byte cost HeXA carries
fewer lessons than CORE:

| | injected block | items | chars/item |
|---|---|---|---|
| HeXA | 4,542 chars (~989 added tokens) | 10 skills + 5 mistakes = **15** | **303** |
| CORE | 4,540 chars (~1,050 est. tokens) | **25** lessons | **182** |

HeXA spends three lines per skill (`Strategy` / `Apply when` / `Example`); CORE spends one
sentence. Same budget, **1.7× fewer lessons**. Note also that neither method retrieves
per problem — both inject a byte-identical block to all 100 eval items (CORE because
`knn_max_z = n_train` makes its neighbourhood the whole training set). So this is a pure
content-density difference, not a retrieval difference.

## 5. Headroom

If HeXA reached CORE's 0.650 on `tautological` and held its current values elsewhere,
its Matchstick score would be **0.760 → ~0.870** — above CORE's measured 0.830 ± 0.031 and
level with CORE's published 0.870. That is the size of the prize, and it is concentrated
in 20 items.

## 6. Honest caveats

- **Retention alone may not be sufficient.** run0's *partial* chain-equality skill
  survived rounds 3–13, spanning four tautological rounds (6, 9, 10, 13), and those rounds
  scored 0/10, 0/10, 0/10, 1/10. The rule has to be stated completely ("make all terms
  identical, a chain equality is trivially true"), not merely gestured at ("turn '-' into
  '=' creating a chain equality"). So the fix is *state it fully, then revisit to validate*
  — not merely *keep it longer*.
- **Tautological training success is sporadic and does not predict eval.** Training
  successes on the subtype were 15/120 (run0), 8/120 (run1), 5/120 (run2), while eval
  tautological accuracy was 0.00, 0.15, 0.15. No relationship — because the rule was
  evicted in every run.
- **11% of recorded `source_seeds` name problems outside the training split** (108 of 998,
  e.g. `matchstick_072/075/078` when the split is 000–059). `compute_initial_confidence_core`
  silently falls back to the level-wide mean for those, so their confidence is
  uninformative. Minor here — confidence is not the discriminator — but it is a real
  bookkeeping defect worth fixing.

## 7. Candidate fixes, in priority order

1. **Difficulty-weighted problem sampling** using the already-computed `b_q`. Mirrors
   CORE's biased sampling. No extra rollouts; changes only which problems the 600 calls
   are spent on. Highest expected value.
2. **Revisit problems** — e.g. 5 attempts × 2 visits instead of 10 × 1, with the second
   visit late enough that the bank has changed. Lets a newly banked rule be validated and
   gain confidence, which is the mechanism CORE's ~6 visits/problem provides.
3. **Protect young and subtype-specific skills from relevance-based eviction.** Either
   show the evolver a stratified trajectory sample spanning subtypes rather than only the
   current round, or exempt skills younger than N rounds from removal, or enforce rule 6b's
   displacement argument (nothing currently checks it).
4. **Compress the injected skill format** to one sentence per lesson to carry ~25 items in
   the same 1,000 tokens, and raise `max_skills` accordingly so the bank is not saturated
   from round 3.

Fixes 1 and 2 are schedule changes, independent of 3 and 4. **But fix 1 is NOT independent
of fix 2** (corrected — an earlier version of this file claimed otherwise). Because
coverage is preserved — every problem keeps at least one visit — difficulty weighting only
has surplus to reallocate when `visits_per_problem > 1`; at `visits=1`, `apportion_visits`
returns exactly one visit per problem and the weighting is inoperative. So the two must be
run jointly, and the decomposition needs a third arm (below). **All of these touch Matchstick's configuration only — none of them may be
applied to the Hanoi or ZebraLogic arms, whose results are frozen and reproducible.**

---

## 8. Implementation of fixes 1 and 2 (applied)

Both are schedule-only changes. Nothing in distillation, injection, retrieval, reward or
evaluation is touched, so the arm differs from `matchstick_v3` in *which problems the 600
calls are spent on and when* — nothing else.

**Code.** `hexa_core/passive_loop.py`:
- `Schedule.visits_per_problem` (+ `problem_visits`), threaded through `make_schedule`.
  Extra visits **widen each round instead of adding rounds**, so the number of rounds and
  therefore the number of evolver calls is held fixed at 30.
- `apportion_visits()` — how many rounds each problem appears in. Every problem gets one
  visit before any gets a second (coverage of D_train is preserved exactly), then the
  surplus is apportioned by hardness `1 - b_q` mixed with uniform at `uniform_mix`, using
  largest-remainder so the slots sum exactly rather than approximately.
- `plan_rounds()` — assigns visits to rounds, spread evenly across the round range (a
  revisit in the next round teaches nothing; the bank has barely moved) and never placing
  a problem twice in one round. With the defaults it *delegates to* `chunk_problems`, so
  the frozen path is a no-op rather than merely an equivalent re-implementation.

**Flags.** `scripts/run_pilot.py`: `--visits-per-problem`, `--problem-sampling
{round_robin,difficulty}`, `--sampling-uniform-mix`; recorded in `provenance.json`.
`scripts/serve_gpt_oss_vllm.sh`: `PIPELINE_VISITS`, `PIPELINE_SAMPLING`,
`PIPELINE_SAMPLEMIX`.

**Budget is preserved exactly.** 600 calls in both arms:

| | rounds | problems/round | attempts/problem | calls | slack |
|---|---|---|---|---|---|
| `matchstick_v3` (frozen) | 30 | 2 | 10 | 600 | 0 |
| visits=2 (new) | 30 | 4 | 5 | 600 | 0 |

`visits=3` would give 30×6×3 = **540**, not 600 — `k` is floor-divided. That arm is *not*
budget-matched and `run_pilot` prints the 60-call slack; a test pins this so it cannot be
run by accident and reported as matched.

**What the reallocation actually buys**, computed against the real measured `b_q` from
`results/matchstick_v3/run0/b_q.json`. The hardness signal is unambiguous:

| subtype | measured mean `b_q` |
|---|---|
| hybrid | 0.983 |
| operator | 0.950 |
| value-loose-chunk | 0.850 |
| value-tight-chunk | 0.686 |
| **tautological** | **0.083** |

| allocation | tautological visits | share of 120 | visits per taut. problem | median revisit gap |
|---|---|---|---|---|
| round_robin (frozen) | 24 | 20% | 1 | — (never revisited) |
| difficulty, mix=0.5 (CORE's published value) | 36 | 30% | 3 | 12 rounds |
| difficulty, mix=0.25 | 47 | 39% | 4 | 8 rounds |
| difficulty, mix=0.0 (pure hardness) | 49 | 41% | 4–5 | — |

A side effect worth noting: rounds containing at least one tautological problem go
**10/30 → 15/30**, so the evolver sees the subtype twice as often. That partially
mitigates cause (c) — relevance-driven eviction — for free, without touching the evolver.

**Why the shares stop at 30–41% rather than CORE's 73%.** Two reasons, and neither is a
shortfall to tune away. First, the min-one-visit coverage guarantee locks 48 of 120 slots
(40%) for the four other subtypes. Second, and more important: **CORE's 73% is its share
of *reflections*, not of rollouts.** CORE reflects only on failures, so failures
concentrate there automatically. The two numbers are not the same quantity and should not
be matched to each other.

**Verification.** `hexa_core/tests/test_problem_sampling.py`, 12 checks: the frozen
default is identical to `chunk_problems`; `round_robin` ignores `b_q` entirely; the budget
never overspends; apportionment sums exactly at every mix; hard problems strictly outrank
easy ones; `mix=1.0` reduces to uniform; an *unmeasured* `b_q` scores as average rather
than easy (so a failed baseline pass does not starve the problem it failed on); rounds are
full and duplicate-free; revisit gaps exceed one round; bad inputs raise. Existing suites
(`test_passive_loop`, `test_core_fidelity`, `test_admission`, `test_budget`, `test_splits`,
`test_train_baseline`, `test_trajectory_selection`, `test_injection`,
`test_provenance_guards`) all still pass, 122 checks total. An in-process end-to-end run
on tower-of-hanoi confirms 48/48 calls, 2 visits per problem, correct spread, and that
`difficulty` without baselines raises rather than silently degrading to uniform.

### The shape change, and the confound inside it

Revisits are paid for out of per-visit depth, so the round shape necessarily changes:

| | rounds | problems/round | attempts/problem | calls |
|---|---|---|---|---|
| `matchstick_v3` (frozen) | 30 | 2 | 10 | 600 |
| `matchstick_v4_sched` | 30 | **4** | **5** | 600 |

Held identical to v3: budget (600), rounds (30, so the same number of evolver calls),
`n_train` (60), `max_tokens` (60k), `n_base` (5), paired contrast, `max_trajs` (12),
`max_skills` (10), the 1000-token injection, and `run-index 0` — so the held-out eval is
the same 100 items under the same seeds and the comparison is paired.

**The confound: `attempts_per_problem` cannot be held constant.** 10 → 5 is forced by
adding revisits at a matched budget. It bears directly on the contrast mechanism, since
`paired` selection wants a success *and* a failure on the same problem. Measured on the v3
logs, over the 36 tautological problem-rounds:

| | pairing rate per visit | visits per taut. problem | expected paired visits |
|---|---|---|---|
| v3, 10 attempts | 42% (15/36) | 1 | 0.42 |
| v4, 5 attempts | 31% (22/72, disjoint halves) | 4 | **1.22** |

So the depth loss is real and is more than repaid — 2.9× more paired opportunities — but
"revisits" and "5-vs-10 attempts" are **not separable within this arm**. A win cannot be
attributed to either alone.

**The decomposition arm**, if the joint result warrants it:
`PIPELINE_VISITS=2 PIPELINE_SAMPLING=round_robin` — the same 30x4x5 shape with uniform
2 visits each. That isolates revisits-plus-shallower-visits from difficulty weighting; the
gap between it and the difficulty arm is fix 1's own contribution.

### Two implementation bugs found before any training compute was spent

Both were in the new scheduling code, and the ~40-call smoke gate (`PIPELINE=full`, which
gates `train` on smoke passing) is what stopped them costing a 7.5 h run. Job 63447401 died
in 10 minutes with 0 training calls spent.

1. **Visits could exceed rounds.** At the smoke shape (`n_train=4, rounds=2`) the hardness
   weighting handed two problems 3 visits each. A problem cannot appear twice in one round,
   so 3 was unsatisfiable and `plan_rounds` asserted. Fixed with a per-problem cap of
   `rounds`, redistributing the excess by weight so the total stays exact.
2. **The placement algorithm was wrong, twice.** Nearest-ideal-round-with-space strands the
   last visit when the only round with capacity already holds the only problem left to
   place — **46 of 168 schedule shapes failed**. Replacing it with
   most-remaining-capacity-first was feasible everywhere but **clustered**: all rounds start
   equally empty, so a problem's four visits landed in rounds 0,1,2,3 for a median revisit
   gap of 1, which destroys the mechanism fix 2 exists to provide. The working version walks
   rounds in order and fills each with the problems owed the most visits, breaking ties
   toward whichever has gone longest unseen.

The clustering bug is the one worth remembering: **it would not have failed anything.** It
would have run to completion and produced a null that read as "revisits don't help." It was
caught by measuring the revisit-gap distribution, not by any assertion — so that
distribution is now a test (`test_difficulty_weighted_revisits_are_also_spread`), as is the
168-shape placement grid and the cap breach. 16 checks in the suite, 138 across `hexa_core`.

Verified on the real config after the fix: 600 calls exact, 30 rounds of width 4, 39% of
visits to `tautological` at 4 visits each, revisit gap min 4 / median 7, 15/30 rounds
containing a tautological problem, no in-round duplicates.

**Status.** Job **63449289** queued 2026-08-22 for `results/matchstick_v4_sched/run0`
(`VISITS=2 SAMPLING=difficulty SAMPLEMIX=0.25 RUNIDX=0`). n=1 when it lands; three runs
needed before the comparison against v3's 0.760 +/- 0.031 means anything, per §6's own
discipline.
