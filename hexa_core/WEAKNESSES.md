# Where the HeXA setup is weak on the CORE tasks — and what to do about it

**Status:** written after the Tower of Hanoi campaign returned a clean null, to record the
reasons while the evidence was fresh. Partially implemented since — see the per-section
markers below.

| § | state |
|---|---|
| 1 evolver starved | **mostly done** — `paired` is now the default and `max_trajs` is 12 (was 5), so the evolver sees 360 trajectories per run instead of 50. `distill_mistakes` still has no call site |
| 2 prompt | **items 1–5 done**, item 6 (confidence requested *and* overridden) open |
| 3 reward gradient | open — deliberately: `paired` supplies the success/failure mixing that 3 was needed for, and a graded reward is a method change with a domain trap |
| 4 schedule | **done** — n=60 / 30 rounds / k=10, `n_base=5` |
| 5 injection cost | open — deliberately: injection is already net-negative, so raising `max_skills` or the token budget makes it worse before retrieval is query-conditional |
| 6 task fit | **done** — default task is now ZebraLogic |
| 7 memorisation | **done** — detector + counterexample `example` field, and the detector is now sound on grid answers |
| 8 measurement | **done** (already fixed when this was written) |

**Replicated on two of three tasks at n=3, with the third a null explained by
headroom.** All three tasks have now been run under the fixed pipeline, and both
positives have been pooled over 3 independent runs — the standard the old Matchstick
null itself used:

| task | HeXA | no-memory | delta | runs | verdict |
|---|---|---|---|---|---|
| ZebraLogic | 0.807 ± 0.018 | 0.560 ± 0.026 | **+0.247 ± 0.023** | 3 | strong positive (§9) |
| Tower of Hanoi | 0.520 ± 0.010 | 0.263 ± 0.007 | **+0.257 ± 0.017** | 3 | strong positive (§10) |
| Matchstick | 0.800 | 0.780 | +0.020 (p = 0.688) | 1 | **null**, headroom-bound (§11) |

Every individual run of both positive tasks is significant on its own and positive in
the same direction; the pooled deltas are ~11× and ~15× their standard errors. §11 is
the important correction: Matchstick first read **−0.250** and that was **my
configuration error, not the method** — the run was launched at `max_tokens=32768` when
60000 had been requested, which starved the evolver (97% of training failures were
truncations) and inflated eval truncation with the bank injected. Retrained at 60000 it
is a clean null, and the bank it produced is *good* (§11). Both campaigns referenced
below (the old Matchstick table, the two Hanoi tables) predate the changes and used
`top5` with 5 trajectories and the pre-rewrite prompt — that is now the baseline the new
configuration had to beat, and on the two tasks with headroom to show it in, it did,
decisively.

**The Matchstick result, for the record.** Three paired runs, same design as Hanoi:

```
run  HeXA   no-mem   delta   discordant     p
 0   0.700  0.780   -0.080   5 / 13      0.096
 1   0.760  0.770   -0.010   7 /  8      1.000
 2   0.710  0.730   -0.020   4 /  6      0.754
MEAN 0.723+/-0.019  0.760+/-0.015  -0.037   pooled 16/27, McNemar p = 0.126
```

Not significant, but negative in all three runs, and truncation was again worse with the
bank (74 vs 65 of 300) -- 5's cost with none of the benefit. `budget_bound` fired on
every attempt of every round, so 3 of the 10 banked skills were never once injected.

**The result this is trying to explain.** Three paired runs, 100 held-out problems,
bank frozen, same problems and seeds in both arms:

```
run  HeXA   no-mem   delta
 0   0.350  0.350   +0.000
 1   0.290  0.340   −0.050
 2   0.300  0.250   +0.050
MEAN 0.313  0.313    0.000        pooled 50 helped / 50 hurt, McNemar p = 1.000
```

Fifty and fifty. The machinery all worked — seeds genuinely paired, 906–979 tokens
injected, banks holding 9–10 skills — so this is a real null, not a broken run.

**An earlier single run read 0.230 vs 0.130 (+0.100, p = 0.053) and was wrong.** Two
artifacts produced it, both since fixed: `max_tokens=32768` truncated 53 of 100 eval
problems identically in both arms, and the eval arms were not actually paired (see §8).
Raising the cap to 60000 lifted the **baseline** by +0.183 and HeXA by +0.083 — the
apparent advantage was HeXA getting the model to finish inside a budget that was too
small. Any future positive result should be checked against a larger token budget
before it is believed.

Ordered by expected cost. §1 and §2 are the ones I would fix first: together they mean
the method currently learns only from successes, when its stated mechanism is contrast.

---

## 1. The evolver is starved of evidence, and of failures entirely

**Evidence.**

- `evolving_distill.py:69` — `format_trajectories_block(successes + failures, max_trajs=5)`.
  The evolver reads **5 trajectories per round**. At 600 training calls it has ever seen
  **50 of 600**; at the earlier 2000-call setting, 50 of 2000.
- Successes always outrank failures on reward — under M2 (+1 vs −1), under M1
  (`1−b_q ≥ 0 ≥ −b_q`), and in InterPhyre too (successes 1.0…0.25, failures −0.5/−0.75).
  So **any round with ≥5 successes sends the evolver zero failures.** Measured: all 10
  rounds, all three Hanoi runs.
- Consequently every *mistake* record in the bank was written without observing a single
  failure.
- `distill.distill_mistakes` — which implements the paper's *"partial skills from
  otherwise-failed trajectories… a learning signal unavailable from successful
  trajectories alone"* — is **never called** by `evolve_skill_bank`. `grep distill_mistakes
  evolving_distill.py` returns nothing. That channel has never run.

**Possible fixes.**

| fix | cost | state |
|---|---|---|
| Raise `max_trajs` 5 → 10–15 | trivial | **DONE, at 12** (`hexa_bridge.DEFAULT_MAX_TRAJS`). Bounded by *input* context, not the teacher's output budget: one Hanoi trajectory renders to ~6.6k chars of reasoning, so this required capping `Thought` too — `_THOUGHT_MAX_CHARS = 2400`, elided from the **middle** so both the setup and the error-bearing tail survive. The override happens in our formatter because `evolving_distill.py:69` hardcodes `max_trajs=5` in shared `skillrl` code. |
| Sample both tails — top-k **and** bottom-k by reward | small | **DONE** — `paired` is the default (`DEFAULT_TRAJECTORY_SELECTION`), `balanced` and `top5` remain as ablation arms. Note `DEFAULT_SUCCESS_SLOTS = 3` was *3 of 5*; widening to 12 made it scale (`DEFAULT_SUCCESS_FRACTION`), or the block would have inverted to 3 successes / 9 failures. |
| Call `distill_mistakes` on the round's failures as a second pass | medium | **OPEN.** Restores a documented mechanism rather than inventing one. Costs one extra teacher call per round. |
| Shrink rounds so successes < 5 naturally | free | Superseded — `paired` gets failures in regardless of the base rate, and §4 shrank k to 10 anyway. |

---

## 2. The prompt does not ask for what the method claims

All in `hexa_core/prompts_core_tasks.py`; the live evolver prompt is
`SKILL_BANK_EVOLUTION_PROMPT_CORE` (installed over HeXA's original by
`hexa_bridge.patch`, `hexa_bridge.py:405`).

**Evidence.**

- The instruction reads *"evolve the bank by **merging** the existing skills with the
  evidence in the new attempts."* The paper says the evolver should *"contrast
  high-reward behaviors against low-reward ones, to learn the unique lessons of success
  over failure."* The word contrast does not appear.
- **No novelty bar.** The final Hanoi bank: *"Use Auxiliary Peg for Temporary Storage"*,
  *"Check Move Legality"*, *"Goal Stack Ordering"*, *"Simulate Final Configuration"* —
  all recitable by gpt-oss-120b unprompted. ~950 tokens telling the model what it knows.
- **No requirement to ground a skill in an observed error**, so nothing forces the
  evolver to look at the reasoning rather than summarise the task.
- **`_CORE_SETTING` contradicts `--m1`.** It states reward is binary ±1.00 and *"every
  accepted attempt is equally strong evidence… weigh attempts by that verdict"* — while
  M1 supplies graded, difficulty-relative rewards (headers read `Reward: +0.85`)
  precisely so that attempts are **not** equally weighted.
- **The requested confidence is discarded for new skills.** `evolving_distill.py:138`
  recomputes it from the source trajectories; only *retained* skills keep the evolver's
  number. The prompt spends a rule on an answer that is thrown away.

**Possible fixes.** A partially-applied draft of these exists in the working tree — check
`git diff` / the file before re-applying.

1. Add an explicit contrast step *before* extraction, including: **if a behaviour appears
   in both accepted and rejected attempts it does not explain the outcome.** That single
   clause disqualifies "use the auxiliary peg".
2. Add a novelty bar: *"would a capable solver have said this before seeing any
   attempts? If yes, drop it."*
3. Require the `principle` to name the concrete error it prevents. Self-policing: a
   generic skill cannot cite one. Also gives a reportable diagnostic — what fraction of
   the bank cites a real failure.
4. Rewrite the reward paragraph to describe the baseline-relative reward actually in use,
   including that `±1.00` exactly means *no baseline was measured*.
5. Make retention adversarial (see §4 — the bank is full from round 0): *"to add a skill,
   name the existing skill it displaces and why the evidence favours it. If none is
   weaker, add nothing."*
6. Stop requesting confidence for new skills, or stop overriding it. Currently both.

---

## 3. Reward has no natural gradient in a single-turn task

**Evidence.** InterPhyre grades success by speed (1.0 / 0.75 / 0.5 / 0.25). One prompt
and one answer has no turns, so M2 collapses every success to +1.0 and "top-5 by reward"
becomes "5 arbitrary successes". M1 (`correct − b_q`) restores an ordering *among*
successes but cannot mix successes and failures, since `1−b_q ≥ 0 ≥ −b_q` always.

**Possible fixes.**

- **Grade by solution quality.** Hanoi ships `optimal_moves` in `Problem.metadata`; a
  40-move solution to a 23-move problem is a worse success. CORE scores with
  `enforce_optimal=False`, so this changes the *reward*, not the verifier. Analogues
  exist elsewhere (sticks moved, clue checks used).
- Combine with `b_q`: `reward = (efficiency if correct else 0) − b_q` keeps the [−1,1]
  domain `_compute_initial_confidence` expects. **Check the domain** — the M1/M2 units
  bug (`reward − b_q` spanning [−2,1] and clamping every mid-difficulty failure to the
  confidence floor) came from exactly this kind of change.

---

## 4. The schedule spends the budget on the wrong axis

**Evidence.**

- The bank reaches `max_skills = 10` **in round 0** and stays flat for nine rounds — on
  both tasks, every run. 90% of training funds revision, not accumulation. The design's
  own risk #2 (Zhang et al.: memories degrade under continuous LLM updating) applies to
  exactly that regime.
- 30 attempts per problem, of which the evolver reads 5. The remainder only sharpens a
  confidence estimate whose standard error moves from 0.09 (n=20) to 0.03 (n=200) — a
  sorting key for ten items.
- 20 training problems against pools of 100 (Matchstick, ZebraLogic) and 200+ (Hanoi).

**Possible fixes.**

| config (600 training calls) | rounds | problems | k | trajectories the evolver ever sees |
|---|---|---|---|---|
| n=20, 10 rounds *(what ran)* | 10 | 20 | 30 | 50 |
| n=40, 20 rounds | 20 | 40 | 15 | 100 |
| **n=60, 30 rounds — ADOPTED** | 30 | 60 | 10 | **360** (at `max_trajs=12`) |
| n=100, 50 rounds | 50 | 100 | 6 | 250 |

**Adopted:** n=60, `round_target=30`, `n_base=5` — `DEFAULT_N_TRAIN` /
`DEFAULT_ROUND_TARGET` / `DEFAULT_N_BASE` in `run_pilot.py`, and the launcher's
`PIPELINE_NTRAIN` / `PIPELINE_ROUNDS` defaults. Derived: 30 rounds × 2 problems × k=10
= exactly 600. Combined with `max_trajs=12` the evolver now reads 360 trajectories
per run rather than 50 — 7×.

**The `b_q` constraint, resolved.** `b_q` costs `n_base × n_train`, so n=60 at
`n_base=10` would be 600 baseline calls, as much as training itself. Taking `n_base=5`
halves that to 300; the cost is `b_q` quantised to steps of 0.2. Acceptable *because of
what `b_q` is used for* — a sort key over at most ten skills and a grading term on the
reward the evolver reads, never a reported number. Raise it back to 10 if `b_q` ever
becomes a quantity the paper states.

**Still unaddressed here:** the bank reaches `max_skills = 10` early regardless, so 30
rounds means ~28 rounds of pure revision — *more* exposure to the degradation regime,
not less. Left as-is deliberately, since raising `max_skills` makes §5 worse; the
adversarial-retention clause from §2 item 5 is what is supposed to carry this.

---

## 5. Injection is currently net-negative

**Evidence.** ~950 tokens per prompt, `budget_bound` firing on **every** attempt (only
~7 of 10 skills fit the 1000-token cap), and **HeXA truncated more than the baseline** —
29/27/28 vs 18/22/18 — because the block pushes borderline responses over the limit. No
accuracy gain to pay for it.

**Possible fixes.** Query-conditional retrieval on the problem's declared `type`
(Matchstick has five: value-loose-chunk, value-tight-chunk, hybrid, operator,
tautological); shorter skill records; raise `max_skills` and the injection budget
*together* rather than separately; or inject only when something type-specific exists.
Note the `task:type` tier is currently always empty because distillation never runs per
type — `injection.tiered_bank` merges the tiers so retrieval works, but nothing fills the
specific one.

---

## 6. Task fit

**Hanoi is a poor showcase.** The model already knows the algorithm, so a strategy
statement adds nothing; its failures are 40-move bookkeeping slips, which a prompt-time
checklist does not fix. Confirmed by the bank's contents and by the null.

**High-base-rate tasks have the opposite problem.** Matchstick measured at **0.760**, not
the 0.681 the paper reports for CORE's provider — leaving only 0.24 of headroom, and
guaranteeing ≥5 successes per round, which starved the mistake channel in every round of
every run (§1). ZebraLogic sits between the two and may be the best fit: hard enough to
leave headroom, knowledge-shaped failures, and five grid sizes for the type hierarchy.

**ADOPTED: ZebraLogic is now the default task** (`run_pilot.DEFAULT_TASK`,
`PIPELINE_TASK`). One prerequisite had to be fixed for it — see §7.

---

## 7. Memorisation

**Evidence.** Verbatim training answers reached the bank **twice**, despite
`_ANTI_MEMORISATION` in the prompt:

- Hanoi, skill `tow_ev_1_006`: `example: "Iterate through the list [[1,2,1],[5,0,2],…]"`
  against s201's answer `moves = [[1, 2, 1], [5, 0, 2], …]`
- Matchstick, skill `mat_ev_2_007`: the string `III = XIV - XI`, verbatim, and from a
  **different** problem than the skill was distilled from — which parent-exclusion cannot
  prevent.

`hexa_core/memorisation.py` now detects both shapes automatically (numeric move runs and
verbatim answer strings) and reports as pilot diagnostic 7. A third case surfaced in the
Matchstick runs: run 0's `mat_ev_5_009` carried `XII = XVIII - VI` from a **different**
training problem than it was distilled from, which parent-exclusion cannot prevent.

**The detector was unsound on ZebraLogic and had to be fixed before §6 could be adopted.**
Its answers are grids, and two things went wrong:

- Every *cell* (`"swimming"`, `"HairColor"`) appears verbatim in the puzzle's own prompt,
  because ZebraLogic enumerates each attribute's possible values up front — **9 of 9**
  candidates from the first training problem. A skill saying "check the FavoriteSport
  column" would have been reported as a leak. Fixed by discarding any candidate that
  appears in the problem's prompt: quoting the prompt back is not memorising the answer.
- That filter alone left ZebraLogic with nothing detectable, tripping the module's own
  all-clear guard. The real unit of secrecy is the *row* — the pairing of a person with
  their sport, colour and food, which the prompt cannot supply. Added `_answer_rows`
  (`DEFAULT_MIN_STRING_RUN = 3` cells of one row, header excluded, word-boundary matched
  above `_MIN_CELL_CHARS`) — the cells are short and ordinary, so `"red"` would otherwise
  match inside "predicted" and the house number `"1"` inside anything.

Verified in both directions: naming columns, one value, or two values is clean; a recited
row is caught; and Matchstick's real leak and Hanoi's numeric run both still fire.

**Possible fixes.** Put a **counterexample** in the `example` field rather than a worked
illustration — the failure mode is the reusable part, and quoting a mistake cannot leak a
correct answer. Keep the JSON key `example`: `evolving_distill.py:170` reads
`raw.get("example", "")`, so renaming it in the prompt silently drops the field.

---

## 8. Measurement fragility — fixed, recorded because it distorted earlier conclusions

- **Token cap manufactured a false positive.** At 32768, 53 of 100 eval problems
  truncated to an empty move list in *both* arms. Fixed by `--max-tokens 60000`;
  truncation fell to 27–29.
- **The eval arms were not paired.** `generate_meta` accepted only `prompt`, so the
  per-item seeding raised `TypeError` and fell back to unseeded calls — while the run
  records stored seeds as though the arms were paired. McNemar assumes pairing. Fixed;
  `test_passive_loop.test_eval_arms_are_paired_on_problem_and_seed` pins it.
- **A reasoning error worth not repeating.** After the first run I argued the effect was
  truncation-independent because 11 of 16 wins came from problems where neither arm
  truncated. That was post-hoc subset selection, and it did not replicate: raising the
  cap changes behaviour on *every* problem, not only previously-truncated ones.

---

## 9. The ZebraLogic result — a real positive, currently n=1

**Run:** `results/zebralogic_v2/run0` — the fixed pipeline (`paired` selection,
`max_trajs=12`, n=60/rounds=30/k=10, `n_base=5`, the memorisation fix from §7) on
ZebraLogic, `max_tokens=32768`. 100 held-out problems, same seeds in both arms, bank
frozen at eval time — the same paired design as the Matchstick/Hanoi null results above.

```
with-memory (HeXA)   0.840
no-memory (baseline) 0.570
delta                +0.270
discordant pairs     37  (HeXA-only-correct 32, baseline-only-correct 5)
McNemar exact p      7.4e-6
```

For calibration, the no-memory baseline (0.570) lands close to CORE's published
gpt-oss-120b ZebraLogic number (0.509 ± 0.014, different provider/split) — the provider
offset this project has been tracking since Phase 2 is small here.

**The effect is broad, not one lucky grid size.** Positive delta on every size in the
xlarge pool:

```
size   n   HeXA   base   delta
5x5   21   0.95   0.71   +0.24
5x6   19   0.79   0.58   +0.21
6x4   21   1.00   0.52   +0.48
6x5   23   0.87   0.65   +0.22
6x6   16   0.50   0.31   +0.19
```

**It is not a truncation artifact — checked directly, per §8's own standing warning.**
finish_reason=length: 9 (HeXA) vs 10 (baseline). Unparseable: 12 vs 13. Nearly identical
between arms, so the gain is not HeXA avoiding truncation the baseline suffers from; it
is coming from the reasoning itself.

**The bank (`round29/bank.json`, the frozen bank eval used): 10 skills, 6 mistakes.**
Splits cleanly into logic skills (`RelativeOrdering`, `EquivalenceLinking`,
`FixedHouseAssignment`, `NegativeHouseAssignment`, `TransitiveEquivalencePropagation`,
`ExactDistanceConstraint` — confidence 1.0, the highest in the bank —
`DerivedEntityResolutionInConstraints`) and format skills (`CompactFinalAnswer`,
`ExactAttributeValueMatching`, `LimitReasoningLength`). Manual read: no skill or mistake
text contains a recited grid row; the only digits present are generic illustrative
examples inside abstracted rules ("e.g., house 4"), not training answers — consistent
with §7's fix, though the automated detector was not run end-to-end against this
specific bank (its `reference_for` lookup needs wiring to `CoreTask`, not just the
adapter `Problem.metadata`, which is empty).

**Training-round lift under-predicted this.** Per-round training accuracy minus the
round's own no-memory `b_q` averaged +0.144 across all 30 rounds — real signal, but far
short of the +0.270 the held-out eval actually showed. This is the opposite of the
overfitting failure mode §1/§7 exist to guard against (in-sample signal not
transferring); here the held-out effect *exceeded* the in-sample one. Also the reason
training lift should not be used to predict eval outcome either way: the historical
counterexample below shows the same magnitude of training lift meaning nothing.

**Replicated at n=3 — pooling held.** `run1` and `run2` were run with the identical
config (`PIPELINE_BUDGET=600`, `max_tokens=60000`, `run_index` 1 and 2 so the eval seed
namespaces are disjoint):

```
run   HeXA   no-mem   delta   discordant      p
 0    0.840  0.570   +0.270    32 /  5     7.4e-6
 1    0.780  0.510   +0.270    31 /  4     3.5e-6
 2    0.800  0.600   +0.200    25 /  5     3.3e-4
MEAN  0.807+/-0.018  0.560+/-0.026  +0.247+/-0.023
```

Truncation stayed low and matched between arms in every run (6-10 HeXA vs 9-11
baseline), so §8's artifact does not apply to any of the three.

**Why this is not yet "HeXA works" without qualification.**

- ~~**n=1 run.**~~ Done — pooled over 3 runs above.
- **Old Hanoi run0 had training lift +0.133** — comparable to this run's +0.144 — **and
  its eval delta was exactly 0.000.** Training lift of this magnitude has already been
  shown once to predict nothing. It is not evidence here either; it is background noise
  that happened to point the right way this time. The eval number is the only reason to
  believe this result, not the training curve that preceded it.
- **Whether this generalises past ZebraLogic is untested.** `results/tower_of_hanoi_v3/run0`
  is running the identical fixed pipeline now specifically to answer that. A positive
  Hanoi result makes this "the fixes work"; a null makes it "the fixes fixed ZebraLogic
  specifically" — still real, narrower, and worth knowing before the paper claims either.

**Update: it generalised.** See §10 — Hanoi came back +0.290, p ≈ 1.5e-5, after the
truncation confound described there was caught and fixed.

---

## 10. The Tower of Hanoi result — a real positive, once truncation was fixed

**Run:** `results/tower_of_hanoi_v3/run0` — same fixed pipeline as §9, n=60/rounds=30/
k=10, `n_base=5`, on Tower of Hanoi, initially at `max_tokens=32768` to match §9
exactly. 100 held-out problems (6-disk, BFS-reachable, `min_distance=1`), same seeds
both arms, bank frozen at round 29.

**The first number, at 32768, is not usable — caught by §8's own standing warning
before it was reported anywhere.**

```
with-memory (HeXA)   0.270
no-memory (baseline) 0.190
delta                +0.080
McNemar p            0.169  (not significant)
finish_reason=length  60/100 (HeXA)   57/100 (baseline)
```

60% of both arms hit the token cap — Hanoi's 6-disk optimal solutions run up to 63
moves, and the model's own verbosity (`run_pilot.py`'s docstring already flagged ~25%
truncation on Hanoi smoke runs at this cap, before this run) pushes most attempts past
32768 regardless of memory. With more than half the sample not finishing, §8's rule
applies directly: *check any positive result against a larger token budget before
believing it.* A quick subset check (accuracy restricted to the 26 problems where
neither arm truncated: delta +0.192, p=0.267) was **not** treated as a fix — it is the
exact post-hoc-subset move §8 already documents as a reasoning error that failed to
replicate once, so it was logged as a curiosity, not evidence, and the eval was rerun
properly instead.

**The rerun did not require retraining.** `_frozen_eval` (`hexa_core/passive_loop.py:606`)
takes a task, a frozen bank, and a generate function — no training state — so
`scripts/rerun_frozen_eval.py` (new) calls it directly against the already-saved
`round29/bank.json`, with the same seeding (`llm_seed=42`, `run_index=0`) so the pairing
matches the original run exactly. Only `max_tokens` changed, from 32768 to 60000. Wired
as `PIPELINE=eval_rerun` in `serve_gpt_oss_vllm.sh`. Cost: 200 calls, not the 900 a full
retrain would have needed.

**At 60000, truncation is matched between arms and the result holds — decisively:**

```
with-memory (HeXA)   0.540
no-memory (baseline) 0.250
delta                +0.290
discordant pairs     45  (HeXA-only-correct 37, baseline-only-correct 8)
McNemar exact p      1.5e-5
finish_reason=length  23/100 (HeXA)   22/100 (baseline)
```

Truncation dropped from 60/57 to 23/22 — no longer a confound — and the delta grew,
not shrank. **This contradicts the a priori expectation set by an earlier finding in
this same file:** raising Hanoi's cap once before (old pipeline, `top5` selection) lifted
the baseline by +0.183 against HeXA's +0.083, collapsing an apparent +0.100 edge to
~0.000. The prediction going into this rerun, made explicitly before it ran, was that
the same asymmetry would likely repeat and shrink or erase this delta. It did not:
with-memory gained +0.270 (0.270→0.540) against baseline's +0.060 (0.190→0.250) — the
opposite asymmetry. Whatever produced the old pattern (a different bank, under `top5`,
with no failures ever reaching the evolver) did not recur under the fixed pipeline.
Recorded here specifically because the prediction was wrong, not adjusted after the
fact to look right.

**So: two of three tasks now show large, truncation-checked, statistically solid
positive deltas under the identical fixed configuration** — ZebraLogic +0.270 (p≈7e-6),
Hanoi +0.290 (p≈1.5e-5). This is a materially stronger claim than §9 alone supported:
"the fixes fixed ZebraLogic specifically" is no longer the leading explanation now that
the same bundle produced a comparably large effect on a second, mechanistically
different task (grid constraint-satisfaction vs. sequential planning). "The fixes work"
is the claim the evidence currently supports, pending the same caveats as §9:

- ~~**n=1 per task.**~~ Done — both pooled at n=3. Hanoi's three runs, each evaluated at
  60000 via `PIPELINE=eval_rerun` against its own frozen `round29/bank.json`
  (`run_index` 0/1/2, disjoint eval seeds — verified zero seed overlap between runs):

  ```
  run   HeXA   no-mem   delta   discordant      p      trunc (wm/nm)
   0    0.540  0.250   +0.290    37 /  8     1.5e-5      23 / 22
   1    0.510  0.270   +0.240    31 /  7     1.2e-4      25 / 19
   2    0.510  0.270   +0.240    36 / 12     7.2e-4      29 / 15
  MEAN  0.520+/-0.010  0.263+/-0.007  +0.257+/-0.017
  ```

  `run1` and `run2` landing on identical 0.510/0.270 is coincidence in the marginal
  totals, not a duplicated run: their eval seed lists are fully disjoint (zero overlap)
  and their discordant and truncation counts differ.
- ~~**Matchstick is the live open question.**~~ Resolved — see §11. It is a null
  (+0.020, p = 0.688) once trained at the right token budget, and §6's task-fit argument
  is now directly supported rather than being a fallback: the no-memory baseline is
  **0.780**, leaving ~0.22 of headroom, and of that remaining headroom almost all is
  truncation rather than reasoning error.

**Caveat discovered afterwards: the 32768 → 60000 comparison is not a clean isolation
of the token budget.** Generation on this server is **not reproducible across server
instances even at a fixed seed** — vLLM's continuous batching makes numerics depend on
batch composition. Measured directly on `run0`: comparing the 32768 eval against the
60000 rerun, **9 of the 15 problems that were correct *and* terminated naturally at
32768 came back wrong at 60000**, with byte-identical prompts (no bank in either arm)
and identical recorded seeds. A generation that stopped well short of 32768 cannot be
altered by raising the cap to 60000, so those are fresh samples, not extensions.

Two consequences, and only the second is a real limitation:

- The **paired within-run** comparisons above are unaffected: both arms of a given run
  execute against the same server instance under the same batching conditions, so
  McNemar on discordant pairs remains valid. Every delta in this file stands.
- The narrative "raising the cap lifted HeXA 0.270 → 0.540 against the baseline's 0.190
  → 0.250" **conflates the cap change with a resample**, and should not be cited as
  evidence about what token budget does. The defensible statement is the one this
  section already rests on: at 60000, with truncation matched between arms, the delta is
  +0.257 ± 0.017 over three runs.
- **Always check truncation before trusting a delta on a task with long-form answers.**
  This is now confirmed twice on Hanoi specifically (the original false positive in §8,
  and the 32768 confound here) — any future Hanoi run at 32768 should be treated as
  provisional until truncation is checked, not just reported.

---

## 11. Matchstick — a null, and a −0.250 that was my own misconfiguration

**This section exists mainly to record an error in how the run was launched, because the
wrong number was interesting enough to have been believed.**

`results/matchstick_v2/run0` was requested with "max tokens same as zebralogic" — i.e.
60000 — and was launched at `max_tokens=32768`. Provenance confirms it
(`results/matchstick_v2/run0/provenance.json`: `max_tokens: 32768`, against
`zebralogic_v2/run0`'s 60000). The mistake was mine and went unnoticed for the whole
run and both of its evals.

**What the bad configuration produced:**

```
                                  HeXA   no-mem   delta   discordant       p      trunc
v2, trained 32k, eval 32k        0.420   0.690   -0.270    3 / 30      1.4e-6   56 / 29
v2, trained 32k, eval 60k        0.470   0.720   -0.250    3 / 28      4.7e-6   36 / 24
v3, trained 60k, eval 60k        0.800   0.780   +0.020    4 /  2      0.688    18 / 20
```

Re-running only the eval at 60000 did **not** fix it (−0.250, still highly significant)
— because the damage was in *training*, not measurement. Retraining at 60000
(`results/matchstick_v3/run0`, provenance verified `max_tokens: 60000`) moved the delta
from −0.250 to +0.020.

**Why the low cap was so destructive here specifically — the evolver had nothing to
learn from.** Decomposing training failures by mode:

```
task / config          failures   truncated   unparseable   WRONG-ANSWER
Matchstick @32k           279     271 (97%)        0            8 (3%)
Matchstick @60k           145     116 (80%)        5           24 (17%)
ZebraLogic @60k           120      48 (40%)       27           45 (38%)
Hanoi @32k                370     172 (46%)        0          198 (54%)
```

At 32768, Matchstick gave the evolver **8 genuine reasoning failures across 30 rounds**
— and those 8 were only 3 distinct problems. The evolver is asked "what did the accepted
attempts do that the rejected ones did not?"; the truthful answer was "the rejected ones
ran out of tokens," which is not a reasoning strategy. Note this is *why* Hanoi survived
the same 32768 cap: 54% of its failures were real wrong answers, so its evolver still
had signal. The cap cost Hanoi some evidence; it destroyed Matchstick's.

**What it wrote instead, and how it did harm.** The v2 bank was 7 of 10 skills restating
constraints **already stated verbatim in the base prompt** ("move exactly one
matchstick", "you cannot simply add a matchstick to the equality sign", "if you take a
stick away from X or V, the remaining stick won't be a valid I", the operator set, the
arithmetic rule). It spent 1016 injected tokens repeating the problem statement, and
converted each restated rule into a per-candidate *verification obligation*. Result: on
identical problems and seeds, injection raised truncation from **29% to 56%** — and
truncation is fatal (accuracy 0.089 with memory, 0.000 without, against 0.841/0.972 when
generation completes). The −0.270 was overwhelmingly a truncation effect the bank caused.

Worse, its only three useful skills — concrete legal-move transformations (V→X, IV↔VI,
X↔V diagonal transfers) — carried the bank's **lowest** confidences (0.225-0.38) and
ranked 8th, 9th and 10th, while the budget walk cut 4 of the 5 mistakes. **Confidence
was anti-correlated with usefulness**, structurally: it is derived from the rewards of
the attempts a skill came from, and on Matchstick@32k nearly every attempt on a hard
problem was a truncation, so skills drawn from the problems that needed help inherited
the worst scores. The prompt's existing generic-skill guard ("would a capable solver have
said this before seeing a single attempt? If yes, drop it") failed because a truncated
ramble contains half-finished validity checks, which makes "verify the shape is valid"
*look* grounded in an attempt.

**At 60000 the evolver produced a genuinely good bank, unprompted.** `matchstick_v3`'s
10 skills are almost the inverse of v2's:

- **A search-order policy** — *"Balance by numeric difference and single-stick
  transformation"* (conf 0.6): compute the numeric imbalance first, then find the single
  move that changes a numeral or operator by exactly that amount. This is precisely the
  inverse of v2's observed error, which was to guess an arithmetically-true equation and
  back-fill a stick story. (All 8 of v2's genuine wrong answers were *arithmetically
  correct* — `X = XI - I`, `V = XVII - XII`, `V = XIII - IIX` — so v2's #2-ranked skill,
  "Arithmetic check after transformation", targeted a failure mode with zero occurrences.)
- **A move catalogue with value deltas** — V→X (+5), X→V (−5), I-transfer (±1),
  I-transfer for imbalance 2, `+`→`−`, and the `−`↔`=` horizontal swap.
- **A tie-break heuristic** — prefer numeral adjustments over operator changes.
- Confidences are 0.5-0.7 (against 0.225-0.55), and the useful skills now rank **first**.
- Only 1 of 10 is a mild rule-restatement, against 7 of 10 in v2.

This substantially undercuts the plan to write a Matchstick-specific evolution prompt:
fixing the token budget produced the bank that prompt was meant to force.

**So why is the delta only +0.020? There is almost no headroom left, and what remains is
truncation, not reasoning.**

```
            acc     on completions    on truncations
HeXA       0.800   0.976 (80/82)     0.000 (0/18)
no-memory  0.780   0.975 (80/82)     0.000 (0/20)
```

**When generation finishes, the model is 97.6% correct.** Exactly **2** eval problems are
wrong-but-completed. Every other loss in both arms is truncation, which costs ~0.176 of
accuracy and caps the achievable score near 0.976. Mean training `b_q` is **0.700**, and
the eval baseline 0.780 — §6's task-fit argument, now with direct evidence: a skill bank
cannot add much to a task the model already solves 78% of the time unaided, and the
residual is a token-budget problem rather than a knowledge problem.

**The gap worth attacking, and it is not skill content.** The v3 bank contains **no
brevity or token-budget skill** (checked: zero of 10 skills mention concision, length,
tokens, or verbosity). ZebraLogic's bank spontaneously produced `LimitReasoningLength`
(conf 0.6) and runs 9-10% truncation; Matchstick has none and runs 18-20%. That is the
one thing a task-specific prompt should still force — and it generalises: *when
truncation is a material share of failures, the bank must carry an early-commit strategy,
not more verification steps.* For Matchstick the natural form falls out of its own skill
#1: once the imbalance is computed and a candidate move closes it exactly, verify that
one candidate and answer, rather than enumerating the rest of the move space.

Caveat on any such improvement: fixing truncation raises **both** arms (it costs the
baseline 0.195), so it improves absolute numbers against CORE's table without
necessarily creating a HeXA-vs-baseline delta. Matchstick's headroom problem is real
independent of the truncation problem.

### 11a. Two follow-up arms, both abandoned before eval — the cap was not the problem

Both are archived under `results/matchstick_v4/` with their own `ABANDONED.md`, and
neither has an eval number.

The plan after §11 was that the v3 bank lacked a brevity skill (unlike ZebraLogic's
`LimitReasoningLength`), so a truncation-aware evolver plus a smaller injected block
should help. Neither half survived contact.

1. **`run0_maxinject5_ABANDONED`** — killed at round 11 for a plumbing fault, not a
   result. `PIPELINE_TRUNCGUIDE=1` did its job: with truncation at ~89% of training
   failures the evolver wrote `Early exit on first valid solution`, `Bounded
   enumeration`, `Prioritize operator moves before numeral moves`. But those skills
   carried the *lowest* confidences in the bank (0.40–0.45) — confidence is derived
   from source-trajectory rewards, and truncation-fixing skills are by construction
   distilled from truncations — so `PIPELINE_MAXINJECT=5` cut every one of them.
   The two "fixes" cancelled: fix 3 filtered out exactly what fix 1 produced.
2. **`run0_maxinject10_ABANDONED`** — the same arm with the cap removed, so the
   brevity skills actually reach the solver (verified: 9/10 injected at round 12).
   Killed at round 16. Paired against v3 on identical problems in identical chunk
   order:

   | paired, 16 rounds | v3 | v4 | delta |
   |---|---|---|---|
   | accuracy | 0.775 | 0.708 | **−0.109** |
   | truncation | 17.5% | 23.8% | **+0.078** |
   | added tokens | 873 | 997 | +124 |

**What this rules out.** "The brevity skills were not reaching the solver" is tested
and refuted. They reach it and truncation still *rises*. The retrieval cap was never
the binding constraint, and neither was the evolver's awareness of truncation.

**What remains.** The bank's content, not its delivery. It is dominated by
per-candidate obligations (`Arithmetic verification`, `Source numeral validity`,
`Destination numeral validity`, `Stick count preservation`), each a check run per
candidate move, each lengthening generation on the one task where truncation *is*
the error mode. Two brevity skills do not outweigh six obligations to check more.
Confidence had also flattened (six skills at 0.50, three at 0.45), so ranking no
longer discriminates and "inject only the best few" is not available anyway.

So on Matchstick the direction is to *shrink* what is injected, not enlarge it —
873 tokens beat 997, and both lose to no-memory's 0. §11's null stands.

**Outstanding:** v3 is n=1. The two positives are pooled at n=3; this null is not.

---

## 12. MathGAP — the task description is load-bearing in a way the others' are not

No MathGAP result yet; this records a caveat that must accompany one whenever it
lands, because it is structural and survives any wording we settle on.

CORE ships MathGAP with **no task instruction at all**. Measured, not assumed:

| task | `task_instruction` the **actor** sees |
|---|---|
| matchstick | 1437 ch |
| tower-of-hanoi | 1120 ch |
| zebralogic | 34 ch + a per-problem JSON schema embedded in the prompt |
| **mathgap** | **0 ch** (`output_instruction` is `""` too; the format line is baked into each problem's text) |

For the other three, `CORE_TASK_DESCRIPTIONS` largely restates rules the actor
*already has in its own prompt* — the evolver is being told what the solver knows.
For MathGAP, everything in that description is net-new information entering a system
that otherwise had none. So a MathGAP gain is attributable partly to the description
and only partly to the learning loop, and it is not directly comparable to the other
three tasks' deltas.

This is not hypothetical. Two runs were abandoned before eval on exactly this axis:

1. **`run0_signbug_ABANDONED`** — the description named the comparison-transfer form
   but never said it was *signed*. The evolver guessed absolute value, banked
   `Treat 'has more than' as absolute difference` at the top confidence (0.52), and
   every later round was solved through it. Training lift **-0.250** over the
   injected rounds. Verified wrong against a train item with a published target:
   the signed reading reproduces it exactly, the absolute reading leaves the answer
   undetermined.
2. **`run0_privinfo_ABANDONED`** — the repair worked (lift **-0.250 → +0.463**, bank
   inverted to `Difference Order Preservation` 0.80 / `Signed Difference Equality`
   0.65) but over-reached. It argued the absolute reading was incorrect, and it
   supplied an **answer-space prior** ("every answer lies between 10 and 30"), which
   the evolver banked as `Sanity Check Answer Range`. On a task with integer answers
   and a **~5% guess floor** that is a benchmark-specific crutch, not reasoning. Its
   provenance was also wrong: the range was computed over all 280 pool rows including
   eval, though train-only gives the same 10..30 with 0/100 eval targets outside it,
   so nothing eval-specific actually reached the prompt.

Both were killed before eval and neither has a quotable number. The lesson is the
rule, not the two instances: **on MathGAP, changes to the task description move the
result as much as changes to the method do.** Any MathGAP number should be reported
with the description it was produced under, and an ablation against a minimal
description would be the honest way to separate the two.

Related: MathGAP's eval split is regime-dependent (`data[n : n+100]` from one 280-row
shuffled pool), so its numbers are comparable only within a regime — including
against CORE's own column, which is n=100 and therefore a different 100 items. See
`RECON.md` §1.2 and `CORE_COMPARISON.md` §1.

---

## Suggested order — steps 1–3 and 5 are now done

1. ~~**§2 items 1–4** — prompt only.~~ Done.
2. ~~**§1** — raise `max_trajs` (with `Thought` truncation) and get failures in front of
   the evolver.~~ Done.
3. ~~**§4** — more problems, fewer attempts.~~ Done.
4. **§3** — efficiency-graded reward, watching the domain. Deferred: `paired` already
   supplies the success/failure mixing this was for, and the earlier M1/M2 units bug
   (`reward − b_q` spanning [−2,1] and clamping) came from exactly this kind of change.
5. ~~**§6** — task choice.~~ Done (ZebraLogic). **§5** — retrieval cost — still open, and
   deliberately: injection is already net-negative, so raising `max_skills` or the token
   budget makes it worse before retrieval is query-conditional.

**What all three tasks tested, and where that leaves the claim.** Everything above
changed at once, which was the wrong shape for attributing a *result* but the right shape
for finding out whether there is one at all. ZebraLogic and Hanoi came back decisively
positive and are now pooled at n=3 (§9, §10: +0.247 ± 0.023 and +0.257 ± 0.017).
Matchstick is a null (§11: +0.020, p = 0.688), on the task whose no-memory baseline is
0.780 and whose residual failures are 97% truncation rather than reasoning error.

That rules out "the fixes do nothing", and rules out "the fixes are ZebraLogic-specific"
— the same bundle produced a comparably large effect on a mechanistically different task
(grid constraint-satisfaction vs. sequential planning). The supported claim is: **the
fixed pipeline produces large, replicated gains on tasks with headroom, and neither helps
nor hurts on a task without it.**

Next, in order:

1. **The two fixes §11 identifies** — force a brevity/early-commit skill when truncation
   is a material share of failures, and shrink the injected block (§5). Both target
   truncation, which is now the binding constraint on Matchstick and a live cost on
   Hanoi (15-29%). Cheap to test and the mechanism is proven on ZebraLogic.
2. **Replicate Matchstick v3 to n=3**, so all three tasks are at the same standard.
3. **The ablation arms** — `PIPELINE_SELECTION=top5`, `PIPELINE_MAXTRAJS=5`,
   `PIPELINE_NTRAIN=20 PIPELINE_ROUNDS=10` — to attribute the win to a specific fix
   rather than to the bundle. This is the biggest remaining gap in the story.
4. **Reconcile the Hanoi provider-gap discrepancy** flagged in `CORE_COMPARISON.md`
   (`PHASE3_DESIGN.md`'s n=20 probe read 0.450 against the paper's 0.179; our n=100
   runs read 0.263 ± 0.007).

Also worth remembering from §8, confirmed again by §10: **check any positive result
against a larger token budget before believing it.** The one apparent win this project
had produced before this session was a truncation
artifact.

A cheap diagnostic still not run: ask the base model, with no memory, whether each banked
skill is already obvious for the task. The fraction it affirms tells you whether the
problem is the distillation or the task, and it is one call per skill.
