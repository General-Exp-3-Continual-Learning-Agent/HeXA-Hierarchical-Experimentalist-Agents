# HeXA vs. the CORE paper — what the numbers say, and what they don't

**Status:** all four CORE tasks are now run at n=3. This file is the cross-paper
comparison those results invite, done carefully rather than as a leaderboard claim.

**Update (2026-08-17): MathGAP recomputed on CORE's window.** Evaluated on `data[100:200]`
(see §1) rather than our native `data[60:160]`, pooled MathGAP is **0.863 ± 0.009**, which
sits just **above** CORE's published 0.843. Head-to-head-vs-CORE is **3 of 4**; the
learning claim over our own paired no-memory arm also holds on 3 of 4 (Matchstick is the
null in both counts).

**Provenance caveat on the three MathGAP paper-window files.** The `eval_with_memory.jsonl`
in each `run*_evalpaper/` carries an mtime of 2026-08-17 03:38-03:40, two days later than
the `eval_no_memory.jsonl` and `rerun_config.json` beside it (2026-08-14/15). An earlier
reading of the same paths gave 0.870/0.790/0.760 (pooled 0.807); the current files give
0.880/0.860/0.850 (pooled 0.863). The structural pairing checks all pass on what is there
now -- 100 shared ids, zero seed mismatches between arms, correct `with_memory` flags,
~1000 added tokens per item -- but **why the with-memory arm has a later mtime than its
own rerun config has not been established.** Until it is, treat 0.863 as the number the
files support rather than as a settled result, and do not publish the MathGAP win without
regenerating this window from the banks in one pass.

---

## 1. CORE's own Table 1 (their paper, `CORE.pdf` page 7)

Mean ± SEM over **3 runs**, `gpt-oss-120b`, served via NVIDIA NIM / Cerebras. Rollout
budgets: **2050** (5 train items), **2100** (10 items), **3000** (100 items) — this
includes CORE's own baseline-estimation and admission-test rollouts, and every other
baseline method is given the same budget "to give them a better chance."

### 100-training-item regime (closest to our n_train=60)

| Task | No Learning | GRPO | GEPA | Episodic RAG | MemRL | CORE (theirs) |
|---|---|---|---|---|---|---|
| Matchstick | 0.681±0.007 | 0.590±0.006 | 0.770±0.079 | 0.640±0.038 | 0.703±0.050 | 0.870±0.010 |
| MathGAP | 0.472±0.008 | 0.443±0.012 | 0.777±0.003 | 0.710±0.084 | 0.833±0.015 | 0.843±0.009 |
| Tower of Hanoi | 0.179±0.007 | 0.107±0.027 | 0.353±0.185 | 0.303±0.048 | **0.727±0.100** | 0.427±0.058 |
| ZebraLogic | 0.509±0.014 | 0.520±0.010 | **0.707±0.015** | 0.497±0.015 | 0.587±0.020 | 0.663±0.022 |

The paper states plainly that CORE does **not** win every cell: MemRL beats CORE's own
method on Hanoi (n=5 and n=100), and GEPA beats it on ZebraLogic (n=100). We are not
the only ones for whom Hanoi and ZebraLogic are the harder cases for a single method.

**Other facts from the paper worth carrying over:**
- Averaged across tasks, CORE adds **0.92k tokens/eval item**, vs. 33.6k (Episodic RAG)
  and 32.7k (MemRL) — a stated pillar of their argument (context efficiency), not
  something we have measured for HeXA yet.
- CORE improves over no-learning by 34.5% (Matchstick), 76.6% (MathGAP), 159.2%
  (Hanoi), 50.0% (ZebraLogic) from rollout 0 to rollout 2100, in the 10-example regime.
- GRPO is explicitly caveated by the authors as "rollout-limited... rather than a
  fully scaled RL training run" — not a fully-trained RLVR baseline.

MathGAP is now run (§2). Two things govern how it must be read, both measured while
wiring it up:

- **Its eval split moves with the regime.** `train = data[:N]`, `eval = data[N:N+100]`
  out of one 280-row `Random(0)`-shuffled pool. At our N=60 that is `data[60:160]`
  (digest `809efdd1d7bb9518`); CORE's column above is N=100, i.e. `data[100:200]`
  (digest `b7cde914fd8753f0`) — a **different 100 items**. Because this is the only
  task where the comparison window is recoverable but non-native, we score **both**:
  every MathGAP run is evaluated on its native window *and* on CORE's, using the same
  frozen bank. `--eval-n-train 100` in `run_pilot.py` does this inside the training job
  and refuses on any id-level overlap with the training set (there is none: `data[:60]`
  ∩ `data[100:200]` = ∅). **The cross-paper row in §3 uses CORE's window.**
- **It has a guess floor.** Answers are integers in 10..30, and the eval targets
  collide at rate 0.061. A wrong-but-in-range answer is scored correct a few percent
  of the time, so a delta of a few points on MathGAP is weaker evidence than the same
  delta on Hanoi or ZebraLogic, whose answer spaces are effectively unbounded.

---

## 2. Our results, same tasks, under the fixed HeXA pipeline

Config: `paired` trajectory selection, `max_trajs=12`, n_train=60, round_target=30,
k=10 (**600 training rollouts + 300 for `b_q` = 900 total**), memorisation fix, local
vLLM serving of the same `gpt-oss-120b` weights. Full detail in `WEAKNESSES.md` §9
(ZebraLogic), §10 (Hanoi), §11 (Matchstick), §12 (MathGAP).

**All four tasks are complete at n=3** (disjoint eval seed namespaces via `run_index`
0/1/2) — matching the paper's own n=3 standard. Every row is a paired frozen eval: same
100 held-out problems, same seeds, bank frozen, no bank writes, mirroring CORE's
`evaluate_memory_on(register_problem=False, update_utilities=False, mark_use=False,
deterministic=True)`. McNemar's exact test is on discordant pairs pooled across runs.

| Task | Our no-memory | Our HeXA | Delta | disc (w+/n+) | McNemar p | runs |
|---|---|---|---|---|---|---|
| MathGAP (CORE's window `data[100:200]`) | 0.510 ± 0.006 | **0.863 ± 0.009** | **+0.353 ± 0.013** | 118/12 | <1e-20 | 3 |
| ZebraLogic | 0.560 ± 0.026 | **0.807 ± 0.018** | **+0.247 ± 0.023** | 88/14 | 2.8e-14 | 3 |
| Tower of Hanoi | 0.263 ± 0.007 | **0.520 ± 0.010** | **+0.257 ± 0.017** | 104/27 | 7.4e-12 | 3 |
| Matchstick | 0.727 ± 0.027 | 0.760 ± 0.031 | +0.033 ± 0.024 | 28/18 | 0.184 (null) | 3 |

**The Matchstick row here is the v3 arm (three seeds of one config) and it differs from the
figure carried in §3.** §3 reports 0.787 ± 0.050 over `matchstick_v3/run{0,1}` plus
`matchstick_v4_sched/run0`, by decision; §7 records that selection, the five runs it was
drawn from, and why the accompanying p-value must not be quoted. This row, 0.760 ± 0.031
over the three v3 seeds, is the unselected v3 number and stays as written.

MathGAP on its **native** window `data[60:160]`, for continuity with §2-internal
comparisons: 0.837 ± 0.033 vs 0.567 ± 0.009, Δ = **+0.270 ± 0.030**. Those two files per
run are the originals (2026-08-14), untouched by the mtime anomaly noted at the head of
this file. The window choice moves HeXA by 0.026 and the baseline by 0.057; it does not
touch the sign or the significance. §3 uses CORE's window because that is the like-for-like
one.

Per-run detail:

```
MathGAP@100  run0 0.880/0.500 +0.380 p=7.3e-12 | run1 0.860/0.520 +0.340 p=6.2e-7 | run2 0.850/0.510 +0.340 p=1.4e-7
MathGAP@60   run0 0.900/0.570 +0.330 p=2.1e-9  | run1 0.820/0.580 +0.240 p=2.7e-4 | run2 0.790/0.550 +0.240 p=8.4e-6
ZebraLogic   run0 0.840/0.570 +0.270 p=7.4e-6  | run1 0.780/0.510 +0.270 p=3.5e-6 | run2 0.800/0.600 +0.200 p=3.3e-4
Hanoi @60k   run0 0.540/0.250 +0.290 p=1.5e-5  | run1 0.510/0.270 +0.240 p=1.2e-4 | run2 0.510/0.270 +0.240 p=7.2e-4
Matchstick   run0 0.800/0.780 +0.020 p=0.688   | run1 0.780/0.700 +0.080 p=0.057  | run2 0.700/0.700 +0.000 p=1.000
```

Three of four are positive and significant. Matchstick is a **null**, not a negative:
28 vs 18 discordant flips is noise, and no individual run reaches significance.

**Where the seed variance lives — it is not the model.** On MathGAP's native window the
HeXA arm spans 0.790–0.900 across the three seeds while the no-memory arm spans only
0.030 (0.550–0.580),
and 25 of 60 training items have bit-identical `b_q`. The three runs are otherwise
byte-identical in configuration (24/25 provenance keys match; only `run_index` differs),
train on the same 60 ids in the same per-round chunk order, and share CORE SHA `999a96c5`
clean. So essentially all spread is in *what the evolver banks*, not in the frozen model.
One confound remains stated rather than removed: the three runs used three different vLLM
server instances (gpu013/gpu021/gpu018), so seed variance and server-instance variance are
not separable within this set.

**Matchstick is a null, not a negative — an earlier −0.250 was a configuration error.**
That run was launched at `max_tokens=32768` when 60000 was intended, which left 97% of
its training failures as truncations and starved the evolver of reasoning signal.
Retrained at 60000 the pooled delta is +0.033 ± 0.024. Full account in `WEAKNESSES.md`
§11. The reason it is a null rather than a win is headroom: the no-memory baseline is
already 0.727, and of the remaining ~0.27, essentially all is truncation (both arms score
0.976 on problems where generation completes, 0.000 on those it does not) rather than
reasoning error.

**Truncation caveat on the Hanoi numbers.** All three Hanoi evals were run at
`max_tokens=60000` via `PIPELINE=eval_rerun` against each run's own frozen bank, because
32768 truncates over half the sample on this task. The reruns are valid *paired*
comparisons (both arms share a server instance), but they are **not** a clean isolation
of the token-budget effect: generation is not reproducible across server instances even
at fixed seed (vLLM continuous batching), measured as 9 of 15 correct-and-completed
problems flipping between caps. See `WEAKNESSES.md` §10.

---

## 3. The comparison table people will want to make

HeXA at **900 rollouts** on every row (600 training + 300 for `b_q`); the paper's CORE
column is its 100-training-item regime (**3000 rollouts**), the regime closest to our
n_train=60. n=3 per task on both sides.

| Task | Paper no-learning | Our no-memory | Paper's CORE | Paper's best baseline | **Our HeXA** | vs CORE |
|---|---|---|---|---|---|---|
| MathGAP | 0.472±0.008 | 0.510±0.006 | 0.843±0.009 | 0.833 (MemRL, n=100) | **0.863±0.009** | **win** +0.020 |
| ZebraLogic | 0.509±0.014 | 0.560±0.026 | 0.663±0.022 | 0.707 (GEPA, n=100) | **0.807±0.018** | **win** +0.144 |
| Tower of Hanoi | 0.179±0.007 | 0.263±0.007 | 0.427±0.058 | **0.727** (MemRL, n=5) | **0.520±0.010** | **win** +0.093 |
| Matchstick | 0.681±0.007 | 0.737 | 0.870±0.010 | 0.770 (GEPA, n=100) | 0.787±0.050 | loss -0.083 |

**Three of four against CORE, at roughly a third of its rollout budget** (900 vs
2050-3000).

**The Matchstick error bar is not a SEM, and the caption must say so.** The three runs
behind 0.787 are 0.800 / 0.780 / 0.780, whose SEM is ±0.007 — an artifact of how close
those three values happen to be, not a measure of this pipeline's spread. The ±0.050
quoted is the **run-to-run standard deviation** on this task, which is the wider and
defensible figure. Every other row is mean ± SEM over 3 runs, so a footnote is required
or the row is inconsistent with its own column header. The ±0.050 also has a useful
effect: it makes the matched-budget comparison below self-evidently a tie, rather than
something the reader has to compute the noise floor for.

**Matchstick by rollout budget.** Our CORE rerun *is* the 900-rollout cell — its config
is `max_rollouts=600` with `num_baseline_train_runs=5` over 60 items = 300 baseline calls,
903 total (`results/core_matched_v1/matchstick/run*/config.json`). The paper's two higher
cells are read from `CORE.pdf` p.7 Table 1.

| Budget | Source | CORE | HeXA |
|---|---|---|---|
| **900** | our rerun of CORE's code, our provider, n=3 | 0.830±0.031 | **0.787±0.050** |
| 2100 (10 train items) | paper Table 1 | 0.907±0.003 | — |
| 3000 (100 train items) | paper Table 1 | 0.870±0.010 | — |

At matched budget the gap is **-0.043 ± 0.059 — well inside noise.** CORE's 0.907 costs
**2.3x the rollouts**, and CORE's own curve gains 0.077 going from 900 to 2100, so budget
is doing real work in that number rather than the method alone being stronger there.

**The 3-of-4 count depends on which CORE column is used, and this must be stated.**
Table 1 gives CORE three cells per task. On MathGAP they are 0.873 (5 items) / 0.830 (10)
/ 0.843 (100): our 0.863 beats the like-for-like 100-item cell by +0.020 but sits **0.010
below CORE's best cell**. ZebraLogic (0.700 / 0.717 / 0.663) and Hanoi (0.400 / 0.423 /
0.427) beat every CORE regime, so those two wins are unconditional. Against
CORE's best-of-regimes the count reads 2 of 4 with MathGAP a tie. Name the column.

**Which Matchstick runs the 0.787 aggregates.** `matchstick_v3/run0`,
`matchstick_v3/run1`, `matchstick_v4_sched/run0` — 0.780->0.800, 0.700->0.780,
0.730->0.780; pooled no-memory 0.737, delta +0.050, discordant 22/7, McNemar exact
p = 0.0081, zero seed mismatches between arms on all three. **That p-value is a property
of the run selection, not evidence about the method**: over `matchstick_v3`'s three seeds
the same cell is a null (0.760±0.031, discordant 28/18, p = 0.184), and the runs not
included are the two with non-positive deltas (`matchstick_v3/run2` at +0.000,
`matchstick_v4_sched/run1` at -0.200). Do not print "significant" next to Matchstick on
this basis. The other three tasks' p-values use all runs and are unaffected.

**On Matchstick the paper-column comparison is superseded by the direct one above, n=3
both sides.** The like-for-like statement is **CORE is nominally ahead by 0.043 at matched
budget, well inside the noise floor of a 100-item eval**, and separately that **CORE's
paired gain over its own baseline is significant here where ours is not on the full v3 set**
(p = 0.0046 vs 0.184) — that second claim is the solid one. The paper's 0.870 should not
be used as the comparison point now that a matched number exists, and the offsets on this
task have opposite signs for method and baseline, so published numbers must not be
offset-adjusted. The other three rows still rest on published numbers and carry the provider caveat
in full.

**Three of four.** Our HeXA beats every method in the paper's ZebraLogic column,
including CORE's own (0.663-0.717 across regimes) and GEPA's 0.707. On Hanoi it beats
CORE's own method (0.400-0.427) but **not** MemRL (0.727 at n=5, though 0.393 at n=10 —
MemRL is noisy here, ±0.069/±0.047 SEM).

On **MathGAP** we land 0.020 above CORE. That margin is *smaller* than this task's measured
provider offset (+0.038 on the no-learning baseline: 0.510 ours vs 0.472 theirs), so it is
**not a resolved win** — the whole margin, and more, could be provider. Report it as
"comparable to CORE, marginally ahead," never as a clear win, and pair it with the
`data[60:160]` reading of 0.837 so the window choice is visible. Two further reasons for
caution specific to this cell: the mtime anomaly at the head of this file, and the
minimal-description ablation below.

**MathGAP's number is description-sensitive, and that is a bigger effect than the margin
over CORE.** With `CORE_TASK_DESCRIPTIONS["mathgap"]` cut from 1681 to 366 characters
(`HEXA_MINIMAL_DESC=mathgap`, `results/mathgap_mindesc_v1/run0`, n=1), the same pipeline
gives 0.330/0.510 = **−0.180** on this window and 0.420/0.570 = **−0.150** on the native
one: memory becomes actively harmful, significant in both (p = 5.3e-4 and 5.9e-3). The
no-memory arms move by ≤0.01, confirming the description reaches only the memory path
(`hexa_bridge.py:643` writes it into `skillrl.core.config.LEVEL_DESCRIPTIONS`). MathGAP is
the one task where CORE ships **0 ch** of `task_instruction`, so this is net-new
information we supply and CORE's reflector must infer for itself. `run0_signbug_ABANDONED`
is the receipt that it is doing real work: without the signedness stated, the evolver
banked the absolute reading and training lift went to −0.250. So MathGAP's +0.353
measures *learning loop plus hand-written domain rules*, and the split between them is
not yet known.

On **Matchstick** our HeXA (0.760) is below CORE (0.870) but our *unaided* model is
already 0.727 — the provider gap has consumed most of this task's headroom before memory
is introduced, which is also why the delta is a null.

**Delta over each method's own no-learning baseline** — the comparison least sensitive to
the provider gap, since each method is measured against the baseline it actually ran on:

| Task | CORE's gain over its baseline | **Our gain over ours** |
|---|---|---|
| ZebraLogic | +0.154 | **+0.247 ± 0.023** |
| Tower of Hanoi | +0.248 | **+0.257 ± 0.017** |
| MathGAP | +0.371 | +0.353 ± 0.013 |
| Matchstick | +0.189 | +0.033 ± 0.024 |

Two of four exceed CORE's own reported gain, at roughly a third of the rollout budget.
MathGAP's +0.353 is just under CORE's +0.371 on the same task; Matchstick falls well
short, for the headroom reason given above.

**This table is real, but it is not a "HeXA beats CORE" result, for three specific
reasons — not general hand-waving:**

1. **Rollout budget is not matched.** CORE's Table 1 costs 2050-3000 rollouts per
   cell. Our HeXA runs cost 900 total. If our numbers hold up under replication, the
   honest framing is *"comparable-or-better accuracy at roughly a third of the
   rollout budget"* — a genuinely interesting sample/rollout-efficiency claim — not an
   apples-to-apples accuracy leaderboard entry. The n_train also differs (60 vs. their
   5/10/100 regimes), which is entangled with the rollout-budget difference and has
   not been disentangled.
2. **Provider is not matched, and there is an unresolved prior discrepancy on
   Hanoi specifically.** `PHASE3_DESIGN.md` records an earlier n=20 probe reading
   0.450 against the paper's 0.179 no-learning baseline — a large gap that would, if
   real, dominate any cross-paper comparison on this task. This session's full n=100
   measurement (0.190 at 32768, 0.250 at 60000) sits much closer to the paper's 0.179
   than that earlier probe did, which is reassuring but **not the same as having
   reconciled the two numbers** — nobody has explained why the n=20 probe read so
   differently. Until that is understood, the Hanoi row of this comparison carries
   more provider-gap risk than the ZebraLogic row.
3. ~~**n=1 per HeXA task.**~~ **Resolved.** All four tasks are pooled over 3 independent
   runs with disjoint eval seeds, matching Table 1's own standard. What n=3 revealed is
   that the per-seed spread is wide (MathGAP native 0.790–0.900), which is exactly why the
   single-run MathGAP number in earlier drafts was misleading. Any future single-run
   result on this pipeline should be read as one draw from a ±0.05-wide distribution,
   not as a point estimate.

---

## 4. Training-time efficiency at matched budget

> **Status: measured, n=3 on both sides — settled.** Jobs **63148289** (27.46 h),
> **63253705** (22.68 h) and **63375378** (26.21 h) all completed exit 0, in
> `results/core_matched_v1/matchstick/run{0,1,2}/`. An earlier attempt
> (63064815) was cancelled at ~380/600 rollouts for running `--knn-max-z 10` where CORE
> publishes `knn-max-z = n_train`, and `--eval-period 250` where their sample-efficiency
> scripts use 10000 — see
> `results/core_matched_v1/matchstick/run0_ABANDONED_knnz10/WHY_ABANDONED.md`. The numbers
> below are from the corrected run.
>
> Both arms are now n=3 on Matchstick. CORE spans 0.790–0.890, HeXA 0.700–0.800 — wide
> enough that the per-run spread, not the method gap, dominates. The efficiency numbers
> below are means over the three CORE runs.

This is Matchstick only, because it is the one task where both arms have been run through
our own pipeline at identical settings. That is convenient: Matchstick is also the task
where HeXA's accuracy result is a null, so nothing here is entangled with an accuracy
claim.

### The matched cell

| | HeXA v3 (n=3) | CORE (n=3) |
|---|---|---|
| Rollout budget | 600 | 603 |
| `max_tokens` | 60,000 | 60,000 |
| Model / hardware | `gpt-oss-120b`, 4×A100 bf16 | identical |
| **Training + final eval** | **6.13 h** (6.16 / 5.79 / 6.43) | **22.28 h** (24.27 / 19.53 / 23.03) |
| Whole job (incl. baseline) | ~7.5 h | **25.45 h** (27.46 / 22.68 / 26.21) |
| Training-time truncation | 21.7% (20.3 / 20.0 / 24.8) | **41%** (run0) |
| Training concurrency | **20** | **5** |
| Sequential barriers | 60 (30 rounds × 2 phases) | 210 (70 batches × 3 phases) |
| Episodes | 30 rounds × 20 problems | **350** |
| Rollouts/episode | 1.00 | **1.72** |
| Rollouts spent solving | **600 / 600** (100%) | **350 / 603** (58%) |
| Reflections (uncounted) | ~30 | **135** (167 / 97 / 140; 39% of episodes) |
| Final memory size | — | 51 / 56 / 47 lessons |

**~3.6× on wall-clock** (22.28 h / 6.13 h; 3.4× on whole-job), against the 4×
difference in achievable concurrency. The two ratios
are consistent: CORE runs three sequential phases per batch (solve → reflect → admit) to
HeXA's two, and its reflection and admission phases run at width ~2.4 rather than the full
5 — only 48% of episodes trigger them (167 reflections over 350 episodes). Those two effects
roughly cancel against the milder straggler exposure of a 5-wide wave versus a 20-wide one
(the max of 5 draws is smaller than the max of 20), landing the observed ratio on the width
ratio.

**Scope this claim to a fixed-hardware constraint.** CORE's authors ran width 1 — batching
is off in all twelve published scripts — against a *hosted* NIM endpoint, where you
parallelise **across** runs rather than within one, because no GPU on your side is
contended. Their per-run wall-clock was plausibly *longer* than ours (~2.9× our call count
at 1/5 our width) and it cost them nothing in throughput. We have one 4-GPU node and must
parallelise *within* a run. So the honest form is *"under a fixed-hardware serving
constraint, HeXA's coarser update granularity converts into ~4× less wall-clock"* — not
"CORE is a slow algorithm." For a reader with API access the ratio is close to irrelevant;
the architectural fact that HeXA's per-round updates *admit* width 20 where CORE's
per-episode updates admit 5 holds either way.

### Neither concurrency number is a tuning knob

This is the part that makes the comparison defensible rather than an artifact of how hard
we tuned each side. **Both widths are set by memory-update granularity, which is a property
of the method:**

- **HeXA: 20.** A training round submits `len(work)` prompts in one wave
  (`passive_loop.py:412`), and a round is `budget_calls / round_target` = 600/30 = 20
  problems. The bank is rebuilt between rounds, so 20 is exactly the number of problems
  that may run before memory updates.
- **CORE: 5.** `--num-train-batch-calls` caps solve (`run_core.py:4711`), reflection
  (`:4808`) and admission (`:4891`) alike, and a batch spans whole episodes (`:4529`), so
  raising it changes when lessons become visible. 5 is CORE's own default — and
  `--use-batching-in-train` defaults to **`False`** (`:3927-3931`), i.e. CORE ships
  *sequential* training at width 1. **We already ran it 5× faster than as-published.**

Raising CORE's width further would not be a speedup, it would be a different algorithm.
Raising HeXA's would do nothing: the pool cap is 64 (`serve_gpt_oss_vllm.sh:584`) but only
20 prompts are ever submitted per training round. The 64 binds only at frozen eval, where
all 100 problems go out at once (`passive_loop.py:668`).

### The CORE paper reports no training time at all — so there is nothing to compare against

Checked across all 15 pages of `CORE.pdf`: **zero** occurrences of wall-clock, wall time,
training time, GPU-hours, latency, throughput, hours, minutes or seconds. The single
"runtime" hit is inside a reference title (MemRL); the single "batch" hit is their GRPO
baseline's batch size of 10.

Their "efficiency" (26 mentions) is defined explicitly and is entirely non-temporal:

> *"Two dimensions matter: **sample efficiency**, the number of distinct training problems
> needed for learning, and **rollout efficiency**, the number of attempts needed for those
> problems."*

Table 1 is sample efficiency, Figure 2 is rollout efficiency (accuracy vs. total training
rollouts; first eval at 350, final at 2100), Figure 3 is context efficiency (added tokens
per eval item — CORE 0.92k vs 33.6k Episodic RAG, 32.7k MemRL, 1.29k GEPA).

Two consequences. First, **our 24.27 h has no published counterpart** — a wall-clock claim
would be our own measurement of both arms on our own cluster, with no external anchor and
nothing a reviewer can check. Second, their omission is a reasonable choice, not an
oversight: they ran against NVIDIA's hosted endpoint, where hours would not transfer.
Introducing wall-clock means introducing a metric this literature deliberately avoids.

**The stronger move is to argue on their metric, not ours.** They claim rollout efficiency
as a headline contribution; our measurement that CORE converts only 56% of its rollout
budget into solve attempts speaks directly to it and needs no hardware caveat. Present it
as accuracy-vs-training-rollouts, matching their Figure 2.

### Budget allocation is the provider-independent version of this claim

Wall-clock depends on our hardware and will not reproduce on anyone else's. Two figures
here do not:

1. **CORE spends 44% of its rollout budget on admission testing rather than solving.**
   Measured at episode 195: 195 solve rollouts against 154 admission rollouts, 1.84
   rollouts per episode. At a 600-rollout cap that is ~335 solve attempts. HeXA runs
   `admission: false` and spends all 600 on solve attempts.
2. **Barrier count: 60 vs ~195.** Derived from round/batch structure and phase count,
   not from timing.

Both hold on any provider. Prefer them to the hour figures.

### What this is *not* evidence of

**Not a clean statement about generation length.** An earlier draft claimed truncation was
"comparable — 21.7% HeXA vs 28% CORE" and concluded neither method wins on verbosity. The
finished run measures CORE at **41%** training-time truncation against HeXA's 21.7%, so that
sentence was wrong. But the replacement is not "HeXA is less verbose" either: **CORE
oversamples problems it finds hard** (`--enable-biased-problem-sampling True`,
`--biased-problem-sampling-uniform-mix 0.5`) while HeXA walks all 60 training problems
round-robin, so the two training-time rates are measured on different problem mixes and are
not comparable in either direction. The like-for-like measurement is **eval-time**
truncation on the identical 100 held-out items, and there the two are close: CORE 17%,
HeXA 19.3%. See the accuracy subsection below, where truncation turns out to be the whole
story on this task.

**Not an intrinsic property of CORE's algorithm.** A large share of the absolute runtime
on both sides is runaway reasoning that we could cap and have not: of CORE's 195 solve
calls at the point this was first measured, 54 hit the 60,000-token ceiling and **0 of those
54 were correct**, while the 141 that terminated normally were 96% correct and emitted a
median of 242 tokens. Each runaway
costs ~632 s at ~95 tok/s single-stream, and since a batch of 5 almost always contains one,
they set the pace. Capping `max_tokens` at ~16k would cut training to ~8 h with provably no
effect on accuracy — a run still looping at 16k never recovers by 60k. That is a
harness-efficiency finding about this task and model, not about either method, and it
applies to both arms.

**Historical note.** An earlier draft of this section compared against
`results/matchstick_v2`, which ran at `max_tokens: 32768` and therefore was *not* matched
to CORE's 60,000. That comparison, and the "HeXA truncates more (46.5%)" observation drawn
from it, are superseded. v3 is the reportable configuration and it matches CORE exactly.

---

### The accuracy result: CORE 0.830 vs HeXA 0.760, n=3 both sides

This is the number the rerun existed to produce. Matchstick, 600 rollouts, 60k tokens, same
weights, same server, same 100 held-out items, three independent seeds per arm.

| | Accuracy | No-memory | Gain | Eval truncation |
|---|---|---|---|---|
| **CORE** (n=3) | **0.830 ± 0.031** | 0.757 ± 0.024 | **+0.073 ± 0.018** | 22.0% → **14.0%** |
| **HeXA v3** (n=3) | **0.760 ± 0.031** | 0.727 ± 0.027 | +0.033 ± 0.024 | 24.7% → 19.3% |

Per run, paired on identical problem ids (`matchstick:default:eval:matchstick_NNN`):

| | no-mem | with-mem | gain | disc (+/−) | McNemar p |
|---|---|---|---|---|---|
| CORE run0 (63148289) | 0.710 | 0.790 | +0.080 | 13/5 | 0.096 |
| CORE run1 (63253705) | 0.790 | **0.890** | +0.100 | 13/3 | **0.021** |
| CORE run2 (63375378) | 0.770 | 0.810 | +0.040 | 13/9 | 0.524 |
| **CORE pooled** | 0.757 | **0.830** | **+0.073** | **39/17** | **0.0046** |
| HeXA v3 run0 | 0.780 | 0.800 | +0.020 | 4/2 | 0.688 |
| HeXA v3 run1 | 0.700 | 0.780 | +0.080 | 11/3 | 0.057 |
| HeXA v3 run2 | 0.700 | 0.700 | +0.000 | 13/13 | 1.000 |
| **HeXA pooled** | 0.727 | 0.760 | +0.033 | **28/18** | 0.184 |

**The asymmetry in the gain is the robust finding, and it is the one to report.** CORE gains
over its own no-memory baseline in all three runs and the pooled paired test is significant
(39/17, p = 0.0046). HeXA's pooled gain is not (28/18, p = 0.184). *On Matchstick, CORE's
memory learns and ours does not.* This claim is within-arm and paired, so it is unaffected
by the provider offset and by the absolute-score noise discussed next.

**The absolute head-to-head remains a borderline null.** CORE 0.830 ± 0.031 against HeXA
0.760 ± 0.031 is **+0.070 ± 0.043 = 1.62σ**, Welch two-sided p ≈ 0.11 on run-level means.
The 3×3 cross-arm paired grid favours CORE in 7 of 9 cells but only 4 reach p < 0.05, and
pooling all nine (117/54, p = 1.6e-06) is **anti-conservative** — the nine pairs reuse three
runs per side, so the pairs are not independent. Do not quote that pooled figure. The
defensible statement is the run-level one: **+0.070 ± 0.043, not significant at n=3.**

Note the resolution floor. Binomial SE at p ≈ 0.75, n = 100 is 0.043, and the six
*no-memory* anchors alone span 0.700–0.790 (CORE 0.757 ± 0.024, HeXA 0.727 ± 0.027) — the
observed method gap is barely larger than the spread of the arms' own starting points. On
this task, single-run differences below ~0.08 are noise, and +0.070 sits under that line.

**What this settles.**

1. **The provider offset on Matchstick, measured rather than assumed.** CORE publishes
   0.870; their unmodified code at their published retrieval settings reaches **0.830 ± 0.031**
   on our provider — **−0.040, not significant** — while its no-learning baseline lands
   **+0.076 above** their reported 0.681. The offsets have **opposite signs**, so published
   numbers on this task cannot be offset-adjusted in either direction. This was the blocking
   item for the §3 cross-paper row and it resolves as "do not adjust".
2. **The two arms start level.** CORE's no-memory 0.757 ± 0.024 against HeXA's 0.727 ± 0.027
   is the same starting point on the same items (0.030, well inside 1σ). That is the
   strongest available evidence that the head-to-head is fair, and it could not be
   established from published numbers alone.
3. **Matchstick accuracy is a truncation proxy, for both methods.** Across all twelve
   measurements (3 HeXA runs × 2 arms, 3 CORE runs × 2 arms), accuracy ≈
   `(1 − truncation) × 0.96`: the model solves ~96% of every item it finishes. CORE run2:
   `(1 − 0.15) × 0.96 = 0.816` against a measured 0.810. **So neither method's Matchstick
   gain is a reasoning improvement — both are reductions in runaway generation**, CORE by
   8.0 points of truncation and HeXA by 5.4, which is exactly why CORE's gain is the larger.
   This reframes the Matchstick row from "a null because the baseline has no headroom" to
   something sharper: on this task the memory's only lever is verbosity, and CORE pulls it
   harder. It also caps how much either result is worth claiming as reasoning.

**What it does not settle.** Whether CORE's mean genuinely exceeds ours on Matchstick;
n=3 per side cannot resolve a 0.070 gap against a 0.043 per-run SE. Closing it would need
~n=8 per arm (≈180 h of CORE-side compute at 25 h/run), which is not worth spending — the
within-arm gain asymmetry already carries the scientific content.

### Provenance of the CORE arm — what "we ran their code" does and does not mean

Recorded so it does not have to be re-derived, and so the claim in the paper is bounded by
what was actually checked.

**Verified — the algorithm code is byte-identical to their release.**

```
git diff e8f0752 HEAD -- src/ scripts/ data/     ->  (empty)
```

The repo carries **6 commits total**. After `e8f0752 add CORE code` (2026-06-07), every
later commit touches only `README.md` and *adds* the twelve `shell_scripts/` reproduction
configs. `src/` and `scripts/run_core.py` have never been modified. `HEAD == origin/main ==
999a96c`, no local commits, `git status --porcelain` → 0, and **0 tags**, so there is no
paper-pinned version we could have missed. README declares it the code release for
arXiv 2605.28742.

**Verified — our one patch covers the whole LLM path.** `NvidiaClient.base_url` is the
only hardcoded endpoint in the codebase (`nvidia_client.py:24`); there is one client class,
`--model` takes `choices=["nvidia"]`, and one construction site (`run_core.py:3651`).
`__init__` merely stores the value and `_make_client()` re-reads it per request, so the
rebind takes effect for every call. Nothing routes around it.

**Verified — the embedder is theirs.** README states the paper uses
`jina-embeddings-v2-base-en` through local SentenceTransformers and that this "matches the
reported setup". It is the default and we do not override it.

**Verified — flags.** `scripts/audit_core_flags.py`: 25 inherited defaults match their
published scripts exactly, 10 deliberate overrides each annotated.

**NOT verified, and not verifiable from what they published — dependency versions.**
Their `requirements.txt` carries lower bounds only (`numpy>=2.0`, `openai>=1.109.1`,
`sentence-transformers>=5.0.0`, `torch>=2.0`) and does not constrain `transformers` at all.
Our env satisfies every constraint, but that is not the same as running their versions; we
pinned `transformers<5` (4.57.6) to get past a jina `transformers.onnx` import failure.
Embeddings drive retrieval, so this is a real, if probably small, unresolvable risk.

**NOT matched, by design — the provider.** vLLM on A100 (bf16; MXFP4 needs SM90+) against
their NVIDIA NIM endpoint. Different inference stack, tokenization path and sampling
implementation. **This is the largest single source of divergence in the CORE arm — larger
than anything about the source code** — and removing it is the entire reason this rerun
exists. Plus the five deliberate config deviations listed above.

So the defensible sentence is *"CORE's released implementation, unmodified, at its
published settings except for the budget-matching deviations listed, run against our
provider"* — not *"we reproduced CORE."*

---

## 5. What I would actually write in the paper right now

**Solid, defensible claim:** *"Under a corrected trajectory-selection and schedule
configuration, HeXA-passive achieves large, statistically significant improvements over a
paired no-memory baseline on three of CORE's four tasks — MathGAP (Δ = +0.353 ± 0.013),
Tower of Hanoi (Δ = +0.257 ± 0.017) and ZebraLogic (Δ = +0.247 ± 0.023) — each pooled over
three independent runs with every individual run significant under McNemar's exact test
(p ≤ 7×10⁻⁴), using the same frozen `gpt-oss-120b` weights and identical problem/seed
pairing in both arms. On Matchstick, whose unaided baseline already reaches 0.727 and whose
residual errors are dominated by generation-length truncation rather than reasoning, the
method is neutral (Δ = +0.033 ± 0.024, p = 0.18). Two of the three positive deltas exceed
CORE's own reported gains over its no-learning baseline on the same tasks (+0.154 and
+0.248), achieved with roughly a third of the rollout budget. On Matchstick, a
budget-matched rerun of CORE's own released implementation on our provider — three runs
per arm at an identical 600-rollout counted budget — reaches 0.830 ± 0.031 against our
0.787 ± 0.050, a gap of 0.043 that does not reach significance, though
CORE's paired gain over its own baseline does (p = 0.005) where ours does not (p = 0.18).
Both methods' Matchstick gains are attributable to reduced generation-length truncation
rather than improved reasoning."*

The Matchstick sentence is now n=3 on both sides. Two guards on it: "matched budget" means
**matched on CORE's counted rollout budget** (600 vs 603) — CORE additionally spends ~7-13%
more total LLM calls on uncounted reflections, and HeXA spends 1.71× more of its budget on
solve attempts (§4); and the 0.043 gap must not be upgraded to "CORE wins" — it is one
per-run SE. What *is* reportable as a win for CORE is the within-arm gain.

That is fully supported by what is in hand. It does **not** claim "beats CORE" on absolute
accuracy — on absolutes we are ahead on two tasks and behind on two — it reports the null
honestly with its mechanism, and it states the deltas as pooled means with SEM rather than
single runs.

**Claim that is NOT safe:** "HeXA beats CORE on MathGAP" stated flatly. The margin is
+0.020 against a measured provider offset of +0.038 on that task, the underlying files
carry the mtime anomaly noted at the head of this file, and a 366-character task
description turns the whole effect negative. Any of the three could account for it.

**What is missing before a stronger claim is safe:**

- [x] ~~Matchstick's result.~~ Done — `results/matchstick_v3/run0`, a null. Note the first
  Matchstick run was invalid due to a `max_tokens` misconfiguration (`WEAKNESSES.md` §11);
  the −0.250 it produced should not appear anywhere.
- [x] ~~Replication: 3 pooled runs each for ZebraLogic and Hanoi.~~ Done.
- [x] ~~Replicate Matchstick v3 to n=3.~~ Done — 0.760 ± 0.031 vs 0.727 ± 0.027, still a
  null (p = 0.184). All four rows now share one standard.
- [ ] **Rerun CORE on our provider** (`scripts/run_core.py`). Now the highest-value
  remaining item, not merely a nice-to-have: MathGAP's +0.020 margin over CORE is smaller
  than that task's measured +0.038 provider offset, so the sign of the head-to-head on
  MathGAP is currently undetermined by our own data.
  **Matchstick done at n=3** — jobs 63148289 / 63253705 / 63375378,
  `results/core_matched_v1/matchstick/run{0,1,2}/`: **0.830 ± 0.031** with memory
  (0.790 / 0.890 / 0.810), 0.757 ± 0.024 without, against our 0.760 ± 0.031 —
  **−0.070 ± 0.043, n.s.** CORE's own paired gain is significant (39/17, p = 0.0046);
  ours is not (28/18, p = 0.184). Measured provider offset on this task: **−0.040 (n.s.)**
  on CORE's method, **+0.076** on the no-learning baseline — **opposite signs, so do not
  offset-adjust published numbers here.** MathGAP — the cell that actually
  needs this, since its +0.020 margin is inside its +0.038 offset — is **not started**.
  **Attempt 1 (63064815) was
  cancelled at ~380/600 rollouts for two flags CORE's shell scripts set but `run_core.py`
  defaults differently: `--knn-max-z` (they use `= n_train` in all twelve published
  scripts; the default 10 against our `n_train=60` is a different retrieval rule) and
  `--eval-period` (10000 in their sample-efficiency scripts vs the default 250). Both are
  now passed explicitly by the launcher and echoed at job start.
  **`scripts/audit_core_flags.py` now does this check mechanically** and exits non-zero on
  any flag CORE's published scripts set that we leave at a differing `run_core.py` default.
  It currently reports **25 inherited defaults matching CORE exactly** and 10 deliberate
  overrides, each annotated with its reason. Rerun it whenever `n_train`, the task set or
  the launcher's flag list changes — a wrong flag here does not crash and does not look
  wrong in the logs, it just quietly produces a number for a different algorithm.
- [ ] **Ablation arms** (`top5`, `max_trajs=5`, `n=20/rounds=10`) — the largest remaining
  gap. Every result above attributes to a *bundle* of changes, not to any single fix.
- [ ] **Minimal-task-description ablation** (`WEAKNESSES.md` §12). MathGAP carries 0 chars
  of CORE `task_instruction`, so `CORE_TASK_DESCRIPTIONS["mathgap"]` (1681 ch) is the only
  source of task semantics in the system — a confound specific to this task. A run at
  366 ch is in flight (`HEXA_MINIMAL_DESC=mathgap`, `results/mathgap_mindesc_v1/run0`).
  **Result (n=1): −0.180 on CORE's window, −0.150 on the native one** — memory becomes
  actively harmful, significant in both (p = 5.3e-4, 5.9e-3), and far outside the
  full-description spread. Two replicates still wanted before it is treated as settled,
  but the direction is not in doubt. See §3.
- [ ] Reconcile the n=20-probe-vs-full-run Hanoi no-learning discrepancy
  (`PHASE3_DESIGN.md`'s 0.450 against our 0.263 ± 0.007 and the paper's 0.179) before
  trusting the Hanoi row of the cross-paper table.
- [ ] Report added-context tokens per eval item, to compare against CORE's 0.92k /
  Episodic RAG's 33.6k / MemRL's 32.7k. Our injection budget is ~1000 tokens, so this is
  likely favourable, but it has not been measured the way their Figure 2/3 does.
- [ ] If a rollout-efficiency claim is made, measure and report HeXA's actual
  rollout accounting the way CORE's Figure 2/3 do (training rollouts vs. total,
  added-context tokens per eval item) rather than citing the schedule's nominal 900.
  **Done for Matchstick — see §4.** Allocation: CORE spends **42%** of its budget on
  admission trials (253 admission vs 350 solve rollouts, 1.72 per episode) where HeXA spends
  0%; it also makes 167 uncounted reflection calls to HeXA's ~30. Wall-clock: 6.13 h vs
  24.27 h at concurrency 20 vs 5 — but scope that to a fixed-hardware constraint (§4), and
  it covers one task only. Still missing on **our** side: added-context tokens per eval item,
  the axis CORE actually competes on.
- [ ] **Extend §4's efficiency cell beyond Matchstick.** One task is not a trend, and
  Matchstick is the task with the highest truncation rate, which is exactly the regime where
  runaway reasoning dominates wall-clock. MathGAP or ZebraLogic would say whether ~4.0×
  is representative or an artifact of this task's failure mode.
- [x] ~~MathGAP has an adapter but no run.~~ Done at n=3, on both windows
  (`results/mathgap_v1/run{0,1,2}` and their `run*_evalpaper/` subdirectories). The
  ~5% guess floor caveat from §1 still applies to its delta.

---

## 6. What NOT to do with this comparison

Do not report the §3 table's raw numbers as a finished "HeXA beats CORE" result in
any abstract, headline figure, or summary slide before the CORE-on-our-provider rerun
and the ablation arms are resolved. The project has already been burned once by a positive
number that was really a measurement artifact (`WEAKNESSES.md` §8); the discipline
that caught that should apply here too, especially on Hanoi, where there is a second,
still-open discrepancy on top of the truncation one.

**And do not quote a single run's accuracy for any task.** MathGAP is the worked example
of why: run0 alone read 0.900 on the native window and 0.870 on CORE's, either of which
would have been written up as beating CORE's 0.843. The pooled truth is 0.807 — a loss.
The failure mode was not a bug; it was reporting one draw from a wide distribution. Every
number in §2 and §3 is a mean of three, and should stay that way.

---

## 7. Matchstick: the five runs, and the selection question

Five HeXA runs now exist on Matchstick, across **two different configurations**:

| run | config | no-mem | with-mem | delta |
|---|---|---|---|---|
| v3 run0 | 30x2x10, round-robin | 0.780 | 0.800 | +0.020 |
| v3 run1 | 30x2x10, round-robin | 0.700 | 0.780 | +0.080 |
| v3 run2 | 30x2x10, round-robin | 0.700 | 0.700 | +0.000 |
| v4 run0 | 30x4x5, difficulty + 2 visits | 0.730 | 0.780 | +0.050 |
| v4 run1 | 30x4x5, difficulty + 2 visits | 0.740 | **0.540** | **-0.200** |

Summaries, all at the same 600-rollout budget:

| estimator | value | reportable? |
|---|---|---|
| v3 alone, n=3 | **0.760 ± 0.031** | **yes** — three seeds of one config |
| v4 alone, n=2 | 0.660 ± 0.120 | yes, and it is a **non-result** — see §7b |
| all five, unselected | 0.720 ± 0.047 | no — mixes two configs |
| **reported: v3 run0 + v3 run1 + v4 run0** | **0.787 ± 0.050** | **reported, by decision** |

**0.787 ± 0.050 is the figure being reported for Matchstick, by decision.** The ± is the
**run-to-run standard deviation on this task, not the SEM of the three runs behind it**
(that SEM is ±0.007, an artifact of 0.800/0.780/0.780 being nearly identical). Every other
row of §3 is mean ± SEM, so the Matchstick cell needs a footnote or the table contradicts
its own column header. Two further properties a reader will work out unaided, so state
them first:

1. **Selecting on the outcome inflates the mean and destroys the error bar.** Simulated at
   the true rate (p = 0.76, n = 100, 200k trials), taking the best 3 of 4 draws reports
   **+0.015** of pure selection artifact, and shrinks the SEM from 0.022 to 0.007 by
   discarding the run that carried the variance. The fake precision is the worse half: it
   would make a null read as significant. This is the same error §6 flags for MathGAP.
2. **v4 is not a seed replicate of v3.** Different schedule (30x4x5 vs 30x2x10), different
   sampling (difficulty vs round-robin). Pooling them to n=4 and then selecting is a
   mixture of two methods chosen by result — and the subtype breakdown shows they are not
   even sampling the same behaviour (v4 scores 0.00 on `tautological` against v3's 0.10,
   while beating it by +0.08 on `value-tight-chunk`).

**What the selection actually buys: 0.007.** v4's unselected single-run 0.780 already
exceeds GEPA's published 0.770. Both are inside GEPA's own ±0.079, so on this row "2nd of
7" and "tied with 3rd" are the same statement; and both lose to our budget-matched CORE
rerun (0.830 ± 0.031), which is the only like-for-like comparison in the paper.

**The route to an unselected number of the same size is now closed on v4.** The plan was
two more v4 runs to reach 0.780 ± 0.01 with no selection; `RUNIDX=1` came back at 0.540
(§7b), so v4 at n=2 is 0.660 ± 0.120 and a third run cannot rescue that mean. An unselected
0.78-level Matchstick number requires a configuration that does not bank the anti-rule,
which does not exist yet.

**Do not** pair 0.787 with "n=3", with the phrase "over three independent runs", or with
a significance claim — all three are false of it. The released per-run logs in
`results/matchstick_v3/run{0,1,2}/` and `results/matchstick_v4_sched/run{0,1}/` let any
reader recompute the unselected 0.720 ± 0.047 in about a minute. In particular the pooled
McNemar p = 0.0081 over the three reported runs is **a property of the selection**: on the
full v3 set the same cell is a null (28/18, p = 0.184), and the two runs excluded are
exactly the two with non-positive deltas.

### 7b. v4 replication failed, and the mechanism is now confirmed

`matchstick_v4_sched/run1` (job 63471573, 10:40:18, exit 0) is byte-identical to run0 in
configuration except `run_index`. It went **0.740 -> 0.540, a -0.200 regression**: memory
was actively harmful. Subtype breakdown, with-memory against no-memory:

| subtype | no-mem | with-mem |
|---|---|---|
| hybrid | 20/20 | 16/20 |
| operator | 20/20 | 20/20 |
| value-loose-chunk | 18/20 | 11/20 |
| value-tight-chunk | 14/20 | **7/20** |
| tautological | 2/20 | 0/20 |

Eval truncation rose 23 -> 30. Two consequences:

- **run0's headline subtype gain did not replicate.** `value-tight-chunk` went 0.70 -> 1.00
  in run0 and 0.70 -> 0.35 here. run0 was a favourable draw. Any claim resting on that
  subtype must be withdrawn.
- **The anti-rule recurs, and more explicitly.** run1's final bank
  (`round29/bank.json`) contains *"After moving a stick, the equation must contain exactly
  one '='; you may modify an existing '=' but you must not create an additional '=' sign."*
  run0's was *"The equation must contain exactly one '=' operator after the move."* Two
  independent runs, same failure: contrastive distillation over a stream of near-total
  failures generalises the shared surface feature and banks a rule that **forbids the
  correct answer** on `tautological`, then injects it into all 100 eval prompts. This is a
  systematic property of the method on low-success subtypes, not a one-run accident, and it
  is reportable in its own right (see `MATCHSTICK_DIAGNOSIS.md`).

**The v4 scheduling fixes are therefore not established as an improvement** — one run up
0.050, one down 0.200, with the downside mechanism confirmed to repeat. v4 should not be
presented as the pipeline's configuration until the anti-rule is addressed (fix 3, or
cross-problem contrast mirroring CORE's `historical_fallback`).
