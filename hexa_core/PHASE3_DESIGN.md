# PHASE 3 DESIGN — the HExA-passive learning loop on CORE tasks

**Status:** build-order steps 1–4 are **built and tested** (teacher vLLM branch,
trajectory writer, `train_baseline`, injection join, parent exclusion, admission gate,
`passive_loop`), and the Stage 3a smoke pilot has run against a live gpt-oss-120b.
Phase 0–2 (adapter + reproduction gate) likewise; see `RECON.md` §4 and §6. Still
unbuilt: M6 biased sampling, `--usage-credit`, `--cycle-chunks`, A5/A6/A8 wiring, and
mid-run resume for `passive_loop`. Sections below marked "now built" record what
shipped and what changed on contact with a real model.

> ## SCOPE CHANGE — read this first
>
> The claim is no longer "HeXA vs CORE". It is: **HeXA generalises beyond InterPhyre
> to three reasoning benchmarks, and beats retrieval-based memory at equal budget.**
> CORE supplies the *tasks* — splits, prompts, verifiers — and nothing else.
>
> Consequences, all in the direction of less machinery:
>
> * **Admission (M5) and failure-biased sampling (M6) are OFF.** Both are CORE's, not
>   HeXA's (§5c: "HeXA has no accept/reject gate at all"). Running them would measure
>   HeXA-plus-CORE's-ideas. `--admission` opts back in.
> * **The admission reserve is gone** with the gate, so `k = budget // (rounds × ppr)`
>   again: 20 / 200 / 400 at n = 100 / 10 / 5, 2000 calls, zero slack.
> * **M1's baseline-relative confidence is opt-in (`--m1`).** Its *pooling* half stays
>   always-on — that is a bug fix for k-attempts-per-problem, not a method change.
> * **Retrieval is flat** (`--hierarchy task`). The `task:type` tier stays empty until
>   distillation runs per type, so it buys nothing yet.
> * **Deferred:** the reproduction gate as a match-the-paper exercise, A5/A6/A8,
>   `--usage-credit`, `--cycle-chunks`.
>
> **Baselines to build:** no-memory (done, `baseline_runner`) and Episodic RAG.
> MemRL, GEPA and ReAct are out — ReAct degenerates to chain-of-thought on
> single-turn tool-free tasks, and reimplementing MemRL well enough to beat it fairly
> is its own project. **Published numbers are not citable**: our 20-problem Hanoi
> gate read 0.450 against the paper's 0.179, so the provider gap would dominate any
> cross-paper comparison.
>
> Sections below still argue CORE parity in places. Where they disagree with this
> box, this box wins.

**Scope:** the loop that turns CORE problems into a skill bank and back into
prompts. Table A (HExA-passive, single-turn, no tools). Tools and Table B are out
of scope; so are ablations, except where a decision below only exists to make one
possible.

**Prerequisite:** the Phase 2 gate must have run against a live server and its
calibration offset recorded. Every number below is uninterpretable without it.

---

## 0. Why this document exists

Porting HeXA to CORE is not a plumbing exercise. HeXA was built around a
simulator that hands out unlimited fresh levels, each attempted once. CORE hands
out **five** problems and a budget of 2000 attempts. Nearly every mechanism in
HeXA that reads naturally under the first regime reads differently under the
second, and four of them break.

The decisions are recorded here with their defaults and their reasoning, because
each one is a place a reviewer can reasonably ask "why that way?" and the answer
should not be "nobody noticed".

---

## 1. Decision: what a round is, when there are no seeds

**The problem.** In InterPhyre, `seeds_per_round` new levels are generated, each
attempted once, and a round boundary falls after them
(`skillrl/loops/evolving_loop.py`). On CORE, `|D_train| ∈ {5, 10, 100}` is fixed
and the budget is 2000 *training calls*. At `|D_train| = 5` that is ~400 attempts
of the same five problems.

**Decision (M3): a round is a chunk of *fresh problems*, each attempted `k` times.**
This is a curriculum: distil from problems 1–10, apply the bank to 11–20, distil
again, and so on. It is HeXA's published method rather than an adaptation of it —
InterPhyre's round boundary also falls after a set of previously-unseen items.

An earlier draft of this section made a round `B = 200` *attempts* over the whole
training set. That is rejected as the primary design, for a reason worth recording
because it is the same reason the curriculum is right: attempt-paced rounds keep
re-attempting the same handful of problems, so within-training improvement **cannot
distinguish learning from memorising**. Every round's skills are applied to problems
they were distilled from. Curriculum rounds give a genuine held-out signal at *every*
round boundary, not only at final eval.

**It also closes §5b structurally.** Parent-problem contamination — a skill distilled
from problem 3 being retrieved for problem 3 — is impossible when the bank only ever
meets problems it has not been distilled from. `--parent-exclusion` (§5b) drops from
being the primary defence to a safety net that should now essentially never fire; if
it fires often, the loop has a bug. **Assert on that**: a non-trivial
`excluded_parent` count under pure streaming is a defect, not a statistic.

### 1a. The regime table

The schedule is **derived, not chosen**, from three quantities: the budget (2000
training calls, CORE's `--max-rollouts`), the training-set size `n`, and a target
round count. Given `rounds` and `problems_per_round`, the attempts-per-problem `k`
follows:

```
rounds              = min(round_target, n)          # streaming cannot exceed n chunks
problems_per_round  = ceil(n / rounds)
reserve             = min(rounds * cap * trials, budget // 5)   # admission, §5c
k                   = (budget - reserve) // (rounds * problems_per_round)
```

**`k` derives from `budget − reserve`, not from `budget`.** CORE charges admission
trials against `--max-rollouts` (§5c), so a schedule that hands the whole budget to
solve calls overruns the moment a round admits a skill — the budget it spends on
admission was already promised to a later round. This is not hypothetical: it is
exactly how the Stage 3a smoke pilot died, at the first round boundary, after the
teacher successfully distilled 8 skills.

The reserve is **uniform per round** on purpose. The obvious alternative — keep the
planned `k` and let later rounds absorb the shortfall — is the worst allocation
available, because it starves precisely the rounds whose bank is most developed.

With `round_target = 10`, `budget = 2000`, `cap = 20`, `trials = 1`:

| regime | rounds × problems/round × k | solve | reserve | total | same-problem depth |
|---|---|---|---|---|---|
| `n = 100` | 10 × 10 × 18 | 1800 | 200 | 2000 | k = 18 |
| `n = 10` | 10 × 1 × 180 | 1800 | 200 | 2000 | k = 180 |
| `n = 5` | 5 × 1 × 380 | 1900 | 100 | 2000 | k = 380 |

The reserve is a **ceiling**, not a floor: a round proposing fewer than `cap` new
skills spends less, and the remainder is reported as unspent rather than reallocated.
When it does bind, the overflow candidates are **rejected** and logged with
`reason="admission-budget-capped"` — the conservative direction, since it can only
cost HeXA a skill it might have kept and never grants one it did not earn.

Note the A6 interaction: with the gate off there is no reserve, so that arm gets a
larger `k` (200 rather than 180 at `n = 10`). The two arms therefore differ in
attempts as well as in gating, and both figures must be reported.

**Budget is held at 2000 in every regime** — that is the constant that matters, since
it is the axis CORE's sample-efficiency comparison is drawn against.

**Rounds is 10, 10, and 5 — not constant, and it cannot be.** Under pure streaming a
round consumes at least one unseen problem, so `n = 5` admits at most 5 rounds. This
is a hard property of the regime, not a knob: manufacturing 10 rounds from 5 problems
requires revisiting, which is `--cycle-chunks` (§1b) and a different experiment. State
the 5 explicitly; do not paper over it by reporting "10 rounds" everywhere.

The residual confound is therefore narrow and nameable: `n = 5` gets half the evolver
rounds. Since it also gets the *most* attempts per problem (380), the two effects push
in opposite directions, so a gain at `n = 5` cannot be attributed to extra evolution.
Report round count per regime alongside accuracy.

> **The regimes learn from structurally different evidence.** `k` spans 18 → 380 from
> `n = 100` to `n = 5`: the small-data regimes see one problem very deeply, the
> large-data regime sees many problems shallowly. That is very likely part of *why*
> sample efficiency behaves the way it does, and it is invisible unless the contrast
> mix is logged per round.

*(An earlier draft of this table read `10 × 10 × 2` for `n = 100` and `5 × 2 × 200`
for `n = 10`. Both were arithmetically wrong — 200 calls and 2000 respectively — and
the first understated same-problem depth at `n = 100` by 10×. The formula above
replaces hand-computed rows so the schedule cannot drift from the budget again;
`test_passive_loop` asserts every regime's solve calls plus admission reserve sum to
exactly 2000.)*

Both contrast sources therefore stay in scope, and CORE already names them:
`contrast_sources` ∈ {`within_problem`, `historical_fallback`} with
`historical_fallback_problem_relations` ∈ {`same_problem`, `different_problem`}
(`memory/localized_ucb.py`). **Log the mix per round and report it per regime.**

Same-problem contrast, where available, is the cleaner of the two: it holds problem
difficulty fixed, so the distilled difference is about *method* rather than about
which problem happened to be easier. Prefer it within a chunk; use cross-problem
across chunks.

Existing unused helpers still apply, now chunking problems rather than attempts:
`trajectory.chunk_attempts`, `assign_rounds`, `round_summary`.

### 1b. Streaming vs cycling

Pure streaming (never revisit a chunk) maximises fresh-problem signal, but a problem
the bank failed in round 1 is never retried with a better bank — so the run cannot
show a problem going **from failed to solved**, which is the most legible evidence of
learning there is.

**Decision.** Pure streaming for the headline; **one cycled run as a pilot
diagnostic** (two passes over the chunks), reported separately. Cycling reintroduces
parent contamination on the second pass, so `--parent-exclusion` is mandatory there —
which is exactly the case the flag is retained for.

### 1c. Which problem to attempt next

Uniform sampling spends budget re-solving what is already solved. CORE weights by
failure rate, `wᵢ = 1 − aᵢ`, mixed with uniform at `ε_mix = 0.5`
(`_sample_biased_training_item`, `run_core.py:1070`; sampler seeded in namespace
7000).

**Decision (M6).** Adopt it, ported not reimplemented in spirit — same weighting,
same default mix, applied **within the current chunk** (across the chunk's `k`
attempts), not across the whole training set: sampling a problem from a future chunk
would break the curriculum's held-out property. **Config flag, default on, ablated.**
If it is not ablated, a reviewer cannot tell whether HExA's gain is method or
sampling.

At `n = 100` with `k = 2` the sampler has almost nothing to do — 10 problems, 20
attempts. It matters at `n ∈ {5, 10}`, where `k` is large. Report it that way rather
than as a single global ablation number.

---

## 2. Decision: how trajectories are rewarded, and what confidence means

Reward itself is settled and built: binary, from CORE's verifier, via
`hexa_bridge.binary_trajectory_reward` (+1.0 / −1.0). M2 is the one complete
M-seam. Verified live: shaped `0.25 / −0.75` → binary `1.0 / −1.0`.

The open decision is **confidence**, and there are two incompatible schemes.

| | HeXA today | CORE |
|---|---|---|
| When credit is assigned | **Generation time.** A skill's confidence is the mean reward of the trajectories it was distilled from, fixed at birth (`distill._compute_initial_confidence`). Retained skills keep a teacher-*self-reported* float (`evolution_prompts.py:76`). | **Usage time.** Every retrieved lesson's advantage is updated after every attempt that used it (`memory.reward_retrieved`). |
| Can a confident-but-useless skill be demoted? | No. | Yes. |

**Decision (M1), now built.** Confidence becomes **baseline-relative utility**
`U(q) = float(correct) − b_q`, pooled over **every** attempt on the source problems,
with `b_q` from `n_base = 10` no-memory samples per training problem. It arrives
through `hexa_bridge.compute_initial_confidence_core`, installed by
`patch(confidence_fn=...)`; `--no-m1` runs the pre-M1 arm.

**A second bug, found only when this was wired up: the stock function keeps one
trajectory per seed.** Under InterPhyre a seed is attempted once, so seed → trajectory
is 1:1. Under CORE a problem is attempted `k` times — 180 at `n = 10`, 380 at `n = 5`
— and `seed_to_traj` keeps the *first* occurrence. Since
`all_trajs = successes + failures`, that first occurrence is a success whenever the
problem was ever solved. Measured against the real functions before the fix:

| evidence for the skill's source problem | confidence |
|---|---|
| solved 3 of 10 | 1.0 |
| solved 1 of 10 | 1.0 |
| solved 0 of 10 | 0.1 |

Confidence answered *"was this problem ever solved?"*, not *"how much did this skill
help?"*, discarding `k − 1` attempts of evidence and saturating at 1.0 for essentially
every skill — while being the retrieval ranking key, the pruning signal, and an input
to admission. Pooling all attempts makes the mean utility exactly
`observed_rate − b_q`, so a skill that merely reproduces the baseline scores the
neutral 0.5, one that beats it scores above, and one that underperforms scores below.

Note this is a **base-rate correction, not a difficulty preference**: a skill from a
problem solved 1/10 at `b_q = 0.05` now outranks one from a problem solved 3/10 at
`b_q = 0.30`, because the first beat its baseline and the second only matched it.

**M1 and M2 stay separate seams.** `reward_fn` remains binary — it feeds the
teacher-facing reward line and the sort choosing which five trajectories the teacher
sees — while `confidence_fn` carries M1. Folding M1 into `reward_fn` would also change
what the teacher reads, confounding the two modifications and making neither
independently ablatable.

Without this, base rates masquerade as skill quality: Matchstick skills
(`b ≈ 0.68`) look strong and Hanoi skills (`b ≈ 0.18`) look weak for reasons that
have nothing to do with the skills.

The arithmetic reuses HeXA's existing map rather than replacing it — but **M1 and M2
use different numeric conventions, and mixing them overflows the domain.**

M2 sets `reward` to **±1**, so it spans the same `[−1,1]` as the shaped HeXA scale it
replaces. `b_q` is a *rate* in `[0,1]`. So `reward − b_q` lands in `[−2, 1]`: a failure
on a `b_q = 0.5` problem gives `−1 − 0.5 = −1.5`. Since
`_compute_initial_confidence` maps `[−1,1] → [0.1,1.0]`, anything past `−1` silently
clamps — pushing *every* failure on a mid-difficulty problem to the floor regardless of
how far below baseline it really was.

**Therefore `U = float(correct) − b_q`, not `reward − b_q`.** `float(correct) ∈ {0,1}`
is CORE's own convention (`_reward_from_metrics` returns `float(bool(is_correct))`),
which puts `U` in `[−1,1]` exactly, so the affine map applies unchanged. Both branches
of `Attempt.utility` then share one range: `reward` is `{−1,+1}`, `U` is `[−1,1]`.

*(This contradiction was live in the design and in `Attempt.utility` until
`test_train_baseline.test_utility_lands_in_the_domain_the_map_expects` caught
`r=−1, b_q=0.5 → −1.5`. The test is kept as a domain guard, since the two conventions
remain adjacent and easy to re-conflate.)*

**Decision (usage-time credit).** **Add it, behind a flag, default on, ablated.**
We already log which skills were injected into which attempt, so the bookkeeping is
nearly free, and without it HeXA's confidence can never be corrected by evidence.
But it **is** a change to HeXA's published method and must be reported as one, not
folded in silently. Name the flag `--usage-credit` and report both settings.

### 2b. What must be built: the `b_q` producer

There is currently **no code that produces `b_q`** (`RECON.md` Known issue 3).
`Attempt.baseline` / `Attempt.utility` and `RolloutBudget.charge_baseline` exist as
fields only. `run_baseline_eval` cannot be reused: `_prepare_problems`
(`baseline_runner.py:852`) calls `task.eval_problems()` and hard-raises on any
problem whose `split != "eval"` — deliberately, to make eval contamination
impossible.

**Decision.** Write a separate train-split baseline pass. It must:
- run `n_base = 10` no-memory attempts per training problem, empty guidelines;
- use CORE's **baseline seed namespace 1000**, not eval's 2000
  (`run_core.py:1193` vs `1462`; namespaces are 1000 baseline / 2000 eval /
  3000 train-solve / 5000–5001 admission / 7000 sampler);
- charge every call to `RolloutBudget.charge_baseline`, which is **outside**
  `max_train` — matching CORE, where baseline estimation is not charged against
  `--max-rollouts`;
- be resumable and cached per problem, like `run_baseline_eval` already is.

Cost: `10 × |D_train|` = 50 / 100 / 1000 calls, i.e. the 2050 / 2100 / 3000 totals.

---

## 3. Decision: general vs task-specific distillation

**The honest position: in Table A the hierarchy is degenerate.** Each headline run
trains on one task, so every skill is task-specific and `general_skills` is empty.
HeXA's own default already reflects this — `skip_general=True` for single-level runs.

So `SkillBank`'s two-level hierarchy, which is a genuine HeXA contribution with no
CORE analogue, **cannot be evaluated by the headline table at all.** Ablation A5
(hierarchical vs flattened) has nothing to measure in that setting.

Three options, and they are not equally good:

1. **Report Table A per-task, and state plainly that the hierarchy is inactive.**
   Honest, cheap, and gives up the A5 claim.
2. **A multi-task run** where all three tasks share one bank, so `general_skills`
   can fill. **Risk: these three tasks share almost nothing.** Roman-numeral
   matchstick manipulation, constraint satisfaction, and recursive planning have no
   common domain content. Anything genuinely general would be meta-strategy
   ("verify the candidate against every constraint before committing"). That may be
   real, or A5 may be a null result — and a null result here is expensive.
3. **Type-level hierarchy within a task.** Every CORE task has internal structure
   we already have in `Problem.metadata`: Matchstick has 5 declared `type`s
   (`value-loose-chunk`, `value-tight-chunk`, `hybrid`, `operator`, `tautological`,
   20 each); ZebraLogic has 5 grid sizes; Hanoi has a spread of solution depths
   (`optimal_moves` 3–63 across the eval set).

**Decision.** Option 3 is the recommended primary, with option 1 as the fallback
framing. Map HeXA's `general → level` hierarchy onto **`task → type`**: the "general"
tier holds skills that apply across types within a task, the "specific" tier holds
per-type skills. This is where transfer plausibly exists, it is testable inside a
single-task run, and it keeps A5 meaningful without betting on cross-task generality.

Option 2 stays available as a separate cross-task experiment, but should not be the
thing A5 depends on.

**Implementation note.** This costs almost nothing: `Skill.source_level` is a free-form
string, so it carries `"matchstick"` for the cross-type tier and
`"matchstick:operator"` for a type-specific one. `SkillBank` needs no schema change.

---

## 4. Decision: retrieval and injection

### 4a. Query conditioning

HeXA retrieves global top-M by confidence — **query-independent**
(`core/retriever.py`). CORE conditions on problem similarity via kNN over problem
embeddings.

But recall finding F3 (`RECON.md` §0): CORE's `knn_max_z = 10`, and the
neighbourhood is drawn over *registered problems*, so `knn_k = min(10, |D_train|)`.
**At `|D_train| ∈ {5, 10}` every query's neighbourhood is the entire training set**
and `Û(q,i)` collapses to a global average. CORE's retrieval is query-independent in
two of its three shipped regimes.

**Decision.** Keep HeXA's global top-M for the headline. It is the honest HeXA
method, and at 5 and 10 it is *structurally equivalent* to CORE's — so those cells
isolate contrastive-generation quality and utility ranking, not conditioning. State
this explicitly in the paper; it is also the correct reading of CORE's own
0.907 → 0.780 retrieval ablation, which was run at 10-example Matchstick.

Add query-conditional retrieval as **A8, at 100-example only**, where conditioning
actually exists. With the §3 decision, the natural conditioning signal is the
problem's `type` from `Problem.metadata` — no embedding model required, which also
sidesteps inheriting an embedding choice from a MemRL code path.

### 4b. Injection site — must change

HeXA appends the skills block to the **system prompt** and adds a
`DIAGNOSTIC_ADDENDUM` to the user turn (`augmented_runner.py:163`).

**Decision.** Neither survives. Injection goes through CORE's own guidelines slot:
`Problem.build_prompt(guidelines)`. That is the only path where "added
task-external tokens" is measured against the same baseline CORE measured against,
which is the whole point of the Figure-3 comparison (CORE 0.92k, GEPA 1.29k,
Episodic RAG 33.6k, MemRL 32.7k).

The `DIAGNOSTIC_ADDENDUM` is dropped entirely — it instructs the agent to review
skills after each *failed simulation*, which does not exist in single-turn.

**Currently missing (Known issue 4).** M7 produces a block
(`hexa_bridge.retrieve_within_token_budget` → `RetrievedBlock`) and
`Problem.build_prompt` accepts one, but **no code joins them**, and nothing has ever
populated `Attempt.added_tokens` from a real injection. The join is a few lines; the
fact that it has never run end to end is why the metric is unverified.

**One live hazard already handled, worth not regressing:** every CORE prompt template
gates the guidelines slot on truthiness, so an empty block makes
`build_prompt(block)` byte-identical to `build_prompt(None)`. A budget too small for
any skill therefore degrades silently to no-memory. `RetrievedBlock.budget_bound`
and the `dropped_*` counts exist to make that visible — the loop must check them.

**And a second instance of that hazard, which those flags could not catch.** The join
existed but read the wrong tier. `evolve_skill_bank` is called with the bare task name,
so every distilled skill lands under `level_skills["tower-of-hanoi"]`; retrieval under
`--hierarchy task:type` asked only for `level_skills["tower-of-hanoi:5"]`, and
`general_skills` is empty for single-level runs (`skip_general`). **No skill ever
reached a prompt.** The smoke pilot proved it: with a 9-skill bank in hand, round 1's
prompt was byte-identical in length to round 0's no-memory prompt and contained no
skills block.

`Injection.starved` could not see it, and that is the instructive part: `starved` means
*the bank offered candidates and the budget dropped them all*. Here the bank offered
nothing **at the level asked for**, which is indistinguishable from an empty bank. A
full pilot would have reported no-memory accuracy under a with-memory label and read as
"the method does not help" — the single most expensive way for this project to be wrong.

Fixed by `injection.tiered_bank`, which serves a `task:type` key as the union of the
type-specific and cross-type tiers (specific first, deduplicated by `skill_id`). Note
that **confidence still ranks across the merged set**, per §4a — tier is only the
stable-sort tiebreak. Note also that until distillation is actually run per type, the
specific tier stays empty and the hierarchy remains degenerate exactly as §3 says.

### 4c. Token budget

**Decision.** Default injection budget **≈1000 tokens**, to sit at CORE's measured
0.92k so the Pareto comparison is like-for-like. Log actual added tokens per eval
item every time.

**Caveat, now resolved for the pilot path:** this said `tiktoken` was absent from the
interpreters that run the gate, making added-token figures *estimates*. It is present
in the vLLM env, and the Stage 3a pilot confirmed the live counter is
`tiktoken:o200k_base` with `exact=True`. Added-token figures from that env are
therefore exact and publishable. The rule is unchanged and still load-bearing: gate any
published figure on `tokens.is_exact()` at runtime, not on the `requirements.txt` pin
and not on this paragraph — a run in a different interpreter silently reverts to the
heuristic.

---

## 5. Decision: contamination and the memorization trap

This is the section most likely to save the project from an embarrassing finding.

### 5a. Eval contamination — guard exists, is never called

`adapter.assert_no_eval_contamination(tasks, attempted_ids)` is defined and tested,
but has **zero production call sites** (Known issue 1). Hard rule 4 currently has no
live enforcement.

**Decision.** The training loop accumulates every attempted problem id — training
solves, admission trials, reflection inputs — and calls the guard **before each
evaluation**, not once at the end. Contamination is the one error that cannot be
detected after the fact from the numbers: a leaked eval item just makes the method
look better.

Also freeze the bank at eval, mirroring CORE exactly (`register_problem=False`,
`update_utilities=False`, `mark_use=False`, `deterministic=True`,
`run_core.py:1604`): no reflection, no evolution, no admission, no bank writes, and
eval rollouts are never stored.

### 5b. The memorization trap — half-mitigated, half-open

Two facts combine badly:

- `Skill` has an `example` field, and most HeXA distil prompts ask for
  *"<optional concrete coordinate example>"* (`evolution_prompts.py:97`,
  `skills_only_prompts.py:108`, and others).
- **HeXA has no parent-problem exclusion.** CORE does:
  `memory.can_apply_lesson_to_problem` gated by
  `--leave-parent-problems-out-in-train` (`run_core.py:4023`, **default False**).

With 5 training problems and ~400 attempts each, nothing prevents the evolver from
writing a training problem's literal answer into a skill's `example`, and then
retrieving that skill for that same problem. Confidence rises. It looks like
learning. It is memorization.

It does **not** leak to eval — eval is a disjoint problem set. But it corrupts every
training-time utility estimate, which is exactly what confidence ranking, pruning,
and admission all consume. And a reviewer who reads one skill bank will see it.

**Status:** the prompt half is already handled. `prompts_core_tasks.py:268` asks for
*"an optional short illustration of the method, not of one problem's answer"*.

**Decision.** Close the other half: implement parent-problem exclusion for HeXA.
`Skill.source_seeds` already carries the originating problem ids (via
`attempt_to_trajectory`, which uses `problem_id` where HeXA expects `seed`), so
`retrieve_within_token_budget` can filter on it. **Flag, default on** — note this is
*stricter* than CORE's default of False, which is a deviation to declare rather than
hide. Add an assertion in the pilot that no admitted skill's text contains a
training problem's answer string.

### 5c. Admission gating (borrowed)

CORE tests a distilled lesson alone, in context, on its originating problem, and
admits only if it succeeds (`run_admission_trials`, `--lesson-admission-trials 1`).
HeXA has **no accept/reject gate at all** — the evolver's output goes straight into
the bank.

**Decision (M5).** Adopt it. **Flag, default on, ablated** — it is CORE's cheapest
structural advantage and A6 exists to find out whether it is what carries
performance. **Charge admission trials to `RolloutBudget.charge_admission`**, which
is inside `max_train`, matching CORE.

Note the interaction with 5b: testing a skill on its originating problem is exactly
the situation parent-problem exclusion forbids at *retrieval* time. These are
consistent — admission is a deliberate one-off probe, not utility accumulation — but
the loop must not let the admission trial write to `problem_table`.

---

## 6. Components to build

| Component | Depends on (exists) | Notes |
|---|---|---|
| `train_baseline.py` — the `b_q` producer | `budget.charge_baseline`, `logging_schema`, `llm/` | §2b. Seed namespace **1000**. Resumable. |
| `call_teacher` vLLM branch | `llm/vllm_client` | **The current hard blocker.** `distill.call_teacher` has only a `claude-` branch plus an HF fallback, so the evolver cannot call gpt-oss-120b — the configuration we committed to. Smallest real unblock; mirror the existing HF branch. |
| Trajectory writer | `trajectory.write_attempts` | `distill.load_trajectories` globs `trajectory_seed*.json` on disk. Our converter makes the right dict; nothing persists it in that layout. Either write that layout or rebind the loader (`trajectory.py:270` anticipates this). |
| `passive_loop.py` — the round loop | everything above | Attempt → score → store → (round boundary) → contrast-pair select → distil/evolve → admit → next round. Calls the contamination guard before each eval. |
| Injection join | `retrieve_within_token_budget`, `Problem.build_prompt` | §4b. Populate `Attempt.added_tokens`. Check `budget_bound`. |
| Parent-problem exclusion | `Skill.source_seeds` | §5b. |
| Admission gate | `budget.charge_admission` | §5c. |

---

## 7. Flags and defaults

| Flag | Default | Ablated | Why it exists |
|---|---|---|---|
| `--round-target` | 10 | — | Capped at `n` by streaming, giving 10 / 10 / 5 rounds at n = 100 / 10 / 5 (§1a) |
| `--problems-per-round` | `ceil(n/rounds)` | — | Curriculum chunk size: 10 / 1 / 1 (M3, §1a) |
| `--attempts-per-problem k` | `(budget−reserve)//(rounds·ppr)` | — | 18 / 180 / 380. Derived from the budget **net of the admission reserve**, so solve + admission is exactly 2000 in every regime (§1a) |
| `--cycle-chunks` | **off** | diagnostic | Second pass over the chunks; needs `--parent-exclusion` (§1b) |
| `--n-base` | 10 | — | CORE parity (M1) |
| `--usage-credit` | on | **yes** | Changes HeXA's method; must be reportable both ways |
| `--biased-sampling` / `--uniform-mix` | on / 0.5 | **yes** | Borrowed from CORE (M6) |
| `--admission-trials` | 1 | **yes (A6)** | Borrowed from CORE (M5) |
| `--no-m1` | off (M1 **on**) | **yes** | Pre-M1 confidence: binary outcome of one trajectory per problem. Saturates at 1.0, so the arm exists to show what the correction buys (§2) |
| `--parent-exclusion` | **on** | yes | Stricter than CORE's default False — declare it. Under pure streaming it should never fire (§1); a non-zero count is a bug. Mandatory under `--cycle-chunks` |
| `--injection-budget-tokens` | 1000 | Pareto sweep | Matches CORE's 0.92k (M7) |
| `--hierarchy` | `task:type` | **yes (A5)** | §3 |
| `--query-conditional` | off | **yes (A8, 100 only)** | §4a |
| `--o2o-seeding` | **off** | — | Pure Online for the headline. If reported, seed-trajectory cost is charged to the budget and shown on the x-axis (M8) |

---

## 8. Must be logged every run

Beyond the CORE-schema fields `logging_schema` already emits:

- bank size (skills, mistakes) vs round; utility distribution over the bank
  (CORE reports >91% non-negative weighted mass — a directly comparable number)
- admission accept rate
- **fraction of reflections with no positive example available for contrast**, split
  by `same_problem` / `different_problem` — and the **contrast mix per round**, since
  §1a predicts it differs systematically by regime (cross-problem at n=100,
  same-problem at n=5) and that is a reportable finding, not noise
- **`excluded_parent` count under pure streaming, which must be 0** (§1). Non-zero
  means a future-chunk problem reached retrieval, i.e. the curriculum leaked
- added tokens per eval item, **with `tokens.is_exact()`** alongside
- `RetrievedBlock.budget_bound` and `dropped_*` counts — silent truncation must be
  visible in the logs, not inferable
- `VLLM_SYSTEM_START_DATE`, vLLM version, TP layout, precision path — a change in
  the harmony date is a change in experimental condition
- pruning events: any skill removed whose measured utility was positive

---

## 9. Build order

1. `call_teacher` vLLM branch, then trajectory writer. Small, and nothing else can
   run without them.
2. `train_baseline.py`. Verify `b_q` against the Phase 2 no-memory eval numbers —
   they should agree on difficulty ordering (Hanoi ≪ ZebraLogic < Matchstick).
3. Injection join + parent exclusion + admission gate, unit-tested individually.
4. `passive_loop.py`. Contamination guard wired before each eval.
5. **Stage 3a pilot: Hanoi, 10-example, 3 runs (~7k calls).** Hanoi because it is
   where CORE is weakest (0.423), so the incumbent to beat is MemRL, not CORE.
   **Inspect before continuing:** bank size trajectory, admission accept rate,
   same-problem contrast availability, added tokens per eval item, and a manual read
   of one full skill bank for memorized answers.
6. Only then the rest of the grid.

---

## 10. Open risks

1. **A5 may be a null result.** If type-level transfer within a task is as weak as
   cross-task transfer, the hierarchy has nothing to show. Cheap to check early:
   inspect whether the pilot's bank naturally splits by type at all.
2. **Bank stability is a known exposure.** CORE deliberately does *not* continuously
   consolidate, citing Zhang et al. 2026 ("useful memories become faulty when
   continuously updated by LLMs"). HeXA's evolver does exactly the merge/revise/prune
   that paper warns about. Add a bank-stability run — held-out accuracy vs evolver
   round, extended past the normal budget, plus a check on whether pruning ever
   removes a high-utility skill. Getting ahead of this is worth more than A7 itself.
3. **The provider offset is not yet known.** If Phase 2 misses badly, `b_q` shifts
   and every utility estimate shifts with it. Do not build step 2 before the gate has
   run.
4. **Nothing has talked to a live server.** All LLM-client behaviour is fake-verified
   (Known issue 11). `probe()` exists to catch a wrong assumption at startup rather
   than 3000 calls in — use it.
