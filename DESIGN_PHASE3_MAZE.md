# Phase 3 design — HExA Design B on ORBIT Maze

> **Scope: Maze only.** Every number, prompt, reward bin and statistic below is specific
> to ORBIT's Maze environment at its published eval config — 9-turn episodes, shortest
> paths of 7–8, 6×6/7×6/7×7/6×7 grids, 207 clean instances of 256. None of it is
> task-general, and none of it transfers to Mastermind or to ORBIT's five training tasks
> without being re-derived.
>
> **Mastermind is not in scope and no Mastermind code exists.** It was dropped in
> `DECISIONS.md` D1: at its eval config there are only P(6,3) = 120 possible secret
> codes, and 96.5% of eval instances share a hidden code with the train pool despite
> zero seed overlap — so a bank can memorise codes and score as cross-instance transfer
> on a task where that claim cannot be tested. Maze's 19.1% layout overlap is a
> controllable confound (§8, §9) rather than a hole in the instance space.
>
> This is also **Design B, not Design C**: transfer is across *instances of one task*,
> never across tasks. Cross-task transfer is out of scope end to end.

Prerequisites: `RECON.md` (env facts), `DECISIONS.md` (D1–D7), Phase 1/2 code in
`hexa_orbit/`. Nothing here is built yet.

**One caveat, stated once.** The teacher-facing pieces — the trajectory formatter's
elision policy, the distillation prompt's emphasis, the reward bin boundaries — depend on
what failures actually look like on this env, which nobody has seen yet. They are marked
**[PROVISIONAL]** with a named revisit trigger. Everything not so marked is settled by
code that already exists and can be built now.

---

## 1. What Design B is here

Round *n* takes a batch of *x* **instances**. Each instance is already a 3-episode object,
so the actor's within-instance ICL (ORBIT's own mechanism) runs untouched inside it; the
bank is what carries across instances.

```
K_0 = {}                                      # empty bank: no offline trajectories exist
for n in 1..R:
    G_n = retrieve(K_{n-1}, budget_tokens)    # M7: token-capped block
    for instance in batch_n:                  # x instances, serial (D5)
        run = actor.run_instance(instance, guidelines=G_n)   # 3 episodes, ≤27 calls
    K_n = evolve(K_{n-1}, [instance_trajectory(r) for r in batch_n])
frozen_eval(K_R, eval_specs)                  # no writes, leaked/clean split
```

Two departures from the InterPhyre loop, both forced:

- **Round 1 distils, it does not start from a seeded bank.** HExA's headline config bootstraps
  from `Initial_trajectories/` (6 InterPhyre seeds). No ORBIT equivalent exists, so round 1
  is `distill_contrastive` over the first batch — the "pure online" configuration
  `docs/HEXA.md` describes as spending round 1 collecting raw data. Budget for that.
- **The evolver's unit is a full 3-episode instance**, not an episode. This is the plan's
  central design choice and it is what makes cross-episode strategy expressible at all
  ("if episode 1 revealed the corridor is a dead end, don't re-walk it") — a skill shape
  no single-episode task can produce.

---

## 2. The instance trajectory dict

`skillrl`'s distil/evolve path consumes trajectory dicts of a fixed shape, and
`hexa_core/trajectory.py` established that string ids round-trip fine
(`Skill.source_seeds` is annotated `list[int]` but never coerced). New module
`hexa_orbit/trajectory.py`, converting one `InstanceRun`:

```python
{
  "seed": run.instance_id,          # D6 identity, not the dataset seed
  "success": run.success_count > 0, # see §3 for why "solved at all"
  "iterations": len(run.steps),     # total steps, all 3 episodes → feeds reward
  "trajectory": [ {thought, action, observation}, ... ],   # one entry per step
  "action": <final action>,
  # additive keys, read by our reward/formatter only:
  "reward": instance_reward(run),
  "episode_successes": [...], "episode_lengths": [...],
  "first_success_episode": 1|2|3|None,
  "adapted_across_episodes": bool,
  "shortest_path_len": int, "leaked": bool,
  "n_unparsed": int, "n_truncated": int,
}
```

`iterations` is the total step count precisely so that a reward accidentally read by
`skillrl`'s stock `compute_trajectory_reward` still lands somewhere sane rather than
throwing — but the stock function's thresholds (1–3 / 4–7 / 8–15 / 16–25 iterations, tuned
to InterPhyre's 25-iteration budget) are wrong here and are replaced in §3.

---

## 3. Reward

`skillrl`'s stock reward is a single-episode outcome×efficiency binning. Every ORBIT
instance would land in its top success bucket or its `iterations < 10` failure bucket, so
it discriminates nothing. Replace via the existing seam:
`hexa_bridge.patch(reward_fn=instance_reward, ...)`, which rebinds
`compute_trajectory_reward` in **both** `teacher_prompts` and `distill` because it is read
in three places.

Hard constraint: output must stay in `[-1, 1]`. `distill._compute_initial_confidence` maps
mean reward from `[-1,1]` onto `[0.1,1.0]`; anything outside silently clamps.

The plan's four bins, made concrete. Keyed on **first success episode**, not
`success_count`:

| Instance outcome | R | rationale |
|---|---|---|
| solved episode 1 | `+1.0 − 0.15·excess` | the head start Design B claims; floor `+0.7` |
| first solved episode 2 | `+0.6` | within-instance learning worked |
| first solved episode 3 | `+0.3` | worked, slowly |
| never solved, episodes differ | `−0.5` | exploratory failure |
| never solved, episodes identical | `−1.0` | no adaptation across its own episodes |

`excess = episode_1_steps − shortest_path_len ∈ {0,1,2}` (a 9-turn budget against a 7–8
path leaves 1–2 moves of slack, RECON §1), so the efficiency term is nearly binary by
construction and is deliberately small.

Three notes on this table:

- **Why first-success-episode and not `success_count`.** The metric Design B lives or dies
  on is episode-1 success on held-out instances. Making that the top of the reward scale
  aligns the evolver's incentive with the claim. This is training on the objective, not
  leaking the test: legitimacy comes from train/eval instance disjointness (guaranteed and
  asserted, `pools.assert_disjoint`), not from the reward being ignorant of the metric.
  `success_count` — ORBIT's own `R(M;τ) ∈ {0,1,2,3}` — is logged separately and unchanged,
  so their reward definition remains reportable.
- **The bottom two rows are the finding, not just a bin.** `−1.0` isolates the failure mode
  ORBIT's Table 2 shows their single-episode RL baseline falling into. Its rate is already
  computed (`report.arm_summary`'s `no_adaptation_rate`), so "HExA's bank prevents
  non-adaptation" is reportable even if cross-instance transfer shows nothing.
- **[PROVISIONAL]** the `0.6 / 0.3` spacing and the `0.15` efficiency slope. **Revisit
  trigger:** if the Phase 2 baseline puts >80% of instances in one bin, the scale is not
  discriminating and the boundaries move.

`adapted_across_episodes` already exists on `InstanceRun` and compares per-episode action
sequences (not positions), so it is meaningful even when every episode fails on turn one.

**Confidence.** Pass `confidence_fn=distill._compute_initial_confidence` (the stock one) for
the first arm. M1's baseline-relative utility is explicitly lower priority here per the
plan — reward is a graded 5-level scale with real variance, not a coin flip, so confidence
is not degenerately confounded with base rate the way it was on Matchstick.

---

## 4. Trajectory formatter — the load-bearing piece

`skillrl`'s stock `format_trajectory_for_teacher` renders the action field as
`x=<char0>, y=<char1>, r=<char2>`: it assumes InterPhyre's `[x,y,r]` triple and emits
silent garbage on anything else. This is a live hazard, not an absence —
`hexa_core/trajectory.py` documents hitting exactly it. Replace through
`hexa_bridge.patch`'s formatter slot.

A maze instance's information content is **the coordinate sequence and what it revealed
about walls**, not prose. Rendering 27 turns of chain-of-thought would blow the teacher's
context and bury the signal. Proposed render, per instance:

```
=== Instance 132dc55db9c9 | grid 7x6 | shortest path 7 | R=+0.60 | solved first on episode 2
--- Episode 1: FAILED (9/9 turns used)
 turn  at      chose  result
    1  (4,3)   up     wall
    2  (4,3)   left   moved → (4,2)
    ...
    9  (2,2)   up     wall            [turn budget exhausted]
--- Episode 2: SOLVED in 8 turns (optimum 7)
    ...
walls found: (3,3) (5,2) ...        # derived, see below
```

Positions are recoverable from observations (`"Now you are at position (x, y)"`), and
outcomes from the four fixed result strings (moved / hit the wall / invalid action / goal).

Two decisions inside this:

- **Reasoning is elided by default.** [PROVISIONAL] — the counter-argument is that the
  *reason* a wrong turn was chosen is exactly what a mistake entry should capture. Plan an
  ablation: reasoning included for the single decisive turn per episode (the first turn
  whose choice diverges from the eventual successful route). **Revisit trigger:** inspect
  round-1 skills; if they are all about geometry and none about decision errors, reasoning
  elision is costing us the mistake entries.
- **`walls found:` is derived, and therefore optional.** It hands the teacher a summary the
  agent had to infer. That is the same latitude CORE's formatter takes, but it risks
  flattering the bank. Make it a flag, default **off** for the headline arm, and report it
  as an ablation.

Selection: `trajectory_selection="paired"` with `max_trajs≈6`. Pairing puts a solved and a
never-solved instance adjacent, which is the contrast that teaches, and 6 keeps the prompt
inside budget given each instance renders ~27 lines.

---

## 5. Retrieval and injection

Already built: `hexa_bridge.retrieve_within_token_budget(bank, task_name, budget_tokens,
max_general=0, max_specific=..., max_mistakes=...)`, and the actor's injection slot
(`MazeActor(guidelines=...)`), which appends to the system prompt only — so per-turn
observations stay byte-identical to the baseline and every added token is attributable to
one measurable block.

- `max_general=0`, no `RELATED_LEVELS` entry. Single task family; cross-task transfer is
  Design C and out of scope.
- `budget_tokens=1000`, tracking CORE's reported 0.92k so the added-token axis stays
  comparable across our own arms.
- `MAX_SKILLS_PER_LEVEL=10` unchanged. Bounded growth is doing real work: forcing the
  teacher to choose is what removes redundancy.
- ⚠ **`tiktoken` resolves only in the `vllm-gptoss` env**, not `phyreagent`. There, the
  counter is `heuristic:words+chars(approx,not-citable)` and the token *budget* is
  enforced against an estimate. Gate any published added-token figure on
  `tokens.is_exact()`, and prefer running Phase 3 from the server env.

---

## 6. Admission gating (M5)

`hexa_core/admission.py` already implements `admit_skills` / `test_skill_admission`: each
candidate is injected **alone** against fresh instances and kept only if it earns its slot.

Cost is the reason this needs a decision rather than a default. One admission trial is a
full instance = up to 27 model calls. Ten candidates × 3 trial instances = **~810 calls per
round**, comparable to the round's own actor budget (§7). Options, cheapest first:

1. **Single-episode trials.** Use `instance.new_episode()` (already on `MazeInstance`, the
   plan's Phase 1 `Episode` shape) and score on episode-1 success alone — ≤9 calls per
   trial, a 3× saving. Well matched to the claim, since episode-1 performance *is* the
   headline.
2. Fewer trial instances (2), accepting noisier accept/reject.
3. Skip M5 for the pilot; enable once the bank is known to be non-degenerate.

**Recommendation: (1).** Trial instances must come from the training pool and never the eval
pool; `seeds.py`'s namespace discipline is the precedent — an admission trial drawn from the
same namespace as a training solve is a *replay* and passes trivially.

---

## 7. Round sizing and cost

Constraints: eval pool 256 (D3 config), train pool 512 available, plan wants 8–12 rounds,
≤27 actor calls per instance, serial (D5).

| | value |
|---|---|
| training instances | 200 (prefix of the 512-instance train pool) |
| rounds *R* | 10 |
| batch *x* | 20 instances/round |
| actor calls/round | ≤540 |
| admission calls/round | ≤270 (option 1: 10 candidates × 3 single-episode trials) |
| evolver calls/round | 2–3 (distil/evolve + mistakes pass) |
| **total, training** | **~8.2k actor calls** |
| frozen eval | 100 instances × ≤27 = ≤2.7k calls |

Real per-instance cost is lower than the cap whenever episodes end early. The binding
practical constraint is wall-clock under D5's serial execution, not tokens — measure it on
the Phase 2 baseline before committing, since that run gives the true calls-per-instance
figure.

---

## 8. Frozen eval and the headline

Identical protocol to Phase 2, one difference: `guidelines=G_R` with the bank **frozen** —
no writes during eval, CORE's freeze convention. `report.aggregate` already produces the
three-curve comparison and the leaked/clean split, so the eval arm needs no new reporting
code, only a second invocation.

The comparison that matters is **episode 1, no-bank vs frozen-bank, on `clean` instances
specifically**. Both arms start every instance from zero within-instance, so ORBIT's
mechanism cannot move episode 1; a rise there is the mechanism their architecture lacks.
Restricting to `clean` is what makes it a strategy claim rather than a recall claim.
`report`'s `episode_1_leakage_gap` is the tripwire: a large positive gap means the pooled
headline is inflated by memorised layouts.

---

## 9. Degenerate-skill audit — the specific risk on this env

The failure mode is route memorisation: a "skill" that is really a remembered path, which
would show up as transfer on the 19.1% leaked instances and nothing on the clean ones.
New module `hexa_orbit/audit.py`, three checks per round:

1. **Literal-route detector.** Flag any skill whose text contains ≥3 consecutive direction
   words, or ≥2 coordinate literals `(x, y)`. Report the count per round; a rising count is
   the bank drifting toward memorisation.
2. **Clean-minus-leaked delta**, already computed. The empirical version of check 1 — and
   the one that matters if 1 misses a paraphrase.
3. **Structural-regularity coverage.** The good outcome, stated positively, because this env
   has genuine map-independent structure a bank *should* find:
   - the goal is always the BFS-farthest reachable cell from the start
     (`_find_farthest_position`) → "expand away from start, toward the far frontier"
   - the shortest path is always 7–8 (the filter guarantees it) → past ~8 moves you have
     detoured, so re-plan rather than push on
   - grids are only 6×6 / 7×6 / 7×7 / 6×7 with wall borders → interior coordinates are
     bounded and an observed `(x,y)` locates you globally
   - 9 turns against a 7–8 path = 1–2 wasted moves total → wall-bumping is nearly fatal

   These are strategy, not recall: they transfer to any layout and are immune to the
   leakage concern. Whether the evolver finds them is the substance of the plan's "inspect
   the bank before scaling" step, and it is the sharpest available test of whether the
   mechanism works at all on this env.

4. **`source_seeds` provenance.** A skill's confidence is the mean reward of the instances
   its `source_seeds` cite. The lookup is a plain dict match and, on a total miss,
   `_compute_initial_confidence` silently substitutes the mean reward over *every*
   trajectory in the round — so an abbreviated or invented id yields the round average
   wearing the costume of evidence. Instance ids are 16-hex strings, exactly the shape a
   model truncates, and confidence is the retriever's ranking key, so this changes what
   gets injected. Handled in two places:
   - **Prevention**, `teacher.make_confidence_fn` (installed by default): a citation that
     is a *unique* prefix or extension of a real id is repaired to that id; ambiguous
     matches are never guessed. On a total miss the stock round-mean value is returned
     unchanged — so no number moves silently — but the event is counted and logged at
     WARNING. Read the tally with `teacher.confidence_stats()`; `fell_back` is the field
     that invalidates a round's confidence ordering.
   - **Reporting**, `audit.validate_source_seeds(skills, known_ids)`: per-skill
     exact/repaired/unknown classification, miss rate, and the list of skills whose
     confidence is not evidence-based. Folded into `audit_bank(..., known_ids=...)`.

Check 3 is a manual read against a written list, not an automated metric. That is
deliberate — automating it would mean scoring skills against a rubric we wrote, which is
how a bank audit becomes circular.

---

## 10. Build order

1. `hexa_orbit/trajectory.py` — `InstanceRun` → trajectory dict (§2). Pure, testable now.
2. `hexa_orbit/reward.py` — `instance_reward` (§3), with bin-boundary tests. Pure.
3. `hexa_orbit/teacher.py` — formatter (§4) + `patch()` wiring + a Maze task description
   for `task_descriptions`. Testable against synthetic `InstanceRun`s.
4. `hexa_orbit/audit.py` — checks 1, 2 and 4 of §9, plus the confidence wrapper
   installed by `teacher.py`. Pure.
5. `hexa_orbit/trajectory.py` — add `load_instance_trajectories` (§12), the whole
   `evolve_skill_bank` integration. Pure, testable now.
6. `hexa_orbit/loop.py` — the round loop (§1, §12), mirroring
   `hexa_core/passive_loop.py`'s round-record, streaming-discipline and frozen-eval
   structure.
7. `scripts/run_orbit_maze_hexa.py` — CLI, with `--stub` parity to the baseline runner.

Items 1–5 are pure functions with no server dependency and can be written and tested
immediately (1–4 are done). Items 6–7 need the Phase 2 baseline's calls-per-instance
figure to size the round budget honestly.

---

## 11. Open, needing the baseline first

- Reward bin spacing and the efficiency slope (§3 [PROVISIONAL]).
- Reasoning elision in the formatter (§4 [PROVISIONAL]).
- Whether the base model clears the plan's Phase 2 gate (~0.55 episode-3). If it does, the
  interesting comparison moves and §8's framing changes before any of this is built.
- Whether `unparsed_rate` is low enough for success rates to mean anything at all. If
  `reasoning_effort="low"` and a 2048-token cap starve the model, that is a Phase 2 fix
  and every number here waits on it.

---

## 12. Round loop mechanics

### The integration seam

`evolve_skill_bank(level_name, prev_bank, new_trajs_dir, output_path, ...)` takes a
**directory** and calls `load_trajectories` on it. `hexa_bridge.patch` accepts
`load_trajectories_fn=`, so writing a round's runs to disk in our own format *is* the
whole integration — no format conversion and no fork of `evolving_distill`. This is
exactly how `hexa_core` wired CORE in (`passive_loop._write_round`).

One small module is therefore still missing from §10's list:

```python
# hexa_orbit/trajectory.py
def load_instance_trajectories(
    traj_dir, task_name=None, *, thought_max_chars=6000
) -> tuple[list[dict], list[dict]]:      # (successes, failures)
```

Signature-compatible with `distill.load_trajectories` (which globs
`trajectory_seed*.json`); ours globs `runs.jsonl`, rehydrates each line into an
`InstanceRun`, and returns `split_successes_failures`. A missing directory warns and
returns empty lists, matching the original.

### Persistence layout

```
results/orbit_maze_hexa/
  manifest.json               # task manifest, actor config, schedule, git revisions
  round0/
    runs.jsonl                # one InstanceRun per line -- what the evolver reads
    bank.json                 # K_0 AFTER this round's evolution
    audit.json                # audit_bank(..., known_ids=<this round's ids>)
    admission.json            # per-candidate accept/reject + trial outcomes
    injection.json            # the retrieved block, its token count, what was dropped
  round1/ ...
  eval_frozen/                # runs.jsonl + summary.json, bank frozen
  eval_no_bank/               # the Phase 2 arm, same instances, same sampling seeds
```

`bank.json` is written **after** evolution, so `roundN/bank.json` is the bank that
round *produced* and `round{N+1}` consumed. Writing it before instead would make an
off-by-one that is invisible until someone plots bank size against success rate.

### RoundRecord

Mirrors `passive_loop.RoundRecord`, dropping what does not apply (no `baseline`
stamping — M1 is out per §3) and adding what this env needs:

```python
@dataclass
class RoundRecord:
    round_index: int
    instance_ids: list[str]          # D6 identities, ordered
    summary: dict                    # report.arm_summary over the round's runs
    reward_bins: dict[str, int]      # {"fast success": 3, "no adaptation": 5, ...}
    n_skills_before: int
    n_skills_after: int
    n_mistakes_after: int
    admission: dict                  # accept_rate(...) over candidates
    audit: dict                      # audit_bank(...) incl. source_seeds provenance
    injection: dict                  # tokens injected, budget_bound, dropped counts
    confidence_stats: dict           # teacher.confidence_stats() delta for this round
```

`reward_bins` is worth its own field rather than being derivable: it is the
distribution the §3 `[PROVISIONAL]` boundaries get revisited against, and having it per
round is what makes "the scale stopped discriminating at round 6" visible.

### Streaming discipline

The bank must never be injected into an instance it was distilled from. Under this loop
that is automatic — round *n* applies `K_{n-1}`, evolved from batches `1..n-1`, to a
fresh batch `n`, and training instances are consumed once and never revisited. So a
`parent_exclusion` check should **never fire**, and if it does that is a bug, not a
safeguard doing its job. Expose it the way `passive_loop.PassiveRunResult.streaming_leak`
does: a boolean property that is expected to be `False`, asserted at the end of a run.

### Resumability

A 10-round run is ~8k model calls and will get interrupted. Resume needs two things,
both already on disk: the last complete `roundN/bank.json`, and the set of instance ids
already consumed (union of `instance_ids` across completed rounds). Restart from
`round{N+1}` with the remaining prefix of the training pool. A round is "complete" only
when its `bank.json` exists — a `runs.jsonl` without a `bank.json` means the run died
mid-evolution, and that round is redone from its runs rather than re-played.

### Budget accounting

Reuse `hexa_core.budget.RolloutBudget`, which already separates the namespaces this
loop needs (`charge_train_solve`, `charge_admission`, `charge_eval`) and raises
`BudgetExhausted` rather than silently overrunning. Charge per **model call**, not per
instance: instances that solve early cost well under the 27-call cap, and a
per-instance charge would over-reserve by roughly a third.

### Admission wiring

Candidates come out of evolution as new skills (`is_new: true`). Per §6, trial them
with `instance.new_episode()` at single-episode granularity, scoring episode-1 success.
Trial instances must come from the **training pool and outside the current round** —
`hexa_core/seeds.py`'s namespace discipline is the precedent: a trial drawn from the
same namespace as a solve already made is a replay and passes trivially.

Rejected candidates are logged with their trial outcomes and dropped, not silently
demoted to low confidence — an admission gate that only adjusts confidence is not a
gate.

---

## 13. Experiment arms

Ordered by whether the headline claim survives without them.

### Essential

| arm | bank | what it isolates |
|---|---|---|
| **A. no-bank** | none | the same-model anchor. Phase 2. |
| **B. frozen-bank** | `K_R`, frozen | the claim: cross-instance transfer |
| **C. length-matched placebo** | scrambled | whether *any* text in the system prompt helps |

**C is not optional**, and it is the arm most likely to be skipped and most likely to
be asked about. Arm B adds ~1000 tokens of confident-sounding task-relevant prose to the
system prompt; arm A adds nothing. If B beats A, "the bank taught it something" and "a
longer, more focused system prompt helps" are both consistent with the data. The
placebo separates them: take `K_R`, keep every skill's title and `when_to_apply`, and
permute which `principle` goes with which — so token count, register, and apparent
specificity are held fixed while the *content* is destroyed. If C also beats A, the
mechanism under test is not what is producing the gain.

### High value, cheap

| arm | what it isolates |
|---|---|
| **D. ORBIT reflection baseline** | `enable_reflection=True`, no bank — *their* mechanism |
| **E. bank-at-round-n** (`K_2`, `K_5`, `K_10`) | does more evolution help, or plateau/decay? |

D matters because it is the honest comparison. ORBIT ships a per-episode Reflexion-style
prompt (`reflection_prompt_2`) that is **off** in their eval path. "HExA beats a method
that had its own reflection mechanism disabled" is a weaker claim than "HExA's
per-round bank beats per-episode self-reflection". Note the cost asymmetry to report
alongside it: reflection spends an extra model call between episodes and does *not*
consume step budget (`multi_episode_env.py:172-190`).

E is three extra frozen evals and gives the learning curve. It is also the cheapest way
to catch bank decay — a bank that peaks at round 5 and degrades by round 10 is a real
result, and reporting only `K_R` would hide it.

### Ablations, if budget allows

| arm | what it isolates |
|---|---|
| F. skills-only (`skills_only_distill`) | do the mistake entries carry weight? |
| G. reasoning included in the formatter | §4's `[PROVISIONAL]` elision decision |
| H. derived `walls found:` summary on | how much the formatter is doing for the teacher |
| I. no admission gating | is M5 earning its ~2.7k calls? |

### Reported for every arm

- Episode 1 / 2 / 3 success rate, **split clean vs leaked** (never pooled only)
- `success_count` distribution over {0,1,2,3}
- no-adaptation rate among never-solved instances
- added tokens per instance, with `tokens.counter_name()`
- `unparsed_rate` and `truncated_rate` — read *before* any of the above

---

## 14. Statistics

### Primary endpoint, pre-registered

**Episode-1 success rate, arm B minus arm A, on `clean` eval instances.** One number.
Everything else in §13 is secondary and descriptive.

Pre-committing matters here because §13 lists nine arms and four reported metrics each,
across three episode indices and two leakage strata. That is enough surface to find a
significant-looking result by accident. The primary endpoint is fixed before the eval
runs; secondary results are reported with effect sizes and no significance claims.

### Pairing, and the trap

The unit is the instance. Both arms see the **same instances**, so the design is paired
and the test is **McNemar's exact test on discordant pairs** — this project's existing
convention (`hexa_core/SESSION_LOG.md`).

⚠ `hexa_core/WEAKNESSES.md:300` records the exact failure to avoid: *"records stored
seeds as though the arms were paired. McNemar assumes pairing."* Pairing here is not
automatic, because sampling is stochastic — at `temperature 0.6` the same arm on the
same instance gives different runs. Pairing therefore requires **the same per-instance
sampling seed in both arms**: `ActorConfig.seed` is already plumbed through to vLLM,
so derive it as `seed = f(instance_id, run_index)` and use the identical value in A, B
and C. Without that the arms are independent samples, McNemar is the wrong test, and the
comparison needs far more instances to say anything.

### Power, and why the eval pool should be the full 256

McNemar's power depends on the discordant count, not on *n* directly. Exact
two-sided p-values for plausible outcomes:

| clean *n* | b (A fail, B succeed) | c (A succeed, B fail) | p |
|---|---|---|---|
| 81 (from a 100-instance prefix) | 14 | 2 | 0.004 |
| 81 | 11 | 3 | 0.057 |
| 207 (full pool) | 24 | 7 | 0.003 |

The lever is the **absolute discordant count**, not the percentage-point difference.
Significance arrives at roughly `b ≥ 10` with few regressions (`b=10, c=2` → p = 0.039),
whatever *n* is — so a larger pool helps because the *same rate* of improvement yields
more discordant pairs. Concretely, at `c ≈ 0.25b`:

- 81 clean instances (a 100-instance prefix): significant from about **+10pp**
- 207 clean instances (the full pool): significant from about **+4pp**

Since nobody knows the effect size yet, and a null result on an underpowered sample is
uninformative rather than evidence of no effect, **the headline arms A/B/C should run the
full 256-instance eval pool**, at ≤6.9k calls per arm. The plan's 50–100 instances remains
right for the Phase 2 sanity check and for pilot inspection — just not for the claim.

These are exact-test boundaries, not a power analysis: they say when an *observed* split
reaches p < 0.05, not the probability of observing it. A real power calculation needs an
assumed effect size, which is what the Phase 2 baseline will finally supply.

### Secondary tests

- `success_count` is ordinal on {0,1,2,3} → **Wilcoxon signed-rank** on paired
  instances. Uses more information than the episode-1 binary, so it is the natural
  companion, but it is *not* the primary: a gain concentrated in episodes 2–3 is
  exactly what ORBIT's own mechanism can produce, so it does not discriminate.
- No-adaptation rate → McNemar on the never-solved subset. Report the subset size; it
  will be small and the test correspondingly weak.

### What is descriptive only, and must not be tested

- **The leaked-vs-clean gap** (`report.episode_1_leakage_gap`). It is a diagnostic on
  whether the pooled number is inflated by memorised layouts, not a hypothesis. The
  strata differ in *n* and in difficulty, and testing across them invites a conclusion
  the design cannot support.
- **Anything against ORBIT's Table 2.** Different model, and their Maze protocol is
  T=5 against our T=3 (D3). Reference lines on the plot, labelled, never differenced.
- **Bank size, confidence trajectories, skill lineage.** Plotted per round to see what
  happened; no tests attached.

### Reporting

For the primary endpoint: both rates with exact binomial CIs, the paired difference, the
discordant counts b and c, and the exact p — not the p alone. And the `unparsed_rate` for
both arms in the same table, because a success-rate difference is uninterpretable if the
two arms differed in how often the model emitted a parseable action at all.

