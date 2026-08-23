# RECON — HExA on the CORE single-turn benchmarks

Phase 0 deliverable. This file answers every Phase 0 recon question with a source
citation, and describes **what actually exists now**, not what was planned.

- HeXA (this repo, where we write): `/home/udhanuka_umass_edu/Scalar_Research/HeXA`
- CORE (read-only, imported in place, never forked):
  `/home/udhanuka_umass_edu/Scalar_Research/core-reasoning`
  at `999a96c527887cd36665eec419960c441529ab01`, `git status --porcelain -uall` empty

Citation convention: paths starting `src/`, `scripts/`, `shell_scripts/` or `data/`
are relative to the CORE root; everything else is relative to the HeXA root. Every
`path:line` below was re-resolved against the working tree while writing this file.
Line numbers are only as stable as the checkout — they are pinned to the SHA above.

---

## 0. Four findings that contradict the plan's assumptions

Read these before anything else. Each one changes a decision the plan already made.

### F1 — Confidence is a hybrid, not mean reward

The plan assumed skill confidence is the mean reward of the source trajectories.
It is that **only for newly minted skills**:

- **New skills** — `skillrl/distillation/distill.py:163-198`
  (`_compute_initial_confidence`) averages `compute_trajectory_reward` over the
  trajectories whose `seed` appears in the skill's `source_seeds` and maps
  `[-1,1] → [0.1,1.0]` via `max(0.1, min(1.0, (avg_reward + 1.0) / 2.0))`
  (`distill.py:198`). If **no** `source_seed` matches a trajectory in the batch it
  silently falls back to the batch-wide mean (`distill.py:191-196`) — not to the
  `fallback_confidence` argument.
- **Retained skills** — `skillrl/distillation/evolving_distill.py:135-145` keeps the
  **teacher's self-reported float**: `confidence = raw.get("confidence", 0.5)`,
  clamped to `[0.1, 1.0]`. The evolution prompt asks for it in prose
  (`skillrl/distillation/evolution_prompts.py:76-82`) and tells the teacher to
  "preserve their existing confidence values (they've been validated)"
  (`evolution_prompts.py:77`).

So the number that drives retrieval ordering (`skillrl/core/retriever.py:36-40`,
`53-58` — both sorts are `key=lambda s: s.confidence`) is **empirical for
generation-0 skills and self-reported for everything that survives an evolve
round**, on the same scale, with nothing in the record distinguishing the two.
Consequence: any claim that retrieval is confidence-*calibrated* is false past
round 1, and M1's baseline-relative utility must replace **both** paths or it will
be diluted by teacher-asserted values.

### F2 — Rollout memory stores only CORRECT attempts

`scripts/run_core.py:2132` gates the persistence of a training attempt on
`if is_correct:` — `_append_past_attempt_record(past_attempts, past_attempts_file,
persisted_attempt)` runs only on success. The attempt dict is fully built either way
(`run_core.py:2100-2131`), so an incorrect attempt is scored, logged into the
episode record, and then dropped from rollout memory.

The paper's prose is right; **Algorithm 1 line 9 is misleading** — it reads as an
unconditional append. Consequence: CORE's episodic store is a success buffer, so any
"replay"/"episodic RAG" comparison must not be described as replaying failures, and
our own contrastive distillation (which *needs* failures) is drawing from a strictly
larger pool than CORE's memory — a difference in kind, not degree, and it must be
stated wherever the two are put side by side.

### F3 — CORE's "localized" retrieval is query-INDEPENDENT in every shipped config

This is the finding the plan got closest to and still got wrong in a way that
inverts the follow-up experiment.

The plan (and `BRIEF.md:158-159`) assumed `knn_max_z = 10` fixed, hence
`knn_k = min(10, |D_train|)`, hence query-independence at N=5 and N=10 only, with
conditioning "existing at 100". What the shipped run scripts actually set:

| regime | script | `--num-train-items` | `--knn-max-z` |
|---|---|---|---|
| N=5 | `shell_scripts/sample_efficiency/5_train/*/run_core.sh` | 5 | **5** |
| N=10 | `shell_scripts/rollout_efficiency/*/run_core.sh` | 10 | **10** |
| N=100 | `shell_scripts/sample_efficiency/100_train/*/run_core.sh` | 100 | **100** |

`knn_max_z` is set to exactly `|D_train|` in all 12 shipped scripts. In
`src/memory/localized_ucb.py:485`, `knn_k = min(self.knn_max_z, m_mem)` where
`m_mem` is the number of *registered* problems. Registration happens only on the
training path with `split="train"` (`run_core.py:4655`, `run_core.py:5212`;
eval passes force `register_problem=False` at `run_core.py:963`, `1266`, `1629`) and
is keyed per problem (`localized_ucb.py:185-235`), so `m_mem ≤ |D_train| = knn_max_z`
**always**. Therefore `knn_k == m_mem`: the neighbourhood is every registered
problem, in all three regimes.

Two further facts make this airtight rather than incidental:

1. **Similarity is discarded in the aggregation.** `localized_ucb.py:500` iterates
   `for neighbor_problem_id, _sim in neighbors:` — the underscore is literal; the
   cosine value is never read again. The averages at `localized_ucb.py:509-520` are
   weighted by `n_eff` (evidence count), not by similarity. Similarity's *only*
   role is to rank-and-truncate at `localized_ucb.py:486-487`, and truncation never
   binds.
2. **The one mechanism that could reintroduce query dependence is off by default.**
   `leave_parent_problems_out_in_train` filters *candidate lessons* by query id
   (`localized_ucb.py:455-463`) and defaults to `False`
   (`localized_ucb.py:173`, argparse default at `run_core.py:4023-4027`); no shipped
   script sets it.

So `Û(q,i)` is the **same global n_eff-weighted average for every query q**, in
every published regime. CORE's 10-example Matchstick retrieval ablation therefore
isolates *ranking and truncation of a global lesson pool*, not per-query
conditioning — which is the correct way to read their result, and is a genuine
contribution rather than a nitpick.

**Consequence for A8, corrected:** running it at 100 examples does **not** restore
conditioning, because `knn-max-z 100` is exactly what the 100-train script uses.
A8 can only isolate conditioning by *deviating* from CORE's config — setting
`knn_max_z < |D_train|` (e.g. 10 at N=100) — and that deviation must be declared as
a deviation, because it is no longer the published configuration. If A8 must stay
inside CORE's shipped settings, it cannot answer the conditioning question at all,
and the honest report is that CORE's localization is inert as published.

Note the same is structurally true on our side: `retrieve_skills`
(`skillrl/core/retriever.py:24-28`) takes `level_name` and no query at all, so HeXA
retrieval is query-independent by construction. That symmetry is worth stating
explicitly rather than letting a reader assume we conditioned and CORE did not.

### F4 — ZebraLogic xlarge is not "5x5 to 6x6", and N=100 exhausts the pool exactly

`_size_group_from_size` (`src/tasks/single_turn/zebralogic_task.py:36-42`) defines
the group by **search-space size**, `houses! ** attributes`, with
`xlarge ⇔ ≥ 10**10`; `_parse_size` reads `houses*attributes`
(`zebralogic_task.py:28-33`). Evaluated over the whole 2..6 grid, xlarge is exactly
`{5*5, 5*6, 6*4, 6*5, 6*6}` — **6 houses x 4 attributes is xlarge**, which the
"5x5 up to 6x6" phrasing excludes. Measured on the bundled pool
(`data/processed/zebralogic/grid_mode.jsonl`, 1000 rows: small 320, medium 280,
large 200, xlarge 200 — 40 of each xlarge size):

| split | 5*5 | 5*6 | 6*4 | 6*5 | 6*6 |
|---|---|---|---|---|---|
| eval (100) | 21 | 19 | **21** | 23 | 16 |
| train pool (100) | 19 | 21 | **19** | 17 | 24 |
| train N=5 | 1 | 1 | **1** | 1 | 1 |
| train N=10 | 2 | 2 | **2** | 2 | 2 |
| train N=100 | 19 | 21 | **19** | 17 | 24 |

So 6*4 is 21% of the eval set. The `{level_description}` slot fed to the teacher
must not misdescribe a fifth of the distribution.

The pool arithmetic (`zebralogic_task.py:521-545`): filter to xlarge (200) →
`random.Random(0).shuffle` → `eval_count = min(100, 200)`, `train_pool = data[:100]`,
`eval = data[100:]`. **At N=100 the train pool is consumed exactly** — 100 train +
100 eval = the entire xlarge population, zero slack. Anything that needs one more
ZebraLogic xlarge training problem than 100 (a dev split, a held-out calibration
set, a resample) is impossible without changing the size group or the split rule.

For the same reason, **Phase 4's enumeration-size table needs revising**, because
the per-task pool sizes are not what a "sample efficiency" framing suggests:

| task | train pool | eval pool | at N=100 |
|---|---|---|---|
| matchstick | 100 rows, `data/processed/matchstick/training/matchstick_train.jsonl` (5 types x 20) | 100 rows, `.../evaluation/matchstick_test.jsonl` (5 types x 20) | pool exhausted exactly |
| zebralogic | 100 (first half of shuffled xlarge) | 100 (second half) | **pool exhausted exactly** |
| tower-of-hanoi | 242 eligible 5-disk states (`3**5 - 1`, measured) | 728 eligible 6-disk states (`3**6 - 1`, measured; max optimal distance 63) | 100 of 242, slack remains |

Hanoi is the only task with train-pool headroom at N=100. Two of the three
"100-example" cells are therefore *whole-population* cells, and a 200-example row
cannot exist for matchstick or zebralogic at all.

---

## 1. CORE side — Phase 0 answers

### 1.1 How CORE is imported, and how tasks are constructed

`src/` mixes package-relative (`from ..base import`) and flat-absolute
(`from reflection.single_turn_prompt_templates import`) imports, so `<core_root>/src`
goes on `sys.path` and modules are imported as `tasks.single_turn.<module>`
(`hexa_core/core_bridge.py:98-115`, which also sets `sys.dont_write_bytecode`).

`scripts/run_core.py` is **not importable** as library code: it calls `os.chdir(ROOT)`
at import time (`run_core.py:24`). Pure helpers are transcribed with a citation
(`_compute_generation_seed`, `run_core.py:1063-1067` →
`hexa_core/baseline_runner.py:core_generation_seed`). The test suite *does* import it
under a stubbed meta-path finder with cwd restored, purely to prove the
transcriptions are byte-exact (`_import_run_core()` in
`hexa_core/tests/test_logging_schema.py`).

Task construction goes through **CORE's own factory**, not through our own mapping:
`tasks.single_turn.get_task` (`src/tasks/single_turn/__init__.py:28-68`) is what
`run_core.py`'s `init_task` calls (`run_core.py:3657-3670`), and
`hexa_core/tasks.py:107` calls the same function. `hexa_core.tasks.TASK_SPECS` carries
only the paper's argument **values**, written in `get_task`'s own vocabulary
(`zebralogic_size_group`, `hanoi_train_disks`, `matchstick_dataset`).

This matters because **`get_task`'s defaults are not the paper config**: Hanoi
defaults to `hanoi_train_disks=[3,4,5,6]`, `hanoi_eval_disks=[7,8]`
(`__init__.py:57-58`). A construction path that re-derived keywords itself would
still look byte-perfect while building a task from the wrong disks.

Paper config, from the shipped scripts (these *are* the published configuration):

```
matchstick     : num_train_items=N, num_eval_items=100, dataset="default"
zebralogic     : num_train_items=N, num_eval_items=100, size_group="xlarge"
tower-of-hanoi : num_train_items=N, num_eval_items=100, variant="state",
                 train_disks=[5], eval_disks=[6], enforce_optimal=False,
                 min_distance=1, seed=42
```

**Gotcha, confirmed:** `TowerOfHanoiTask(variant="state").name` is
`"tower-of-hanoi-state"` (`src/tasks/single_turn/tower_of_hanoi_task.py:631-634`),
not `tower-of-hanoi`. Our registry key is the short form and is what lands in
`Problem.task_name` / `Skill.source_level`; CORE's name appears only as
`CoreTask.manifest()["core_task_name"]`. Anything joining our logs to CORE's must
translate.

### 1.2 Eval splits and their exact recoverability, per task

All four are deterministic. Three are also **regime-independent**: eval id lists are
byte-identical across N ∈ {5,10,100}, asserted in `hexa_core/tests/test_splits.py`,
digests (sha256 over `"\n".join(eval_ids)`, first 12 hex) —
matchstick `4f7cb366114f`, zebralogic `265f1343ee32`, tower-of-hanoi `bd7be6454bd3`.
`train ∩ eval == ∅` by id for every task and regime (also asserted).

**MathGAP is the exception, and it was added later than the other three.** Its eval
set is *not* regime-independent: `_prepare_data` slices one shuffled pool as
`train = data[:N]` and `eval = data[N : N+100]` (`mathgap_task.py:163-164`), so
raising N slides the eval window forward. The splits stay disjoint at every N, so
this is a **comparability** constraint, not a contamination risk: no MathGAP number
may be compared across regimes, and a run's manifest `eval_ids_sha256` is the thing
to check before pooling two runs. Declared as
`TaskSpec.eval_split_regime_invariant=False` and asserted in *both* directions
(`test_splits.py::test_mathgap_eval_window_slides`) so the difference cannot be
quietly forgotten. At our standard N=60 the eval set is `data[60:160]`, digest
recorded per run.

| task | mechanism | exact recoverability |
|---|---|---|
| **matchstick** | Checked-in JSONL, read in file order. `_load_data` (`matchstick_task.py:236-263`) returns `items[:num_eval_items]` for the eval split. Train uses `_select_balanced_rows_by_type` (def `:73`, called `:243`). Paths at `matchstick_task.py:135-136`. | **Fully recoverable, no RNG.** Eval = all 100 rows of `evaluation/matchstick_test.jsonl`, in file order. Train is a deterministic balanced take over 5 types x 20. |
| **mathgap** | `_prepare_data` (`mathgap_task.py:150-184`): load `data/processed/mathgap/reasoning.json` (280 rows, all `task=="mathgap"`), `random.Random(0).shuffle(data)` (`:160-161`), then `train = data[:N]`, `eval = data[N : N+100]`. Answer target is `item['text_target']`; ids are `item['custom_id']`. | **Fully recoverable** by replaying `random.Random(0).shuffle` over the file in order. Seed 0 is hardcoded and independent of `--seed`. **Regime-dependent** (see above). Pool is only 280 rows, so `N <= 180`; `make_task` raises above that rather than letting CORE soft-degrade to a short eval split. Note also `mathgap_task.py:14` calls `random.seed(0)` at *module import*, mutating the global RNG — harmless for us because every sampler we own uses a local `random.Random`, but it means import order can perturb any code that relies on the global one. |
| **zebralogic** | `_prepare_data` (`zebralogic_task.py:520-560`): filter `size_group=="xlarge"` → `random.Random(0).shuffle(data)` (`:531`) → `eval = data[100:]`, `train_pool = data[:100]` → `_select_balanced_train_subset(train_pool, target_total=N)`. | **Fully recoverable**, but only by replaying `random.Random(0).shuffle` over the pool in file order. Seed 0 is hardcoded, independent of `--seed`. Reproduced independently while writing this file; distribution table in F4. |
| **tower-of-hanoi** | BFS state index from the goal (`tower_of_hanoi_task.py:378-404`), eligible states filtered on `distance_by_state[state] >= min_distance` (`:882`), then a per-disk RNG seeded by the **string** `f"{seed}:tower-of-hanoi:{variant}:{split}:n{disk}:d{min_distance}"` (`:905-907`), shuffle, take quota, then `items.sort(key=task_item_id)` (`:918`). | **Fully recoverable** given `(seed, variant, split, disk, min_distance)`. Train disks `[5]` and eval disks `[6]` make the splits disjoint *by construction* — different state spaces, not a partition. `seed` is the only knob that changes the sample; it is 42 in the paper config. |

Pool sizes measured, not assumed: Hanoi eligible states are 242 (n=5) and 728 (n=6),
i.e. `3**n - 1` — the full state graph is connected and only the goal state is
excluded by `min_distance=1`.

### 1.3 Verifiers

Instance methods on the task, pure, no hidden state:

```python
pred = task.parse_answer(full_raw_model_text)      # -> str
ok   = task.evaluate(pred, item.all_valid_outputs) # -> bool
```

| task | `parse_answer` | `evaluate` |
|---|---|---|
| matchstick | `matchstick_task.py:171` | `matchstick_task.py:196-204` |
| zebralogic | `zebralogic_task.py:463` | `zebralogic_task.py:489-505` |
| tower-of-hanoi | `tower_of_hanoi_task.py:672-677` | `tower_of_hanoi_task.py:679-714` |

**Two independent numbers, not one.** `run_core.py:1782-1789`
(`_score_task_prediction`) calls `task.score_prediction(pred, target)` for
`verifier_score` and `task.evaluate(pred, target)` for `is_correct` — two separate
calls. Probed live: **none** of the three tasks overrides `score_prediction`, so all
inherit the binary base implementation (`src/tasks/base.py:93`) and the two agree
today. `hexa_core.adapter.Problem.score()` delegates to the originating CORE
instance rather than deriving one from the other
(`hexa_core/adapter.py:112-126`, bound at `:202`), so if CORE ever grants partial
credit we inherit it instead of silently reporting binary under CORE's key.

**Rewards are binary** on all three: `uses_verifier_score_for_reward()`
(`src/tasks/base.py:100`) is False, so `_reward_from_metrics` returns
`float(bool(is_correct))`.

**What text gets scored — the parity detail that decides everything.**
`NvidiaClient.generate_structured` builds
`parts = [part for part in (reasoning_text, final_text) if part]` then
`raw_text = "".join(parts)` (`src/model/nvidia_client.py:143-144`) — concatenated,
**no separator** — and `generate_answer` (`run_core.py:433`, `out = primary.raw_text`
at `:469`) scores `raw_text`. The
chain-of-thought is part of the string the verifier parses. This is not cosmetic:
Hanoi's `_extract_moves_from_text` returns `candidates[-1]`
(`tower_of_hanoi_task.py:325-350`) — the **last** parseable move list in the text —
so including or excluding the CoT changes which candidate wins. Note the `if part`
filter keeps whitespace-only parts, so `(" ", "F")` must yield `" F"`; a client that
"tidies up" with `.strip()` diverges. The single definition of this join in our code
is `hexa_core/llm/base.py::Generation.from_parts`.

### 1.4 Generators and difficulty parameters

- **matchstick** — no generator; a fixed checked-in dataset. Difficulty is the
  `type` field, 5 classes x 20 per split (`value-loose-chunk`, `value-tight-chunk`,
  `hybrid`, `operator`, `tautological`), and the train selector balances across them
  (`_select_balanced_rows_by_type`, `matchstick_task.py:73`). `dataset="default"`
  is the only shipped variant.
- **zebralogic** — no generator; a fixed pool with a difficulty *classifier*.
  `_size_group_from_size` (`zebralogic_task.py:36-42`) buckets on
  `houses! ** attributes`: `<10**3` small, `<10**6` medium, `<10**10` large, else
  xlarge. `size_group="xlarge"` is the paper setting; `VALID_SIZE_GROUPS` also allows
  `"all"`, which switches off the balanced train selector (`:541-548`).
- **tower-of-hanoi** — a real generator: BFS over the 3^n state space from the
  canonical goal (`tower_of_hanoi_task.py:378-404`). Difficulty parameters:
  `train_disks` / `eval_disks` (the state space itself), `min_distance` (minimum
  optimal-move distance from goal, `:882`), `enforce_optimal` (`:712` — whether a
  correct-but-suboptimal solution scores 0), and `variant`
  (`state` vs the canonical prompt, `:636-648`). Paper: disks 5/6,
  `min_distance=1`, `enforce_optimal=False`. Optimal distance reaches 63 at n=6.

### 1.5 Answer-format instructions

Verbatim, from the class attribute `output_instruction`:

| task | source | text |
|---|---|---|
| matchstick | `matchstick_task.py:26` (`OUTPUT_INSTRUCTION`), exposed at `:114` | `Please think step by step. At the end of your response, write your final answer in this format: 'Correct version of the equation: <your solution>'` |
| zebralogic | `zebralogic_task.py:148-151`, exposed at `:414` | `Present your reasoning and solution in the requested paper-style JSON format with top-level keys "reasoning" and "solution".` |
| tower-of-hanoi | `tower_of_hanoi_task.py:76-79`, exposed at `:587` | `Ensure your final answer includes the complete list of moves in the format: moves = [[disk_id, from_peg, to_peg], ...]` |

**`format_instructions in prompt` does NOT hold for zebralogic** (the design brief
assumed it did). ZebraLogic's `output_instruction` is a prose summary, while
`_build_problem_prompt` (`zebralogic_task.py:199-212`) embeds a **per-problem JSON
schema** instead. `Problem.format_instructions` still holds
`task.output_instruction` verbatim, but the substring check applies only to
matchstick and tower-of-hanoi (`_EMBEDS_OUTPUT_INSTRUCTION` in
`hexa_core/tests/test_foundation.py`).

**The guidelines slot, and the silent-no-op hazard.** All three builders gate the
memory block on plain truthiness — `if guidelines:` at `matchstick_task.py:144`,
`zebralogic_task.py:201`, `tower_of_hanoi_task.py:637`. So passing `""` produces a
prompt **byte-identical** to passing `None`: a misconfigured token budget silently
turns the memory-augmented arm into its own no-memory baseline, with every log line
looking healthy. That is why `hexa_bridge.retrieve_within_token_budget` warns when a
populated bank renders to an empty block (`hexa_core/hexa_bridge.py:300-327`) and
reports `dropped_skills` / `dropped_mistakes` / `budget_bound`.

**`ZebraLogicTask.build_solve_prompt` is stateful** — it reads
`self._problem_output_schema_json[problem]` (`zebralogic_task.py:436-461`, table
built at `:526-529`). All 100 eval problems have distinct schemas, and a *fresh*
task instance asked to build another instance's prompt emits a placeholder schema
with **no error**. Prompt builders must stay bound to the instance the problem came
from; `hexa_core.adapter.Problem` binds them per instance.

### 1.6 Rollout accounting, including the baseline-outside-`max_rollouts` subtlety

The rollout counter advances only by solve + admission:
`solve_rollout_count = len(attempts)`, `admission_rollout_count` from the episode
log, `episode_rollout_count = solve + admission` (`run_core.py:2487-2493`), and the
driver's `current_rollout_count` starts from `start_rollout`
(`run_core.py:4243`, `4499`) — which is 0 on a fresh run and only non-zero when
restored from a checkpoint (`run_core.py:689`). Baseline estimation
(`run_baseline_training`, `run_core.py:1145-1200`) runs before that loop and its
calls are **never added to the counter**.

`--num-baseline-train-runs` defaults to **10** (`run_core.py:4036`) and the shipped
scripts pass `--run-baseline-train True` with `--max-rollouts 2000`.

So `--max-rollouts 2000` means **2000 training calls**, and the true cost per run is
`2000 + 10*|D_train|`:

| regime | gated (solve + admission) | baseline (outside) | total LLM calls |
|---|---|---|---|
| N=5 | 2000 | 50 | **2050** |
| N=10 | 2000 | 100 | **2100** |
| N=100 | 2000 | 1000 | **3000** |

`hexa_core.budget.RolloutBudget` encodes exactly this: `training_rollouts` is
`train_solve + admission` and is the only quantity `max_train` gates;
`charge_baseline` is never gated (`hexa_core/budget.py`). It raises *before*
incrementing, so a caller that catches `BudgetExhausted` can still write a truthful
manifest. `report()["all_in_total"]` additionally includes eval, which CORE's figure
omits — do not quote it as CORE's number.

Other shipped budget-relevant settings: `--val-runs 1`, `K_train=K_eval=25`,
`--min-usages 25`, `--ucb-beta 0.025`, `--lesson-admission-trials 1`,
`--num-lessons-per-reflection 3`, `--eval-period 10000`, `--snapshot-period 50`,
biased problem sampling on with `--biased-problem-sampling-uniform-mix 0.5`,
`--lesson-duplication-threshold 0.95`, `--test-lessons-individually True`,
`--always-accept-lessons False`. `--run-initial-eval` is `True` for matchstick and
zebralogic and `False` for tower-of-hanoi.

**Seed namespaces are per phase and unrelated** — verified at each call site, all
through `_compute_generation_seed(base, *components)` (`run_core.py:1063-1067`):

| phase | namespace | call site | base seed |
|---|---|---|---|
| baseline estimation | **1000** | `run_core.py:1193` (`run_baseline_training`) | `args.llm_seed` |
| eval / validation passes | 2000 | `run_core.py:1421` (`_run_memory_passes_on_items`) | `args.llm_seed` |
| training solves | 3000 | `run_core.py:1993` | `args.llm_seed` |
| reflection | 4000 | `run_core.py:2603` (`seed_namespace=`) | `args.llm_seed` |
| lesson admission | 5000 / 5001 | `run_core.py:3290-3291` | `args.llm_seed` |
| biased item sampler | 7000 | `run_core.py:1134` | **`args.seed`**, not `llm_seed` |

All six are constants in `hexa_core/baseline_runner.py:69-84`, each with its
enclosing-function provenance. **M1 must use `CORE_BASELINE_SEED_OFFSET = 1000`**:
`b_q` is a training-item quantity drawn from the baseline distribution, and taking it
from namespace 2000 would produce plausible numbers from the eval sampling
distribution — exactly the confound M1 exists to remove.

### 1.7 Eval is frozen

`evaluate_memory_on` (`run_core.py:1604-1631`) passes `deterministic=True`,
`mark_use=False`, `register_problem=False`, `update_utilities=False`
(`run_core.py:1626-1630`). `deterministic=True` also zeroes the UCB exploration
bonus (`localized_ucb.py:561-567`: the bonus is computed only
`if mark_use and not deterministic`). We mirror all four; the bank is never written
during evaluation.

### 1.8 Inference provider

CORE's published numbers come from **NVIDIA NIM**: `NvidiaClient` with
`base_url = "https://integrate.api.nvidia.com/v1"`
(`src/model/nvidia_client.py:24`), `--model nvidia --nvidia-model "openai/gpt-oss-120b"`
in every shipped script, `NVIDIA_API_KEY` required (`run_core.py:3650-3655`).

Generation config from the shipped scripts: temperature 0.6 (reflection 0.6, except
Hanoi's reflection at **0.8**), top_p 0.95, solve/eval/reflection tokens 32768,
`reasoning_effort="high"` injected for model names containing `gpt-oss-120b`,
embeddings `jinaai/jina-embeddings-v2-base-en` with `--embedding-provider auto`.
`max_tokens=32768` spans reasoning + final **combined**.

We serve `gpt-oss-120b` on **local vLLM** instead. See DECISIONS.md for why, and for
the citability consequence.

### 1.9 Logging schema

Matching CORE's schema is what makes their plotting code reusable.

- `validation.jsonl` — 12 keys in insertion order, plus `note` appended 13th when
  not None. Written by `_log_validation_results` (`run_core.py:1923`) via
  `_append_jsonl_record` (`run_core.py:1853`); file path set at `run_core.py:4219`.
  Keys: `episode`, `rollout`, `val_acc_with_mem`, `val_std_with_mem`,
  `val_mean_verifier_score`, `val_verifier_score_std`, `val_runs`,
  `individual_run_accuracies`, `individual_run_mean_verifier_scores`,
  `detailed_results`, `timestamp`, `memory_size`.
- `validation_summary.jsonl` — `run_core.py:1898-1922`, path at `run_core.py:4220`.
  **The brief was wrong** to call this "the same minus `detailed_results`"
  (`BRIEF.md:167`). It is a 6-key subset (`episode`, `val_acc_with_mem`,
  `val_std_with_mem`, `individual_run_accuracies`, `timestamp`, `memory_size`), then
  `rollout`/`val_runs`/`note` only `if key in record`, then six diagnostics keys
  merged from `_summarize_validation_attempt_diagnostics` (`run_core.py:1865`):
  `total_attempt_count`, `primary_finish_reason_counts`, `prediction_source_counts`,
  `primary_length_count`, `missing_usable_answer_count`,
  `missing_usable_answer_rate`. **14 keys, and it does not carry
  `val_mean_verifier_score`.**

`hexa_core/logging_schema.py` reproduces both, and the parity is **asserted against
CORE's own functions**, not eyeballed (`hexa_core/tests/test_logging_schema.py`,
including byte-equality of the written line). `val_std_with_mem` is CORE's `np.std`
(ddof=0) over run accuracies (`run_core.py:1601`); our `population_std` reproduces
it and appears under that key only.

`primary_length_count` is the truncation diagnostic the Phase-2 ladder needs. It only
becomes non-zero if the attempt dicts carry `primary_finish_reason` and
`prediction_source`; CORE itself sets those only for structured-generation models, so
`{"null": N}` is legitimate *there* but would mean the diagnostic is dead *here*.
`hexa_core/baseline_runner.py` populates them.

---

## 2. HeXA side — Phase 0 answers

### 2.1 Actor loop cap

`DEFAULT_MAX_ITERATIONS = 25` (`skillrl/core/config.py:92`), threaded through every
loop as `max_iterations` (`skillrl/loops/evolving_loop.py:105` signature default,
`:283` passed into `run_batch`, `--max-iterations` at `:402`). This is an InterPhyre
ReAct budget — 25 tool-using turns per episode.

**On CORE tasks the cap is 1.** All three CORE tasks are single-turn: one prompt,
one generation, verify. `Attempt` records exactly one step, and `Problem.tools`
defaults to `None`, which is the passive mode (Table A). Consequently the shaped
reward's iteration-count term has no signal at all on CORE tasks — see 2.6.

### 2.2 Evolver batch unit and round boundary

The round unit is `seeds_per_round` **episodes**, not rollouts
(`skillrl/loops/evolving_loop.py:99`, default 5). One round, in source order:

1. **Phase 1** — evolve the *previous* round's bank from the *previous* round's
   trajectories (`evolving_loop.py:219-244`, calling
   `evolving_distill.evolve_skill_bank`). Round 1 instead distils from a seed
   trajectory directory (`:199-207`).
2. **Phase 2** — run the agent on `seeds_per_round` **fresh** seeds with that bank
   injected (`evolving_loop.py:248-303`; `run_batch(..., skill_bank=bank, ...)`).
3. **Phase 3** — `round_accuracy = len(successes)/len(results)` over exactly those
   seeds (`evolving_loop.py:305-323`).
4. Advance: `current_seed = max(seeds) + 1` (`evolving_loop.py:329`).

So the **round boundary is a batch of B episodes**, and the evolve step is
**one round lagged** behind the trajectories it consumes. M3 makes a round a batch of
B *attempts*, with B chosen to give 8-12 rounds inside 2000 calls.

### 2.3 Skill / Mistake schema

`skillrl/core/skill_bank.py` — plain dataclasses, `asdict`/`**` JSON round-trip
(`:57`, `:62`, `:75`, `:80`).

- `Skill` (`:11-24`): `skill_id`, `title`, `principle`, `when_to_apply`,
  `example=""`, `source_level=""`, `source_type=""`
  (`"success"`/`"failure"`/`"contrastive"`), `source_seeds: list[int] = []`,
  `generation: int = 0`, `confidence: float = 0.5`.
- `Mistake` (`:27-37`): `mistake_id`, `description`, `why_it_happens`,
  `how_to_avoid`, `source_level=""`, `source_seeds: list[int] = []`,
  `generation: int = 0`.
- `SkillBank` (`:40`): `general_skills` plus per-level `level_skills` /
  `level_mistakes` dicts. The general/level hierarchy maps to general/per-task
  unchanged; `source_level` carries the CORE task name (`"matchstick"`).

**`source_seeds` is annotated `list[int]` but never coerced**, and
`_compute_initial_confidence` matches by plain dict lookup
(`distill.py:177-188`), so **string problem ids work end to end** — verified. A
source id that does not appear in the batch falls back silently to the batch mean
(F1), which is why both prompt variants instruct the teacher to copy the
`--- Problem <id>` headers verbatim.

### 2.4 How confidence is ACTUALLY computed

See **F1**. Empirical for new skills (`distill.py:163-198`), teacher-self-reported
for retained ones (`evolving_distill.py:135-145`, prompt at
`evolution_prompts.py:76-82`), both clamped to `[0.1, 1.0]`, both consumed by the
same `key=lambda s: s.confidence` sorts in the retriever.

M1 replaces this with baseline-relative utility `U(q) = r - b_q`, `b_q` from
`n_base=10` no-memory samples per training problem. Without it, matchstick skills
(b ≈ 0.68) look strong and Hanoi skills (b ≈ 0.18) look weak purely from base
rates. The baseline rollouts are charged to the budget
(`RolloutBudget.charge_baseline`), outside `max_train`, mirroring CORE (1.6).

### 2.5 Retrieval and injection site

**Retrieval** — `skillrl/core/retriever.py:24-60`:
top-`max_specific` level skills by confidence, top-`max_general` general skills by
confidence, plus a `RELATED_LEVELS` top-up if fewer than
`MIN_SPECIFIC_SKILLS = 3` specific skills exist (`:21`, `:44-53`). It takes
`level_name` and **no query** — query-independent by construction (F3).
`RELATED_LEVELS` (`retriever.py:10-19`) is deliberately left unpatched: it is
`.get(level_name, [])` and no CORE task name is a key, so cross-task transfer is
already inert. Mistakes come separately from
`SkillBank.get_mistakes_for_level` (`augmented_runner.py:144-147`).

**Injection, in HeXA** — the block goes into the **system prompt**:
`augmented_system += "\n\n" + skills_block` (`skillrl/runner/augmented_runner.py:165`),
rendered by `SkillBank.format_skills_as_prompt` (`skill_bank.py:128-133`), with a
separate `DIAGNOSTIC_ADDENDUM` appended to the user turn
(`augmented_runner.py:172`, constant at `:40`).

**Injection, for CORE tasks** — must instead go through CORE's own guidelines slot,
`task.build_solve_prompt(problem, guidelines)`, exposed as
`Problem.build_prompt(guidelines)`. That is the only way the added-token metric stays
comparable to CORE's reported 0.92k, and it is also the only slot CORE's templates
have. `format_skills_as_prompt` is a *method*, which the `patch()` idiom cannot
reach, so its hardcoded `=== LEARNED PHYSICS SKILLS ===` heading is relabelled after
rendering and **before** token counting (relabelling changes the count) — at
`hexa_core/hexa_bridge.py:335`, the same fix
`discovery-agents/hexa_discovery/discovery_runner.py:96-100` already uses on its own
header.

Measured with the live heuristic counter: 6 realistic skills + 3 mistakes render to
**702 tokens**, so the default 1000-token budget fits roughly 6 skills + 1 mistake.
On a full bank the greedy walk **will** bind, and because skills are walked before
mistakes, the mistakes starve first.

### 2.6 Reward coupling

`compute_trajectory_reward` (`skillrl/distillation/teacher_prompts.py:197-223`) is
InterPhyre-shaped: success → `{1.0, 0.75, 0.5, 0.25}` by iteration count
(`≤3 / ≤7 / ≤15 / else`), failure → `-0.75` if it gave up under 10 iterations else
`-0.5`. On a single-turn CORE task every trajectory has `iterations == 1`, so every
success scores 1.0 and every failure `-0.75` — the shaping is degenerate but not
*neutral*, and `-0.75` biases `_compute_initial_confidence` low. **M2** replaces it
with `+1.0` / `-1.0`.

It is read in **two modules and seven call sites**, which is why `patch()` rebinds it
in both: `teacher_prompts.py:234` (inside `format_trajectory_for_teacher`) and
`:278` (the `format_trajectories_block` sort key) resolve it in `teacher_prompts`;
`distill.py:186`, `:193` (`_compute_initial_confidence`) and `:227`, `:228`, `:230`,
`:231` (trajectory ordering and log lines) resolve it in `distill`. The design's
"three places" was a summary, not a count.

**The patch is load-bearing, not cosmetic**, and the test asserts it so it stays
that way: unpatched, `format_trajectories_block` on our trajectory dicts prints
`Final action: x=m, y=o, r=v` — the first three characters of `"moves = ..."` —
because the stock formatter indexes `action[0..2]` for InterPhyre's `[x, y, r]`
(`teacher_prompts.py:267`). Silent garbage, not a crash.

Other InterPhyre couplings and their neutral replacements:
`LEVEL_DESCRIPTIONS` (`skillrl/core/config.py:36`) → CORE task descriptions
(`hexa_core/prompts_core_tasks.py`); "You are a physics teacher…" in every prompt
template → task-neutral wording; `load_trajectories` globbing
`trajectory_seed*.json` (`distill.py:135-147`) → our attempt records
(`hexa_core/trajectory.py::load_attempt_trajectories`, signature-compatible).

`call_teacher` (`distill.py:51-57`) dispatches on model name: `claude-*` → Claude CLI
subprocess, anything else → local HuggingFace. A gpt-oss-120b-on-vLLM evolver needs a
**third branch**, by patching or by passing an explicit callable. Not yet written.

### 2.7 Train/eval seed separation

**There is none, but the precise shape matters.** The brief says each round "runs
fresh seeds and reports accuracy on those same seeds"; read against
`evolving_loop.py:248-329`, the accurate statement is:

- Round k reports accuracy on seeds **never previously run**, solved with the bank
  built from round k-1 — so it is not literally train-on-test within a round.
- But there is **no fixed evaluation set at all**: every reported number comes from
  a different sample of seeds, so round-to-round differences mix learning with
  sampling noise, and no two rounds are comparable as measurements.
- Every evaluated seed is then absorbed into the next round's training data
  (Phase 1 of round k+1 consumes round k's trajectories), so nothing is ever held
  out permanently, and there is no final evaluation on a frozen set.

M4 fixes this by splitting train/eval with disjoint id sets and reporting on the
frozen CORE eval split only. The mechanism is
`hexa_core.adapter.assert_no_eval_contamination(tasks, attempted_ids)`, and hard
rule 4 ("never evaluate on a problem seen during training/evolution/admission")
has **no live enforcement point yet** — see Known issues.

---

## 3. Interpreters (measured on this machine, not assumed)

| interpreter | py | vllm | openai | tiktoken | pytest | torch |
|---|---|---|---|---|---|---|
| `/usr/bin/python` (system) | 3.12.3 | – | – | – | – | – |
| `~/.conda/envs/phyreagent` | 3.11 | – | 2.16.0 | – | – | 2.10.0 |
| `/work/pi_sniekum_umass_edu/$USER/.conda/envs/vllm-gptoss` | 3.12 | 0.26.0 | 2.53.0 | 0.13.0 | – | 2.11.0 |

Re-verified while writing this file: under the system interpreter,
`find_spec` returns None for all of `vllm`, `openai`, `tiktoken`; the `vllm-gptoss`
env reports `0.26.0 / 2.53.0 / 0.13.0`.

Consequences, all load-bearing:

- **Tests** run under the system interpreter via `python -m`, with no pytest, no
  network, no GPU, no third-party imports. `hexa_core.llm` lazy-imports `openai`
  inside the constructor for exactly this reason.
- **The gate driver** needs `openai`: run it from `phyreagent` or `vllm-gptoss`.
  From `phyreagent` the token counter is the heuristic; from `vllm-gptoss` it is
  exact. The manifest records which, via `tokens.is_exact()`.
- **The server** needs `vllm`: only `vllm-gptoss` has it, and that path is the
  serve script's default `CONDA_ENV`.
- The system interpreter is PEP-668 locked. **No dependency was added anywhere.**

---

## 4. What exists

```
hexa_core/
  __init__.py            PEP-562 lazy re-exports (cheap import: no sys.path, no CORE)
  core_bridge.py         locate CORE, sys.path, git SHA + dirty, dont_write_bytecode
  adapter.py             Problem, Tool, Task protocol, CoreTask, contamination guard
  tasks.py               TASK_SPECS + make_task() via CORE's own get_task
  budget.py              RolloutBudget: train_solve / baseline / admission / eval
  tokens.py              added-token metric, is_exact(), added_tokens_raw(), heuristic
  logging_schema.py      CORE-parity validation.jsonl + summary + RunManifest + MeanSEM
  trajectory.py          Attempt record, Attempt -> HeXA trajectory dict, batch helpers
  hexa_bridge.py         patch(): binary reward, neutral prompts, token-budgeted retrieval
  prompts_core_tasks.py  task-neutral distil/evolve prompts + CORE task descriptions
  llm/                   base.py (Generation parity rule), vllm_client.py, nim_client.py
  baseline_runner.py     Phase 2 no-learning eval, resumable, CORE-shaped logs
  tests/                 13 suites + shared harness, plain `python -m`
  RECON.md  DECISIONS.md
scripts/
  serve_gpt_oss_vllm.sh  sbatch vLLM server (4xA100 TP=4 default, H100=1 -> 2xH100 TP=2)
  run_repro_gate.py      Phase 2 driver
```

---

## 5. Known issues

Open, unfixed, and listed rather than omitted.

1. **`assert_no_eval_contamination` is never called from production code.**
   `hexa_core/adapter.py` defines it; the only call sites are
   `tests/test_adapter.py` and `tests/test_foundation.py`. Hard rule 4 therefore has
   no live enforcement point. Correct-by-construction for Phase 2 (which does no
   training), but Phase 3's training loop **must** call it with the accumulated
   attempted-id set before each evaluation or the guard is decoration. It could not
   be wired without a training loop, which does not exist.
2. **The ZebraLogic teacher-facing task description is wrong about 21% of the eval
   set.** `hexa_core/prompts_core_tasks.py` describes xlarge as running "from 5
   houses x 5 attributes up to 6 x 6", which excludes `6*4` — 21 of 100 eval and 1
   of 5 train problems (F4). This string fills the `{level_description}` slot in the
   distillation and evolution prompts, so it is what the teacher is told the task
   distribution is. **Not yet fixed.**
3. **No producer for `b_q`.** M1's seam is fields-only:
   `Attempt.baseline` / `Attempt.utility` and `RolloutBudget.charge_baseline` exist,
   but nothing runs the `n_base=10` no-memory passes per training problem.
   `run_baseline_eval` cannot be reused — `_prepare_problems` calls
   `task.eval_problems()` and hard-raises on any problem whose `split != "eval"`.
   Phase 3 must write a train-split rollout driver.
4. **The added-token metric has never run against a live prompt.** M7 produces a
   block and `Problem.build_prompt(guidelines)` accepts one, but no code joins them,
   and nothing has ever populated `Attempt.added_tokens` from a real injection —
   only from synthetic banks. The metric that makes the Figure-3 comparison possible
   is untested end to end.
5. **The token counter is the heuristic, so added-token figures are estimates.**
   `tiktoken` is absent from the system interpreter and from `phyreagent`, so
   `counter_name()` returns `heuristic:words+chars(approx,not-citable)` and
   `count_tokens` is `max(words*1.35, chars//4)`. Read
   `manifest["token_counter"]["citable"]` (wired to `tokens.is_exact()`) before
   quoting any added-token number against CORE's 0.92k. The heuristic is fine for
   its other job — enforcing an injection budget. Note `added_tokens` is not
   additive (the winning branch can differ between base and augmented), and it
   clamps at 0; `added_tokens_raw` exposes the unclamped delta so a genuinely
   shrinking augmentation is distinguishable from no injection at all.
6. **`probe()`'s reasoning-effort verdict rests on an unvalidated 1.5x threshold**,
   and `_probe_effort_size` silently prefers
   `usage["completion_tokens_details"]["reasoning_tokens"]` →
   `usage["reasoning_tokens"]` → `len(reasoning_text)` **without recording which
   source it used**. The raw low/high numbers *are* recorded, so the threshold is
   recalibratable from the first real run; the source field is not.
7. **The gate's startup probe takes ~138 s to fail when no server is listening**
   (measured, with `openai` installed): connection-refused is classed transient, so
   `VLLMClient`'s backoff runs all 6 attempts (5/10/20/40/60 s). **Deliberately not
   changed** — vLLM takes ~10 min to load gpt-oss-120b, so an operator who submits
   sbatch and then starts the gate wants the probe to wait through warmup. It prints
   per-attempt progress to stderr and ends with an actionable message and exit 2, no
   traceback.
8. ~~Seven stale `.pyc` files remain under `core-reasoning/src/**/__pycache__/`.~~
   **RESOLVED.** They were written by the original recon pass, before
   `install_core_on_path()` set `sys.dont_write_bytecode`
   (`hexa_core/core_bridge.py:111`). The three `__pycache__/` directories were
   removed; `find core-reasoning -name '*.pyc' | wc -l` is now `0` and
   `git -C ../core-reasoning status --porcelain -uall` is 0 lines at
   `999a96c527887cd36665eec419960c441529ab01`.

   The decision this issue asked for: deleting them is the *lesser* write. They were
   our own build artifacts, not CORE content, so leaving them is what actually
   leaves the dependency modified. Nothing new is written now, so this cannot recur.

   **The durable lesson survives the fix and is the reason this entry is kept rather
   than deleted:** `core-reasoning/.gitignore:1` is `__pycache__/`, so
   `git -C ../core-reasoning status --porcelain` reported a clean tree throughout the
   period when the repo *was* being written to. **A clean git status is not evidence
   that CORE is untouched.** Any future claim of non-modification must also check
   `find core-reasoning -name '*.pyc'`, as `tests/test_provenance_guards.py` does.
9. **M5, M6 and M8 are undefined anywhere.** The brief specifies M1 (baseline-
   relative utility), M2 (binary reward), M3 (round = batch of B), M4 (train/eval
   separation) and M7 (token-budgeted injection). `grep -rn 'M5|M6|M8' hexa_core/`
   returns nothing. Their absence cannot be scored as "skipped" or "narrowed"
   because there is no specification to compare against.
10. **No aggregate test runner.** 13 suites, each individually `python -m`-runnable,
    with no driver, so "all tests pass" is a claim only a pasted shell loop can
    check and a new suite is easy to leave out of it.
11. **Nothing in this package has ever talked to a live vLLM server.** No server has
    been launched; every LLM-client behaviour (streaming chunk shape, the
    `reasoning_content` delta field, `stream_options` acceptance, effort transport)
    is fake-verified only. `probe()` exists to catch a wrong assumption at startup
    rather than 3000 calls in.

---

## 6. Status / next steps

### Built and verified

- **Phase 0** — this file and `DECISIONS.md`.
- **Phase 1** — the task adapter, end to end: `core_bridge`, `adapter`, `tasks`,
  `budget`, `tokens`, `trajectory`, `logging_schema`, `prompts_core_tasks`,
  `hexa_bridge`, `llm/`, `baseline_runner`, and both scripts.
- **13 test suites pass** under the system interpreter. Re-run while writing this
  file: `foundation, adapter, splits, verifiers, budget, tokens, core_fidelity,
  provenance_guards, llm_clients, llm_parity, logging_schema, hexa_seam,
  baseline_runner` — 13 PASS, 0 FAIL, with `PYTHONDONTWRITEBYTECODE=1`, no network
  and no GPU.
- **Split determinism** asserted for all four tasks, `train ∩ eval == ∅`, eval size
  100. **Regime independence** asserted for the three that have it (digests in 1.2);
  MathGAP's regime *dependence* is asserted positively instead.
- **Prompt byte-parity** with `task.build_solve_prompt(problem, None)` over all 1660
  prompts (12 x 100 eval + 4 x (5+10+100) train).
- **Verifier round-trip** per task: a reference answer scores 1, a wrong one scores 0.
  Cross-problem negative control tightened when MathGAP was added: a scored-1 pair is
  only permitted when the two problems *provably share an answer*, which happens on
  MathGAP alone (12/300 pairs) because its answers are integers confined to 10..30.
  That is a **guess floor of ~4-6% under MathGAP accuracy** — a property of the
  benchmark, and the reason a small MathGAP delta means less than the same delta
  elsewhere.
- **CORE logging parity** asserted against `run_core.py`'s own functions, including
  byte-equality of the written JSONL line.
- **`hexa_bridge.patch()`** verified load-bearing (2.6) and reversible.

### NOT built

- **M1** — no `b_q` producer (Known issues 3). Fields only.
- **M2** — built and wired (the one M-seam that is complete).
- **M3** — `chunk_attempts` / `assign_rounds` / `round_summary` exist in
  `trajectory.py`, but no loop calls them. No round boundary is ever crossed.
- **M4** — the contamination guard exists but is never called (Known issues 1).
- **M5, M6, M8** — undefined (Known issues 9).
- **M7** — retrieval and budgeting built; **no injection call site** (Known issues 4).
- **The passive loop** — no training loop, no evolve loop, no vLLM branch in
  `call_teacher`. Nothing has ever run a HExA round on a CORE task.
- **Stage 3a** — not started.
- **Tools** — `Problem.tools` is always `None`; the `Tool` protocol has no
  implementations. Everything is passive mode (Table A).
- **Ablations A1-A8** — none. A8's design needs revising first (F3).
- **Phase 2 results** — the gate has never been run against a live server, so there
  is no measured accuracy for any task and no calibration offset.

### Ordered next actions

1. **Install vLLM into the serving env** — it is already there
   (`vllm 0.26.0` in `/work/pi_sniekum_umass_edu/$USER/.conda/envs/vllm-gptoss`,
   re-verified), so this step is *confirming* rather than installing. Confirm the
   env still resolves `vllm`, `openai`, `tiktoken` and that
   `scripts/serve_gpt_oss_vllm.sh`'s `CONDA_ENV` default points at it. Do **not**
   pip-install into `phyreagent`.
2. **Launch the server.** `sbatch scripts/serve_gpt_oss_vllm.sh` (4xA100, TP=4,
   bf16 dequant — the available path) or `H100=1 bash scripts/serve_gpt_oss_vllm.sh`
   (2xH100, TP=2, native MXFP4 — faster, scarce). Allow ~10 min to load. Confirm
   `VLLM_SYSTEM_START_DATE` is printed and lands in
   `logs/vllm_endpoint_<jobid>.json`, and note the node and port.
3. **Probe it.** Run the gate's startup probe (any gate invocation does this, or
   `VLLMClient.probe(strict=True)` directly). It must confirm the reasoning parser,
   that a garbage `reasoning_effort` is *rejected*, and a low-vs-high differential.
   Record `reasoning_effort_transport` and the raw low/high numbers — the first is
   what makes "we ran at high effort" auditable, the second is what lets the 1.5x
   threshold be recalibrated instead of re-guessed.
4. **Smoke gate.** 20 calls, one task, one run:
   `python scripts/run_repro_gate.py --smoke 20 --runs 1 --tasks tower-of-hanoi
   --base-url http://<node>:8000/v1 --endpoint-json logs/vllm_endpoint_latest.json`
   from `phyreagent` or `vllm-gptoss`. This is a wiring check, not a result — the
   manifest says so. Confirm non-zero `primary_finish_reason` counts, that
   `reasoning_content` is present in `raw_text`, and that nothing exits 3.
5. **Full gate.** `python scripts/run_repro_gate.py --runs 3 --base-url ...`
   — 3 tasks x 3 runs x 100 problems = 900 calls. Exit 0 = all `PASS*`, 1 = some
   `MISS*`, 2 = setup/probe failure (nothing measured), 3 = some task `ERROR`
   (a generation failed; not quotable either way). Results are cached and
   fingerprinted, so an interrupted gate resumes on an identical rerun.
   On a miss, work the fixed ladder the gate prints, in order and without skipping:
   reasoning effort not taking effect → `reasoning_content` missing from `raw_text`
   → prompt mismatch → truncation at 32768 (`finish_reason == "length"`, worst on
   Hanoi n=6, where optimal solutions reach 63 moves) → generator parameters.
6. **Record the calibration offset** per task, with the provider footnote, before
   any Phase 3 work. Then fix Known issues 1-4 in that order, since each is a
   prerequisite for a Phase 3 claim.
