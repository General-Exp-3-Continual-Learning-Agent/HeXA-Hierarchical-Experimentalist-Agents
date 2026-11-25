# Ablations

Every ablation keeps the whole pipeline fixed and changes exactly one thing, so a
drop in solve rate is attributable to that one component. Only the reward
ablation has its own wrapper; everything else is a flag on a standard entry point
or a small module. This file gives, for each ablation, what it removes, what it
is meant to show, the reported result, and how to run it.

The skill bank has three orthogonal axes of instantiation — how it is
initialized before experimentation, how it is updated between rounds, and
whether interaction reward labels are attached to skills. Sections 1 and 2 below
ablate those axes; sections 3-5 ablate what the bank contains and how it is
built.

Two reference points the drops are read against, both on `catapult` with Claude
as actor and teacher: within-level HeXA scores **82.7 ± 9.3%** held-out, and
cross-level joint synthesis scores **42.7 ± 1.9%**.

---

## 1. Reward labels on the skill bank

**Run with:** `scripts/run_hexa_contrastive.py`

The contrastive-only variant removes two things from the standard loop: the
per-trajectory reward score that ranks trajectories before distillation, and the
per-skill confidence calibrated from those rewards. What is left is pure
contrast — the teacher sees successes and failures and extracts skills and
mistakes by comparing them, with no environment feedback and no confidence for
the retriever to rank by.

**Purpose.** Separate the contrastive scaffolding from the reward signal. If the
variant lands close to HeXA, success/failure pairing is doing the work and the
reward machinery is bookkeeping. If it lands far below, rewards and confidence
are load-bearing.

**Result** (solve rate, both arms scored on the same training seeds, n=3 banks
per arm):

| Actor / level | No reward | Reward |
|---|---|---|
| Claude, `catapult` | 51.0 ± 10.1 | **78.0 ± 10.7** |
| Qwen-2.5-7B, `down_to_earth` | 64.0 ± 1.7 | **72.0 ± 5.2** |
| Qwen-2.5-7B, `two_body_problem` | 20.0 ± 2.7 | **42.0 ± 3.3** |

Rewards help on every level and every actor, and the no-reward arm also needs
more actor turns per seed. Note these are training-seed rates, not held-out: no
contrastive bank was evaluated held-out, so training is the only matched basis.

```bash
python scripts/run_hexa_contrastive.py --level catapult \
    --initial-traj-dir Initial_trajectories/catapult

python scripts/run_hexa_contrastive.py --level down_to_earth \
    --initial-traj-dir Initial_trajectories/down_to_earth \
    --teacher-model "Qwen/Qwen2.5-7B-Instruct"
```

Running the teacher on Qwen requires a local OpenAI-compatible server (vLLM);
see INSTALL.md.

Per-round banks are written as `skill_bank_contrastive_only_{round}.json` with a
`progress_contrastive_only.json` summary. The schema matches standard HeXA
output minus the `confidence` fields.

---

## 2. Initialization and update rule

**Run with:** one wrapper per configuration, listed below.

*Initialization* is how the bank starts: **Off2On** warm-starts from skill-free
base-actor trajectories and keeps evolving from online interaction, **Online**
starts from an empty bank and bootstraps entirely from the agent's own
interaction, and **Offline** distils once from the warm-start batch and then
freezes. *Update* is what the evolver is given each round: **Evolving** receives
both the accumulated bank and the newest trajectory batch, while **Iterative**
rebuilds the bank from the newest batch alone.

**Purpose.** Separate the value of warm-start experience from the value of
accumulating across rounds. Offline isolates the warm start with no online
learning at all; Online isolates online learning with no warm start; Iterative
keeps both but discards the accumulated bank each round, so it tests whether
carrying prior skills forward matters or whether the latest rollouts suffice.

**Result** (Claude Sonnet, solve rate % and mean turns per seed under the
round-by-round protocol):

| Configuration | Catapult | Iters. | Pass the Parcel | Iters. |
|---|---|---|---|---|
| **Off2On Evolving** (= HeXA) | **84.0** | **11.8** | **68.0** | **12.7** |
| Off2On Iterative | 56.0 | 15.7 | 48.0 | 17.7 |
| Online Evolving | 44.0 | 18.2 | 58.0 | 16.4 |
| Online Iterative | 36.0 | 20.4 | 56.0 | 18.9 |
| Offline Static | 60.0 | 17.0 | 48.0 | 18.5 |

Off2On + Evolution wins on both levels on both metrics, and this is the
configuration written simply as HeXA everywhere else. Both halves contribute:
dropping the warm start (Online Evolving) and dropping accumulation (Off2On
Iterative) each cost heavily, and dropping both is worst.

```bash
python scripts/run_hexa.py                   --level catapult   # Off2On Evolving
python scripts/run_hexa_off2on_iterative.py  --level catapult   # Off2On Iterative
python scripts/run_hexa_online_evolving.py   --level catapult   # Online Evolving
python scripts/run_hexa_online_iterative.py  --level catapult   # Online Iterative
python scripts/run_hexa_offline.py           --level catapult   # Offline Static
```

Each takes `--initial-traj-dir` for the warm start; the Online wrappers ignore it
and start from an empty bank.

---

## 3. Within-level: how skills are built and revised

All four arms run on `catapult` against the 82.7 ± 9.3% reference, and all are
flags on `scripts/run_hexa.py` except the rollback arm, which rewrites a finished
bank.

| Arm | What changes | Success (%) | Drop |
|---|---|---|---|
| **Full HeXA** | — | **82.7 ± 9.3** | — |
| Skill-free collection | Bank hidden from the actor while collecting training experience; used normally at evaluation | 63.3 ± 3.3 | 19.4 |
| Earliest skill versions | Each final skill's text reverted to its earliest version; same entries, order and confidences | 56.7 ± 16.5 | 26.0 |
| Append-only bank updates | New skills appended; existing entries never edited, merged or evolved | 48.0 ± 12.8 | 34.7 |
| Summary bank | One accumulating free-text summary of the same experience, at matched context length | 62.7 ± 0.9 | 20.0 |

**Skill-free collection** asks whether skills improve the *experience* gathered
for later learning, not just the final prompt. The evolver still sees the
previous bank and the new trajectories, but the updated bank never guides the
next round's interactions. The drop says skill-guided interaction yields more
useful experience to distil from.

**Earliest skill versions** isolates the content of the revisions. Membership,
identifiers, order and confidences are preserved and only the actor-visible text
is rolled back, so what remains is whether later wording of a skill is better
than its first.

**Append-only updates** remove revision and merging entirely. This is the largest
drop of the four, and the arm also burns the most actor turns per seed — an
unrevised bank accumulates entries that do not get sharpened.

**Summary bank** removes structure rather than revision: no separate skills and
mistakes, no confidence, no retrieval, just a rolling free-text summary at
matched length. It tests whether the structured bank beats simply remembering
the same experience in prose.

```bash
# skill-free collection
python scripts/run_hexa.py --level catapult \
    --initial-traj-dir Initial_trajectories/catapult \
    --condition-collection-on-bank false

# append-only / summary bank
python scripts/run_hexa.py --level catapult \
    --initial-traj-dir Initial_trajectories/catapult \
    --update-strategy append_only
python scripts/run_hexa.py --level catapult \
    --initial-traj-dir Initial_trajectories/catapult \
    --update-strategy summary --summary-max-words 1000

# earliest skill versions: rewrite a finished bank, then evaluate it frozen
python -m skillrl.distillation.rollback_skill_versions \
    --run-dir <finished-run-dir> --level catapult --all --out-dir <out>
```

`--update-strategy` also accepts `single_parent`, which permits per-entry
revision but forbids merging. It is implemented and runnable but not reported in
the paper.

Each round writes an update log beside the bank recording candidates, dedup
verdicts, revisions with their parent entry, and evictions.

---

## 4. Cross-level: whether transfer is really hierarchical

All three arms synthesise a `catapult` bank and are evaluated frozen on held-out
seeds against the 42.7 ± 1.9% joint-synthesis reference.

| Arm | What changes | Success (%) | Drop |
|---|---|---|---|
| **Full HeXA** | Joint synthesis from three source banks | **42.7 ± 1.9** | — |
| Per-level transfer | Each source bank adapted to the target separately, then concatenated; no cross-level merging | 25.3 ± 10.9 | 17.4 |
| Description-only synthesis | Bank written from the level description alone; no source banks | 18.0 ± 2.0 | 24.7 |
| Invocation-based hierarchical skills | Source skills injected verbatim at a matched budget; no synthesis step | 34.7 ± 6.4 | 8.0 |

**Per-level transfer** is the direct test of merging. Both arms see the same
sources and the same target description under the same budgets, and both adapt
the sources to the target; only the combining differs. The gap is what joint
integration buys over adapting each source in isolation and presenting the
results side by side.

**Description-only synthesis** is the floor: it shows how much of the bank's
value comes from the source skills at all, rather than from the teacher writing
plausible advice off the level description. It is the largest drop of the three,
larger than removing joint synthesis.

**Invocation-based hierarchical skills** tests implicit hierarchy — the actor
invoking source skills as sub-skills without any new skill being written. Source
entries go into the prompt as written, labelled with their origin level, at the
same injected budget. It loses the least, which is the point: implicit
invocation recovers much of the benefit, and explicit synthesis adds the rest.

```bash
# per-level transfer vs. joint, under matched budgets
python -m skillrl.distillation.ablation_joint_vs_independent \
    --target catapult --condition both \
    --source-bank down_to_earth=<bank.json> \
    --source-bank two_body_problem=<bank.json> \
    --source-bank pass_the_parcel=<bank.json> \
    --out-dir <out>

# description-only
python -m skillrl.distillation.cross_level_synthesis \
    --target catapult --description-only --output-dir <out>

# invocation-based: build the raw bank, evaluate with the same provenance header
python -m skillrl.distillation.build_raw_source_bank \
    --target catapult --budget matched --output <bank.json>
python -m skillrl.runner.run_skill_agent \
    --level catapult --skill-bank <bank.json> --xl-framing --no-evolver
```

`scripts/run_cross_level.py` is a convenience wrapper for the unablated pipeline
and does not forward these flags — run the synthesis module directly, then point
the wrapper at the resulting bank with `--skip-synthesis`.

Two further synthesis flags exist for controlling what the teacher may use:
`--prompt-variant` chooses the synthesis template (`full-neutral` is the full
structure with every worked example stripped, for hint-free numbers;
`text-game` is the domain-neutral version for non-physics tasks), and
`--transfer-guard` forbids carrying source-calibrated numbers into the target
bank and validates the output.

---

## 5. Which parts of the bank matter at injection time

These arms do not retrain. Within each block every arm retrieves from a single
frozen bank, so no difference can come from a different training run. Qwen-2.5-7B
throughout, held-out solve rate, n=3 evaluations.

| Injected bank | Two Body Problem | Down to Earth |
|---|---|---|
| Full bank | **50.0 ± 3.3** | **82.7 ± 5.2** |
| Skills only | 46.0 ± 5.9 | 71.4 ± 6.0 |
| Mistakes only | 17.3 ± 4.1 | 74.8 ± 5.4 |
| Wrong-level skills, own mistakes | 9.3 ± 1.9 | 27.9 ± 6.7 |
| ReAct (empty bank) | 12.0 ± 0.0 | 63.3 ± 3.4 |

The wrong-level arm is the one that matters: it swaps in fluent, well-formed
skills distilled from a *different* level while keeping this level's own
mistakes, at comparable injected length. It lands at or below the empty-bank
baseline, so the gain comes from what was distilled, not from having competent
text in the prompt.

A second block varies the confidence ranking on a frozen bank. Dropping the two
lowest-confidence skills costs nothing (76.7 → 76.7) while dropping the two
highest costs 11.4 points, at injected lengths within 1% of each other; shuffling
the confidences while keeping every skill costs 6.0. The ranking carries real
information where confidences are spread; on Two Body they are nearly flat and
the same comparison is a pre-registered negative check.

The skills-only arm has a wrapper, `scripts/run_hexa_skills_only.py`, which
trains a bank from successes alone. The remaining arms here are surgery on an
already-frozen bank rather than separate training runs.
