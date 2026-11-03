# ACE vs HExA on InterPhyre (Qwen2.5-7B)

ACE (Agentic Context Engineering, Zhang et al., ICLR 2026) run as a baseline against
HExA on two InterPhyre levels: **down_to_earth (DTE)** and **two_body_problem (TBP)**.
Last updated 2026-10-11.

## TL;DR

| Level | HExA | ACE (paper config) | No-memory ReAct | ACE online simulator calls vs HExA |
|---|---|---|---|---|
| DTE | **83%** (78 / 80 / 90) | 63% (56 / 76 / 56) | 63% | 1.4–2× |
| TBP | **37%** (50 / 30 / 32) | 26% (28 / 16 / 34) | 12% | 3–4× |

These are held-out success rates on seeds 57–106, with 3 independently built memories per
method per level. HExA beats ACE on both levels while using less environment interaction.
ACE only beats the no-memory agent on TBP.

## Protocol (identical for both methods)

- **Model:** Qwen2.5-7B-Instruct for every role. In ACE that is the Generator, Reflector and
  Curator. In HExA it is the actor and the teacher. No stronger model is used anywhere.
- **Initial trajectories:** the same 5 trajectories seed both methods,
  `physics-reasoning-agents/Initial_trajectories/{DTE_Qwen7B,TwoBody_7B}`.
- **Online learning:** seeds 6–56, 17 rounds of 3 seeds. Memory updates after every seed
  (ACE) or every round (HExA).
- **Held-out eval:** seeds 57–106. The final memory is frozen, so no updates happen during
  the eval.
- **Actor:** `ReactAgent`, 25 iterations, temperature 0.3. The memory is appended to the
  system prompt.
- **Replicates:** n = 3, and each replicate is a separately built memory, not a re-run of the
  eval on the same memory.

## ACE implementation

The code is in `baselines/ace/`, and the runner is `baselines/runner/run_ace.py`. It follows
the reference repo `github.com/ace-agent/ace @ 82709de` in online mode without ground-truth
labels.

- **`playbook.py`:** the reference `playbook_utils.py`, copied verbatim. A test checks its
  bytecode matches the original.
- **`prompts.py`:** the reference Generator, Reflector and Curator prompts in their no-GT
  variants.
- **`dedup.py`:** the reference `BulletpointAnalyzer` (grow-and-refine), ported. It uses
  `all-mpnet-base-v2` embeddings, a cosine threshold of 0.9, and an LLM merge at
  temperature 0.3.
- **`loop.py`:** the Generator → Reflector → Curator loop. The Curator emits ADD-only
  deltas, which are merged deterministically. With `--resolve-rounds N`, a failed seed is
  reflected on and re-solved up to N times before curation. Re-solves are not scored, but
  their simulator calls are counted.
- **Role temperature:** 0, with greedy decoding. Before the 2026-10-10 fix to
  `load_qwen_model`, the roles actually ran at temperature 1e-5. The paper-config runs below
  were made after that fix.

**Paper config** is `--resolve-rounds 5 --dedup` (dedup threshold 0.9), which is the
paper's best setting. **Default config** is the reference repo's defaults: dedup off and no
re-solves.

Every run's `manifest.json` lists its deviations from the reference implementation:

- **No ground-truth answer.** InterPhyre has no reference answer, so the Reflector sees the
  simulator verdict and the observations instead.
- **Renamed example IDs.** The Reflector prompt's example bullet IDs were changed to
  `xmpl-0000N`. The reference's `calc-00001` matched a real bullet ID, and the model
  credited it on every seed.
- **No post-curation re-generation.** The reference loop re-runs the Generator after each
  curation only to log training accuracy, so it is skipped here.

## Results: DTE

**Held-out (57–106):**

| Method | Rep 1 | Rep 2 | Rep 3 | Mean |
|---|---|---|---|---|
| HExA | 78% | 80% | 90% | **83%** |
| ACE, paper config | 56% | 76% | 56% | **63%** |
| No-memory ReAct | – | – | – | 63% |
| ACE, default config, seeded | 70% | 4% | 34% | 36% |
| ACE, default config, no initial trajectories | 36% | 32% | 32% | 33% |

**Online learning (6–56) and cost:**

| | Online accuracy | Online simulator calls | Held-out simulator calls per seed | Final memory size |
|---|---|---|---|---|
| HExA | 73 / 61 / 57% | 583–624 | 8.3–10.3 | ≤10 skills + 5 mistakes; ~0.6k tokens injected |
| ACE, paper config | 67 / 65 / 71% | 878 / 951 / 1239 (325 / 384 / 746 are re-solves) | 9.3–14.2 | 27–32 bullets, ~2.2–2.3k tokens |

## Results: TBP

**Held-out (57–106):**

| Method | Rep 1 | Rep 2 | Rep 3 | Mean |
|---|---|---|---|---|
| HExA | 50% | 30% | 32% | **37%** |
| ACE, paper config | 28% | 16% | 34% | **26%** |
| ACE, default config, seeded | 18% | 12% | 4% | 11% |
| ACE, default config, no initial trajectories | 6% | 26% | 10% | 14% |
| No-memory ReAct | – | – | – | 12% |

**Online learning (6–56) and cost:**

| | Online accuracy | Online simulator calls | Held-out simulator calls per seed | Final memory size |
|---|---|---|---|---|
| HExA | 43 / 27 / 35% | 784 / 937 / 876 | 13.8–18.7 | ≤10 skills + 5 mistakes |
| ACE, paper config | 41 / 16 / 22% | 2700 / 3562 / 2698 (2018 / 2829 / 1866 are re-solves) | 14.4–16.3 | 28–36 bullets, ~2.8–3.2k tokens |

Both methods' simulator calls are counted with `baselines.memory.trajectory_fmt.count_simulator_calls`.

## Results: Qwen2.5-3B (paper config, 2026-10-12)

Same protocol as 7B. The initial trajectories are the ones HExA's 3B builds used
(`DTE_Qwen3B`, identical to `DTE_Qwen7B`, and `TwoBody_7B`; no 3B TBP set exists).
HExA and ReAct numbers are from `ICLR/tables/interphyre_qwen.tex` (± population SD, n = 3).

| Level | HExA | ACE, paper config | No-memory ReAct |
|---|---|---|---|
| DTE held-out | **23.3 ± 0.9** | 2.7 ± 2.5 (6 / 0 / 2) | 8.3 ± 0.9 |
| TBP held-out | **12.0 ± 0.0** | 1.3 ± 0.9 (0 / 2 / 2) | 7.3 ± 0.9 |

| | Online accuracy (6–56) | Online simulator calls (re-solves) | Final playbook |
|---|---|---|---|
| ACE DTE | 2 / 8 / 8% | 4954 / 5143 / 4586 (4146 / 4077 / 3703) | 13 / 11 / 21 bullets, 2.0–3.7k tokens |
| ACE TBP | 2 / 4 / 6% | 4724 / 4659 / 4431 (3820 / 3768 / 3532) | 10 / 13 / 7 bullets, 1.5–2.2k tokens |

- At 3B, ACE is below no-memory ReAct on both levels. The 3B actor solves almost
  nothing during training, so nearly every seed runs all 5 re-solves; the Reflector
  has few successes to learn from and frequently returns invalid JSON or credits
  bullets the actor never cited.
- Held-out episodes mostly run to the 25-iteration limit (42–48 of 50 seeds), as
  ReAct 3B's do; there are no harness errors behind the low scores.
- Harness note: DTE run 1 hung at seed 24 on `simulate_partial(stop_step=1e9)`.
  The tool now clamps `stop_step` to the level's 2000-step limit (as
  `simulate_with_trace` already did); run 1 resumed from seed 24 with the cap.
- Directories: `baselines/data/results/ace_{down_to_earth,two_body_problem}_qwen3b_s6-56_warmup_paperconfig{,_run2,_run3}`
  and `..._s57-106_frozen_paperconfig{,_run2,_run3}`.

## Results: catapult, Claude Sonnet 4.6 (2026-10-13, 1 run)

Claude Sonnet 4.6 for every role via `claude -p` (the same actor loader as HExA's
catapult Claude runs), seeded with the 5 trajectories in `Initial_trajectories/catapult`,
online on 6–56, frozen eval on 57–106.

| Method | Held-out 57–106 | Online 6–56 |
|---|---|---|
| HExA (3 runs) | **86 / 92 / 70% (83%)** | 40 / 45 / 33 of 51 |
| RAG | 59% | |
| **ACE (1 run, mixed config)** | **54%** (27 / 50) | 14 / 51 (27%) |
| MemRL | 50 / 48% | |
| ReAct | 8 / 14% | |

- **Not paper config.** Re-solve rounds were cut during training at the user's request,
  to bound Claude usage and wall time: seeds 6–13 used 5 rounds (2/8 solved), 14–31 used
  2 rounds (4/18), 32–56 used 0 rounds (8/25). Dedup (0.9) stayed on throughout. The
  held-out eval is one attempt per seed, as for every method.
- **Single run, n = 1.** No ±.
- **Playbook:** 151 bullets, ~24.7k tokens at the end. Dedup ran after every curation and
  never merged a bullet on this level, so every actor prompt grew all run (prompts reached
  ~80k characters). At the end the reference analyzer counts 11 high-performing, 19
  problematic and 93 never-cited bullets.
- **Interruptions:** the job stopped on Claude session limits six times and once on an
  unexplained crash (log lost). Each resume redraws the seed that was in progress.
  Seed 9's scored attempt was redrawn after the crash; both draws failed. The log of switches
  is in `RESOLVE_SWITCH.txt`, and the 5- and 2-round manifests are saved beside `manifest.json`.
- Directories: `baselines/data/results/ace_catapult_claude_s6-56_warmup_paperconfig/` and
  `ace_catapult_claude_s57-106_frozen_paperconfig/`.

## Findings

1. **HExA beats ACE on both levels.** The gap is +20 points on DTE and +11 on TBP, even
   though ACE used 1.4–4× more simulator calls while learning. The extra calls come from
   ACE's reflect-and-re-solve rounds, and they cost more on TBP because more seeds fail
   there.
2. **ACE needs the paper config at 7B.** With the default config, the add-only playbook
   grows without bound, and early bullets get locked in. The most-cited bullets account for
   about 40% of all citations. The model also never acts on its own "harmful" counts.
   Default-config ACE is at or below ReAct on both levels, and its replicates swing widely
   (4–70% on DTE).
3. **Paper-config ACE stabilizes but stays behind.** It matches ReAct on DTE (63%) and
   beats it on TBP (26% vs 12%), but it trails HExA on both levels.
4. **ACE varies more across builds.** On TBP it ranges from 16% to 34%. Its best build beat
   two of HExA's builds, and its worst is close to ReAct.
5. **Online vs held-out.** ACE's held-out accuracy fell below its online accuracy in most
   runs: 2 of 3 on DTE paper config, all 3 on TBP default config, and 1 of 3 on TBP paper
   config. HExA rose on every DTE build.
   - Because online accuracy includes the early, nearly empty memory, this suggests ACE's
     final playbook is worse than its average playbook during learning.
   - This is not conclusive. The seed sets may differ in difficulty, no ReAct run on 6–56
     exists to calibrate against, and HExA also dipped slightly on 2 of its 3 TBP builds.

## Caveats

- **Initial trajectories are from an older level generator.** In DTE the green ball starts
  at y = 4.5 rather than today's 4.0, and the TBP files' "seed 76–84" labels are wrong. They
  can't be regenerated, but they match no seed in 1–300, so they don't leak held-out
  instances. Both methods used the same files.
- **HExA replicate independence.** vLLM 0.26 seeds its sampler with 0 by default, and HExA
  sends requests one at a time. So two builds with the same code on the same GPU type come
  out byte-identical.
  - TBP builds 2 and 3 were rerun with explicit vLLM seeds 0 and 3. The first build-3 run
    was a duplicate of build 2 and is kept as
    `two_body_qwen7b_build3_DUPLICATE_of_build2_seed0`.
  - TBP build 1 and the DTE builds differ from each other only because they ran on
    different GPU types or code versions.
  - ACE samples with transformers and sets no seed, so its replicates are independent.
- **TBP build 1 predates one code change.** Build 1 (2026-09-12, L40S) ran the same
  training and eval commands and the same evolver, including the feasibility tool. The only
  later change is the 2026-10-02 contact-normal fix in `predict_first_contact`, which should
  be negligible for TBP's ball-on-ball contacts.
- **Not budget-matched.** ACE got more environment interaction than HExA. That favours
  ACE, so it only strengthens HExA's result, but it should be reported.

## Where the results are

- **ACE, DTE, paper config:**
  - `baselines/data/results/ace_dte_qwen7_run{1,2,3}/`, with each frozen eval nested
    inside.
  - The default-config and no-initial-trajectory DTE runs, and the unseeded TBP training
    runs, were moved out of `baselines/data/results/` on 2026-10-10.
- **ACE, TBP, paper config:** `baselines/data/results/ace_tbp_qwen7_run{1,2,3}/`, with each frozen eval nested
  inside. The default-config seeded TBP runs (`..._hexainit5*`) are no longer in that folder.
- **HExA:** `physics-reasoning-agents/skillrl/RESULTS/results_qwen7b/HeXA/`
  - DTE: `down_to_earth_qwen7b/run{1,2,3}`
  - TBP: `two_body_qwen7b` (eval in `Ablations/final_eval_REF`) and
    `two_body_qwen7b_build{2,3}` (eval in `final_eval`)
- **No-memory ReAct:** `physics-reasoning-agents/skillrl/RESULTS/results_qwen7b/ReAct_heldout/`

Each ACE run directory contains:

- `manifest.json`: config and deviations
- `episodes.jsonl`: one record per seed
- `ace_calls.jsonl`: every Reflector, Curator and dedup call
- `ace_state.json`: counters
- `playbooks/`: playbook snapshots
- `code_rev.txt`, `code_diff.patch`, `code_sha1.txt`: the exact code used

## Reproducing

Submit from the repo root.

ACE, paper config, training followed by the frozen held-out eval:

```bash
sbatch -t 16:00:00 baselines/bash_scripts/sbatch_ace_qwen.sh --level two_body_problem \
    --init-dir /home/udhanuka_umass_edu/Scalar_Research/physics-reasoning-agents/Initial_trajectories/TwoBody_7B \
    --extra "--resolve-rounds 5 --dedup" --tag paperconfig
sbatch --dependency=afterok:<jobid> baselines/bash_scripts/sbatch_ace_qwen.sh --level two_body_problem \
    --seeds 57-106 --frozen-from baselines/data/results/ace_two_body_problem_qwen7b_s6-56_warmup_paperconfig \
    --tag paperconfig
```

HExA TBP builds. Run from `physics-reasoning-agents/`, and give each build its own vLLM seed:

```bash
BUILDS=2 VLLM_SEED=0 PORT=8002 sbatch skillrl/run_tbp_builds.sh
BUILDS=3 VLLM_SEED=3 PORT=8003 sbatch skillrl/run_tbp_builds.sh
```

Offline logic tests for ACE (49 checks):

```bash
timeout 300 /project/pi_sniekum_umass_edu/vgandhi/conda/envs/VerlToolInterphyre/bin/python baselines/tests/test_ace_logic.py
```
