# DECISIONS.md — HExA Design B on ORBIT

Resolves the six open items in `RECON.md` §9. Written before any Phase 1 adapter code.
Supersedes the corresponding assumptions in `hexa_design_b_orbitplan.md`.

---

## D1. Mastermind's 120-code space (RECON §0c)

**Decision: drop Mastermind as the Design-B pilot. Maze only, for now.**

At the eval config (`code_length=3, num_numbers=6, duplicate_numbers=False`) there are
only P(6,3) = 120 possible codes. RECON measured 96.5% of eval instances (247/256) share
a hidden code with an instance in the train pool, despite 0 seed overlap. A skill bank can
memorize codes rather than strategy and would score as "cross-instance transfer" on a
task where that claim cannot actually be tested at this config.

Rejected mitigations and why:

- *Widen `num_numbers`* — breaks Table 2 comparability. RECON §2 established the
  eval-pool seed hash covers the entire task config, so changing `num_numbers` produces a
  different 256-instance eval set than ORBIT's published one. This removes the reason
  Mastermind was attractive (a reference line against Table 2) without removing the need
  to re-validate a new eval pool from scratch.
- *Redact revealed codes from teacher-visible transcripts* — treats a symptom, not the
  cause. Even without the literal `"Congratulations! ... code [4,1,3]"` string, an
  evolver can correlate partial-guess feedback patterns with previously-seen codes across
  96.5% of the eval pool. This trades an obvious leak for a harder-to-detect one, which is
  worse for a paper, not better.
- *Audit the bank for literal digit triples* — a detection method, not a prevention
  method. Catches verbatim memorization only, not weighted preference toward the ~105
  codes the bank has effectively seen.

Maze's leakage (19.1% of eval instances share a layout with the train pool; see RECON §10) is structurally different and tolerable: a
shared *map* lets a bank at most memorize one route, which is a normal thing to control
for (see D6), not a categorical hole in the instance space the way 120 codes is for a
disjoint-hidden-state claim.

**Action:** Maze becomes the sole Design-B pilot. Do not build the Mastermind adapter.
If a second task is wanted later, pick one of ORBIT's five *training* tasks (e.g.
Minesweeper) as an internal harness sanity check — not for a Table 2 comparison, and not
as a Design-B transfer claim, since those tasks were in ORBIT's own meta-training
distribution.

---

## D2. `T` is not fixed at 3 (RECON §0d)

**Decision: enforce a hard 3-episode cap in our own loop. Do not accept variable `T`.**

`MultiEpisodeEnv` ends on `total_steps >= total_step_cap`, with no episode counter cap —
an instance that resolves in one guess/move gets many more than 3 episodes, and
`success_count` can exceed the nominal 0–3 range. Accepting this and normalizing the
reward would make "Episode 1 success rate" and "Episode 3 success rate" refer to
different underlying conditions across instances (an easy instance's "episode 1" and a
hard instance's "episode 1" would no longer be comparable slices), which breaks the one
metric this whole design is built around: the Episode-1 delta between no-bank and HExA.

**Action:** wrap `MultiEpisodeEnv` (or reimplement the outer loop) so it terminates after
exactly 3 completed episodes regardless of remaining step budget, independent of
`total_step_cap`. Document this as a deliberate deviation from ORBIT's exact step-cap
semantics in the eventual writeup: *"we cap at 3 episodes per instance regardless of
step count, to keep the episode-indexed metric well-defined."*

---

## D3. Maze `total_step_cap`: moot under D2 — the real finding is that ORBIT ran T=5

**Decision: `total_step_cap` is immaterial; set it to 27 and treat D2's hard 3-episode
cap as the binding constraint. Accept that ORBIT's Table 2 Maze figure is a 5-episode
number cited beside our 3-episode curve.**

The original framing of this item conflated two independent knobs, and the correction
changes what the decision is about.

**`total_step_cap` is not the per-episode budget.** `max_turns_per_episode: 9` reaches
`MazeEnvAdapter.max_turns`, and the inner env terminates each episode itself at
`current_turn >= self.max_turns` (`envs/maze_env_adapter.py:493`). `total_step_cap` bounds
only the *outer* trajectory (`envs/multi_episode_env.py:219`). So each episode is 9 turns
whatever the cap is — 45 does **not** buy 15 steps/episode. With D2's hard T=3, an
instance consumes at most 3 × 9 = 27 steps, so a cap of 45 never binds and the 45-vs-27
choice has no behavioural effect. The plan's `H=9, T=3 → 27 steps/instance` assumption was
correct as written; do **not** rewrite it to 45.

The binding budget is `max_turns=9` against resolved shortest paths of 7–8 (RECON §1) —
1–2 moves of slack per episode. That, not the step cap, is what makes Maze hard.

**What the 45 actually reveals.** `minimum_episodes = total_step_cap // max_turns`
(`envs/multi_episode_env.py:370`), so `eval_config.yaml`'s 45 // 9 = **5**: `get_metrics`
emits `episode_1..episode_5`. Since `eval_config.yaml` is what `scripts/eval/openai.sh`
defaults to, **ORBIT's published Maze protocol is T=5, not T=3.** This was not visible in
RECON §2, which recorded the 45-vs-27 discrepancy without deriving the episode count from
it.

That collides with D2. Two coherent resolutions:

- *Match ORBIT (T=5, cap 45)* — like-for-like reference lines, at ~67% more steps per
  instance, and it gives up D2's clean 0–3 reward binning.
- *Keep D2's T=3* — **chosen.** D2's argument about comparable episode slices holds, the
  Episode-1 delta is the actual claim and is unaffected by the choice (episode 1 is
  episode 1 under either protocol), and it is cheaper.

**Action:** set `total_step_cap=27` for all Maze runs, with D2's hard 3-episode cap as the
real terminator. In the writeup, label the Table 2 Maze reference line explicitly as a
**5-episode** figure — this is a second reason, on top of the model-identity confound
already noted in the plan's Phase 2, that it is a cited reference and not a direct diff.
Any "final success rate" or episode-3 framing must carry that caveat; the Episode-1
comparison does not need it.

---

## D4. Environment install: shim, don't install `rllm[verl]`

**Decision: write a ~40-line shim providing `Action` (dataclass) and `BaseEnv` (ABC),
matching the `mini/` stub pattern already built in scratch for GEM. Do not run
`git submodule update --init --recursive` on ORBIT's `rllm` submodule.**

`MultiEpisodeEnv` and the Maze adapter only import two objects from rLLM. A full
`-e .[verl]` install pulls verl/vLLM/torch for no benefit over a trivial shim, and adds a
large, fragile dependency surface to a component (`rllm`, on ORBIT's own
`fix/token-in-token-out` branch) that Design B doesn't otherwise touch.

**Action:** extract `Action` and `BaseEnv`'s interface into a local shim module,
import `MazeGenerator` and `MazeEnvAdapter` against it, verify with the same
determinism checks RECON already ran (`reset(seed=N)` reproducibility).

---

## D5. Parallelism: serialize instance resets

**Decision: run one instance at a time. Do not parallelize across instances for the
pilot.**

Seeding is global (`random.seed`, `np.random.seed`), not per-instance. ORBIT's own
`--n-parallel 256` eval is therefore not independently reproducible at the instance
level under concurrency — a finding worth flagging in the writeup regardless of what we
do, since it means their own published eval runs have this same property.

**Action:** serial execution for all Phase 2/3 runs. Revisit parallelization later via
`np.random.default_rng` isolation per instance or one-instance-per-subprocess, only
after the pilot's numbers are trusted — correctness first, speed later.

---

## D6. Maze instance identity: resolved seed / maze bytes

**Decision: key all disjointness, dedup, and leakage checks on the resolved seed (post
`seed += 1` walk) or a hash of the maze layout itself — never the dataset seed passed
in.**

RECON found 181/256 eval dataset seeds resolve to a different effective seed via
`MazeEnvAdapter`'s shortest-path filter, and 256 dataset seeds collapse to 243 distinct
mazes. Using dataset seed as instance identity would silently miscount both
train/eval disjointness and within-pool duplicates.

**Action:** every instance the adapter constructs carries `resolved_seed` and a
`maze_hash` (e.g. SHA-256 of the wall/path/goal grid) as its identity, computed once at
construction and used everywhere the plan's adapter tests check disjointness.

---

## D7. Actor and evolver: gpt-oss-120b via vLLM, both

**Decision: actor and evolver are the same model, `openai/gpt-oss-120b`, served by
`scripts/serve_gpt_oss_vllm.sh` and reached through `hexa_core.llm.VLLMClient`. No Claude
in the recorded runs.**

This is the same actor/evolver configuration `hexa_core` already uses, so the ORBIT arm is
comparable to the CORE arm without a model-identity confound *between our own arms* — a
separate concern from the cross-model confound against ORBIT's Table 2, which stands
regardless (RECON §2, D3).

Rejected: the Claude CLI shim (`react_agent/run_react_claude.py:39`). It needs no API key,
but its own `--model` flag is documented as "purely for logging since Claude Code chooses
the model automatically" (:238-239) — the model is not pinned, so a run is not
reproducible. Acceptable for a throwaway smoke test, not for a recorded number.

Mechanics already in place, to be reused rather than rebuilt:
- `VLLMClient.generate_structured(messages=[...])` (hexa_core/llm/vllm_client.py:394)
  accepts a full message list, which is what a multi-turn cumulative-chat actor needs.
- Endpoint resolution: `HEXA_VLLM_URL` → `VLLM_BASE_URL` → `http://localhost:8000/v1`
  (:142). Served name `openai/gpt-oss-120b`, port 8000 (serve script :105-106).
- Evolver routing is automatic: `resolve_teacher_model` sends any bare model name
  containing `gpt-oss` to vLLM when a base URL is set (`hexa_core/llm/teacher.py:98-117`,
  `VLLM_MODEL_HINTS = ("gpt-oss",)`), so `--teacher-model openai/gpt-oss-120b` needs no
  extra plumbing.
- Sampling defaults are CORE's paper config (0.6 / 0.95 / 32768 / effort `high`,
  `hexa_core/llm/base.py:35-38`). **Open**: whether to keep those or match ORBIT's eval
  sampling (`temperature 0.6`, `top_p 0.95` — `scripts/eval/openai.sh`). They agree on
  temperature and top_p, so only `max_tokens` and reasoning effort differ; ORBIT caps
  responses hard and its system prompt says "Overlong responses will be penalized"
  (RECON §3). Not blocking Phase 1; settle before the Phase 2 baseline.

**Client env:** `~/.conda/envs/phyreagent` (python 3.11, numpy, openai — no vllm, which is
correct for the client side; the server env is
`/work/pi_sniekum_umass_edu/$USER/.conda/envs/vllm-gptoss`).

---

## New, resulting from D1: leakage-split reporting

Because Maze retains 19.1% train/eval map overlap even with correct seed-disjointness,
**every headline metric must be reported split by leaked vs. clean eval instances**
(byte-identical to some train instance vs. not), not just pooled. This should be built
into the metrics logging from Phase 2 onward, not retrofitted after the no-bank
baseline is already run — the split needs to exist before any HExA-vs-no-bank
comparison is trusted, since a bank could otherwise inflate its transfer claim purely on
the 19.1% of instances where it can partially recall a specific route rather than a
general strategy.

---

## Net effect on `hexa_design_b_orbitplan.md`

- **Task staging**: Maze only. Mastermind dropped, not deferred — do not revisit without
  a genuinely different Mastermind config that also gets its own fresh eval-pool
  validation.
- **Phase 2 (no-bank baseline)**: run at `total_step_cap=27` with the hard `T=3` cap as the
  binding terminator (per D3), serial execution, leakage-split reporting from the first
  run. Table 2's Maze line is a 5-episode figure — label it as such wherever it appears.
- **Phase 3 (HExA Design B)**: same environment settings; retriever config simplified per
  RECON §7 (single task family, `max_general_skills=0`, no `RELATED_LEVELS` map needed).
- **Actor**: reimplement a `GEMTextAgent`-equivalent (cumulative chat, `\boxed{}` parsing
  via `extract_last_boxed` semantics, last-match) against our own LLM client and the
  shimmed env — `hexa_core`'s single-turn actor and `ReactAgent`'s InterPhyre binding are
  both wrong fits, confirmed in RECON §7.
- **Reward**: rebind `compute_trajectory_reward` via `hexa_core/hexa_bridge.py`'s existing
  monkeypatch seam, using per-episode success + step count (both confirmed available,
  RECON §5), output kept inside `[-1, 1]` since `_compute_initial_confidence` clamps to
  that range.
- **Oracle**: port Maze's free BFS-shortest-path oracle for an optional regret metric;
  skip Mastermind's DP oracle (moot, Mastermind is dropped).

## Still open, not blocking Phase 1

- Round size `x` and total instance-pool sizes for Maze (train pool doesn't exist yet,
  per RECON §0b — must be generated via the `is_train=True` seed-doubling scheme).
  Suggest: instantiate the plan's earlier proposal (~200 training instances, 8–12 rounds,
  100 held-out) once real per-instance cost is measured on a handful of pilot instances.
  Note the step budget is the plan's original ≤27 per instance, not the wider figure D3
  first assumed — so cost estimates should not be revised upward on that account.
- Whether to eventually revisit Mastermind under a genuinely different, freshly-validated
  config — out of scope until Maze results exist.
