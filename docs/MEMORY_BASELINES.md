# Memory baselines: Episodic RAG and MemRL

Two cross-episode memory baselines for InterPhyre, in `baselines/`. They exist to
separate *"HeXA works because it distils skills"* from *"HeXA works because it has
any memory at all"* — the three baselines in [BASELINES.md](BASELINES.md) either
have no memory (ReAct, Direct) or memory confined to a single seed (Reflexion).

Both are lifelong learners over the same 50-seed grid HeXA uses, and both inject
into the same seam HeXA's skill bank uses, so the only thing that differs between
methods is the injected block.

| | Stored | Retrieval | Update | Bank size |
|---|---|---|---|---|
| Episodic RAG | raw rollouts, verbatim | nearest success + nearest failure | append only | unbounded |
| MemRL | LLM-summarized experiences | top-k1 by cosine, then top-k2 by similarity+utility | Monte-Carlo Q update per episode | unbounded, near-duplicates merged |
| HeXA | distilled skills + mistakes | top-K by confidence | teacher evolves the bank per round | capped at 10 |

## Provenance

**Episodic RAG** follows the CORE paper (arXiv 2605.28742, §4.1): all rollouts go
into a memory indexed by problem embedding; each inference gets the most similar
past *successful* rollout and the most similar past *unsuccessful* rollout with
their verifier feedback; if either class is empty, retrieve by similarity alone.
No summarization, no utility.

**MemRL** is ported from the official release (MemTensor/MemRL, arXiv
2601.03192) using its **ALFWorld** configuration — of its four benchmarks, the
closest analogue to InterPhyre. Memory holds Intent-Experience-Utility triplets:

- **Phase A** — candidates `= TopK_k1({i : cos(z_q, z_i) > δ})`; empty ⇒ inject
  nothing and let the frozen model explore.
- **Phase B** — pick top-`k2` by `(1−λ)·ẑ(sim) + λ·ẑ(Q)`. Similarity is z-scored
  against **precomputed dataset statistics**; Q is z-scored against the
  **candidate pool**, clipped to ±3.
- **Update** — `Q ← (1−α)Q + α(r + γ·maxQ')` with `γ = 0` (Monte-Carlo), applied
  to every memory that was injected. `r = +1` success, `−1` failure.
- **Write-back** — success ⇒ `SCRIPT + TRAJECTORY` (proceduralization); failure ⇒
  a reflection note (adjustment). Only the *intent* is embedded, not the content.

Defaults: `k1=5, k2=3, λ=0.5, α=0.3, Q_init=0`, near-duplicate merge at 0.90,
reward attribution to merge targets at 0.85 with gain 0.1.

## Run it

```bash
# 1. Calibrate the retrieval key + MemRL's constants (no rollouts, no LLM calls)
sbatch baselines/bash_scripts/sbatch_calibrate.sh --level pass_the_parcel

# 2. Claude actor — comparable with the published HeXA / ReAct numbers
sbatch baselines/bash_scripts/sbatch_baseline_claude.sh \
    --method episodic_rag --level pass_the_parcel \
    --initial-traj-dir Initial_trajectories/pass_the_parcel

# 3. Local Qwen2.5-7B-Instruct actor on one GPU
sbatch baselines/bash_scripts/sbatch_baseline_qwen.sh \
    --method memrl --level catapult \
    --initial-traj-dir Initial_trajectories/catapult
```

Or directly, without Slurm:

```bash
python scripts/calibrate_intents.py --level catapult --seeds 1-105 \
    --intent-mode geometry --embedding-model numeric
python scripts/run_episodic_rag.py --level catapult --seeds 6-55 \
    --initial-traj-dir Initial_trajectories/catapult \
    --output-dir baselines/data/rag_catapult
python scripts/run_memrl.py --help
```

Re-running resumes: seeds whose trajectory JSON exists are skipped.

### Step 1 is not optional for MemRL

MemRL's Phase-A threshold δ is dataset-density dependent — the paper sets it to
the top-20% quantile of the benchmark's pairwise task-description similarities,
and its config hardcodes per-benchmark similarity-normalization statistics.
`baselines/calibrate.py` measures all three on InterPhyre intents;
`run_memrl.py` refuses to start on ALFWorld's values, and refuses a calibration
measured with a different embedder or intent mode.

## The retrieval key: why it is not a text embedding

Both papers embed a *task description*. On InterPhyre that does not work, and the
calibration run is what establishes it. Within one level, every intent is the
same template with different numbers, so language-model embeddings come out
nearly collinear — and, more importantly, their **ranking barely tracks scene
similarity**, which is the only thing retrieval uses:

| key | pairwise cosine (105 seeds) | Spearman ρ vs geometry | top-1 overlap |
|---|---|---|---|
| jina-v2-base-en, `--intent-mode full` — catapult | 0.9994 ± 0.0003 | — | — |
| jina-v2-base-en, `--intent-mode geometry` — catapult | 0.9986 ± 0.0005 | **+0.11** | 11% |
| jina-v2-base-en, `--intent-mode geometry` — pass_the_parcel | 0.9979 ± 0.0007 | **+0.40** | 10% |
| `numeric` (45 dims) — catapult | −0.007 ± 0.494 | ground truth | — |
| `numeric` (29 dims) — pass_the_parcel | −0.007 ± 0.508 | ground truth | — |

A compressed range alone would be survivable: δ is a quantile and z-normalization
rescales whatever variance exists. Retrieving at ρ ≈ 0.11 is not — it makes
Phase A draw near-arbitrary memories, which would turn both baselines into straw
men and make any HeXA win meaningless.

So the **default key is `numeric`**: the standardized geometry vector parsed out
of the geometry intent, cosine-compared like any other embedding. It is a
documented deviation from "embed the task description", recorded in every run
manifest, and it preserves what Phase A is *for*. The text embedders remain
available (`--embedding-model jinaai/jina-embeddings-v2-base-en`, or
`openai:text-embedding-3-large` to match MemRL's own config) — worth running as a
secondary configuration precisely because the gap between them is a finding.

Note that the choice was made from a diagnostic over **geometry alone** — no
rewards, no evaluation labels, no success rates — so it does not constitute
tuning on the evaluation set.

`baselines/memory/features.py` holds the key, the scaler, and the diagnostic;
`rank_agreement` scores both sides with cosine (the metric retrieval actually
uses) so identical keys score ρ = 1.

## Protocol

Seeds **6–55 in order** (the grid the other baselines used), optionally preloaded
with the level's `Initial_trajectories/` for the same Off2On head start HeXA
gets. Retrieve → solve (≤25 iterations) → update utilities → write back, once per
seed. Results are grouped into rounds of 3 so the curve overlays HeXA's rounds.

`--freeze-after N` makes memory read-only from stream index N, which is MemRL's
"Transferring" setting (frozen bank, unseen instances).

`--lambda-q 0` is the runtime-RL ablation: Q-blind retrieval, everything else
identical.

## What gets logged

Per episode, in `episodes.jsonl` and each trajectory JSON:

- `success`, `iterations`
- `simulator_calls` — the real rollout budget in an experimentation environment
  (a `simulate_action` call, not a ReAct iteration)
- `added_context_tokens` — CORE's context-efficiency axis; raw-rollout replay
  costs an order of magnitude more prompt than a bounded skill bank, and this
  measures it on our environment
- `retrieval_diagnostics` — which memories were considered, their similarity, Q,
  and composite score; and *why* nothing was injected when nothing was
- `utility_updates` — old Q, new Q, reward, visit count per injected memory

`summary_<method>.json` adds per-round accuracy, the running success rate, and
memory-bank statistics. `manifest.json` records the full config and an explicit
list of deviations from the published setups.

## Deviations, by actor

| | Claude actor | Qwen actor |
|---|---|---|
| MemRL memory-building temperatures (0.0 / 0.3) | **not reproducible** — `claude -p` has no temperature flag | honoured |
| Retrieval key | `numeric` by default, for the reason measured above; text embedders available | same |
| Comparability with published HeXA numbers | direct (same actor) | separate comparison; run HeXA/ReAct on Qwen too |

HeXA's `DIAGNOSTIC_ADDENDUM` is deliberately **not** injected into either
baseline: it instructs the agent to cite skills, which is method-specific
scaffolding.

## Layout

```
baselines/
├── calibrate.py            step 1: δ, sim_norm_mean, sim_norm_std per level
├── memory/
│   ├── intent.py           (level, seed) -> intent text  [no rollouts]
│   ├── features.py         numeric geometry key, scaler, rank-agreement diagnostic
│   ├── embedder.py         jina local / OpenAI, with a disk cache
│   ├── store.py            JSON-backed records + cosine
│   ├── rag.py              Episodic RAG
│   ├── memrl.py            two-phase retrieval + MC utility update
│   ├── summarize.py        memory-building calls (claude CLI or local model)
│   ├── trajectory_fmt.py   trajectory text + rollout accounting
│   └── blocks.py           prompt blocks + token counting
├── runner/
│   ├── memory_runner.py    the injection seam (augmented_runner analogue)
│   ├── stream.py           the shared online seed-stream loop
│   ├── run_episodic_rag.py
│   └── run_memrl.py
├── bash_scripts/           sbatch entry points
└── tests/test_memory_logic.py   54 checks, no GPU, ~2s
```
