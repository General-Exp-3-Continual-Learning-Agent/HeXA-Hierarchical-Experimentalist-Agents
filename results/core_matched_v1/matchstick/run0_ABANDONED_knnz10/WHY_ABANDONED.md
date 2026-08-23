# Abandoned: `--knn-max-z` did not match CORE's published configuration

**Job 63064815, cancelled 2026-08-18 at ~380/600 training rollouts (~16 h in).**

## What was wrong

Our launcher passed 17 flags to `run_core.py` and inherited argparse defaults for the
rest. Two of those defaults do not match what CORE publishes in `shell_scripts/`:

| Flag | CORE's published value | What this run used | Consequence |
|---|---|---|---|
| `--knn-max-z` | **= `num_train_items`** in every script (5 / 10 / 100) | **10** (argparse default), against `n_train=60` | **Different retrieval rule** |
| `--eval-period` | 10000 in the sample-efficiency scripts | **250** (argparse default) | Extra eval passes; wall-clock only |

### `--knn-max-z` — why this invalidates the accuracy result

CORE sets `knn_max_z = num_train_items` in all twelve published scripts:

```
rollout_efficiency/*       n_train=10   knn-max-z=10
sample_efficiency/5_train  n_train=5    knn-max-z=5
sample_efficiency/100_train n_train=100 knn-max-z=100
```

Since `knn_k = min(knn_max_z, |registered problems|)`, their convention makes the
neighbourhood **the entire training set**, so an insight's utility `Û(q,i)` is a global
average — query-independent by construction, at every regime. Similarity still ranks
candidates; the utility term does not condition on the query.

This run used `z=10` against 60 training items, so `knn_k = 10`: utility conditioned on the
10 nearest problems. That is a **different algorithm from the one CORE reports**, and the
whole point of this run was to remove exactly that class of objection. The correct value
for our `n_train=60` cell is `--knn-max-z 60`.

Direction of the bias is **unknown** — more conditioning could help or hurt CORE. That is
precisely why it cannot be reported: an unfavourable CORE number would be fairly
attributable to our misconfiguration rather than to the method.

### `--eval-period` — wall-clock contamination only

`validation_summary.jsonl` records eval passes at rollout 0 and rollout 254. CORE's
sample-efficiency scripts set 10000 specifically to suppress intermediate evaluation and
keep only the initial and final passes. At 250 against a 600-rollout budget we picked up
roughly two extra 100-problem eval passes, ~1.4 h of the ~21 h projection.

Accuracy is **not** affected — `evaluate_memory_on` runs with `register_problem=False,
update_utilities=False, mark_use=False, deterministic=True`, so no eval pass writes to the
bank. Only the timing number is contaminated.

## What is still usable from this directory

The **efficiency measurements**, with the eval-period caveat applied:

- 195 solve rollouts against 154 admission rollouts → **1.84 rollouts/episode**, i.e. CORE
  spends ~44% of its budget on admission testing. Unaffected by either flag.
- Truncation: of 195 solve calls, 141 terminated normally (**136 correct, 96%**) and 54 hit
  the 60,000-token ceiling (**0 correct**). Runaway reasoning costs ~632 s per call at
  ~95 tok/s and sets the pace of every batch.
- ~1161 s per 5-episode batch, ~22.5 rollouts/h, ~21 h projected for 600 rollouts —
  **inflated by ~1.4 h** of unintended eval passes, so read as ~19.5 h.

`knn_max_z` affects retrieval, not scheduling, so the barrier-structure and
budget-allocation figures carry over to the corrected run. The accuracy numbers do not.

## Deliberate deviations from CORE's scripts (unchanged in the rerun)

These are intentional and documented in `CORE_COMPARISON.md` §4:

| Flag | CORE | Ours | Why |
|---|---|---|---|
| `--solve/eval/reflection-tokens` | 32768 | 60000 | match HeXA v3 |
| `--num-train-items` | 5/10/100 | 60 | match HeXA schedule |
| `--max-rollouts` | 2000 | 600 | match HeXA budget |
| `--num-baseline-train-runs` | 10 (default) | 5 | match HeXA `n_base` |
| `--use-batching-in-*` | off (shipped default) | on (5 / 25 / 30) | tractability; 5 is their own default width |
| `NvidiaClient.base_url` | NVIDIA hosted NIM | local vLLM | provider constraint; `nvidia_client.py:24` is hardcoded |
