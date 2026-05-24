# Ablations

## Reward ablation: contrastive-only

**Wrapper:** `scripts/run_hexa_contrastive.py`
**Loop:** [skillrl/loops/contrastive_only_loop.py](../skillrl/loops/contrastive_only_loop.py)
**Variant distill / evolve:** [skillrl/distillation/contrastive_only_distill.py](../skillrl/distillation/contrastive_only_distill.py), [skillrl/distillation/contrastive_only_evolve.py](../skillrl/distillation/contrastive_only_evolve.py)
**Variant teacher prompt:** [skillrl/distillation/teacher_prompts_contrastive_only.py](../skillrl/distillation/teacher_prompts_contrastive_only.py)

This is the same offline-to-online evolving loop as HeXA, with two parts of the standard pipeline removed:

| Removed | Where it normally lives | What replaces it |
|---|---|---|
| Per-trajectory **reward score** (used to rank trajectories before distillation, see [SKILLRL_SUMMARY § Reward calculation](../skillrl/SKILLRL_SUMMARY.md#reward-calculation-trajectory-scoring)) | `distill.py` | None — the teacher just gets the trajectories |
| Per-skill **confidence** (calibrated from source-seed rewards, used by the agent's retriever to rank skills) | `_compute_initial_confidence` in `distill.py` | Skills are unranked by confidence; retrieval falls back to recency / contrastive importance only |

What's left is pure CONTRAST: the teacher is shown successes and failures and asked to extract skills/mistakes purely by comparing them. No reward signal, no confidence calibration.

### What this ablation tells us

If the contrastive variant lands far below HeXA, it means rewards & confidence calibration are doing real work — they're not just bookkeeping. If it lands close, the contrastive scaffolding alone (success vs failure trajectories) is the load-bearing piece, and the reward stuff is dressing.

### Where it was run

In the paper, the contrastive ablation was run on **Qwen-7B** (not Claude) on two levels:

- `down_to_earth`
- `two_body_problem`

This was a deliberate choice — the smaller teacher model is more sensitive to prompt scaffolding, so reward/confidence removal hits harder there and gives a clearer signal.

```bash
python scripts/run_hexa_contrastive.py \
    --level down_to_earth \
    --initial-traj-dir Initial_trajectories/down_to_earth \
    --num-rounds 3 --seeds-per-round 5 \
    --teacher-model "Qwen/Qwen2.5-7B-Instruct"
```

Note: running the teacher on Qwen-7B requires a local OpenAI-compatible server (e.g. vLLM). See [INSTALL.md § Optional Qwen / local vLLM](../INSTALL.md).

### Per-round outputs

The loop writes one bank per round in `<output-dir>/skill_bank_contrastive_only_{round}.json`, with `progress_contrastive_only.json` summarising round-by-round accuracy. The schema mirrors the standard HeXA outputs but without `confidence` fields.
