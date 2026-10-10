# Interphyre RL Training

GRPO training of `Qwen2.5-3B-Instruct` as a ReAct agent on InterPhyre physics-puzzle
levels, built on `verl` / `verl_tool`. The agent calls a `simulate_action` tool to place
objects and observes the physics outcome.

Two levels: `down_to_earth` (**dte**) and `two_body_problem` (**tbp**).

## Variants

| Variant | What it does | Train scripts (dte / tbp) |
|---|---|---|
| **Baseline** | Plain GRPO, no skills | `slurm_train_3b_a100_1gpu_dte_bsz1.sh` / `..._tbp_bsz1.sh` |
| **Skilled** | Injects a fixed, pre-built skill bank into the prompt | `slurm_train_3b_a100_1gpu_dte_bsz1_skilled.sh` / `..._tbp_bsz1_skilled.sh` |
| **Evolving** | Skill bank is regenerated *during* training by a vLLM 7B teacher from rollout trajectories | `slurm_evolving_grpo_dte.sh` / `slurm_evolving_grpo_tbp.sh` |

All variants: 1×A100, batch size 1, 500 steps, train seeds 1–50, val seeds 51–100.

## Setup

Scripts use a `VerlToolInterphyre` conda env (set `CONDA_ENV` to its path) and
auto-set `PYTHONPATH` to the repo-level `../interphyre` package (so `import
interphyre` resolves); the evolving scripts also add the repo root so the shared
`skillrl` package imports. Run all `sbatch` commands from this `training/`
directory. Other paths are overridable: `CKPT_ROOT` (default `checkpoints/`),
`HF_HOME`, and for the teacher `MODEL_PATH`.

## Data

Each script expects pre-built parquet data under `data/` (not shipped). To
build it, e.g. for baseline dte:

```bash
python examples/data_preprocess/interphyre_data.py \
    --output_dir data/interphyre_dte_s1_50 \
    --levels down_to_earth \
    --num_train_per_level 50 --num_val_per_level 50 \
    --train_seed_start 1 --val_seed_start 51
```

Data dirs per variant: `interphyre_{dte,tbp}_s1_50` (baseline),
`interphyre_{dte,tbp}_s1_50_skilled` (skilled, add `--skill_bank <path>`),
`interphyre_{dte,tbp}_evolving_continuous` (evolving).

## Run training

```bash
# Baseline
sbatch examples/train/interphyre/slurm_train_3b_a100_1gpu_dte_bsz1.sh
sbatch examples/train/interphyre/slurm_train_3b_a100_1gpu_tbp_bsz1.sh

# Skilled
sbatch examples/train/interphyre/slurm_train_3b_a100_1gpu_dte_bsz1_skilled.sh
sbatch examples/train/interphyre/slurm_train_3b_a100_1gpu_tbp_bsz1_skilled.sh

# Evolving — START THE TEACHER FIRST, then the trainer
sbatch examples/train/interphyre/slurm_vllm_7b_teacher.sh   # writes logs/vllm_teacher_endpoint.txt
sbatch examples/train/interphyre/slurm_evolving_grpo_dte.sh
sbatch examples/train/interphyre/slurm_evolving_grpo_tbp.sh
```

**Outputs:** per-run logs, metrics, and trajectories in `logs/run<N>/`
(`trajectories.jsonl`); checkpoints (every 50 steps) under
`$CKPT_ROOT/interphyre/<run_name>/run<N>/`.

The evolving variant starts from 5 base-model rollouts per level: point
`INIT_TRAJ_JSONL` at them (default `skill_data/first_five_trajs/<dte|tbp>_epoch0_trajectories.jsonl`).
It then writes the converted trajectories to `skill_data/` and the evolving banks to
`skill_banks/`.

> Don't launch multiple runs in the same second — the run-dir picker isn't atomic
> and concurrent jobs can collide on `logs/run<N>`.

## Run eval

One general script: `examples/eval/slurm_eval.sh`. Pass `--checkpoint` (HF Hub id or
local `.../actor/huggingface` path) and `--level`; add `--skill_bank` for
skilled/evolving checkpoints. Defaults: seeds 51–100, 25 max turns.

```bash
# Baseline / evolving checkpoint
sbatch examples/eval/slurm_eval.sh \
    --checkpoint <hub_id_or_path> --level two_body_problem

# Skilled checkpoint (match the bank used at train time)
sbatch examples/eval/slurm_eval.sh \
    --checkpoint <hub_id_or_path> --level down_to_earth \
    --skill_bank <path/to/skill_bank.json>
```

Results are written to `logs/eval/eval<N>.jsonl` with a printed solve rate.
