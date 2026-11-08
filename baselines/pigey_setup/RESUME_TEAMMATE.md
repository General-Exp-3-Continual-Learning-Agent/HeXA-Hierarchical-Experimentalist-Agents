# Resume the two unfinished Pigey evaluations

Handoff for `udhanuka_umass_edu`, prepared 2026-11-08. Use your own Unity account
and your own authenticated Claude Code account with available Sonnet/Haiku quota.
The shared scratch environments and model weights are already installed. Code is
on the GitHub branch `robotics_baseline`; experiment outputs are not in Git.

## What remains

Shared runtime root:

```text
/scratch/workspace/vgandhi_umass_edu-roboprobe-sim/pigey
```

| Run | Finished bank eval | Remaining | Last failed job | Reason |
|---|---:|---:|---|---|
| `goal_swap` | 249/250 | `t00_e35` | 66918619 | Claude CLI exceeded 300 seconds |
| `spatial_task` | 117/250 | 133 episodes, starting `t04_e42` | 66930225 | Claude rate limit |

Both seed/training phases and both no-bank eval arms are complete. Use the existing
`results/hexa/<run>_train/bank_round_16.json`. The resume helper submits only the
with-bank evaluation and its dependent CPU comparison. It preserves finished
trials and retries backend errors; no-bank eval and training are not rerun.

## Get the code and check access

On a Unity login node, clone into your own persistent source directory:

```bash
git clone --branch robotics_baseline --single-branch \
  https://github.com/General-Exp-3-Continual-Learning-Agent/HeXA-Hierarchical-Experimentalist-Agents.git \
  ~/HeXA-Pigey-resume
cd ~/HeXA-Pigey-resume
git submodule update --init baselines/Pigey
export PIGEY_ROOT=/scratch/workspace/vgandhi_umass_edu-roboprobe-sim/pigey
export PATH="$HOME/.local/bin:$PATH"
claude auth status
```

If needed, install/authenticate your own Claude Code CLI before proceeding; do not
copy Varun's credentials or point HOME at his account. `CLAUDE_BINARY` can identify
your own CLI executable if it is not on PATH. Authentication status alone does not
prove remaining usage quota; confirm your account has quota before launching.

Varun approved write access for your account to the mutable `cache/`, `tmp/`, the
two unfinished bank result trees, new logs, and the two comparison/job-list files.
The ACL preparation job is **66943467**. Verify it completed successfully and that
you can write these paths before submission:

```bash
sacct -j 66943467 -X --format=JobID,State,ExitCode
for path in "$PIGEY_ROOT/logs" "$PIGEY_ROOT/tmp" \
  "$PIGEY_ROOT/cache" "$PIGEY_ROOT/results/eval/goal_swap_bank" \
  "$PIGEY_ROOT/results/eval/spatial_task_bank"; do
  test -w "$path" && echo "writable: $path" || echo "NO WRITE ACCESS: $path"
done
sinfo -p superpod-a100,cpu -o '%P %a %l %G'
sacctmgr show assoc where user="$USER" format=Account,Partition,QOS
squeue -u vgandhi_umass_edu,udhanuka_umass_edu -o '%.18i %.34j %.12T %.24R'
```

Use the `pi_sniekum_umass_edu` Slurm account if it is available to you, otherwise
set `SLURM_ACCOUNT` to an account with access to `superpod-a100` and `cpu`.
Check that neither suite already has an active resume job. Submit each suite once;
concurrent jobs must not write the same output directory.

## Submit the remaining evaluations

Still in your clone:

```bash
bash baselines/pigey_setup/experiments/resume_eval.sh goal_swap --dry-run
bash baselines/pigey_setup/experiments/resume_eval.sh spatial_task --dry-run

bash baselines/pigey_setup/experiments/resume_eval.sh goal_swap
bash baselines/pigey_setup/experiments/resume_eval.sh spatial_task
```

Each invocation prints an eval job ID and a dependent compare job ID, and appends
them to `results/eval/<run>_jobs.txt`. Defaults are one A100-80GB, 8 CPUs, 64 GB,
`superpod-a100`, and exclusion of `gpu023` (which previously lacked `/project`).
Walltimes are 1 hour for the single goal-swap episode and 14 hours for spatial task.
The helper derives REPO_DIR and PIGEY_REPO from your clone, so it uses your code and
upstream Pigey submodule while reusing the shared installed environments.

Protocol remains Sonnet 5.5, Haiku 4.5 verifier, Qwen3-VL-32B-FP8 detector at 0.55 GPU
share, pi0.5 at 0.35, SAM as released, phase 0 skipped, 35 LLM steps, 500 steps per
VLA rollout, CLI timeout 300 seconds and max-turns 3. Eval init states are 25-49
for all ten tasks. Do not change the final banks or use the invalid goal-swap run
folders suffixed `.invalid_ratelimit_20261107`.

## Monitor and confirm completion

```bash
squeue -u "$USER" -o '%.18i %.34j %.12T %.10M %.24R'
# Substitute the IDs printed by the helper:
sacct -j EVAL_JOB_ID,COMPARE_JOB_ID -X --format=JobID,JobName,State,ExitCode,Elapsed
cat "$PIGEY_ROOT/results/eval/goal_swap_comparison.txt"
cat "$PIGEY_ROOT/results/eval/spatial_task_comparison.txt"
```

Logs are `logs/<run>-eval-bank-resume-<job>.{out,err}` and
`logs/<run>-compare-resume-<job>.{out,err}`. Server logs use the same job ID.
Each final comparison must show **250 paired episodes**, no unfinished episodes,
and a clean injection audit. Comparisons run after any eval exit, so a completed
compare job can still contain partial results. If needed, resubmit the same
helper after resolving the infrastructure/quota issue; it skips completed pairs.
The rate-limit handler waits up to six hours per CLI call, but fails immediately
if the reported reset exceeds that cap. Do not launch when quota is exhausted.

As of handoff, the partial paired goal-swap result is 184/249 with bank vs 108/249
without (+30.5 points); spatial task is 91/117 vs 80/117 (+9.4 points). The complete
spatial-task no-bank arm is 176/250. Do not compare 91/117 against 176/250 as if they
covered the same tasks/states.

## Preserve outputs

The shared source workspace expires **2026-11-26**. Before then, copy important
results (including new trial JSONs, videos, banks, comparison reports, logs and job
IDs) to an existing persistent `/work` directory you can write. Run substantial
copies through a CPU Slurm job. Reusing the installed environments does not extend
the source workspace lifetime. Record final results back in the project handoff.
