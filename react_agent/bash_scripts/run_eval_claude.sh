#!/bin/bash
#SBATCH --partition=cpu
#SBATCH -c 2
#SBATCH -t 24:00:00
#SBATCH --mem 8G
#SBATCH --begin=now+150minutes

# ReAct evaluation using Claude Code CLI (no GPU needed - API calls)

module load conda/latest

# Activate virtual environment
conda activate phyreagent

# Where to save results. On compute nodes $HOME is often read-only; use $SCRATCH so files are visible.
# After the job, from a login node: cp -r $SCRATCH/hexa/results_claude_two_body/* <REPO_DIR>/results_claude_two_body/
if [ -n "$SCRATCH" ]; then
  EVAL_DIR="$SCRATCH/hexa/results_claude_two_body"
  echo "Saving results to SCRATCH: $EVAL_DIR"
else
  EVAL_DIR="${REPO_DIR:-$PWD}/results_claude_two_body"
fi

# Run ReAct Agent on 10 seeds (0 to 9)
# Use --resume to skip seeds that already have a completed trajectory file.
# This is useful when the experiment is interrupted by Claude API rate limits.
python -m react_agent.run_react_claude \
    --level down_to_earth \
    --seeds 34 35\
    --max-iterations 25 \
    --eval-dir "$EVAL_DIR" \
    --resume --verbose

# Generate plot from the summary json
python react_agent/plot_eval.py \
    --eval-dir "$EVAL_DIR" \
    --model claude-sonnet-4-6
