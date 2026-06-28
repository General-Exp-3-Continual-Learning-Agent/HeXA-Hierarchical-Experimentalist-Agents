"""Convert training-framework JSONL trajectories to SKILL-RL individual JSON format.

The training framework saves trajectories as JSONL with fields:
  step, level, score, response_length, tool_calls, n_turns, response_snippet

SKILL-RL distillation expects individual trajectory_seed{N}.json files with:
  success, seed, iterations, level, trajectory (list of {thought,action,observation}),
  raw_response (fallback for the teacher formatter)

Usage:
    python SKILL-RL/tools/convert_training_trajs.py \
        --input SKILL-RL/FirstFiveTrajs/dte_epoch0_trajectories.jsonl \
        --output SKILL-RL/data/initial/down_to_earth/ \
        --seeds 1 2 3 4 5
"""

import argparse
import json
import re
import sys
from pathlib import Path


def parse_react_steps(text: str) -> list[dict]:
    """Parse a raw ReAct text string into thought/action/observation dicts."""
    steps = []
    # Split on Thought: boundaries (each new Thought starts a step)
    chunks = re.split(r"(?=Thought:)", text.strip())
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        thought_m = re.search(r"Thought:(.*?)(?=Action:|$)", chunk, re.DOTALL)
        action_m  = re.search(r"Action:(.*?)(?=Action Input:|Observation:|Thought:|$)", chunk, re.DOTALL)
        obs_m     = re.search(r"Observation:(.*?)(?=Thought:|$)", chunk, re.DOTALL)

        thought = thought_m.group(1).strip() if thought_m else ""
        action  = action_m.group(1).strip()  if action_m  else ""
        obs     = obs_m.group(1).strip()     if obs_m     else ""

        if thought or action:
            steps.append({"thought": thought, "action": action, "observation": obs})
    return steps


def convert(jsonl_path: Path, output_dir: Path, seeds: list[int]):
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [l for l in jsonl_path.read_text().splitlines() if l.strip()]

    if len(seeds) != len(lines):
        print(f"Warning: {len(seeds)} seeds given but {len(lines)} lines in {jsonl_path}. "
              f"Using min({len(seeds)}, {len(lines)}) entries.")

    count = min(len(seeds), len(lines))
    written = 0

    for i in range(count):
        entry = json.loads(lines[i])
        seed  = seeds[i]
        snippet = entry.get("response_snippet", "")
        steps   = parse_react_steps(snippet)

        record = {
            "success":     entry.get("score", 0.0) >= 1.0,
            "seed":        seed,
            "iterations":  entry.get("n_turns", 0),
            "level":       entry.get("level", "unknown"),
            "trajectory":  steps,
            "raw_response": snippet,
        }

        out_path = output_dir / f"trajectory_seed{seed}.json"
        out_path.write_text(json.dumps(record, indent=2))
        status = "SUCCESS" if record["success"] else "FAIL"
        print(f"  seed={seed} {status} ({len(steps)} parsed steps) → {out_path.name}")
        written += 1

    print(f"\nConverted {written}/{count} trajectories to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert training JSONL to SKILL-RL JSON files")
    parser.add_argument("--input",  type=str, required=True, help="Path to JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--seeds",  type=int, nargs="+", required=True,
                        help="Seed numbers to assign to each line (by order)")
    args = parser.parse_args()

    convert(Path(args.input), Path(args.output), args.seeds)


if __name__ == "__main__":
    main()
