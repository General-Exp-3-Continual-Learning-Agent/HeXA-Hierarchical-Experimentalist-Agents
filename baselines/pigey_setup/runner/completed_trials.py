"""Print ``task episode`` for every finished trial under a results directory.

The sbatch loops use this to resume: a (task, init state) pair listed here is
skipped. A trial counts as finished when its ``trial.json`` has ``success`` true
or false; backend errors (``success: null``) are rerun. Other suites are ignored.

    python runner/completed_trials.py RESULTS_DIR [--suite libero_goal_task]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def completed(root: str | Path, suite: str | None = None) -> list[tuple[int, int]]:
    pairs = set()
    for path in Path(root).rglob("trial.json"):
        try:
            trial = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if suite and trial.get("suite") != suite:
            continue
        if trial.get("status") == "backend_error" or trial.get("success") is None:
            continue
        pairs.add((int(trial["task_id"]), int(trial["episode"])))
    return sorted(pairs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root")
    parser.add_argument("--suite", default=None)
    args = parser.parse_args(argv)
    if Path(args.root).is_dir():
        for task, episode in completed(args.root, args.suite):
            print(task, episode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
