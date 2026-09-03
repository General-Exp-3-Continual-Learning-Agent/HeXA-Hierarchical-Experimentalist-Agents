#!/usr/bin/env python
"""Rewrite Mastermind `runs.jsonl` success fields from recovered ground truth.

Why the stored fields are wrong
-------------------------------
Episode success was recorded from the env's `reward` on the step that ends an
episode, and that value does not reliably describe the guess made on that step.
Demonstrated on instance b734e37459e7d143: 49 pooled peg observations pin the code
to (2,1,8,6) uniquely, the agent guessed exactly 2 1 8 6, and the env returned
reward 0.5. It is not a simple one-step lag either -- across v3 the terminal reward
matches the current guess 191 times, the previous guess 77, and neither 269.

Two label rules have been used across the project, both derived from that field:
`reward > 0` (any peg correct -- inflates ~1.9x) and `reward >= 1.0` (misses solves
that land on the final allowed turn -- 53 of 256 instances in v3). Arms recorded
under different rules are not comparable, which is the immediate reason to rescore
everything with one rule.

The rule used here
------------------
An episode is solved iff some guess in it equals the instance's secret code. The
code is recovered by intersecting every peg constraint observed for that
`instance_id` across *all* arms -- the id is the code's identity, so pooling is
valid and pins 255/256 codes. This depends on no reward or label field.

`episode_successes_confirmed` additionally requires that the env ended the episode
before its turn budget was spent, i.e. it excludes final-turn solves the env may
not have credited. Report the primary rule; use this for a robustness check.

Instances whose code cannot be pinned keep their original fields and are marked
`scoring.rescored = false`.

    python scripts/rescore_mastermind_runs.py --apply      # writes, keeps .bak
    python scripts/rescore_mastermind_runs.py              # dry run
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import re
import shutil
from pathlib import Path

CODES = list(itertools.permutations(range(1, 9), 4))
_OBS = re.compile(
    r"At turn (\d+), you guessed \[([0-9, ]+)\]\. This guess receives "
    r"(\d+) black peg\(s\) and (\d+) white peg\(s\)"
)
MAX_TURNS = 3


def pegs(guess: tuple, code: tuple) -> tuple[int, int]:
    black = sum(g == c for g, c in zip(guess, code))
    return black, len(set(guess) & set(code)) - black


def guess_of(step: dict):
    digits = [int(x) for x in re.findall(r"\d", str(step.get("parsed_action") or ""))]
    return tuple(digits) if len(digits) == 4 else None


def recover_secrets(paths) -> dict[str, tuple]:
    constraints: dict[str, list] = {}
    for path in paths:
        for line in Path(path).open():
            rec = json.loads(line)
            if rec.get("error"):
                continue
            bucket = constraints.setdefault(rec["instance_id"], [])
            for step in rec["steps"]:
                for m in _OBS.finditer(step["observation"]):
                    bucket.append((
                        tuple(int(x) for x in m.group(2).split(",")),
                        (int(m.group(3)), int(m.group(4))),
                    ))
    secrets = {}
    for iid, cons in constraints.items():
        cand = [c for c in CODES if all(pegs(g, c) == bw for g, bw in cons)]
        if len(cand) == 1:
            secrets[iid] = cand[0]
    return secrets


def rescore(rec: dict, secret) -> dict:
    episodes: dict[int, list] = {}
    for step in rec["steps"]:
        episodes.setdefault(step["episode_index"], []).append(step)
    n = len(rec.get("episode_successes") or []) or 3

    solved, confirmed = [], []
    for i in range(n):
        steps = episodes.get(i, [])
        hit = secret is not None and any(guess_of(s) == secret for s in steps)
        solved.append(hit)
        confirmed.append(hit and len(steps) < MAX_TURNS)

    rec["scoring"] = {
        "rescored": secret is not None,
        "rule": "episode solved iff a guess equals the recovered secret code",
        "secret_recovered": list(secret) if secret else None,
        "episode_successes_original": list(rec.get("episode_successes") or []),
    }
    if secret is None:
        return rec

    rec["episode_successes"] = solved
    rec["episode_successes_confirmed"] = confirmed
    rec["success_count"] = sum(solved)
    metrics = rec.setdefault("metrics", {})
    metrics["episode/success_rate"] = 1.0 if any(solved) else 0.0
    metrics["episode/success_count"] = sum(solved)
    for i, hit in enumerate(solved):
        metrics[f"episode_{i + 1}/success_rate"] = 1.0 if hit else 0.0
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write files (keeps .bak)")
    ap.add_argument("--glob", default="results/orbit_mastermind_*",
                    help="result directories to rescore")
    args = ap.parse_args()

    paths = sorted(set(
        glob.glob(f"{args.glob}/runs.jsonl")
        + glob.glob(f"{args.glob}/eval_frozen/runs.jsonl")
        + glob.glob(f"{args.glob}/round*/runs.jsonl")
    ))
    secrets = recover_secrets(paths)
    print(f"files: {len(paths)}   codes pinned: {len(secrets)}")

    for path in paths:
        recs = [json.loads(l) for l in Path(path).open()]
        before = sum(1 for r in recs if any(r.get("episode_successes") or []))
        out = [rescore(r, secrets.get(r["instance_id"])) if not r.get("error") else r
               for r in recs]
        after = sum(1 for r in out if any(r.get("episode_successes") or []))
        skipped = sum(1 for r in out if not r.get("scoring", {}).get("rescored", True))
        n = len(recs)
        print(f"  {path}")
        print(f"      any-solve {before}/{n} ({before/n:.3f}) -> {after}/{n} "
              f"({after/n:.3f})   unpinned kept as-is: {skipped}")
        if args.apply:
            shutil.copy2(path, path + ".bak")
            with Path(path).open("w") as fh:
                for r in out:
                    fh.write(json.dumps(r) + "\n")
    if not args.apply:
        print("\ndry run -- pass --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
