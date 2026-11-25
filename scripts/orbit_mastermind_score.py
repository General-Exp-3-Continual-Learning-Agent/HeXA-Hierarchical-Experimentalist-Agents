#!/usr/bin/env python
"""Score Mastermind arms on *true solves*, paired, with McNemar exact p-values.

Why this exists: ORBIT's `MultiEpisodeEnv._is_episode_success` is
`terminated and reward > 0`. GEM's Mastermind reward is black_pegs/4, and an
episode that exhausts its three guesses still *terminates*, so a final guess with
one peg in the right place is flagged a success. Measured inflation on the
gpt-oss baseline: 3.51 flagged successes per code actually cracked.

The rule here is `max step reward >= 1.0`, i.e. 4/4 black pegs, i.e. the code was
named. Both metrics are printed so the discrepancy stays visible rather than
being quietly swapped.
"""

from __future__ import annotations

import argparse
import json
import re
from math import comb
from pathlib import Path


import itertools

_CODES = list(itertools.permutations(range(1, 9), 4))
_OBS = re.compile(
    r"At turn (\d+), you guessed \[([0-9, ]+)\]\. This guess receives "
    r"(\d+) black peg\(s\) and (\d+) white peg\(s\)"
)
SECRETS: dict[str, tuple] = {}


def _pegs(guess: tuple, code: tuple) -> tuple[int, int]:
    black = sum(g == c for g, c in zip(guess, code))
    return black, len(set(guess) & set(code)) - black


def _guess(step: dict):
    digits = [int(x) for x in re.findall(r"\d", str(step.get("parsed_action") or ""))]
    return tuple(digits) if len(digits) == 4 else None


def recover_secrets(paths) -> None:
    """Pin each instance's code by intersecting peg constraints across all arms.

    An instance_id is the code's identity, so feedback from every arm that played it
    constrains the same secret. Pooling them pins 253/256 instances; the rest are
    dropped from scoring rather than guessed at.
    """
    constraints: dict[str, list] = {}
    for path in paths:
        for line in path.open():
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
    for iid, cons in constraints.items():
        cand = [c for c in _CODES if all(_pegs(g, c) == bw for g, bw in cons)]
        if len(cand) == 1:
            SECRETS[iid] = cand[0]


def load(path: Path) -> dict:
    out = {}
    for line in path.open():
        rec = json.loads(line)
        if rec.get("error"):
            continue
        eps: dict[int, list] = {}
        for step in rec["steps"]:
            eps.setdefault(step["episode_index"], []).append(step)
        # NOT `reward >= 1.0`. GEM's reward is black_pegs/4, but the value returned
        # on the step that ENDS an episode does not reliably carry the peg count:
        # measured on stored runs, an agent that named the code exactly received
        # 0.25 on that step. Checked against brute-forced ground truth, `reward>=1.0`
        # missed 58 of 420 episodes in which the agent did name the secret.
        # Ground truth is instead "did any guess equal the code", where the code is
        # recovered by intersecting every peg constraint observed for that instance
        # across all arms (same instance_id == same code; 253/256 pin uniquely).
        secret = SECRETS.get(rec["instance_id"])
        solved = [
            secret is not None and any(
                _guess(s) == secret for s in eps.get(i, ())
            )
            for i in range(3)
        ]
        out[rec["instance_id"]] = {
            "solved": solved,
            "flagged": rec["episode_successes"],
            "truncated": rec.get("n_truncated", 0),
        }
    return out


def mcnemar(keys, a, b, pred) -> tuple[int, int, float]:
    """Two-sided exact test on the discordant pairs; b gains, a loses."""
    gained = sum(1 for k in keys if not pred(a[k]) and pred(b[k]))
    lost = sum(1 for k in keys if pred(a[k]) and not pred(b[k]))
    n = gained + lost
    if n == 0:
        return gained, lost, 1.0
    lo = min(gained, lost)
    p = 2 * sum(comb(n, i) for i in range(lo + 1)) / 2 ** n
    return gained, lost, min(1.0, p)


METRICS = [
    ("ep1", lambda d, k: d[k]["solved"][0]),
    ("ep2", lambda d, k: d[k]["solved"][1]),
    ("ep3", lambda d, k: d[k]["solved"][2]),
    ("any", lambda d, k: any(d[k]["solved"])),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path)
    ap.add_argument("arms", type=Path, nargs="+")
    args = ap.parse_args()

    def runs(p: Path) -> Path:
        return p if p.suffix == ".jsonl" else p / "runs.jsonl"

    all_paths = [runs(args.baseline)] + [runs(a) for a in args.arms]
    recover_secrets(all_paths)
    print(f"codes pinned from pooled feedback: {len(SECRETS)}")
    base = load(runs(args.baseline))
    for arm_path in args.arms:
        arm = load(runs(arm_path))
        keys = sorted(set(base) & set(arm))
        n = len(keys)
        print(f"\n=== {arm_path} vs {args.baseline}  (paired n={n}) ===")
        print("  TRUE SOLVE (code named, reward 4/4):")
        for label, pred in METRICS:
            f = lambda d, p=pred: (lambda x: p({0: x}, 0))
            b_rate = sum(1 for k in keys if pred(base, k)) / n
            a_rate = sum(1 for k in keys if pred(arm, k)) / n
            g, l, p = mcnemar(keys, base, arm,
                              lambda d, pr=pred: pr({0: d}, 0))
            print(f"    {label}: base={b_rate:.3f} arm={a_rate:.3f} "
                  f"delta={100 * (a_rate - b_rate):+5.1f}  "
                  f"gained={g} lost={l} p={p:.4f}")
        fb = sum(1 for k in keys if any(base[k]["flagged"])) / n
        fa = sum(1 for k in keys if any(arm[k]["flagged"])) / n
        print(f"  ORBIT-FLAGGED any: base={fb:.3f} arm={fa:.3f} "
              f"delta={100 * (fa - fb):+5.1f}   (partial credit -- not a solve rate)")
        tb = sum(base[k]["truncated"] for k in keys) / n
        ta = sum(arm[k]["truncated"] for k in keys) / n
        print(f"  truncated turns/instance: base={tb:.2f} arm={ta:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
