#!/usr/bin/env python
"""Pair two DiscoverPhysics arms on their seeds and test the difference.

    python scripts/compare_discovery_arms.py \
        --a discovery-agents/hexa_discovery/data/runs/osc_eval:baseline \
        --b discovery-agents/hexa_discovery/data/runs/osc_episodic_rag:eval_frozen

Each side is `<run-dir>:<arm>`, where `<arm>` is the value `rejudge.py` stored
(`baseline`, `with_bank`, or `eval_frozen` for a memory arm, which it takes from
the episode's parent directory).  Everything is read from the independent judge's
`rejudge_claude-opus-4-6.json`, never from the self-judged episode files.

Both tests are EXACT, which matters at n = 15:

`explanation`, `reward`
    Paired sign-flip permutation on the mean difference.  All 2^15 = 32,768 sign
    assignments are enumerated, so the p-value is exact rather than sampled.
    This is the test the paper's table reports; a sign test is a different test
    and gives materially different answers on this data (hubble reward 0.0060
    under permutation, 0.1185 under a sign test).

`pass@1`
    Exact McNemar on the discordant pairs, at both the released code's 0.75 gate
    and the paper's 0.90.

Arms must cover the SAME seeds; an unpaired comparison is refused rather than
silently computed on whatever overlaps, because the two arms differ in which
seeds errored and a quiet intersection would not be the protocol either was run
under.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

JUDGE = "claude-opus-4-6"


def load(spec: str, judge: str) -> tuple[str, dict[int, dict]]:
    run_dir, _, arm = spec.partition(":")
    path = Path(run_dir) / f"rejudge_{judge}.json"
    if not path.exists():
        raise SystemExit(f"no re-judge file at {path}; run hexa_discovery.rejudge first")
    rows = json.loads(path.read_text())["rows"]
    picked = {r["seed"]: r for r in rows if not arm or r["arm"] == arm}
    if not picked:
        raise SystemExit(
            f"no rows with arm={arm!r} in {path}; present: "
            f"{sorted({r['arm'] for r in rows})}"
        )
    return f"{Path(run_dir).name}:{arm or '*'}", picked


def permutation_p(diffs: list[float]) -> float:
    """Exact two-sided sign-flip permutation p on the mean paired difference.

    Under the null the sign of each difference is exchangeable, so enumerating
    every assignment gives the exact distribution. Zero differences are kept:
    they are uninformative about direction but they do dilute the mean, which
    is the statistic being tested.
    """
    n = len(diffs)
    if n == 0:
        return float("nan")
    if n > 22:                       # 4M+ assignments; switch to sampling first
        raise SystemExit(f"{n} pairs is too many to enumerate exactly")
    observed = abs(sum(diffs))
    hits = sum(
        1 for signs in itertools.product((1, -1), repeat=n)
        if abs(sum(s * d for s, d in zip(signs, diffs))) >= observed - 1e-12
    )
    return hits / (2 ** n)


def mcnemar_p(a_pass: list[bool], b_pass: list[bool]) -> tuple[float, int, int]:
    """Exact two-sided McNemar on the discordant pairs."""
    b01 = sum(1 for x, y in zip(a_pass, b_pass) if not x and y)
    b10 = sum(1 for x, y in zip(a_pass, b_pass) if x and not y)
    n = b01 + b10
    if n == 0:
        return 1.0, b01, b10
    k = min(b01, b10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail), b01, b10


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--a", required=True, help="<run-dir>:<arm>  (the reference)")
    p.add_argument("--b", required=True, help="<run-dir>:<arm>  (the arm under test)")
    p.add_argument("--judge", default=JUDGE)
    p.add_argument("--per-seed", action="store_true")
    args = p.parse_args()

    name_a, rows_a = load(args.a, args.judge)
    name_b, rows_b = load(args.b, args.judge)
    seeds_a, seeds_b = set(rows_a), set(rows_b)
    if seeds_a != seeds_b:
        raise SystemExit(
            f"arms are not paired: {name_a} has {len(seeds_a)} seeds, "
            f"{name_b} has {len(seeds_b)}; "
            f"only in A {sorted(seeds_a - seeds_b)}, "
            f"only in B {sorted(seeds_b - seeds_a)}"
        )
    seeds = sorted(seeds_a)
    n = len(seeds)

    print(f"A = {name_a}\nB = {name_b}\nn = {n} paired seeds "
          f"({seeds[0]}-{seeds[-1]}), judge {args.judge}\n")

    if args.per_seed:
        print(f"{'seed':>5} {'expl A':>7} {'expl B':>7} {'rew A':>7} {'rew B':>7}")
        for s in seeds:
            a, b = rows_a[s], rows_b[s]
            print(f"{s:5d} {a['expl_independent']:7.2f} {b['expl_independent']:7.2f} "
                  f"{a['reward_independent']:+7.3f} {b['reward_independent']:+7.3f}")
        print()

    for metric, key in (("explanation", "expl_independent"),
                        ("reward", "reward_independent")):
        va = [rows_a[s][key] for s in seeds]
        vb = [rows_b[s][key] for s in seeds]
        diffs = [y - x for x, y in zip(va, vb)]
        up = sum(1 for d in diffs if d > 0)
        down = sum(1 for d in diffs if d < 0)
        mean_a, mean_b = sum(va) / n, sum(vb) / n
        print(f"{metric:12s} {mean_a:+.3f} -> {mean_b:+.3f}  "
              f"(delta {mean_b - mean_a:+.3f}; {up} up / {down} down)  "
              f"permutation p = {permutation_p(diffs):.4f}")

    for gate, key in (("0.75", "pass_indep_075"), ("0.90", "pass_indep_090")):
        pa = [bool(rows_a[s][key]) for s in seeds]
        pb = [bool(rows_b[s][key]) for s in seeds]
        pval, b01, b10 = mcnemar_p(pa, pb)
        print(f"pass@1 (>={gate}) {sum(pa):2d}/{n} -> {sum(pb):2d}/{n}  "
              f"(+{b01} / -{b10})  exact McNemar p = {pval:.4f}")


if __name__ == "__main__":
    main()
