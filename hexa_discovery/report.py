"""Paper-ready transfer table, following the DiscoverPhysics reporting protocol.

Matches the benchmark paper where it can:
  - judge: claude-opus-4-6, deliberately outside the set of agents under test
  - pass: normalized MSE < 0.1 AND explanation score >= 0.9 (the paper's number;
    scripts/run_benchmark.py hardcodes 0.75, so both are reported)
  - pass@k: sample k of the n attempts per world without replacement, a single
    passing attempt marks the world passed, averaged over worlds, 1000 resamples
  - mean +/- bootstrapped standard error, 5000 resamples

Deviations, stated rather than hidden: 10 experiment rounds vs the paper's 16,
and 2 held-out worlds vs their 22. Both make our numbers a lower bound.

The baseline can be POOLED across runs (--pool-baselines). The same no-bank
config at the same seed scored -0.20 in one run and +1.00 in another, so a single
run's baseline is not a stable reference; pooling uses every no-bank episode
available for a world and differences both bank arms against it.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path

PAPER_SCORE, CODE_SCORE, ERR = 0.90, 0.75, 0.1
N_BOOT, N_PASSK = 5000, 1000


def _boot_se(vals: list[float]) -> float:
    if len(vals) < 2:
        return float("nan")
    ms = [st.mean(random.choices(vals, k=len(vals))) for _ in range(N_BOOT)]
    return st.pstdev(ms)


def load(run: Path) -> list[dict]:
    f = run / "rejudge_claude-opus-4-6.json"
    if not f.exists():
        raise SystemExit(f"missing {f}; run hexa_discovery.rejudge first")
    return json.loads(f.read_text())["rows"]


def passes(r: dict, thr: float) -> bool:
    nm = r.get("norm_mse")
    return bool(nm is not None and nm < ERR and r["expl_independent"] >= thr)


def pass_at_k(by_world: dict, k: int, thr: float) -> float:
    """Expected fraction of worlds passed from k attempts (the paper's estimator)."""
    tot = 0.0
    for _ in range(N_PASSK):
        got = 0
        for rs in by_world.values():
            if len(rs) < k:
                continue
            got += any(passes(r, thr) for r in random.sample(rs, k))
        tot += got / max(1, len(by_world))
    return tot / N_PASSK


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--pool-baselines", action="store_true")
    args = p.parse_args()
    random.seed(0)

    runs = {Path(r).name: load(Path(r)) for r in args.runs}

    pooled: dict = {}
    if args.pool_baselines:
        for rows in runs.values():
            for r in rows:
                if r["arm"] == "baseline":
                    pooled.setdefault(r["world"], []).append(r)

    for name, rows in runs.items():
        print(f"\n{'='*78}\n{name}\n{'='*78}")
        worlds = sorted({r["world"] for r in rows})
        print(f"{'world':8s} {'arm':10s} {'n':>2s} {'reward':>16s} {'expl':>6s} "
              f"{'p@.75':>6s} {'p@.90':>6s}")
        deltas = []
        for w in worlds:
            bank = [r for r in rows if r["world"] == w and r["arm"] == "with_bank"]
            base = (pooled.get(w) if args.pool_baselines
                    else [r for r in rows if r["world"] == w and r["arm"] == "baseline"])
            for lab, rs in (("with_bank", bank), ("baseline", base)):
                rw = [r["reward_independent"] for r in rs]
                print(f"{w:8s} {lab:10s} {len(rs):2d} {st.mean(rw):+8.3f} ± {_boot_se(rw):.3f} "
                      f"{st.mean(r['expl_independent'] for r in rs):6.2f} "
                      f"{sum(passes(r, CODE_SCORE) for r in rs):3d}/{len(rs):<2d} "
                      f"{sum(passes(r, PAPER_SCORE) for r in rs):3d}/{len(rs):<2d}")
            rb = [r["reward_independent"] for r in bank]
            rz = [r["reward_independent"] for r in base]
            d = st.mean(rb) - st.mean(rz)
            se = (_boot_se(rb) ** 2 + _boot_se(rz) ** 2) ** 0.5
            deltas.append(d)
            print(f"{'':8s} {'DELTA':10s}    {d:+8.3f} ± {se:.3f}   "
                  f"= {d/se:.1f} sigma" if se == se else "")
        print(f"\n  mean delta over {len(deltas)} world(s): {st.mean(deltas):+.3f}   "
              f"({sum(1 for d in deltas if d > 0)}/{len(deltas)} positive)")

        for thr, lab in ((PAPER_SCORE, "0.90 (paper)"), (CODE_SCORE, "0.75 (code)")):
            bw = {w: [r for r in rows if r["world"] == w and r["arm"] == "with_bank"]
                  for w in worlds}
            zw = ({w: pooled[w] for w in worlds} if args.pool_baselines else
                  {w: [r for r in rows if r["world"] == w and r["arm"] == "baseline"]
                   for w in worlds})
            ks = [k for k in (1, 3, 5) if k <= min(len(v) for v in bw.values())]
            print(f"  pass@k, threshold {lab}: " + "  ".join(
                f"k={k}: bank {pass_at_k(bw,k,thr):.0%} vs base {pass_at_k(zw,k,thr):.0%}"
                for k in ks))


if __name__ == "__main__":
    main()
