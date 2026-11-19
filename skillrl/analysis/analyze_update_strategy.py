#!/usr/bin/env python
"""Report the skill-bank update-strategy ablation (catapult, Claude Sonnet 4.6).

Arms x replicates (run2/run4/run5 share K0 per replicate):
  full          : RESULTS/.../HeXA_catapult_claude_<run>/frozen_eval_seeds57-106 (existing)
  append_only   : data/ablation/update_strategy/catapult/append_only/<run>/eval_seeds57-106
  single_parent : data/ablation/update_strategy/catapult/single_parent/<run>/eval_seeds57-106

Per arm: held-out success (mean +- SAMPLE s.d. over replicates), mean turns,
injected tokens (cl100k block the eval actor receives), final bank size
(skills + mistakes). Paired bootstrap across seeds for full vs single_parent and
single_parent vs append_only: per seed, success averaged over that arm's
replicates; resample the 50 seeds with replacement.

    python -m skillrl.analysis.analyze_update_strategy [--boot 10000] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path

from skillrl.analysis.analyze_unconditioned_collection import injected_block, load_eval

PROJ = Path(__file__).resolve().parents[2]
FULL_BASE = PROJ / "skillrl/RESULTS/catapult_claude/HeXA_catapult"
ABL_BASE = PROJ / "skillrl/data/ablation/update_strategy/catapult"
RUNS = ("run2", "run4", "run5")
ARMS = ("full", "append_only", "single_parent")
SEEDS = list(range(57, 107))
LEVEL = "catapult"


def arm_paths(arm: str, run: str) -> tuple[Path, Path]:
    if arm == "full":
        d = FULL_BASE / f"HeXA_catapult_claude_{run}"
        return d / "skill_bank_evolving_17.json", d / "frozen_eval_seeds57-106" / LEVEL
    d = ABL_BASE / arm / run
    return d / "skill_bank_evolving_17.json", d / "eval_seeds57-106" / LEVEL


def crashed(t: dict) -> bool:
    return ((t.get("iterations") or 0) <= 1 and (t.get("elapsed_time") or 0) < 10) \
        or "error communicating" in json.dumps(t)


def sd(xs):
    return st.stdev(xs) if len(xs) > 1 else float("nan")


def paired_bootstrap(a: dict, b: dict, n_boot: int, rng: random.Random):
    seeds = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in seeds]
    obs = sum(d) / len(d)
    boots = []
    for _ in range(n_boot):
        smp = [d[rng.randrange(len(d))] for _ in d]
        boots.append(sum(smp) / len(smp))
    boots.sort()
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    p = min(1.0, 2 * min(sum(x <= 0 for x in boots), sum(x >= 0 for x in boots)) / n_boot)
    return {"n_seeds": len(seeds), "diff": obs, "ci95": [lo, hi], "p_two_sided": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    rep = {}        # (arm, run) -> metrics
    per_seed = {}   # arm -> seed -> list of successes over replicates
    for arm in ARMS:
        per_seed[arm] = {}
        for run in RUNS:
            bank, ev = arm_paths(arm, run)
            trajs = load_eval(ev, SEEDS)
            if not trajs:
                print(f"  [missing] {arm}/{run}: no eval at {ev}")
                continue
            bad = sorted(s for s, t in trajs.items() if crashed(t))
            ok = {s: t for s, t in trajs.items() if s not in bad}
            _, tok = injected_block(bank)
            b = json.loads(bank.read_text()) if bank.exists() else {}
            rep[(arm, run)] = {
                "n": len(ok), "missing": [s for s in SEEDS if s not in trajs], "crashed": bad,
                "success": sum(bool(t.get("success")) for t in ok.values()) / max(1, len(ok)),
                "turns": st.mean(t.get("iterations", 0) for t in ok.values()) if ok else float("nan"),
                "inj_tokens": tok,
                "bank_skills": len(b.get("level_skills", {}).get(LEVEL, [])),
                "bank_mistakes": len(b.get("level_mistakes", {}).get(LEVEL, [])),
            }
            for s, t in ok.items():
                per_seed[arm].setdefault(s, []).append(float(bool(t.get("success"))))

    print(f"\n{'arm/run':24s} {'n':>3s} {'succ%':>6s} {'turns':>6s} {'inj_tok':>7s} {'bank':>6s}  issues")
    for (arm, run), m in rep.items():
        issues = (f"crashed={m['crashed']} " if m["crashed"] else "") + \
                 (f"missing={len(m['missing'])}" if m["missing"] else "")
        print(f"{arm + '/' + run:24s} {m['n']:3d} {100 * m['success']:6.1f} {m['turns']:6.2f} "
              f"{m['inj_tokens']:7d} {m['bank_skills']:>2d}+{m['bank_mistakes']:<2d}  {issues}")

    summary = {}
    print(f"\n{'arm':14s} {'reps':>4s} {'success % (mean±sd)':>20s} {'turns':>12s} {'inj tokens':>16s} {'bank size':>12s}")
    for arm in ARMS:
        ms = [rep[(arm, r)] for r in RUNS if (arm, r) in rep]
        if not ms:
            continue
        g = lambda k: [m[k] for m in ms]
        size = [m["bank_skills"] + m["bank_mistakes"] for m in ms]
        summary[arm] = {
            "reps": len(ms),
            "success_mean": 100 * st.mean(g("success")), "success_sd": 100 * sd(g("success")),
            "turns_mean": st.mean(g("turns")), "turns_sd": sd(g("turns")),
            "tokens_mean": st.mean(g("inj_tokens")), "tokens_sd": sd(g("inj_tokens")),
            "bank_size_mean": st.mean(size), "bank_size_sd": sd(size),
            "bank_skills_mean": st.mean(g("bank_skills")), "bank_mistakes_mean": st.mean(g("bank_mistakes")),
        }
        s = summary[arm]
        print(f"{arm:14s} {s['reps']:4d} {s['success_mean']:9.1f} ± {s['success_sd']:5.1f}   "
              f"{s['turns_mean']:5.2f} ± {s['turns_sd']:4.2f} {s['tokens_mean']:7.0f} ± {s['tokens_sd']:5.0f} "
              f"{s['bank_size_mean']:5.1f} ± {s['bank_size_sd']:3.1f}")

    rng = random.Random(args.seed)
    tests = {}
    for a, b in (("full", "single_parent"), ("single_parent", "append_only")):
        if not per_seed.get(a) or not per_seed.get(b):
            continue
        ma = {s: st.mean(v) for s, v in per_seed[a].items()}
        mb = {s: st.mean(v) for s, v in per_seed[b].items()}
        r = paired_bootstrap(ma, mb, args.boot, rng)
        tests[f"{a}_vs_{b}"] = r
        print(f"\npaired bootstrap {a} - {b}: diff {100 * r['diff']:+.1f} pp, "
              f"95% CI [{100 * r['ci95'][0]:+.1f}, {100 * r['ci95'][1]:+.1f}], "
              f"p={r['p_two_sided']:.4f} (n_seeds={r['n_seeds']}, B={args.boot})")

    if args.json:
        args.json.write_text(json.dumps({"replicates": {f"{a}/{r}": m for (a, r), m in rep.items()},
                                         "arms": summary, "tests": tests}, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
