"""Loose composition scan over the FULL prior history of a run.

For every skill C appearing at round k:
  1. Build C's own lineage L  (all earlier versions of the same skill, matched
     across rounds by title-weighted similarity).
  2. own = union of word n-grams over every version of L at rounds < k.
     This is everything C could have inherited from itself.
  3. new = grams(C) - own        <- content C did NOT inherit from its own past
  4. For every OTHER lineage M with at least one version before round k,
     m = union of M's grams over rounds < k.
     shared = new & m            <- content traceable to a different skill

C is a composition event at round k when >=1 other lineage contributes
>= MIN_SHARED n-grams. Donors may come from ANY earlier round, may still be
alive, and may contribute partially — the loosened definition.

Because `own` accumulates, an import is counted only in the round it first
appears; later rounds see it as inherited. No double counting.

Caveat to state when reporting: this cannot distinguish content that came from
another SKILL from content that came from round k's TRAJECTORIES and happened
to match an earlier skill. Add the trajectory-exclusion check before quoting a
number as causal.

Usage:
    python -m skillrl.analysis.find_composition_history --run run2
    python -m skillrl.analysis.find_composition_history            # all runs
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
BASE = PROJ / "skillrl/RESULTS/catapult_claude/HeXA_catapult"
RUNS = ("run2", "run4", "run5")
LEVEL = "catapult"
FIELDS = ("title", "principle", "when_to_apply", "example")
STOP = set("the a an to of and or for with at in on is as by when then it this "
           "that be are if from can will not its into than".split())
LINEAGE_TAU = 0.30


def skills(run, n):
    p = BASE / f"HeXA_catapult_claude_{run}" / f"skill_bank_evolving_{n}.json"
    return json.load(open(p))["level_skills"].get(LEVEL, []) if p.exists() else []


def n_rounds(run):
    d = BASE / f"HeXA_catapult_claude_{run}"
    ns = [int(m.group(1)) for p in d.glob("skill_bank_evolving_*.json")
          if (m := re.search(r"evolving_(\d+)\.json$", p.name))]
    return max(ns) if ns else 0


def _w(t):
    t = (t or "").lower().replace("→", " ").replace("—", " ")
    return [w for w in re.findall(r"[a-z0-9=.\-]+", t) if w not in STOP]


def body(s):
    return _w(" ".join(str(s.get(f) or "") for f in FIELDS))


def jac(a, b):
    return len(a & b) / max(1, len(a | b)) if (a or b) else 0.0


def match(a, b):
    return (0.60 * jac(set(_w(a.get("title"))), set(_w(b.get("title"))))
            + 0.40 * jac(set(body(a)), set(body(b))))


def grams(s, n=4):
    w = body(s)
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def build_lineages(run, N, gram_n):
    """[{versions: [(round, skill)], title: str}] — greedy across rounds."""
    lins = []
    for k in range(1, N + 1):
        for s in skills(run, k):
            best, bs = None, 0.0
            for L in lins:
                if L["versions"][-1][0] == k:      # already has a member this round
                    continue
                v = match(L["versions"][-1][1], s)
                if v > bs:
                    best, bs = L, v
            if best is not None and bs >= LINEAGE_TAU:
                best["versions"].append((k, s))
                best["title"] = s["title"]
            else:
                lins.append({"versions": [(k, s)], "title": s["title"]})
    for L in lins:
        L["grams_upto"] = {}
        acc = set()
        for (k, s) in L["versions"]:
            L["grams_upto"][k] = set(acc)          # grams strictly BEFORE round k
            acc |= grams(s, gram_n)
        L["rounds"] = {k for k, _ in L["versions"]}
        L["acc_all"] = acc
    return lins


def scan(run, min_shared, gram_n):
    N = n_rounds(run)
    lins = build_lineages(run, N, gram_n)
    events = []
    for li, L in enumerate(lins):
        for (k, s) in L["versions"]:
            own = L["grams_upto"][k]
            if not own and k > 1:
                own = set()
            new = grams(s, gram_n) - own
            if not new:
                continue
            donors = []
            for mj, M in enumerate(lins):
                if mj == li:
                    continue
                m = set()
                for (kk, ss) in M["versions"]:
                    if kk < k:
                        m |= grams(ss, gram_n)
                if not m:
                    continue
                shared = new & m
                if len(shared) >= min_shared:
                    alive = any(kk >= k for kk, _ in M["versions"])
                    donors.append(dict(donor=M["title"], donor_alive=alive,
                                       n=len(shared),
                                       sample=sorted(shared, key=len, reverse=True)[:4]))
            if donors:
                donors.sort(key=lambda d: -d["n"])
                events.append(dict(
                    run=run, round=k, child=s["title"], child_id=s.get("skill_id"),
                    is_first_version=(k == L["versions"][0][0]),
                    n_donors=len(donors), total_shared=sum(d["n"] for d in donors),
                    donors=donors[:4]))
    events.sort(key=lambda e: (-e["total_shared"], e["round"]))
    return lins, events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=RUNS)
    ap.add_argument("--min-shared", type=int, default=3)
    ap.add_argument("--gram", type=int, default=4)
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()
    out = PROJ / "skillrl/data/ablation/lineage_audit"
    out.mkdir(parents=True, exist_ok=True)

    for run in ([a.run] if a.run else list(RUNS)):
        lins, ev = scan(run, a.min_shared, a.gram)
        json.dump(ev, open(out / f"composition_history_{run}.json", "w"), indent=1)
        multi = [e for e in ev if e["n_donors"] >= 2]
        newsk = [e for e in ev if e["is_first_version"]]
        print(f"\n######## {run}: {n_rounds(run)} rounds, {len(lins)} distinct skill lineages")
        print(f"  composition events (>= {a.min_shared} shared {a.gram}-grams "
              f"from a different lineage): {len(ev)}")
        print(f"    with >= 2 distinct donors : {len(multi)}")
        print(f"    at the skill's FIRST round: {len(newsk)}")
        for e in ev[:a.top]:
            tag = "NEW" if e["is_first_version"] else "   "
            print(f"  [{tag}] r{e['round']:<2} donors={e['n_donors']} "
                  f"shared={e['total_shared']:<3} {e['child'][:62]}")
            for d in e["donors"][:2]:
                al = "alive" if d["donor_alive"] else "gone "
                print(f"            <- ({d['n']:>2}, {al}) {d['donor'][:60]}")
        print(f"  -> {out}/composition_history_{run}.json")


if __name__ == "__main__":
    main()
