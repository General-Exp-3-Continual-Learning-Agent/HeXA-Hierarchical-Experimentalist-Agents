"""Exhaustive adjacent-round composition scan over the evolving skill banks.

PURE COMPOSITION (what this looks for), done lexically so it needs no model:
  For each round k and each skill C in bank k:
    P  = C's predecessor in bank k-1 (title-weighted best match)
    new = word n-grams in C that are ABSENT from P      <- content C did not inherit
    D  = any OTHER skill in bank k-1 containing those n-grams
  C is a composition candidate when it has a dominant predecessor P *and*
  content traceable to a different earlier skill D. That is clauses A1-A3 of
  the audit definition, checked mechanically.

Two shapes are reported:
  INHERIT_PLUS  C continues P and imports content from D   (>=2 parents)
  NEW_FROM_N    C has no predecessor but its content is covered by >=2 earlier
                skills (a skill assembled out of existing ones)

The donor may survive or be retired; both count. Each hit records the shared
n-grams so a human can confirm clause A3 by reading the three texts.

NOTE on matching: predecessor identification MUST weight the title. Flat
bag-of-words Jaccard put run2's r15 "Use r=1.5 as Starting Probe..." closer to
"Use trace_green_ball to Gate Full Simulation Runs" (0.281) than to its own
r14 self (0.256), because the body was heavily rewritten that round.

Usage:
    python -m skillrl.analysis.find_pure_composition                # all runs
    python -m skillrl.analysis.find_pure_composition --run run5 --min-shared 2
Outputs:
    skillrl/data/ablation/lineage_audit/pure_composition_<run>.json
    ICLR/notes/pure_composition_candidates.md
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
BASE = PROJ / "skillrl/RESULTS/catapult_claude/HeXA_catapult"
RUNS = ("run2", "run4", "run5")
LEVEL = "catapult"
FIELDS = ("title", "principle", "when_to_apply", "example")
STOP = set("the a an to of and or for with at in on is as by when then it this "
           "that be are if from can will not its into than".split())


def run_dir(run: str) -> Path:
    return BASE / f"HeXA_catapult_claude_{run}"


def skills(run: str, n: int) -> list[dict]:
    p = run_dir(run) / f"skill_bank_evolving_{n}.json"
    return json.load(open(p))["level_skills"].get(LEVEL, []) if p.exists() else []


def n_rounds(run: str) -> int:
    ns = [int(m.group(1)) for p in run_dir(run).glob("skill_bank_evolving_*.json")
          if (m := re.search(r"evolving_(\d+)\.json$", p.name))]
    return max(ns) if ns else 0


def _w(text: str) -> list[str]:
    t = (text or "").lower().replace("→", " ").replace("—", " ")
    return [w for w in re.findall(r"[a-z0-9=.\-]+", t) if w not in STOP]


def body_words(s: dict) -> list[str]:
    return _w(" ".join(str(s.get(f) or "") for f in FIELDS))


def jac(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b)) if (a or b) else 0.0


def match(a: dict, b: dict) -> float:
    """Title-weighted similarity. The title is the stable part of a skill's
    identity; the body gets rewritten wholesale some rounds."""
    return (0.60 * jac(set(_w(a.get("title"))), set(_w(b.get("title"))))
            + 0.40 * jac(set(body_words(a)), set(body_words(b))))


def grams(s: dict, n: int = 4) -> set[str]:
    w = body_words(s)
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


PRED_TAU = 0.30


def scan(run: str, min_shared: int, gram_n: int) -> list[dict]:
    hits: list[dict] = []
    N = n_rounds(run)
    for k in range(2, N + 1):
        ps, cs = skills(run, k - 1), skills(run, k)
        if not ps or not cs:
            continue
        pg = [grams(p, gram_n) for p in ps]
        cg_next = cs
        for c in cs:
            scores = [match(p, c) for p in ps]
            best = max(range(len(ps)), key=lambda i: scores[i]) if ps else None
            cgrams = grams(c, gram_n)

            if best is not None and scores[best] >= PRED_TAU:
                new = cgrams - pg[best]
                if not new:
                    continue
                for j, p2 in enumerate(ps):
                    if j == best:
                        continue
                    shared = new & pg[j]
                    if len(shared) < min_shared:
                        continue
                    survives = any(match(p2, x) > 0.50 for x in cg_next)
                    hits.append(dict(
                        run=run, kind="INHERIT_PLUS", round_from=k - 1, round_to=k,
                        child=c["title"], child_id=c.get("skill_id"),
                        predecessor=ps[best]["title"], predecessor_id=ps[best].get("skill_id"),
                        pred_score=round(scores[best], 3),
                        donor=p2["title"], donor_id=p2.get("skill_id"),
                        donor_survives=survives, n_shared=len(shared),
                        shared=sorted(shared, key=len, reverse=True)[:8]))
            else:
                donors = []
                for j, p2 in enumerate(ps):
                    shared = cgrams & pg[j]
                    if len(shared) >= min_shared:
                        donors.append((p2, len(shared),
                                       sorted(shared, key=len, reverse=True)[:5]))
                if len(donors) >= 2:
                    donors.sort(key=lambda t: -t[1])
                    hits.append(dict(
                        run=run, kind="NEW_FROM_N", round_from=k - 1, round_to=k,
                        child=c["title"], child_id=c.get("skill_id"),
                        predecessor=None, predecessor_id=None, pred_score=None,
                        donor=" + ".join(d[0]["title"] for d in donors[:3]),
                        donor_id=[d[0].get("skill_id") for d in donors[:3]],
                        donor_survives=None,
                        n_shared=sum(d[1] for d in donors),
                        shared=[g for d in donors[:3] for g in d[2]][:8]))
    hits.sort(key=lambda h: (-h["n_shared"], h["round_to"]))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=RUNS)
    ap.add_argument("--min-shared", type=int, default=3)
    ap.add_argument("--gram", type=int, default=4)
    a = ap.parse_args()

    out_dir = PROJ / "skillrl/data/ablation/lineage_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [a.run] if a.run else list(RUNS)
    allhits, md = {}, []

    md.append("# Pure-composition candidates (adjacent rounds, lexical scan)\n")
    md.append(f"Generated by `skillrl/analysis/find_pure_composition.py` "
              f"(min_shared={a.min_shared}, {a.gram}-grams).\n")
    md.append("A candidate is a skill at round k that **continues a predecessor at k-1** "
              "*and* contains content **absent from that predecessor** but present in a "
              "**different** skill at k-1 (`INHERIT_PLUS`), or a skill with no predecessor "
              "whose content is covered by two or more earlier skills (`NEW_FROM_N`).\n")
    md.append("These are candidates, not confirmed events: read the three texts and check "
              "the shared phrase really is absent from the predecessor.\n")

    for run in runs:
        h = scan(run, a.min_shared, a.gram)
        allhits[run] = h
        json.dump(h, open(out_dir / f"pure_composition_{run}.json", "w"), indent=1)
        print(f"{run}: {len(h)} candidates "
              f"({sum(1 for x in h if x['kind']=='INHERIT_PLUS')} INHERIT_PLUS, "
              f"{sum(1 for x in h if x['kind']=='NEW_FROM_N')} NEW_FROM_N)")
        md.append(f"\n## {run} — {len(h)} candidates\n")
        for x in h:
            surv = ("donor survives" if x["donor_survives"] else
                    "donor retired" if x["donor_survives"] is False else "n/a")
            md.append(f"\n### r{x['round_from']}->{x['round_to']} · {x['kind']} · "
                      f"shared={x['n_shared']} · {surv}\n")
            md.append(f"- **C** ({x['child_id']}): {x['child']}")
            if x["predecessor"]:
                md.append(f"- **P** ({x['predecessor_id']}, match={x['pred_score']}): "
                          f"{x['predecessor']}")
            md.append(f"- **D** ({x['donor_id']}): {x['donor']}")
            md.append("- shared phrases absent from P:")
            for g in x["shared"]:
                md.append(f"  - `{g}`")

    notes = PROJ / "ICLR/notes/pure_composition_candidates.md"
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_text("\n".join(md))
    total = sum(len(v) for v in allhits.values())
    print(f"\nTOTAL {total} candidates")
    print(f"  json: {out_dir}/pure_composition_<run>.json")
    print(f"  md:   {notes}")


if __name__ == "__main__":
    main()
