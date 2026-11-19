"""Raw-source transfer ablation: NO synthesis into the target space.

Cross-level HExA runs an evolver that rewrites source-level banks (down_to_earth,
two_body_problem, pass_the_parcel) into a target-level (catapult) bank. This
ablation skips that conversion: the SOURCE skills/mistakes are put into the
actor's prompt exactly as written, labelled with the puzzle they came from, and
the actor decides how to use them while solving each catapult seed. No teacher
call, no rewriting, no target-specific edits.

Budget (--budget):
  matched : same injected size as a synthesized bank (default 6 skills + 4
            mistakes). Skills = top-confidence round-robin across the sources
            (2 per source); mistakes = round-robin across sources in list order
            (the retriever takes the first N mistakes in list order).
  all     : every source skill and mistake (larger than the HExA budget).

The output is an ordinary SkillBank JSON (level_skills[target]) so it loads with
the existing runner; evaluate with --xl-framing (same provenance header as the
synthesized cross-level eval).
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from skillrl.core.skill_bank import SkillBank

DEFAULT_SOURCES = ["down_to_earth", "two_body_problem", "pass_the_parcel"]


def _pretty(level: str) -> str:
    return level.replace("_", " ").title()


def build(source_dir: Path, sources: list[str], target: str, budget: str,
          n_skills: int, n_mistakes: int) -> tuple[SkillBank, dict]:
    per_src_skills, per_src_mist = {}, {}
    for lv in sources:
        b = SkillBank.load(source_dir / f"{lv}.json")
        per_src_skills[lv] = sorted(b.level_skills.get(lv, []), key=lambda s: s.confidence, reverse=True)
        per_src_mist[lv] = list(b.level_mistakes.get(lv, []))

    def round_robin(lists: dict, k):
        out, i = [], 0
        while (k is None or len(out) < k) and any(i < len(v) for v in lists.values()):
            for lv in sources:
                if i < len(lists[lv]) and (k is None or len(out) < k):
                    out.append((lv, lists[lv][i]))
            i += 1
        return out

    k_s = None if budget == "all" else n_skills
    k_m = None if budget == "all" else n_mistakes
    sk_sel = round_robin(per_src_skills, k_s)
    mi_sel = round_robin(per_src_mist, k_m)

    bank = SkillBank()
    for lv, s in sk_sel:
        s = copy.deepcopy(s)
        s.title = f"({_pretty(lv)}) {s.title}"
        bank.level_skills.setdefault(target, []).append(s)
    for lv, m in mi_sel:
        m = copy.deepcopy(m)
        m.description = f"({_pretty(lv)}) {m.description}"
        bank.level_mistakes.setdefault(target, []).append(m)
    audit = {"target": target, "budget": budget, "sources": sources,
             "skills": [{"source": lv, "id": s.skill_id, "confidence": s.confidence} for lv, s in sk_sel],
             "mistakes": [{"source": lv, "id": m.mistake_id} for lv, m in mi_sel]}
    return bank, audit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir", default="skillrl/cross_banks")
    ap.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    ap.add_argument("--target", default="catapult")
    ap.add_argument("--budget", choices=["matched", "all"], default="matched")
    ap.add_argument("--skills", type=int, default=6)
    ap.add_argument("--mistakes", type=int, default=4)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    bank, audit = build(Path(a.source_dir), a.sources, a.target, a.budget, a.skills, a.mistakes)
    out = Path(a.output)
    bank.save(out)
    out.with_name(out.stem + "_audit.json").write_text(json.dumps(audit, indent=2))
    print(f"saved {out}: {len(bank.level_skills.get(a.target, []))} skills, "
          f"{len(bank.level_mistakes.get(a.target, []))} mistakes")


if __name__ == "__main__":
    main()
