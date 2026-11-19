"""Ablation: final vs. original skill versions (within-level).

Tests whether *refining skill content* across training rounds helps, beyond
merely retaining the same collection of skill entries. We build a `rollback`
bank that is identical to the final bank in every structural way (which skill
IDs exist, their order, confidences, source_seeds, generations, and the mistake
records) but whose *actor-visible strategy text* (title / principle /
when_to_apply / example) is reverted to each skill's earliest traceable
ancestor across the evolving banks of the same run.

Because we keep confidences and ordering identical, confidence-ranked retrieval
selects exactly the same skill IDs for both banks — so both conditions receive
*corresponding* skills and only the wording differs. (A post-hoc audit against
the frozen final eval's `skills_used` confirms this; see `audit_injection`.)

Lineage is traced loosely: a final skill is linked back round-by-round to the
best-matching predecessor using (a) the trailing slot index of the
`cat_ev_<gen>_<slot>` id, (b) title/principle token overlap, and (c)
source_seeds overlap. The earliest skill reachable in that chain supplies the
rollback text. A final skill with no earlier appearance at all is left
unchanged and recorded (its rollback would be a no-op).

Outputs (under --out-dir):
    rollback_bank.json   frozen bank with reverted text
    lineage.json         per-skill ancestor chain + match scores
    exclusions.json      skills left unchanged, with reason
    text_diffs.md        human-readable final-vs-original diffs + token deltas

This does NOT evaluate. Use jobs/ablation_rollback_eval_array_catapult.sh to
eval the rollback bank and skillrl/analysis/analyze_rollback.py to report.

Example:
    python -m skillrl.distillation.rollback_skill_versions \\
        --run-dir skillrl/RESULTS/catapult_claude/HeXA_catapult/HeXA_catapult_claude_run4 \\
        --level catapult --all \\
        --out-dir skillrl/data/ablation/rollback/catapult/run4
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import asdict
from pathlib import Path

from skillrl.core.skill_bank import Skill, SkillBank

# Actor-visible fields that rollback reverts. Everything else (skill_id,
# confidence, generation, source_seeds, source_level, source_type) is metadata
# and is kept from the final bank so retrieval selection is unchanged.
TEXT_FIELDS = ("title", "principle", "when_to_apply", "example")

_STOP = {"the", "a", "an", "to", "of", "and", "or", "for", "with", "at", "in",
         "on", "is", "as", "by", "when", "then", "x", "y", "r"}


def estimate_tokens(text: str) -> int:
    """Deterministic token proxy (~4 chars/token), matches the jvi ablation."""
    return max(1, round(len(text or "") / 4))


def _trailing_slot(skill_id: str) -> str | None:
    """`cat_ev_9_000` -> '000'; `cat_sk_003` -> '003'; else None."""
    m = re.search(r"_(\d+)$", skill_id or "")
    return m.group(1) if m else None


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if w not in _STOP and len(w) > 1}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _match_score(final: Skill, cand: Skill) -> float:
    """Loose similarity in [0,1+]: slot bonus + title/principle/seed overlap.

    Loose on purpose (user: 'even if remotely related, that works') — the slot
    bonus alone clears the acceptance threshold so an unbroken id-slot chain is
    always followed, while content/seed overlap recovers cross-slot migrations.
    """
    slot_bonus = 0.6 if (_trailing_slot(final.skill_id)
                         and _trailing_slot(final.skill_id) == _trailing_slot(cand.skill_id)) else 0.0
    title = _jaccard(_tokens(final.title), _tokens(cand.title))
    princ = _jaccard(_tokens(final.principle), _tokens(cand.principle))
    seeds = _jaccard(set(final.source_seeds or []), set(cand.source_seeds or []))
    content = 0.5 * title + 0.35 * princ + 0.15 * seeds
    return slot_bonus + content


# Loose acceptance: a shared slot (0.6) always passes; a cross-slot link needs
# real content/seed overlap (>=0.18) to count.
_ACCEPT = 0.18


def load_round_banks(run_dir: Path, level: str) -> list[tuple[int, list[Skill]]]:
    """Return [(round_number, [skills for level])] for every skill_bank_evolving_N.json, ascending."""
    out = []
    for p in run_dir.glob("skill_bank_evolving_*.json"):
        m = re.search(r"evolving_(\d+)\.json$", p.name)
        if not m:
            continue
        n = int(m.group(1))
        bank = SkillBank.load(p)
        out.append((n, bank.level_skills.get(level, [])))
    out.sort(key=lambda t: t[0])
    return out


def trace_lineage(final_skill: Skill, rounds: list[tuple[int, list[Skill]]]) -> list[dict]:
    """Walk backward from the final round to the earliest reachable ancestor.

    Returns the chain newest->oldest as dicts {round, skill, score}. The last
    element is the earliest actionable version; its text supplies the rollback.
    """
    chain: list[dict] = []
    # rounds is ascending; the final skill lives in the last round.
    cur = final_skill
    # Index of the final round in `rounds`:
    last_round = rounds[-1][0]
    chain.append({"round": last_round, "skill": cur, "score": None})
    for rnum, skills in reversed(rounds[:-1]):  # earlier rounds, newest first
        if not skills:
            continue
        best, best_score = None, 0.0
        for cand in skills:
            s = _match_score(cur, cand)
            if s > best_score:
                best, best_score = cand, s
        if best is not None and best_score >= _ACCEPT:
            chain.append({"round": rnum, "skill": best, "score": round(best_score, 3)})
            cur = best  # continue tracing from the matched predecessor
    return chain


def build_rollback(run_dir: Path, level: str, only_slots: set[str] | None):
    rounds = load_round_banks(run_dir, level)
    if not rounds:
        raise SystemExit(f"no skill_bank_evolving_*.json under {run_dir}")
    final_round, final_skills = rounds[-1]

    final_bank = SkillBank.load(run_dir / f"skill_bank_evolving_{final_round}.json")
    rollback = copy.deepcopy(final_bank)

    lineage, exclusions, diffs = [], [], []
    rb_skills = rollback.level_skills[level]

    for i, fs in enumerate(final_skills):
        slot = _trailing_slot(fs.skill_id)
        chain = trace_lineage(fs, rounds)
        ancestor = chain[-1]["skill"]  # earliest reachable
        lineage.append({
            "final_id": fs.skill_id,
            "final_title": fs.title,
            "slot": slot,
            "ancestor_id": ancestor.skill_id,
            "ancestor_round": chain[-1]["round"],
            "ancestor_title": ancestor.title,
            "chain": [
                {"round": c["round"], "id": c["skill"].skill_id,
                 "title": c["skill"].title, "score": c["score"]}
                for c in chain
            ],
        })

        selected = (only_slots is None) or (slot in only_slots)
        no_earlier = ancestor.skill_id == fs.skill_id
        changed = any(getattr(ancestor, f) != getattr(fs, f) for f in TEXT_FIELDS)

        if not selected:
            exclusions.append({"final_id": fs.skill_id, "slot": slot,
                               "reason": "not selected (--only filter)"})
            continue
        if no_earlier or not changed:
            exclusions.append({
                "final_id": fs.skill_id, "slot": slot,
                "reason": "no earlier version (born at final round)" if no_earlier
                          else "text identical to earliest ancestor (no-op)"})
            continue

        # Revert ONLY actor-visible text; keep all metadata from the final skill.
        before = {f: getattr(fs, f) for f in TEXT_FIELDS}
        for f in TEXT_FIELDS:
            setattr(rb_skills[i], f, getattr(ancestor, f))
        after = {f: getattr(ancestor, f) for f in TEXT_FIELDS}

        tok_final = estimate_tokens(" ".join(before[f] for f in TEXT_FIELDS))
        tok_roll = estimate_tokens(" ".join(after[f] for f in TEXT_FIELDS))
        diffs.append({
            "final_id": fs.skill_id, "slot": slot,
            "ancestor_id": ancestor.skill_id, "ancestor_round": chain[-1]["round"],
            "before": before, "after": after,
            "tok_final": tok_final, "tok_rollback": tok_roll,
            "tok_delta": tok_roll - tok_final,
        })

    return rollback, lineage, exclusions, diffs


def audit_contradictions(rollback: SkillBank, level: str, diffs: list[dict]) -> list[str]:
    """Cheap check: a rolled-back skill shouldn't duplicate another entry's id or
    produce an empty actor field. (Semantic contradictions are flagged for human
    review via text_diffs.md; this just catches structural breakage.)"""
    warns = []
    seen = {}
    for s in rollback.level_skills.get(level, []):
        if s.skill_id in seen:
            warns.append(f"duplicate skill_id after rollback: {s.skill_id}")
        seen[s.skill_id] = s
        for f in TEXT_FIELDS[:3]:  # example may legitimately be empty
            if not (getattr(s, f) or "").strip():
                warns.append(f"empty {f} after rollback: {s.skill_id}")
    return warns


def write_diffs_md(path: Path, run_dir: Path, level: str, diffs: list[dict],
                   exclusions: list[dict], warns: list[str]) -> None:
    lines = [f"# Rollback diffs — {run_dir.name} / {level}", ""]
    lines.append(f"Rolled back **{len(diffs)}** skills; excluded **{len(exclusions)}**.")
    net = sum(d["tok_delta"] for d in diffs)
    lines.append(f"Net injected-token delta (rollback − final) across changed skills: "
                 f"**{net:+d}** tokens.")
    if warns:
        lines += ["", "## ⚠ audit warnings", *[f"- {w}" for w in warns]]
    lines += ["", "## Changed skills", ""]
    for d in diffs:
        lines.append(f"### {d['final_id']}  (slot {d['slot']}) ← {d['ancestor_id']} "
                     f"(round {d['ancestor_round']})  [{d['tok_delta']:+d} tok]")
        for f in TEXT_FIELDS:
            b, a = d["before"][f], d["after"][f]
            if b != a:
                lines += [f"- **{f}**",
                          f"    - final:    {b}",
                          f"    - rollback: {a}"]
        lines.append("")
    if exclusions:
        lines += ["## Excluded (left unchanged)", ""]
        for e in exclusions:
            lines.append(f"- {e['final_id']} (slot {e['slot']}): {e['reason']}")
    path.write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path,
                   help="HeXA run dir containing skill_bank_evolving_*.json")
    p.add_argument("--level", default="catapult")
    p.add_argument("--out-dir", required=True, type=Path)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true",
                   help="roll back every eligible skill together (default)")
    g.add_argument("--only", nargs="+", metavar="SLOT",
                   help="roll back only these trailing-slot indices, e.g. 000 005")
    args = p.parse_args()

    only_slots = set(args.only) if args.only else None
    rollback, lineage, exclusions, diffs = build_rollback(
        args.run_dir, args.level, only_slots)
    warns = audit_contradictions(rollback, args.level, diffs)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    rollback.save(out / "rollback_bank.json")
    (out / "lineage.json").write_text(json.dumps(lineage, indent=2))
    (out / "exclusions.json").write_text(json.dumps(exclusions, indent=2))
    write_diffs_md(out / "text_diffs.md", args.run_dir, args.level,
                   diffs, exclusions, warns)

    print(f"run: {args.run_dir.name}  level: {args.level}")
    print(f"  rolled back : {len(diffs)} skills")
    print(f"  excluded    : {len(exclusions)} skills")
    if warns:
        print(f"  ⚠ warnings  : {len(warns)} (see text_diffs.md)")
    print(f"  wrote       : {out}/rollback_bank.json, lineage.json, "
          f"exclusions.json, text_diffs.md")


if __name__ == "__main__":
    main()
