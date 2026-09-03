"""Sanity-check a cross-level synthesised skill bank against its audit sidecar.

Verifies that every cited `(source_level, skill_id)` in the audit resolves to
a real entry in the corresponding source bank, and flags violations of the
synthesis prompt's hard constraints (empty citations, missing rationale,
numeric coordinate leakage in `example`, weak corroboration on high-confidence
skills).

Exit code:
- 0 if no errors (warnings are allowed).
- 1 if any error-severity issue is found.

Usage:

    python -m skillrl.distillation.check_synthesis_audit \\
        --audit skillrl/data/cross_level/catapult/skill_bank_xl_catapult_audit.json

The audit sidecar records ``source_bank_paths`` so the source banks are
loaded automatically. To override (e.g. banks have moved), pass
``--source-bank level=path`` repeatedly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from skillrl.core.skill_bank import SkillBank


# A digit followed by a digit/dot pattern catches "0.5", "1.5", "x=0.5", "r=2".
# Also catches integer coordinates like "y=3". Bare "3 seconds" inside a
# physics description is rare but possible — treated as warning, not error.
_NUMERIC_LEAK_RE = re.compile(r"\b-?\d+(?:\.\d+)?\b")


def _load_source_banks(
    paths: dict[str, Path],
) -> dict[str, set[str]]:
    """Return {source_level: set of skill_ids found in that bank}."""
    valid_ids: dict[str, set[str]] = {}
    for level, path in paths.items():
        bank = SkillBank.load(path)
        skill_ids = {s.skill_id for s in bank.level_skills.get(level, [])}
        # Mistakes get their own ID space; the audit only cites skills,
        # but include mistake_ids so a teacher that cites a mistake by ID
        # doesn't get flagged as hallucinated.
        mistake_ids = {m.mistake_id for m in bank.level_mistakes.get(level, [])}
        valid_ids[level] = skill_ids | mistake_ids
    return valid_ids


def _check_entry(
    entry: dict,
    kind: str,
    valid_ids: dict[str, set[str]],
    errors: list[str],
    warnings: list[str],
) -> None:
    """Check a single audit entry (skill or mistake) and append findings."""
    label = entry.get("skill_id") or entry.get("mistake_id") or "<unknown>"
    citations = entry.get("source_skills", [])

    if not citations:
        errors.append(f"[{kind} {label}] empty source_skills (hard constraint)")
        return

    cited_levels: set[str] = set()
    for cite in citations:
        if not isinstance(cite, dict):
            errors.append(f"[{kind} {label}] malformed citation entry: {cite!r}")
            continue
        src_level = cite.get("source_level")
        src_id = cite.get("skill_id")
        if not src_level or not src_id:
            errors.append(
                f"[{kind} {label}] citation missing source_level/skill_id: {cite!r}"
            )
            continue
        if src_level not in valid_ids:
            errors.append(
                f"[{kind} {label}] cites unknown source_level '{src_level}'. "
                f"Known: {sorted(valid_ids)}"
            )
            continue
        if src_id not in valid_ids[src_level]:
            errors.append(
                f"[{kind} {label}] hallucinated citation: "
                f"({src_level}, {src_id}) not found in source bank"
            )
            continue
        cited_levels.add(src_level)

    if not entry.get("transfer_rationale", "").strip():
        warnings.append(f"[{kind} {label}] empty transfer_rationale")

    if kind == "skill":
        confidence = float(entry.get("confidence", 0.0))
        if confidence >= 0.7 and len(cited_levels) < 2:
            warnings.append(
                f"[{kind} {label}] confidence={confidence:.2f} but only "
                f"{len(cited_levels)} source level(s) cited "
                f"(prompt requires ≥2 for ≥0.7 confidence)"
            )


def _check_numeric_leak_in_examples(
    bank: SkillBank, target_level: str, warnings: list[str]
) -> None:
    """Warn if any synthesised skill's `example` contains a number."""
    for skill in bank.level_skills.get(target_level, []):
        if skill.example and _NUMERIC_LEAK_RE.search(skill.example):
            warnings.append(
                f"[skill {skill.skill_id}] example contains numeric values "
                f"(prompt forbids invented coordinates): {skill.example!r}"
            )


def _citation_stats(audit: dict) -> dict:
    """Compute per-source-level citation counts and top-cited source skills."""
    per_level: dict[str, int] = {}
    per_skill: dict[str, int] = {}
    for entry in audit.get("skills", []) + audit.get("mistakes", []):
        for cite in entry.get("source_skills", []) or []:
            if not isinstance(cite, dict):
                continue
            lvl = cite.get("source_level", "?")
            sid = cite.get("skill_id", "?")
            per_level[lvl] = per_level.get(lvl, 0) + 1
            key = f"{lvl}:{sid}"
            per_skill[key] = per_skill.get(key, 0) + 1
    return {"per_level": per_level, "per_skill": per_skill}


def check_synthesis(
    audit_path: Path,
    bank_path: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
) -> int:
    audit = json.loads(audit_path.read_text())
    target_level = audit["target_level"]

    if bank_path is None:
        # Default: same directory, conventional filename.
        bank_path = audit_path.parent / f"skill_bank_xl_{target_level}.json"
    bank = SkillBank.load(bank_path)

    declared_paths = {
        lvl: Path(p) for lvl, p in audit.get("source_bank_paths", {}).items()
    }
    if source_overrides:
        declared_paths.update(source_overrides)

    missing = [str(p) for p in declared_paths.values() if not p.exists()]
    if missing:
        print("[FATAL] Source bank file(s) not found:")
        for m in missing:
            print(f"  - {m}")
        print("Pass --source-bank level=path to override.")
        return 1

    valid_ids = _load_source_banks(declared_paths)

    errors: list[str] = []
    warnings: list[str] = []

    audit_skill_ids = {s.get("skill_id") for s in audit.get("skills", [])}
    bank_skill_ids = {s.skill_id for s in bank.level_skills.get(target_level, [])}
    only_in_audit = audit_skill_ids - bank_skill_ids
    only_in_bank = bank_skill_ids - audit_skill_ids
    if only_in_audit:
        errors.append(
            f"[bank/audit mismatch] skills in audit but not bank: {sorted(only_in_audit)}"
        )
    if only_in_bank:
        errors.append(
            f"[bank/audit mismatch] skills in bank but not audit: {sorted(only_in_bank)}"
        )

    for entry in audit.get("skills", []):
        _check_entry(entry, "skill", valid_ids, errors, warnings)
    for entry in audit.get("mistakes", []):
        _check_entry(entry, "mistake", valid_ids, errors, warnings)

    _check_numeric_leak_in_examples(bank, target_level, warnings)

    stats = _citation_stats(audit)

    print("=" * 70)
    print(f"Cross-level synthesis audit — target: {target_level}")
    print("=" * 70)
    print(f"  Bank:  {bank_path}")
    print(f"  Audit: {audit_path}")
    print(f"  Teacher: {audit.get('teacher_model', '?')}")
    print(f"  Source levels: {audit.get('source_levels', [])}")
    print(f"  Synthesised: {len(audit.get('skills', []))} skills, "
          f"{len(audit.get('mistakes', []))} mistakes")
    print()
    print("  Source-bank skill counts (input):")
    for lvl, n in audit.get("source_bank_skill_counts", {}).items():
        print(f"    {lvl}: {n}")
    print()
    print("  Citation distribution (target → source):")
    for lvl, count in sorted(
        stats["per_level"].items(), key=lambda kv: kv[1], reverse=True
    ):
        print(f"    {lvl}: {count} citations")
    if stats["per_skill"]:
        print("\n  Top-cited source skills:")
        top = sorted(stats["per_skill"].items(), key=lambda kv: kv[1], reverse=True)[:5]
        for key, n in top:
            print(f"    {key}: cited {n}×")

    print()
    if errors:
        print(f"[ERRORS] {len(errors)}:")
        for e in errors:
            print(f"  ✗ {e}")
    else:
        print("[ERRORS] none")

    if warnings:
        print(f"\n[WARNINGS] {len(warnings)}:")
        for w in warnings:
            print(f"  ! {w}")
    else:
        print("\n[WARNINGS] none")

    print()
    if errors:
        print(f"FAIL — {len(errors)} error(s), {len(warnings)} warning(s).")
        return 1
    print(f"PASS — {len(warnings)} warning(s). Review warnings before reporting results.")
    return 0


def _parse_source_bank_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"--source-bank expects 'level=path', got: {raw!r}"
        )
    level, path = raw.split("=", 1)
    return level.strip(), Path(path.strip())


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity-check a cross-level synthesised skill bank.",
    )
    parser.add_argument("--audit", required=True, type=Path,
                        help="Path to *_audit.json sidecar emitted by cross_level_synthesis.")
    parser.add_argument("--bank", type=Path, default=None,
                        help="Path to the synthesised SkillBank JSON "
                             "(default: same dir, skill_bank_xl_<target>.json).")
    parser.add_argument("--source-bank", action="append", type=_parse_source_bank_arg,
                        default=[],
                        help="Override source bank path. Format: '<level>=<path>'. "
                             "Repeatable. Defaults read from audit.source_bank_paths.")
    args = parser.parse_args()

    overrides = dict(args.source_bank) if args.source_bank else None
    sys.exit(check_synthesis(args.audit, args.bank, overrides))


if __name__ == "__main__":
    _main()
