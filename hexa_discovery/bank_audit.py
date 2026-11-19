"""Audit a skill bank for law-form leakage.

Input redaction (`traj_adapter.redact`) is the first layer: it strips the
answer's name out of everything the teacher reads. It cannot be the only layer,
for two reasons — the blocklist is necessarily incomplete, and the teacher can
*infer* a model class from redacted observations and then name it itself.

So the bank is **scanned, not scrubbed**. Scrubbing a skill would mangle prose
that cannot be regenerated and would hide the problem; a flag tells you which
run to distrust and which term to add to the blocklist. A flagged run is not
automatically invalid — a skill may name a form belonging to a world that is in
the TRAIN set, which leaks nothing. It is invalid when the named form belongs to
a HELD-OUT world, which is why `audit_bank` takes the held-out list.
"""

from __future__ import annotations

import re

from typing import Iterable, Optional

from hexa_discovery.traj_adapter import scan_for_law_forms

#: Terms that identify a particular world's answer, for attributing a flag to
#: the world it would leak. Only worlds whose answer has a nameable form appear;
#: a hit on a term not listed here is reported without attribution.
WORLD_LAW_TERMS: dict[str, tuple[str, ...]] = {
    "yukawa": ("yukawa", "screened", "screening length", "exp(-", "helmholtz"),
    "gravity": ("newtonian", "poisson"),
    "coulomb_easy": ("coulomb", "inverse-square", "inverse square"),
    # circle and fractional share the SAME operator (the paper calls circle
    # "Fractional gravitation with ring of particles"), so a skill naming the
    # fractional Laplacian gives away BOTH. Attributing it to `fractional` alone
    # would let it leak into a held-out `circle` undetected.
    "fractional": ("fractional laplacian", "riesz", "anomalous diffusion"),
    "circle": ("fractional laplacian", "riesz", "anomalous diffusion"),
    "extra_dimensions": ("kaluza", "extra dimension", "compactification"),
    "dark_matter": ("dark matter", "dark halo"),
    "ether": ("ether", "luminiferous", "preferred direction"),
    "hubble": ("hubble", "hubble flow"),
    "oscillator": ("sinusoid", "periodicity", "period", "oscillat",
                   "time-depend", "time depend", "time-vary", "time modulat",
                   "sign-flip", "reverses sign", "phase",
                   "time-dependent coupling", "sinusoidally varying",
                   "periodically revers"),
}


def _skill_text(skill) -> str:
    parts = [
        getattr(skill, "title", "") or "",
        getattr(skill, "principle", "") or "",
        getattr(skill, "when_to_apply", "") or "",
        getattr(skill, "example", "") or "",
    ]
    return "\n".join(parts)


def _mistake_text(mistake) -> str:
    parts = [
        getattr(mistake, "description", "") or "",
        getattr(mistake, "why_it_happens", "") or "",
        getattr(mistake, "how_to_avoid", "") or "",
    ]
    return "\n".join(parts)


def _normalise(text: str) -> str:
    """Lowercase and collapse every separator run to one space.

    `LAW_FORM_RE` matches separator variants (`fractional[\\s_-]*laplacian`) and
    `scan_for_law_forms` returns the matched text verbatim, so a bank that wrote
    "fractional-Laplacian" yields the term `fractional-laplacian`. The needles
    below are stored with spaces. Raw substring matching then fails in BOTH
    directions -- `"fractional laplacian" in "fractional-laplacian"` is False and
    so is the reverse -- and the hit is detected but silently unattributed,
    which drops it out of `leaks_held_out` and makes `n_critical` zero.

    That is not hypothetical. It is exactly what happened to the within-circle
    run: `dis_ev_5_000` named the fractional Laplacian, the only law-form hit in
    circle's frozen bank, and the audit reported it as
    `[train-world only] (coulomb_easy)` -- attributed to a train world through
    the co-occurring `inverse-square` -- instead of `[LEAKS HELD-OUT] (circle)`.
    Measured across `WORLD_LAW_TERMS`, 22 (needle, separator) combinations lost
    attribution this way, spanning yukawa, coulomb_easy, fractional, circle,
    extra_dimensions, dark_matter, ether and oscillator.
    """
    return re.sub(r"[\s_-]+", " ", text.lower()).strip()


def attribute(term: str) -> list[str]:
    """Worlds whose answer the term would name. Empty when unattributable.

    Both sides are separator-normalised; see `_normalise`.
    """
    t = _normalise(term)
    return sorted(
        world for world, needles in WORLD_LAW_TERMS.items()
        if any(_normalise(n) in t or t in _normalise(n) for n in needles)
    )


def audit_bank(
    bank,
    level: str,
    held_out_worlds: Optional[Iterable[str]] = None,
) -> dict:
    """Scan every skill and mistake in `level` for law-form terms.

    Returns a JSON-serialisable report: per-entry hits, and the subset that
    names a HELD-OUT world's answer — the only ones that invalidate a transfer
    number.
    """
    held = set(held_out_worlds or ())
    findings: list[dict] = []

    entries = [("skill", s, _skill_text(s), getattr(s, "skill_id", "?"))
               for s in bank.level_skills.get(level, [])]
    entries += [("mistake", m, _mistake_text(m), getattr(m, "mistake_id", "?"))
                for m in bank.level_mistakes.get(level, [])]

    for kind, _entry, text, entry_id in entries:
        terms = scan_for_law_forms(text)
        if not terms:
            continue
        worlds = sorted({w for t in terms for w in attribute(t)})
        findings.append({
            "kind": kind,
            "id": entry_id,
            "terms": terms,
            "worlds": worlds,
            "leaks_held_out": sorted(set(worlds) & held),
        })

    critical = [f for f in findings if f["leaks_held_out"]]
    return {
        "n_flagged": len(findings),
        "n_critical": len(critical),
        "findings": findings,
        "held_out_worlds": sorted(held),
    }


def report(audit: dict, *, where: str = "bank") -> None:
    """Print an audit result. Loud for held-out leaks, quiet otherwise."""
    if not audit["n_flagged"]:
        print(f"  Law-form audit ({where}): clean.")
        return

    print(f"  Law-form audit ({where}): {audit['n_flagged']} entr(ies) name a "
          f"law form.")
    for f in audit["findings"]:
        tag = "LEAKS HELD-OUT" if f["leaks_held_out"] else "train-world only"
        worlds = ", ".join(f["worlds"]) or "unattributed"
        print(f"    [{tag}] {f['kind']} {f['id']}: "
              f"{', '.join(f['terms'])}  ({worlds})")
    if audit["n_critical"]:
        print(f"  [WARNING] {audit['n_critical']} entr(ies) name the answer of a "
              f"HELD-OUT world ({', '.join(audit['held_out_worlds'])}). The "
              f"transfer number from this bank is confounded — treat it as "
              f"invalid rather than as a result.")
